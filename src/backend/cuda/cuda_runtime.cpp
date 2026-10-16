#include "tinyinfer/backend/cuda/cuda_runtime.h"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cstring>
#include <exception>
#include <functional>
#include <limits>
#include <stdexcept>
#include <utility>

namespace tinyinfer {
namespace {
void checked(cudaError_t status, const char* action) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(action) + ": " + cudaGetErrorString(status));
}
struct DeviceGuard {
    int previous{};
    explicit DeviceGuard(int ordinal) {
        checked(cudaGetDevice(&previous), "cudaGetDevice");
        checked(cudaSetDevice(ordinal), "cudaSetDevice");
    }
    ~DeviceGuard() { (void)cudaSetDevice(previous); }
};
struct CudaQueue {
    int ordinal;
    cudaStream_t stream{};
    std::exception_ptr failure;
    explicit CudaQueue(int id) : ordinal(id) {
        DeviceGuard guard(ordinal);
        checked(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
    }
    ~CudaQueue() {
        try { DeviceGuard guard(ordinal); (void)cudaStreamSynchronize(stream); (void)cudaStreamDestroy(stream); }
        catch (...) {}
    }
};
struct Stage {
    void* pointer{};
    std::size_t capacity{};
    bool leased{false};
    explicit Stage(std::size_t bytes) : capacity(bytes) {
        checked(cudaMallocHost(&pointer, bytes), "cudaMallocHost");
    }
    ~Stage() { (void)cudaFreeHost(pointer); }
};
struct StageLease {
    std::shared_ptr<Stage> stage;
    explicit StageLease(std::shared_ptr<Stage> value) : stage(std::move(value)) { stage->leased = true; }
    ~StageLease() { stage->leased = false; }
};
class CudaOperation final : public CompletionState {
public:
    std::shared_ptr<CudaQueue> queue;
    cudaEvent_t event{};
    bool recorded{false}, done{false};
    std::exception_ptr failure;
    std::function<void()> finish;
    CudaOperation(std::shared_ptr<CudaQueue> q, std::function<void()> callback)
        : queue(std::move(q)), finish(std::move(callback)) {
        DeviceGuard guard(queue->ordinal);
        checked(cudaEventCreateWithFlags(&event, cudaEventDisableTiming), "cudaEventCreate");
    }
    ~CudaOperation() {
        // The enclosing Completion/queue drains before releasing operands.
        try { wait(); DeviceGuard guard(queue->ordinal); (void)cudaEventDestroy(event); }
        catch (...) { try { DeviceGuard guard(queue->ordinal); (void)cudaEventDestroy(event); } catch (...) {} }
    }
    void fail(std::exception_ptr error) noexcept {
        failure = error; if (!queue->failure) queue->failure = error;
        finish = {}; done = true;
    }
    bool ready() const noexcept override {
        if (done) return true;
        if (!recorded) return false;
        try {
            DeviceGuard guard(queue->ordinal);
            const auto status = cudaEventQuery(event);
            if (status == cudaErrorNotReady) return false;
            if (status != cudaSuccess) {
                (void)cudaStreamSynchronize(queue->stream);
                try { checked(status, "CUDA completion query"); }
                catch (...) { const_cast<CudaOperation*>(this)->fail(std::current_exception()); }
                return done;
            }
            // wait() also performs download publication or records sticky failure.
            const_cast<CudaOperation*>(this)->wait();
        } catch (...) {}
        return done;
    }
    void wait() override {
        if (!done) {
            try {
                DeviceGuard guard(queue->ordinal);
                if (recorded) {
                    const auto status = cudaEventSynchronize(event);
                    if (status != cudaSuccess) {
                        // Drain before reporting terminal failure/releasing operands.
                        (void)cudaStreamSynchronize(queue->stream);
                        checked(status, "CUDA completion");
                    }
                    if (finish) finish();
                }
                finish = {}; done = true;
            } catch (...) { fail(std::current_exception()); }
        }
        if (failure) std::rethrow_exception(failure);
    }
};
class CudaAllocation final : public DeviceBuffer {
public:
    std::shared_ptr<CudaQueue> owner;
    void* pointer{};
    CudaAllocation(std::shared_ptr<CudaQueue> queue, std::size_t bytes, std::size_t alignment)
        : DeviceBuffer({DeviceKind::Cuda, static_cast<std::uint32_t>(queue->ordinal)}, bytes, alignment), owner(std::move(queue)) {
        DeviceGuard guard(owner->ordinal);
        if (bytes) checked(cudaMalloc(&pointer, bytes), "cudaMalloc");
    }
    ~CudaAllocation() override {
        try { DeviceGuard guard(owner->ordinal); if (pointer) (void)cudaFree(pointer); } catch (...) {}
    }
};
void range(std::size_t capacity, std::size_t offset, std::size_t bytes) {
    if (offset > capacity || bytes > capacity - offset) throw std::out_of_range("CUDA transfer exceeds capacity");
}
unsigned char* pointer(const std::shared_ptr<CudaAllocation>& a, std::size_t offset) {
    return static_cast<unsigned char*>(a->pointer) + offset; // Nonempty spans only.
}
} // namespace
namespace detail {
struct CudaRuntimeState {
    std::shared_ptr<CudaQueue> queue;
    DeviceInfo info;
    std::size_t max_allocation{};
    DeviceRuntimeStatistics stats;
    CudaTransferStatistics transfers;
    std::vector<std::shared_ptr<Stage>> stages;
    std::vector<std::shared_ptr<CudaOperation>> pending;
    void drain(bool all) {
        for (auto it = pending.begin(); it != pending.end();) {
            if (all || (*it)->ready()) {
                try { (*it)->wait(); } catch (...) { if (!queue->failure) queue->failure = std::current_exception(); }
                it = pending.erase(it);
            } else ++it;
        }
        if (queue->failure) std::rethrow_exception(queue->failure);
    }
    std::shared_ptr<StageLease> lease(std::size_t bytes) {
        for (auto& s : stages) if (!s->leased && s->capacity >= bytes) {
            ++transfers.staging_reuses; return std::make_shared<StageLease>(s);
        }
        auto stage = std::make_shared<Stage>(bytes);
        auto result = std::make_shared<StageLease>(stage);
        auto free = std::find_if(stages.begin(), stages.end(), [](const auto& s) { return !s->leased; });
        if (free != stages.end()) *free = stage;
        else if (stages.size() < 2) stages.push_back(stage);
        ++transfers.staging_allocations;
        return result;
    }
    ~CudaRuntimeState() { try { drain(true); } catch (...) {} }
};
} // namespace detail
namespace {
std::shared_ptr<CudaAllocation> allocation(const std::shared_ptr<detail::CudaRuntimeState>& state, const DeviceSpan& span) {
    span.validate();
    auto result = std::dynamic_pointer_cast<CudaAllocation>(span.buffer);
    if (!result || result->owner != state->queue) throw std::invalid_argument("foreign CUDA runtime/device allocation");
    return result;
}
template<class Enqueue>
Completion submit(const std::shared_ptr<detail::CudaRuntimeState>& state,
                  std::vector<std::shared_ptr<const void>> retained,
                  std::function<void()> finish, Enqueue enqueue) {
    // All throwing host bookkeeping precedes native submission. Completion owns
    // operands; operations own the queue independently, avoiding ownership cycles.
    auto operation = std::make_shared<CudaOperation>(state->queue, std::move(finish));
    state->pending.push_back(operation);
    Completion result(operation, std::move(retained));
    try {
        enqueue();
        checked(cudaEventRecord(operation->event, state->queue->stream), "cudaEventRecord");
        operation->recorded = true;
    } catch (...) {
        const auto error = std::current_exception();
        (void)cudaStreamSynchronize(state->queue->stream);
        operation->fail(error);
        std::rethrow_exception(error);
    }
    return result;
}
} // namespace
std::vector<DeviceInfo> CudaDeviceRuntime::enumerate_devices() {
    int count{};
    const auto status = cudaGetDeviceCount(&count);
    if (status == cudaErrorNoDevice) return {};
    checked(status, "cudaGetDeviceCount");
    std::vector<DeviceInfo> result;
    for (int i = 0; i < count; ++i) {
        cudaDeviceProp properties{}; checked(cudaGetDeviceProperties(&properties, i), "cudaGetDeviceProperties");
        result.push_back({{DeviceKind::Cuda, static_cast<std::uint32_t>(i)}, properties.name, true, {}});
    }
    return result;
}
CudaDeviceRuntime::CudaDeviceRuntime(std::uint32_t ordinal) {
    const auto devices = enumerate_devices();
    if (ordinal >= devices.size()) throw std::invalid_argument("CUDA device ordinal unavailable");
    auto state = std::make_shared<detail::CudaRuntimeState>();
    state->info = devices[ordinal]; state->queue = std::make_shared<CudaQueue>(static_cast<int>(ordinal));
    cudaDeviceProp properties{};
    checked(cudaGetDeviceProperties(&properties, static_cast<int>(ordinal)), "cudaGetDeviceProperties");
    state->max_allocation = properties.totalGlobalMem;
    state_ = std::move(state);
}
CudaDeviceRuntime::~CudaDeviceRuntime() { try { synchronize(); } catch (...) {} }
DeviceId CudaDeviceRuntime::device() const noexcept { return state_->info.id; }
DeviceInfo CudaDeviceRuntime::device_info() const { return state_->info; }
std::shared_ptr<DeviceBuffer> CudaDeviceRuntime::allocate(std::size_t bytes, std::size_t alignment) {
    if (!alignment || (alignment & (alignment - 1)) || alignment > 256)
        throw std::invalid_argument("CUDA alignment must be power-of-two <= 256");
    if (bytes > state_->max_allocation)
        throw std::out_of_range("CUDA allocation exceeds physical device capacity");
    state_->drain(false);
    auto result = std::make_shared<CudaAllocation>(state_->queue, bytes, alignment);
    ++state_->stats.allocations; state_->stats.allocated_bytes += bytes;
    return result;
}
Completion CudaDeviceRuntime::upload(DeviceSpan destination, std::shared_ptr<const Buffer> source, std::size_t offset) {
    auto dst = allocation(state_, destination);
    if (!source) throw std::invalid_argument("CUDA upload requires host ownership");
    range(source->size_bytes(), offset, destination.bytes);
    DeviceGuard guard(static_cast<int>(device().ordinal)); state_->drain(false);
    if (!destination.bytes) { ++state_->stats.uploads; return {}; }
    auto lease = state_->lease(destination.bytes);
    std::memcpy(lease->stage->pointer, static_cast<const unsigned char*>(source->data()) + offset, destination.bytes);
    auto result = submit(state_, {dst, source}, [lease] {}, [&] {
        checked(cudaMemcpyAsync(pointer(dst, destination.offset), lease->stage->pointer, destination.bytes,
                                cudaMemcpyHostToDevice, state_->queue->stream), "CUDA upload");
    });
    ++state_->stats.uploads; state_->stats.uploaded_bytes += destination.bytes; return result;
}
Completion CudaDeviceRuntime::download(std::shared_ptr<Buffer> destination, DeviceSpan source, std::size_t offset) {
    auto src = allocation(state_, source);
    if (!destination) throw std::invalid_argument("CUDA download requires host ownership");
    range(destination->size_bytes(), offset, source.bytes);
    DeviceGuard guard(static_cast<int>(device().ordinal)); state_->drain(false);
    if (!source.bytes) { ++state_->stats.downloads; return {}; }
    auto lease = state_->lease(source.bytes);
    auto result = submit(state_, {src, destination}, [destination, offset, lease, bytes=source.bytes] {
        std::memcpy(static_cast<unsigned char*>(destination->data()) + offset, lease->stage->pointer, bytes);
    }, [&] {
        checked(cudaMemcpyAsync(lease->stage->pointer, pointer(src, source.offset), source.bytes,
                                cudaMemcpyDeviceToHost, state_->queue->stream), "CUDA download");
    });
    ++state_->stats.downloads; state_->stats.downloaded_bytes += source.bytes; return result;
}
Completion CudaDeviceRuntime::copy(DeviceSpan destination, DeviceSpan source) {
    auto dst = allocation(state_, destination), src = allocation(state_, source);
    if (destination.bytes != source.bytes) throw std::invalid_argument("CUDA copy sizes differ");
    DeviceGuard guard(static_cast<int>(device().ordinal)); state_->drain(false);
    if (!source.bytes) { ++state_->stats.copies; return {}; }
    std::shared_ptr<CudaAllocation> scratch;
    if (dst == src && destination.offset < source.offset + source.bytes && source.offset < destination.offset + destination.bytes)
        scratch = std::make_shared<CudaAllocation>(state_->queue, source.bytes, kDefaultBufferAlignment);
    std::vector<std::shared_ptr<const void>> retained{dst, src}; if (scratch) retained.push_back(scratch);
    auto result = submit(state_, std::move(retained), {}, [&] {
        const void* from = pointer(src, source.offset);
        if (scratch) {
            checked(cudaMemcpyAsync(scratch->pointer, from, source.bytes, cudaMemcpyDeviceToDevice, state_->queue->stream), "CUDA overlap scratch");
            from = scratch->pointer;
        }
        checked(cudaMemcpyAsync(pointer(dst, destination.offset), from, source.bytes,
                                cudaMemcpyDeviceToDevice, state_->queue->stream), "CUDA copy");
    });
    ++state_->stats.copies; state_->stats.copied_bytes += source.bytes; return result;
}
void CudaDeviceRuntime::synchronize() {
    DeviceGuard guard(static_cast<int>(device().ordinal));
    const auto status = cudaStreamSynchronize(state_->queue->stream);
    if (status != cudaSuccess && !state_->queue->failure) {
        try { checked(status, "CUDA stream synchronize"); } catch (...) { state_->queue->failure = std::current_exception(); }
    }
    state_->drain(true);
}
DeviceRuntimeStatistics CudaDeviceRuntime::statistics() const noexcept { return state_->stats; }
CudaTransferStatistics CudaDeviceRuntime::transfer_statistics() const noexcept {
    auto result = state_->transfers;
    for (const auto& s : state_->stages) result.cached_bytes += s->capacity;
    return result;
}
} // namespace tinyinfer
