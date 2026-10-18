#include "tinyinfer/backend/cuda/cuda_runtime.h"
#include "tinyinfer/backend/cuda/operator_contract.h"
#include "kernels.h"
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
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
void checked(cublasStatus_t status, const char* action) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(action) + ": cuBLAS status " + std::to_string(static_cast<int>(status)));
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
    cublasHandle_t blas{};
    std::exception_ptr failure;
    explicit CudaQueue(int id) : ordinal(id) {
        DeviceGuard guard(ordinal);
        checked(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
        try {
            checked(cublasCreate(&blas), "cublasCreate");
            checked(cublasSetStream(blas, stream), "cublasSetStream");
            checked(cublasSetPointerMode(blas, CUBLAS_POINTER_MODE_HOST), "cublasSetPointerMode");
            checked(cublasSetMathMode(blas, CUBLAS_PEDANTIC_MATH), "cublasSetMathMode");
            checked(cublasSetAtomicsMode(blas, CUBLAS_ATOMICS_NOT_ALLOWED), "cublasSetAtomicsMode");
        } catch (...) {
            if (blas) (void)cublasDestroy(blas);
            (void)cudaStreamDestroy(stream); throw;
        }
    }
    ~CudaQueue() {
        try { DeviceGuard guard(ordinal); (void)cudaStreamSynchronize(stream); (void)cublasDestroy(blas); (void)cudaStreamDestroy(stream); }
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
    CudaKernelStatistics kernels;
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
CudaKernelStatistics CudaDeviceRuntime::kernel_statistics() const noexcept { return state_->kernels; }
NodeCapability CudaDeviceRuntime::query_capability(const CapabilityQuery& query) const {
    if (query.device != device()) return {false, false, "CUDA query targets another device"};
    auto result = cuda::validate_operator_contract(query);
    if (!result.supported) return result;
    for (const auto* list : {&query.inputs, &query.outputs}) for (const auto& value : *list)
        if (TensorLayout(value.spec.shape).size_bytes(value.spec.dtype) > state_->max_allocation)
            return {false, false, "CUDA tensor exceeds physical device capacity"};
    result.reason = "CUDA FP32 device kernel supported; runtime spans/layouts must be validated";
    return result;
}
Completion CudaDeviceRuntime::execute(OpType op, const std::vector<CudaTensorView>& inputs,
                                     const CudaTensorView& output, NodeAttributes attributes) {
    Node node; node.op = op; node.attributes = std::move(attributes);
    node.inputs.resize(inputs.size()); node.outputs.resize(1);
    CapabilityQuery query{device(), node, {}, {{output.spec, output.strides}}};
    std::vector<std::shared_ptr<CudaAllocation>> allocations;
    std::vector<std::shared_ptr<const void>> retained;
    auto validate = [&](const CudaTensorView& view) {
        auto buffer = allocation(state_, view.storage);
        if (view.storage.offset % sizeof(float) ||
            view.storage.bytes != TensorLayout(view.spec.shape).size_bytes(view.spec.dtype))
            throw std::invalid_argument("CUDA view requires aligned offset and exact spec byte size");
        retained.push_back(buffer); return buffer;
    };
    for (const auto& input : inputs) {
        query.inputs.push_back({input.spec, input.strides}); allocations.push_back(validate(input));
    }
    auto dst = validate(output);
    const auto capability = query_capability(query);
    if (!capability.supported) throw std::invalid_argument(capability.reason);
    for (const auto& input : inputs)
        if (input.storage.buffer == output.storage.buffer && input.storage.bytes && output.storage.bytes &&
            input.storage.offset < output.storage.offset + output.storage.bytes &&
            output.storage.offset < input.storage.offset + input.storage.bytes)
            throw std::invalid_argument("CUDA kernel output must not alias input");
    DeviceGuard guard(static_cast<int>(device().ordinal)); state_->drain(false);
    const TensorLayout layout(output.spec.shape);
    if (!layout.numel()) return {};
    cuda::native::Params p; p.count = static_cast<int>(layout.numel()); p.rank = static_cast<int>(layout.rank());
    for (int i = 0; i < p.rank; ++i) p.dims[i] = static_cast<int>(output.spec.shape[i]);
    auto kind = cuda::native::Kernel::Epilogue;
    const bool matrix = op == OpType::MatMul || op == OpType::Gemm || op == OpType::FusedGemmActivation;
    cuda::GemmDescriptor d;
    if (matrix) {
        d = cuda::make_gemm_descriptor(query);
        p.columns = d.columns; p.bias_rows = d.bias_rows; p.bias_columns = d.bias_columns;
        p.zero_k = d.work == cuda::GemmWork::ZeroK; p.has_bias = d.has_bias; p.relu = d.relu;
        p.alpha = d.epilogue_alpha; p.beta = d.epilogue_beta;
    } else if (op == OpType::Add) {
        kind = cuda::native::Kernel::Add;
        auto broadcast = [&](const CudaTensorView& view, int* strides) {
            const auto shift = layout.rank() - view.spec.shape.size();
            for (std::size_t i = 0; i < view.spec.shape.size(); ++i)
                if (view.spec.shape[i] != 1) strides[shift+i] = static_cast<int>(view.strides[i]);
        };
        broadcast(inputs[0], p.a_strides); broadcast(inputs[1], p.b_strides);
    } else if (op == OpType::ReLU) kind = cuda::native::Kernel::Relu;
    else if (op == OpType::Tanh) kind = cuda::native::Kernel::Tanh;
    else {
        kind = cuda::native::Kernel::Softmax;
        auto axis = std::int64_t{-1}; const auto found = node.attributes.find("axis");
        if (found != node.attributes.end()) axis = std::get<std::int64_t>(found->second);
        if (axis < 0) axis += static_cast<std::int64_t>(layout.rank());
        p.axis_size = p.dims[axis];
        for (int i = static_cast<int>(axis) + 1; i < p.rank; ++i) p.inner *= p.dims[i];
        p.groups = p.count / p.axis_size;
    }
    auto input_pointer = [&](std::size_t i) -> const float* {
        return i < inputs.size() && inputs[i].storage.bytes
            ? reinterpret_cast<const float*>(pointer(allocations[i], inputs[i].storage.offset)) : nullptr;
    };
    auto* out = reinterpret_cast<float*>(pointer(dst, output.storage.offset));
    const bool product = matrix && d.work == cuda::GemmWork::MatrixProduct;
    auto result = submit(state_, std::move(retained), {}, [&] {
        if (product) {
            checked(cublasSgemm(state_->queue->blas, d.transpose_left ? CUBLAS_OP_T : CUBLAS_OP_N,
                d.transpose_right ? CUBLAS_OP_T : CUBLAS_OP_N, d.native_m, d.native_n, d.native_k,
                &d.product_alpha, input_pointer(d.left_input), d.lda, input_pointer(d.right_input), d.ldb,
                &d.product_beta, out, d.ldc), "CUDA SGEMM");
            ++state_->kernels.cublas_calls;
        }
        {
            checked(cuda::native::launch(kind, input_pointer(0), input_pointer(1), input_pointer(2),
                                        out, p, state_->queue->stream), "CUDA kernel launch");
            ++state_->kernels.kernel_launches;
        }
    });
    ++state_->kernels.dispatches; return result;
}
CudaBackend::CudaBackend(std::shared_ptr<CudaDeviceRuntime> runtime) : runtime_(std::move(runtime)) {
    if (!runtime_) throw std::invalid_argument("CUDA backend requires runtime");
}
bool CudaBackend::supports(OpType op) const noexcept {
    return op == OpType::MatMul || op == OpType::Gemm || op == OpType::FusedGemmActivation ||
           op == OpType::Add || op == OpType::ReLU || op == OpType::Tanh || op == OpType::Softmax;
}
DeviceInfo CudaBackend::device_info() const { return runtime_->device_info(); }
NodeCapability CudaBackend::query_capability(const CapabilityQuery& query) const { return runtime_->query_capability(query); }
void CudaBackend::execute(const Node&, ExecutionContext&) {
    throw std::runtime_error("CUDA requires device views; host Executor is not a GPU model execution plan");
}
} // namespace tinyinfer
