#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "tinyinfer/backend/metal/metal_runtime.h"
#include "tinyinfer/backend/cpu_backend.h"
#include "tinyinfer/ops/operator_schema.h"
#include "kernels.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <functional>
#include <exception>
#include <limits>
#include <stdexcept>
#include <utility>

namespace tinyinfer {
namespace {
std::string message(NSError* error) {
    return error ? std::string(error.localizedDescription.UTF8String) : "unknown Metal error";
}
class MetalOperation final : public CompletionState {
public:
    MetalOperation(id<MTLCommandBuffer> command, id<MTLDevice> device,
                   id<MTLCommandQueue> queue, std::function<void()> finish = {})
        : command_(command), device_(device), queue_(queue), finish_(std::move(finish)) {}
    bool ready() const noexcept override {
        if (done_) return true;
        const auto status = command_.status;
        if (status != MTLCommandBufferStatusCompleted && status != MTLCommandBufferStatusError) return false;
        // Complete any host-download publication before advertising readiness.
        try { const_cast<MetalOperation*>(this)->wait(); } catch (...) {}
        return done_;
    }
    void wait() override {
        if (!done_) {
            [command_ waitUntilCompleted];
            done_ = true;
            try {
                if (command_.status == MTLCommandBufferStatusError)
                    throw std::runtime_error("Metal execution failed: " + message(command_.error));
                if (command_.status != MTLCommandBufferStatusCompleted)
                    throw std::runtime_error("Metal command did not complete");
                if (finish_) finish_();
            } catch (...) { failure_ = std::current_exception(); }
            finish_ = {};
        }
        if (failure_) std::rethrow_exception(failure_);
    }
private:
    id<MTLCommandBuffer> command_;
    [[maybe_unused]] id<MTLDevice> device_;
    [[maybe_unused]] id<MTLCommandQueue> queue_;
    std::function<void()> finish_;
    bool done_{false};
    std::exception_ptr failure_;
};
bool kernel_supported(OpType op) noexcept {
    return op == OpType::Add || op == OpType::ReLU || op == OpType::Tanh ||
           op == OpType::MatMul || op == OpType::Gemm || op == OpType::FusedGemmActivation || op == OpType::Softmax;
}
void range(std::size_t size, std::size_t offset, std::size_t bytes) {
    if (offset > size || bytes > size - offset) throw std::out_of_range("Metal copy range exceeds capacity");
}
struct Params {
    std::uint32_t count{}, rank{}, m{}, k{}, n{}, trans_a{}, trans_b{}, has_bias{}, relu{};
    std::uint32_t dims[4]{}, a_strides[4]{}, b_strides[4]{};
    std::uint32_t bias_rows{1}, bias_cols{1}, inner{1}, axis_size{1}, groups{};
    float alpha{1}, beta{1};
};
static_assert(sizeof(Params) == 112, "Metal parameter ABI changed");
template<class T> T attr(const NodeAttributes& attrs, const char* name, T fallback) {
    auto found = attrs.find(name); return found == attrs.end() ? fallback : std::get<T>(found->second);
}
} // namespace
namespace detail {
struct MetalRuntimeState {
    DeviceId identity;
    id<MTLDevice> native_device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;
    std::array<id<MTLComputePipelineState>, 5> pipelines;
    id<MTLBuffer> dummy;
    std::vector<std::shared_ptr<MetalOperation>> pending;
    DeviceRuntimeStatistics statistics;
    MetalKernelStatistics kernels;
    std::exception_ptr failure;
    void drain(bool wait_all) {
        for (auto it = pending.begin(); it != pending.end();) {
            auto operation = *it;
            if (wait_all || operation->ready()) {
                try { operation->wait(); }
                catch (...) { if (!failure) failure = std::current_exception(); }
                it = pending.erase(it);
            } else ++it;
        }
        if (failure) std::rethrow_exception(failure);
    }
    ~MetalRuntimeState() { try { drain(true); } catch (...) {} }
};
} // namespace detail
namespace {
class MetalAllocation final : public DeviceBuffer {
public:
    MetalAllocation(std::shared_ptr<detail::MetalRuntimeState> state, std::size_t bytes, std::size_t alignment)
        : DeviceBuffer(state->identity, bytes, alignment), owner(std::move(state)) {
        // Zero-size views still have a valid native dummy allocation for zero-K bindings.
        native = [owner->native_device newBufferWithLength:std::max<std::size_t>(bytes, 4)
                                                  options:MTLResourceStorageModePrivate];
        if (!native) throw std::runtime_error("Metal device allocation failed");
    }
    std::shared_ptr<detail::MetalRuntimeState> owner;
    id<MTLBuffer> native;
};
std::shared_ptr<MetalAllocation> allocation(const std::shared_ptr<detail::MetalRuntimeState>& state,
                                           const DeviceSpan& span, bool shader = false) {
    span.validate();
    auto result = std::dynamic_pointer_cast<MetalAllocation>(span.buffer);
    if (!result || result->owner != state || span.buffer->device() != state->identity)
        throw std::invalid_argument("resource belongs to another Metal runtime/device");
    if (shader && span.offset % alignof(float)) throw std::invalid_argument("FP32 binding offset must be aligned");
    return result;
}
id<MTLCommandBuffer> command(const std::shared_ptr<detail::MetalRuntimeState>& state) {
    state->drain(false);
    auto result = [state->queue commandBuffer];
    if (!result) throw std::runtime_error("Metal command buffer creation failed");
    return result;
}
Completion submit(const std::shared_ptr<detail::MetalRuntimeState>& state, id<MTLCommandBuffer> cb,
                  std::vector<std::shared_ptr<const void>> retained, std::function<void()> finish = {}) {
    auto operation = std::make_shared<MetalOperation>(cb, state->native_device, state->queue, std::move(finish));
    // Allocate all C++ bookkeeping before commit; exceptions cannot leave unowned work.
    state->pending.push_back(operation);
    Completion completion(operation, std::move(retained));
    [cb commit];
    return completion;
}
void blit(id<MTLCommandBuffer> cb, id<MTLBuffer> destination, std::size_t destination_offset,
          id<MTLBuffer> source, std::size_t source_offset, std::size_t bytes) {
    auto encoder = [cb blitCommandEncoder];
    if (!encoder) throw std::runtime_error("Metal blit encoder creation failed");
    [encoder copyFromBuffer:source sourceOffset:source_offset toBuffer:destination
         destinationOffset:destination_offset size:bytes];
    [encoder endEncoding];
}
void broadcast_strides(const TensorSpec& input, const Shape& output, std::uint32_t* result) {
    auto strides = TensorLayout(input.shape).strides();
    const auto padding = output.size() - input.shape.size();
    for (std::size_t i = 0; i < input.shape.size(); ++i)
        result[padding + i] = input.shape[i] == 1 ? 0 : static_cast<std::uint32_t>(strides[i]);
}
} // namespace
std::vector<DeviceInfo> MetalDeviceRuntime::enumerate_devices() {
    @autoreleasepool {
        std::vector<DeviceInfo> result;
        for (id<MTLDevice> device in MTLCopyAllDevices())
            result.push_back({{DeviceKind::Metal, static_cast<std::uint32_t>(result.size())},
                              std::string(device.name.UTF8String), true, {}});
        return result;
    }
}
MetalDeviceRuntime::MetalDeviceRuntime(std::uint32_t ordinal) {
    @autoreleasepool {
        auto devices = MTLCopyAllDevices();
        if (ordinal >= devices.count) throw std::runtime_error("requested Metal device is unavailable");
        auto state = std::make_shared<detail::MetalRuntimeState>();
        state->identity = {DeviceKind::Metal, ordinal}; state->native_device = devices[ordinal];
        state->queue = [state->native_device newCommandQueue];
        if (!state->queue) throw std::runtime_error("Metal command queue creation failed");
        auto options = [MTLCompileOptions new];
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
        if (@available(macOS 15.0, *)) {
            options.mathMode = MTLMathModeSafe;
            options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
        } else
#endif
        {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            options.fastMathEnabled = NO;
#pragma clang diagnostic pop
        }
        NSError* error = nil;
        state->library = [state->native_device newLibraryWithSource:@(detail::metal_kernel_source)
                                                          options:options error:&error];
        if (!state->library) throw std::runtime_error("Metal shader compilation failed: " + message(error));
        const char* names[] = {"add_f32", "relu_f32", "tanh_f32", "gemm_f32", "softmax_f32"};
        for (std::size_t i = 0; i < state->pipelines.size(); ++i) {
            auto function = [state->library newFunctionWithName:@(names[i])];
            if (!function) throw std::runtime_error("Metal kernel function missing");
            state->pipelines[i] = [state->native_device newComputePipelineStateWithFunction:function error:&error];
            if (!state->pipelines[i]) throw std::runtime_error("Metal pipeline creation failed: " + message(error));
            ++state->kernels.pipeline_creations;
        }
        state->dummy = [state->native_device newBufferWithLength:4 options:MTLResourceStorageModePrivate];
        if (!state->dummy) throw std::runtime_error("Metal dummy allocation failed");
        state_ = std::move(state);
    }
}
MetalDeviceRuntime::~MetalDeviceRuntime() = default;
DeviceId MetalDeviceRuntime::device() const noexcept { return state_->identity; }
DeviceInfo MetalDeviceRuntime::device_info() const { return {device(), state_->native_device.name.UTF8String, true, {}}; }
DeviceRuntimeStatistics MetalDeviceRuntime::statistics() const noexcept { return state_->statistics; }
MetalKernelStatistics MetalDeviceRuntime::kernel_statistics() const noexcept { return state_->kernels; }
void MetalDeviceRuntime::synchronize() { state_->drain(true); }
std::shared_ptr<DeviceBuffer> MetalDeviceRuntime::allocate(std::size_t bytes, std::size_t alignment) {
    @autoreleasepool {
        state_->drain(false);
        if (alignment > 256) throw std::invalid_argument("Metal allocation alignment exceeds 256");
        if (std::max<std::size_t>(bytes, 4) > state_->native_device.maxBufferLength)
            throw std::invalid_argument("Metal allocation exceeds device maxBufferLength");
        auto result = std::make_shared<MetalAllocation>(state_, bytes, alignment);
        ++state_->statistics.allocations; state_->statistics.allocated_bytes += bytes;
        return result;
    }
}
Completion MetalDeviceRuntime::upload(DeviceSpan destination, std::shared_ptr<const Buffer> source,
                                      std::size_t source_offset) {
    @autoreleasepool {
        auto dst = allocation(state_, destination);
        if (!source) throw std::invalid_argument("Metal upload requires owned host buffer");
        range(source->size_bytes(), source_offset, destination.bytes);
        state_->drain(false);
        if (!destination.bytes) { ++state_->statistics.uploads; return {}; }
        auto staging = [state_->native_device newBufferWithBytes:static_cast<const unsigned char*>(source->data()) + source_offset
                                                        length:destination.bytes options:MTLResourceStorageModeShared];
        if (!staging) throw std::runtime_error("Metal upload staging allocation failed");
        auto cb = command(state_); blit(cb, dst->native, destination.offset, staging, 0, destination.bytes);
        auto completion = submit(state_, cb, {dst, source});
        ++state_->statistics.uploads; state_->statistics.uploaded_bytes += destination.bytes;
        return completion;
    }
}
Completion MetalDeviceRuntime::download(std::shared_ptr<Buffer> destination, DeviceSpan source,
                                        std::size_t destination_offset) {
    @autoreleasepool {
        auto src = allocation(state_, source);
        if (!destination) throw std::invalid_argument("Metal download requires owned host buffer");
        range(destination->size_bytes(), destination_offset, source.bytes);
        state_->drain(false);
        if (!source.bytes) { ++state_->statistics.downloads; return {}; }
        auto staging = [state_->native_device newBufferWithLength:source.bytes options:MTLResourceStorageModeShared];
        if (!staging) throw std::runtime_error("Metal download staging allocation failed");
        auto cb = command(state_); blit(cb, staging, 0, src->native, source.offset, source.bytes);
        auto finish = [destination, destination_offset, staging, bytes = source.bytes] {
            std::memcpy(static_cast<unsigned char*>(destination->data()) + destination_offset, staging.contents, bytes);
        };
        auto completion = submit(state_, cb, {src, destination}, std::move(finish));
        ++state_->statistics.downloads; state_->statistics.downloaded_bytes += source.bytes;
        return completion;
    }
}
Completion MetalDeviceRuntime::copy(DeviceSpan destination, DeviceSpan source) {
    @autoreleasepool {
        auto dst = allocation(state_, destination), src = allocation(state_, source);
        if (destination.bytes != source.bytes) throw std::invalid_argument("Metal copy sizes differ");
        state_->drain(false);
        if (!source.bytes) { ++state_->statistics.copies; return {}; }
        auto cb = command(state_);
        // Scratch storage provides memmove semantics for overlapping ranges.
        if (dst == src && destination.offset < source.offset + source.bytes && source.offset < destination.offset + destination.bytes) {
            auto scratch = [state_->native_device newBufferWithLength:source.bytes options:MTLResourceStorageModePrivate];
            if (!scratch) throw std::runtime_error("Metal overlap-copy staging allocation failed");
            blit(cb, scratch, 0, src->native, source.offset, source.bytes);
            blit(cb, dst->native, destination.offset, scratch, 0, source.bytes);
        } else blit(cb, dst->native, destination.offset, src->native, source.offset, source.bytes);
        auto completion = submit(state_, cb, {dst, src});
        ++state_->statistics.copies; state_->statistics.copied_bytes += source.bytes;
        return completion;
    }
}
NodeCapability MetalDeviceRuntime::query_capability(const CapabilityQuery& query) const {
    if (query.device != device()) return {false, false, "Metal query targets another device"};
    if (!kernel_supported(query.node.op)) return {false, false, "operator has no Metal kernel"};
    auto cpu_query = query; cpu_query.device = {};
    auto result = CpuBackend().query_capability(cpu_query);
    if (!result.supported) return result;
    bool runtime_check = false;
    auto check = [&](const TensorDescriptor& descriptor) -> std::string {
        const auto layout = TensorLayout(descriptor.spec.shape);
        if (layout.rank() > 4) return "Metal initial kernel rank limit is four";
        if (layout.numel() > std::numeric_limits<std::uint32_t>::max() ||
            layout.size_bytes(descriptor.spec.dtype) > state_->native_device.maxBufferLength)
            return "Metal shape exceeds kernel/device size limit";
        for (auto dim : descriptor.spec.shape)
            if (dim > std::numeric_limits<std::uint32_t>::max()) return "Metal dimension exceeds uint32";
        if (!descriptor.strides) runtime_check = true;
        else if (!TensorLayout(descriptor.spec.shape, *descriptor.strides).is_contiguous())
            return "Metal initial kernels require contiguous physical layout";
        return {};
    };
    for (const auto& descriptor : query.inputs) {
        auto reason = check(descriptor); if (!reason.empty()) return {false, false, reason};
    }
    for (const auto& descriptor : query.outputs) {
        auto reason = check(descriptor); if (!reason.empty()) return {false, false, reason};
    }
    return {true, runtime_check, "Metal FP32 kernel supported; validate runtime spans/layouts"};
}
Completion MetalDeviceRuntime::execute(OpType op, const std::vector<MetalTensorView>& inputs,
                                       const MetalTensorView& output, NodeAttributes attributes) {
    @autoreleasepool {
        Node node; node.op = op; node.attributes = attributes;
        node.inputs.resize(inputs.size()); node.outputs.resize(1);
        CapabilityQuery query{device(), node, {}, {{output.spec, output.strides}}};
        std::vector<std::shared_ptr<MetalAllocation>> allocations;
        std::vector<std::shared_ptr<const void>> retained;
        for (const auto& input : inputs) {
            query.inputs.push_back({input.spec, input.strides});
            allocations.push_back(allocation(state_, input.storage, true)); retained.push_back(input.storage.buffer);
            if (input.storage.bytes != TensorLayout(input.spec.shape).size_bytes(input.spec.dtype))
                throw std::invalid_argument("Metal input view byte size differs from spec");
        }
        auto dst = allocation(state_, output.storage, true); retained.push_back(dst);
        auto capability = query_capability(query);
        if (!capability.supported) throw std::invalid_argument(capability.reason);
        const auto layout = TensorLayout(output.spec.shape);
        if (output.storage.bytes != layout.size_bytes(output.spec.dtype))
            throw std::invalid_argument("Metal output view byte size differs from spec");
        for (const auto& input : inputs) {
            if (input.storage.buffer == output.storage.buffer && input.storage.bytes && output.storage.bytes &&
                input.storage.offset < output.storage.offset + output.storage.bytes &&
                output.storage.offset < input.storage.offset + input.storage.bytes)
                throw std::invalid_argument("Metal kernel output must not alias input");
        }
        state_->drain(false);
        if (!layout.numel()) return {};
        Params p; p.count = static_cast<std::uint32_t>(layout.numel()); p.rank = layout.rank();
        for (std::size_t i = 0; i < layout.rank(); ++i) p.dims[i] = output.spec.shape[i];
        std::size_t pipeline = 0, threads = p.count;
        if (op == OpType::Add) {
            broadcast_strides(inputs[0].spec, output.spec.shape, p.a_strides);
            broadcast_strides(inputs[1].spec, output.spec.shape, p.b_strides);
        } else if (op == OpType::ReLU) pipeline = 1;
        else if (op == OpType::Tanh) pipeline = 2;
        else if (op == OpType::Softmax) {
            pipeline = 4;
            auto axis = attr<std::int64_t>(attributes, "axis", -1);
            if (axis < 0) axis += output.spec.shape.size();
            p.axis_size = output.spec.shape[axis];
            for (std::size_t i = axis + 1; i < layout.rank(); ++i) p.inner *= output.spec.shape[i];
            p.groups = p.count / p.axis_size; threads = p.groups;
        } else {
            pipeline = 3;
            p.trans_a = attr<std::int64_t>(attributes, "transA", 0);
            p.trans_b = attr<std::int64_t>(attributes, "transB", 0);
            p.m = output.spec.shape[0]; p.n = output.spec.shape[1];
            p.k = inputs[0].spec.shape[p.trans_a ? 0 : 1];
            p.alpha = attr<float>(attributes, "alpha", 1); p.beta = attr<float>(attributes, "beta", 1);
            p.relu = op == OpType::FusedGemmActivation;
            p.has_bias = inputs.size() == 3;
            if (p.has_bias) {
                const auto& shape = inputs[2].spec.shape;
                if (shape.size() == 2) p.bias_rows = shape[0];
                if (!shape.empty()) p.bias_cols = shape.back();
            }
        }
        auto cb = command(state_);
        auto encoder = [cb computeCommandEncoder];
        if (!encoder) throw std::runtime_error("Metal compute encoder creation failed");
        auto kernel = state_->pipelines[pipeline];
        [encoder setComputePipelineState:kernel];
        [encoder setBuffer:inputs[0].storage.bytes ? allocations[0]->native : state_->dummy
                     offset:inputs[0].storage.bytes ? inputs[0].storage.offset : 0 atIndex:0];
        const bool second_has_storage = inputs.size() >= 2 && inputs[1].storage.bytes;
        [encoder setBuffer:second_has_storage ? allocations[1]->native : state_->dummy
                     offset:second_has_storage ? inputs[1].storage.offset : 0 atIndex:1];
        [encoder setBuffer:dst->native offset:output.storage.offset atIndex:2];
        [encoder setBytes:&p length:sizeof(p) atIndex:3];
        const bool bias_has_storage = p.has_bias && inputs[2].storage.bytes;
        [encoder setBuffer:bias_has_storage ? allocations[2]->native : state_->dummy
                     offset:bias_has_storage ? inputs[2].storage.offset : 0 atIndex:4];
        const auto width = std::min<NSUInteger>(64, kernel.maxTotalThreadsPerThreadgroup);
        // Uniform groups work on Intel Mac GPUs too; kernels guard padded lanes.
        [encoder dispatchThreadgroups:MTLSizeMake((threads + width - 1) / width, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
        [encoder endEncoding];
        auto completion = submit(state_, cb, std::move(retained));
        ++state_->kernels.dispatches;
        return completion;
    }
}
MetalBackend::MetalBackend(std::shared_ptr<MetalDeviceRuntime> runtime) : runtime_(std::move(runtime)) {
    if (!runtime_) throw std::invalid_argument("Metal backend requires runtime");
}
bool MetalBackend::supports(OpType op) const noexcept { return kernel_supported(op); }
DeviceInfo MetalBackend::device_info() const { return runtime_->device_info(); }
NodeCapability MetalBackend::query_capability(const CapabilityQuery& query) const { return runtime_->query_capability(query); }
void MetalBackend::execute(const Node&, ExecutionContext&) {
    throw std::runtime_error("Metal requires device views; host Executor is not a GPU model execution plan");
}
} // namespace tinyinfer
