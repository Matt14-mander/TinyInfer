#pragma once
#include <utility>
#include "tinyinfer/backend/backend.h"
#include "tinyinfer/runtime/device/device_runtime.h"

namespace tinyinfer {
// An explicit device view, never a host Tensor. Physical layout is validated on dispatch.
struct MetalTensorView {
    TensorSpec spec;
    DeviceSpan storage;
    Strides strides;
    MetalTensorView(TensorSpec value_spec, DeviceSpan span)
        : spec(std::move(value_spec)), storage(std::move(span)), strides(TensorLayout(spec.shape).strides()) {}
};
struct MetalKernelStatistics {
    std::size_t dispatches{0};
    std::size_t pipeline_creations{0};
};
struct MetalTransferStatistics {
    std::size_t staging_allocations{0}, staging_reuses{0}, cached_bytes{0};
};
struct MetalCommandProfile {
    std::size_t commands{0}, timestamped_commands{0};
    double gpu_us{0}, host_wait_us{0};
};
namespace detail { struct MetalRuntimeState; }
class MetalDeviceRuntime final : public DeviceRuntime {
public:
    static std::vector<DeviceInfo> enumerate_devices();
    explicit MetalDeviceRuntime(std::uint32_t ordinal = 0);
    ~MetalDeviceRuntime() override;
    DeviceId device() const noexcept override;
    DeviceInfo device_info() const;
    std::shared_ptr<DeviceBuffer> allocate(std::size_t bytes,
        std::size_t alignment = kDefaultBufferAlignment) override;
    Completion upload(DeviceSpan, std::shared_ptr<const Buffer>, std::size_t = 0) override;
    Completion download(std::shared_ptr<Buffer>, DeviceSpan, std::size_t = 0) override;
    Completion copy(DeviceSpan, DeviceSpan) override;
    void synchronize() override;
    DeviceRuntimeStatistics statistics() const noexcept override;
    MetalKernelStatistics kernel_statistics() const noexcept;
    MetalTransferStatistics transfer_statistics() const noexcept;
    // Optional diagnostics; leave disabled for acceptance timing.
    void enable_profiling(bool);
    MetalCommandProfile command_profile() const noexcept;
    NodeCapability query_capability(const CapabilityQuery&) const;
    // A single device operation, not a model executor. No host fallback on failure.
    Completion execute(OpType, const std::vector<MetalTensorView>& inputs,
                       const MetalTensorView& output, NodeAttributes attributes = {});
private:
    std::shared_ptr<detail::MetalRuntimeState> state_;
};
// Capability bridge for preparation. The legacy host Executor is explicitly rejected.
class MetalBackend final : public Backend {
public:
    explicit MetalBackend(std::shared_ptr<MetalDeviceRuntime> runtime);
    std::string_view name() const noexcept override { return "metal"; }
    bool supports(OpType) const noexcept override;
    DeviceInfo device_info() const override;
    NodeCapability query_capability(const CapabilityQuery&) const override;
    void execute(const Node&, ExecutionContext&) override;
private:
    std::shared_ptr<MetalDeviceRuntime> runtime_;
};
} // namespace tinyinfer
