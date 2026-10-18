#pragma once
#include "tinyinfer/backend/backend.h"
#include "tinyinfer/runtime/device/device_runtime.h"
#include <utility>

namespace tinyinfer {
namespace detail { struct CudaRuntimeState; }
struct CudaTransferStatistics {
    std::size_t staging_allocations{}, staging_reuses{}, cached_bytes{};
};
struct CudaTensorView {
    TensorSpec spec;
    DeviceSpan storage;
    Strides strides;
    CudaTensorView(TensorSpec value, DeviceSpan span)
        : spec(std::move(value)), storage(std::move(span)), strides(TensorLayout(spec.shape).strides()) {}
};
struct CudaKernelStatistics {
    std::size_t dispatches{}, cublas_calls{}, kernel_launches{};
};
// One non-default ordered stream. External serialization is required. This
// runtime executes device views, not a host Tensor/Executor or prepared model.
class CudaDeviceRuntime final : public DeviceRuntime {
public:
    static std::vector<DeviceInfo> enumerate_devices();
    explicit CudaDeviceRuntime(std::uint32_t ordinal = 0);
    ~CudaDeviceRuntime() override;
    DeviceId device() const noexcept override;
    DeviceInfo device_info() const;
    std::shared_ptr<DeviceBuffer> allocate(std::size_t bytes,
        std::size_t alignment = kDefaultBufferAlignment) override;
    Completion upload(DeviceSpan, std::shared_ptr<const Buffer>, std::size_t = 0) override;
    Completion download(std::shared_ptr<Buffer>, DeviceSpan, std::size_t = 0) override;
    Completion copy(DeviceSpan, DeviceSpan) override;
    void synchronize() override;
    DeviceRuntimeStatistics statistics() const noexcept override;
    CudaTransferStatistics transfer_statistics() const noexcept;
    CudaKernelStatistics kernel_statistics() const noexcept;
    NodeCapability query_capability(const CapabilityQuery&) const;
    Completion execute(OpType, const std::vector<CudaTensorView>&,
                       const CudaTensorView&, NodeAttributes = {});
private:
    std::shared_ptr<detail::CudaRuntimeState> state_;
};
class CudaBackend final : public Backend {
public:
    explicit CudaBackend(std::shared_ptr<CudaDeviceRuntime>);
    std::string_view name() const noexcept override { return "cuda"; }
    bool supports(OpType) const noexcept override;
    DeviceInfo device_info() const override;
    NodeCapability query_capability(const CapabilityQuery&) const override;
    void execute(const Node&, ExecutionContext&) override;
private:
    std::shared_ptr<CudaDeviceRuntime> runtime_;
};
} // namespace tinyinfer
