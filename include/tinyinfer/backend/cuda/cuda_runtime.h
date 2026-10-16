#pragma once
#include "tinyinfer/runtime/device/device_runtime.h"

namespace tinyinfer {
namespace detail { struct CudaRuntimeState; }
struct CudaTransferStatistics {
    std::size_t staging_allocations{}, staging_reuses{}, cached_bytes{};
};
// One non-default ordered stream. External serialization is required. This
// first slice owns resources/transfers; it is not yet a CUDA operator backend.
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
private:
    std::shared_ptr<detail::CudaRuntimeState> state_;
};
} // namespace tinyinfer
