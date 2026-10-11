#pragma once
#include <memory>
#include <vector>
#include "tinyinfer/runtime/device/device.h"
#include "tinyinfer/core/memory/buffer.h"

namespace tinyinfer {
// Separate from Tensor/Storage: native GPU handles belong to backend subclasses.
class DeviceBuffer {
public:
    DeviceBuffer(DeviceId device, std::size_t bytes, std::size_t alignment);
    virtual ~DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&&) = delete;
    DeviceBuffer& operator=(DeviceBuffer&&) = delete;
    DeviceId device() const noexcept { return device_; }
    std::size_t size_bytes() const noexcept { return bytes_; }
    std::size_t alignment() const noexcept { return alignment_; }
    // Only CPU resources can expose a host pointer; GPU resources always return null.
    void* host_data() noexcept;
    const void* host_data() const noexcept;
protected:
    virtual void* host_pointer() const noexcept { return nullptr; }
private:
    DeviceId device_;
    std::size_t bytes_, alignment_;
};
struct DeviceSpan {
    std::shared_ptr<DeviceBuffer> buffer;
    std::size_t offset{0}, bytes{0};
    void validate() const;
};

// Backend wait() must return/throw only after work is terminal and cannot access
// retained resources again. A failure is sticky: repeated wait() reports it again.
class CompletionState {
public:
    virtual ~CompletionState() = default;
    virtual bool ready() const noexcept = 0;
    virtual void wait() = 0;
};
class Completion {
public:
    Completion() = default; // Already-complete synchronous operation.
    Completion(std::shared_ptr<CompletionState> state,
               std::vector<std::shared_ptr<const void>> retained);
    ~Completion();
    Completion(Completion&& other) noexcept;
    Completion& operator=(Completion&& other) noexcept;
    Completion(const Completion&) = delete;
    Completion& operator=(const Completion&) = delete;
    bool ready() const noexcept;
    void wait(); // Required to observe execution failures before exposing output.
private:
    void drain_noexcept() noexcept;
    std::shared_ptr<CompletionState> state_;
    std::vector<std::shared_ptr<const void>> retained_;
};

// One ordered queue per runtime; async implementations must retain all operands
// and queue state in the returned completion. Host buffers must not be modified
// while queued copies are pending. Public inference must wait before returning.
struct DeviceRuntimeStatistics {
    // Successful logical requests and cumulative payload bytes, not resident memory.
    std::size_t allocations{0}, allocated_bytes{0};
    std::size_t uploads{0}, uploaded_bytes{0};
    std::size_t downloads{0}, downloaded_bytes{0};
    std::size_t copies{0}, copied_bytes{0};
};
class DeviceRuntime {
public:
    DeviceRuntime() = default;
    virtual ~DeviceRuntime() = default;
    DeviceRuntime(const DeviceRuntime&) = delete;
    DeviceRuntime& operator=(const DeviceRuntime&) = delete;
    virtual DeviceId device() const noexcept = 0;
    virtual std::shared_ptr<DeviceBuffer> allocate(std::size_t bytes,
        std::size_t alignment = kDefaultBufferAlignment) = 0;
    virtual Completion upload(DeviceSpan destination, std::shared_ptr<const Buffer> source,
                              std::size_t source_offset = 0) = 0;
    virtual Completion download(std::shared_ptr<Buffer> destination, DeviceSpan source,
                                std::size_t destination_offset = 0) = 0;
    virtual Completion copy(DeviceSpan destination, DeviceSpan source) = 0;
    virtual void synchronize() = 0;
    virtual DeviceRuntimeStatistics statistics() const noexcept = 0;
};
class CpuDeviceRuntime final : public DeviceRuntime {
public:
    explicit CpuDeviceRuntime(std::shared_ptr<Allocator> allocator = default_allocator());
    DeviceId device() const noexcept override { return {}; }
    std::shared_ptr<DeviceBuffer> allocate(std::size_t bytes,
        std::size_t alignment = kDefaultBufferAlignment) override;
    Completion upload(DeviceSpan, std::shared_ptr<const Buffer>, std::size_t = 0) override;
    Completion download(std::shared_ptr<Buffer>, DeviceSpan, std::size_t = 0) override;
    Completion copy(DeviceSpan, DeviceSpan) override;
    void synchronize() override {} // Copies execute inline.
    DeviceRuntimeStatistics statistics() const noexcept override { return statistics_; }
private:
    std::shared_ptr<Allocator> allocator_;
    DeviceRuntimeStatistics statistics_;
};
} // namespace tinyinfer
