#include "tinyinfer/runtime/device/device_runtime.h"
#include <cstring>
#include <stdexcept>
#include <utility>

namespace tinyinfer {
std::string_view device_kind_name(DeviceKind kind) {
    switch (kind) {
    case DeviceKind::Cpu: return "cpu";
    case DeviceKind::Cuda: return "cuda";
    case DeviceKind::Metal: return "metal";
    }
    throw std::invalid_argument("unknown device kind");
}
void validate_device_id(DeviceId device) {
    (void)device_kind_name(device.kind);
    if (device.kind == DeviceKind::Cpu && device.ordinal != 0)
        throw std::invalid_argument("CPU device ordinal must be zero");
}
DeviceBuffer::DeviceBuffer(DeviceId device, std::size_t bytes, std::size_t alignment)
    : device_(device), bytes_(bytes), alignment_(alignment) {
    validate_device_id(device);
    if (!alignment || (alignment & (alignment - 1)))
        throw std::invalid_argument("device allocation alignment must be a power of two");
}
void* DeviceBuffer::host_data() noexcept {
    return device_.kind == DeviceKind::Cpu ? host_pointer() : nullptr;
}
const void* DeviceBuffer::host_data() const noexcept {
    return device_.kind == DeviceKind::Cpu ? host_pointer() : nullptr;
}
namespace {
void range(std::size_t capacity, std::size_t offset, std::size_t bytes) {
    if (offset > capacity || bytes > capacity - offset)
        throw std::out_of_range("copy range exceeds buffer capacity");
}
class CpuDeviceBuffer final : public DeviceBuffer {
public:
    CpuDeviceBuffer(std::size_t bytes, std::size_t alignment, std::shared_ptr<Allocator> allocator)
        : DeviceBuffer({}, bytes, alignment), storage_(bytes, std::move(allocator), alignment) {}
protected:
    void* host_pointer() const noexcept override { return const_cast<void*>(storage_.data()); }
private:
    Buffer storage_;
};
void validate_cpu_span(const DeviceSpan& span) {
    span.validate();
    if (span.buffer->device() != DeviceId{})
        throw std::invalid_argument("CPU queue cannot copy a foreign device resource");
    if (span.bytes && !span.buffer->host_data())
        throw std::invalid_argument("CPU resource has no host mapping");
}
unsigned char* pointer(const DeviceSpan& span) {
    return static_cast<unsigned char*>(span.buffer->host_data()) + span.offset;
}
} // namespace
void DeviceSpan::validate() const {
    if (!buffer) throw std::invalid_argument("device span requires a buffer");
    range(buffer->size_bytes(), offset, bytes);
}
Completion::Completion(std::shared_ptr<CompletionState> state,
                       std::vector<std::shared_ptr<const void>> retained)
    : state_(std::move(state)), retained_(std::move(retained)) {
    if (!state_) throw std::invalid_argument("pending completion requires state");
}
Completion::~Completion() { drain_noexcept(); }
Completion::Completion(Completion&& other) noexcept
    : state_(std::move(other.state_)), retained_(std::move(other.retained_)) {}
Completion& Completion::operator=(Completion&& other) noexcept {
    if (this != &other) {
        drain_noexcept();
        state_ = std::move(other.state_);
        retained_ = std::move(other.retained_);
    }
    return *this;
}
bool Completion::ready() const noexcept { return !state_ || state_->ready(); }
void Completion::wait() {
    try {
        if (state_) state_->wait();
    } catch (...) {
        retained_.clear(); // wait failure is terminal by the backend contract.
        throw;
    }
    retained_.clear();
}
void Completion::drain_noexcept() noexcept {
    try { wait(); } catch (...) {} // Destructors protect lifetimes, not error reporting.
}
CpuDeviceRuntime::CpuDeviceRuntime(std::shared_ptr<Allocator> allocator)
    : allocator_(std::move(allocator)) {
    if (!allocator_) throw std::invalid_argument("device allocator must not be null");
}
std::shared_ptr<DeviceBuffer> CpuDeviceRuntime::allocate(std::size_t bytes, std::size_t alignment) {
    auto buffer = std::make_shared<CpuDeviceBuffer>(bytes, alignment, allocator_);
    ++statistics_.allocations;
    statistics_.allocated_bytes += bytes;
    return buffer;
}
Completion CpuDeviceRuntime::upload(DeviceSpan destination, std::shared_ptr<const Buffer> source,
                                    std::size_t source_offset) {
    validate_cpu_span(destination);
    if (!source) throw std::invalid_argument("upload requires owned host storage");
    range(source->size_bytes(), source_offset, destination.bytes);
    if (destination.bytes) std::memmove(pointer(destination),
        static_cast<const unsigned char*>(source->data()) + source_offset, destination.bytes);
    ++statistics_.uploads; statistics_.uploaded_bytes += destination.bytes;
    return {};
}
Completion CpuDeviceRuntime::download(std::shared_ptr<Buffer> destination, DeviceSpan source,
                                      std::size_t destination_offset) {
    validate_cpu_span(source);
    if (!destination) throw std::invalid_argument("download requires owned host storage");
    range(destination->size_bytes(), destination_offset, source.bytes);
    if (source.bytes) std::memmove(static_cast<unsigned char*>(destination->data()) + destination_offset,
                                 pointer(source), source.bytes);
    ++statistics_.downloads; statistics_.downloaded_bytes += source.bytes;
    return {};
}
Completion CpuDeviceRuntime::copy(DeviceSpan destination, DeviceSpan source) {
    validate_cpu_span(destination); validate_cpu_span(source);
    if (destination.bytes != source.bytes) throw std::invalid_argument("copy span sizes differ");
    if (source.bytes) std::memmove(pointer(destination), pointer(source), source.bytes);
    ++statistics_.copies; statistics_.copied_bytes += source.bytes;
    return {};
}
} // namespace tinyinfer
