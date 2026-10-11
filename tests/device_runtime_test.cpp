#include "tinyinfer/tinyinfer.h"
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <type_traits>
static_assert(!std::is_copy_constructible_v<tinyinfer::CpuDeviceRuntime>, "queue must have one owner");
static_assert(!std::is_copy_constructible_v<tinyinfer::Completion>, "completion must not duplicate ownership");
using namespace tinyinfer;
namespace {
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
template<class F> void rejects(F fn) {
    try { fn(); } catch (const std::exception&) { return; }
    throw std::runtime_error("invalid device operation accepted");
}
class OpaqueBuffer final : public DeviceBuffer {
public:
    OpaqueBuffer() : DeviceBuffer({DeviceKind::Cuda, 0}, 16, 16) {}
protected:
    void* host_pointer() const noexcept override { return reinterpret_cast<void*>(0x1234); }
};
class DeferredState final : public CompletionState {
public:
    DeferredState(std::weak_ptr<const void> resource, bool fail = false)
        : resource_(std::move(resource)), fail_(fail) {}
    bool ready() const noexcept override { return done_; }
    void wait() override {
        if (!done_) {
            require(!resource_.expired(), "pending work lost its resource");
            done_ = true;
        }
        if (fail_) throw std::runtime_error("terminal device failure");
    }
private:
    std::weak_ptr<const void> resource_;
    bool fail_, done_{false};
};
void copies() {
    CpuDeviceRuntime runtime;
    auto host = std::make_shared<Buffer>(16);
    auto* bytes = static_cast<unsigned char*>(host->data());
    for (int i = 0; i < 16; ++i) bytes[i] = static_cast<unsigned char>(i);
    auto first = runtime.allocate(16, 16), second = runtime.allocate(16, 16);
    require(first->device() == DeviceId{} && first->alignment() == 16, "allocation metadata differs");
    auto upload = runtime.upload({first, 4, 8}, host, 2);
    require(upload.ready(), "CPU copy is not synchronous"); upload.wait();
    runtime.copy({second, 0, 8}, {first, 4, 8}).wait();
    auto output = std::make_shared<Buffer>(16);
    std::memset(output->data(), 99, 16);
    runtime.download(output, {second, 0, 8}, 3).wait();
    auto* result = static_cast<unsigned char*>(output->data());
    for (int i = 0; i < 8; ++i) require(result[i + 3] == i + 2, "copy offset lost");
    require(result[2] == 99 && result[11] == 99, "copy overwrote outside destination span");
    // Overlapping copies within one allocation are defined as memmove.
    runtime.copy({second, 1, 7}, {second, 0, 7}).wait();
    require(static_cast<unsigned char*>(second->host_data())[7] == 8, "overlap copy differs");
    auto zero = runtime.allocate(0);
    runtime.upload({zero, 0, 0}, std::make_shared<Buffer>(0)).wait();
    runtime.download(std::make_shared<Buffer>(0), {zero, 0, 0}).wait();
    runtime.copy({zero, 0, 0}, {zero, 0, 0}).wait();
    runtime.synchronize();
    // Allocation ownership is independent of queue/runtime handles.
    std::shared_ptr<DeviceBuffer> surviving;
    { CpuDeviceRuntime temporary; surviving = temporary.allocate(8); }
    require(surviving->host_data() != nullptr, "runtime destruction invalidated allocation");
    rejects([&] { runtime.allocate(8, 3); });
    rejects([&] { runtime.upload({first, 12, 8}, host); });
    rejects([&] { runtime.upload({first, 0, 8}, host, 9); });
    rejects([&] { runtime.download(output, {first, 0, 8}, 9); });
    rejects([&] { runtime.copy({first, 0, 8}, {second, 0, 4}); });
    rejects([&] { runtime.upload({first, 0, 8}, {}); });
    rejects([&] { runtime.download({}, {first, 0, 8}); });
    rejects([&] { DeviceSpan{first, std::numeric_limits<std::size_t>::max(), 1}.validate(); });
    rejects([&] { DeviceSpan{}.validate(); });
    rejects([] { CpuDeviceRuntime bad(nullptr); });
    auto opaque = std::make_shared<OpaqueBuffer>();
    require(!opaque->host_data() && !std::as_const(*opaque).host_data(), "GPU handle exposed as host data");
    rejects([&] { runtime.copy({first, 0, 8}, {opaque, 0, 8}); });
    rejects([] { validate_device_id({DeviceKind::Cpu, 1}); });
    rejects([] { device_kind_name(static_cast<DeviceKind>(99)); });
    const auto stats = runtime.statistics();
    require(stats.allocations == 3 && stats.allocated_bytes == 32, "allocation accounting differs");
    require(stats.uploads == 2 && stats.uploaded_bytes == 8, "upload accounting differs");
    require(stats.downloads == 2 && stats.downloaded_bytes == 8, "download accounting differs");
    require(stats.copies == 3 && stats.copied_bytes == 15, "copy accounting differs");
}
void completions() {
    for (bool fail : {false, true}) {
        auto resource = std::make_shared<Buffer>(16);
        std::weak_ptr<const void> weak = resource;
        auto state = std::make_shared<DeferredState>(weak, fail);
        Completion completion(state, {resource});
        resource.reset();
        require(!completion.ready() && !weak.expired(), "pending resource not retained");
        Completion moved(std::move(completion));
        require(completion.ready(), "moved-from completion is not empty");
        if (fail) { rejects([&] { moved.wait(); }); rejects([&] { moved.wait(); }); }
        else { moved.wait(); moved.wait(); }
        require(moved.ready() && weak.expired(), "terminal operation retained resources");
    }
    // Destructor and move assignment drain outstanding work before releasing it.
    auto resource = std::make_shared<Buffer>(8);
    std::weak_ptr<const void> weak = resource;
    auto state = std::make_shared<DeferredState>(weak);
    {
        Completion destination(state, {resource}); resource.reset();
        destination = Completion{};
        require(state->ready() && weak.expired(), "move assignment dropped live resource");
    }
    resource = std::make_shared<Buffer>(8); weak = resource;
    state = std::make_shared<DeferredState>(weak, true);
    { Completion abandoned(state, {resource}); resource.reset(); }
    require(state->ready() && weak.expired(), "destructor did not drain failed work");
    rejects([] { Completion invalid(nullptr, {}); });
}
} // namespace
int main() { copies(); completions(); }
