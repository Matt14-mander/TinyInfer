#include "tinyinfer/backend/metal/metal_runtime.h"
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace tinyinfer;
namespace {
void require(bool value, const char* text) { if (!value) throw std::runtime_error(text); }
template<class F> void rejects(F fn) {
    try { fn(); } catch (const std::exception&) { return; }
    throw std::runtime_error("invalid Metal operation accepted");
}
void run() {
    auto runtime = std::make_shared<MetalDeviceRuntime>();
    auto info = runtime->device_info();
    std::cout << "Device: " << info.name << '\n';
    require(info.id.kind == DeviceKind::Metal && info.available, "invalid device metadata");
    auto input = std::make_shared<Buffer>(32);
    auto* bytes = static_cast<unsigned char*>(input->data());
    for (int i = 0; i < 32; ++i) bytes[i] = static_cast<unsigned char>(i);
    auto a = runtime->allocate(32), b = runtime->allocate(32);
    require(!a->host_data(), "Metal native allocation exposed a host pointer");
    auto upload = runtime->upload({a, 3, 19}, input, 1);
    std::weak_ptr<const Buffer> input_weak = input; input.reset();
    require(!input_weak.expired(), "upload did not retain input ownership");
    auto copied = runtime->copy({b, 5, 19}, {a, 3, 19});
    auto output = std::make_shared<Buffer>(32); std::memset(output->data(), 99, 32);
    auto downloaded = runtime->download(output, {b, 5, 19}, 2);
    runtime->synchronize(); // Must finalize host downloads, even before wait().
    auto* actual = static_cast<unsigned char*>(output->data());
    for (int i = 0; i < 19; ++i) require(actual[i + 2] == i + 1, "queued copy offset/order mismatch");
    require(actual[1] == 99 && actual[21] == 99, "copy wrote outside destination");
    upload.wait(); copied.wait(); downloaded.wait(); downloaded.wait();
    require(input_weak.expired(), "terminal upload retained the host input");
    // Device overlap must behave as memmove (scratch resource is native and retained).
    runtime->copy({b, 6, 18}, {b, 5, 18}).wait();
    runtime->download(output, {b, 6, 18}).wait();
    for (int i = 0; i < 18; ++i) require(actual[i] == i + 1, "overlapping device copy differs");
    // ready() publishes the complete download, never only the native command status.
    auto publish = runtime->download(output, {b, 6, 18});
    while (!publish.ready()) {} // Small transfer; wait() would be preferred in an application.
    publish.wait();
    // Completed stages are reused while still-live Completion wrappers retain no lease.
    auto stable = runtime->transfer_statistics();
    for (int i=0;i<8;++i) {
        runtime->upload({a,0,18}, output).wait();
        runtime->download(output,{a,0,18}).wait();
    }
    require(runtime->transfer_statistics().staging_allocations == stable.staging_allocations,
            "completed transfer stages were not reused");
    require(runtime->transfer_statistics().staging_reuses >= stable.staging_reuses+16, "staging reuse counter");
    // Multiple pending downloads may complete in any host wait order. Their
    // publication must precede stage reuse, including synchronize/ready/destruction.
    std::vector<Completion> transfers;
    std::vector<std::shared_ptr<Buffer>> hosts;
    for (int i=0;i<12;++i) {
        auto source=std::make_shared<Buffer>(32); std::memset(source->data(),i,32);
        transfers.push_back(runtime->upload({a,0,32},source));
        auto host=std::make_shared<Buffer>(32); hosts.push_back(host);
        transfers.push_back(runtime->download(host,{a,0,32}));
    }
    for (auto it=transfers.rbegin();it!=transfers.rend();++it) it->wait();
    for (int i=0;i<12;++i) for (int j=0;j<32;++j)
        require(static_cast<unsigned char*>(hosts[i]->data())[j]==i,"busy stage was overwritten");
    runtime->synchronize();
    require(runtime->transfer_statistics().cached_bytes <= 64,"unbounded staging cache");
    runtime->enable_profiling(true);
    runtime->upload({a,0,32},hosts[0]).wait();
    require(runtime->command_profile().commands==1,"profile did not observe completion");
    runtime->enable_profiling(false);
    auto empty = runtime->allocate(0);
    runtime->upload({empty, 0, 0}, std::make_shared<Buffer>(0)).wait();
    runtime->download(std::make_shared<Buffer>(0), {empty, 0, 0}).wait();
    runtime->copy({empty, 0, 0}, {empty, 0, 0}).wait();
    const auto before = runtime->statistics();
    rejects([&] { runtime->allocate(4, 3); });
    rejects([&] { runtime->allocate(4, 512); });
    rejects([&] { runtime->copy({b, 0, 20}, {a, 30, 20}); });
    rejects([&] { runtime->upload({a, 0, 8}, std::make_shared<Buffer>(4)); });
    rejects([&] { runtime->download({}, {a, 0, 4}); });
    rejects([&] { runtime->copy({b, 0, 4}, {a, 0, 8}); });
    rejects([&] { runtime->allocate(std::numeric_limits<std::size_t>::max()); });
    CpuDeviceRuntime cpu;
    rejects([&] { runtime->copy({a, 0, 4}, {cpu.allocate(4), 0, 4}); });
    auto other = std::make_shared<MetalDeviceRuntime>();
    rejects([&] { runtime->copy({a, 0, 4}, {other->allocate(4), 0, 4}); });
    require(runtime->statistics().allocations == before.allocations &&
            runtime->statistics().copies == before.copies && runtime->statistics().uploads == before.uploads,
            "failed request changed counters");
    // Pending completion/allocations remain valid after wrapper destruction.
    auto host = std::make_shared<Buffer>(4); *static_cast<float*>(host->data()) = 2.5F;
    std::shared_ptr<DeviceBuffer> survivor;
    Completion pending;
    {
        auto short_runtime = std::make_unique<MetalDeviceRuntime>();
        survivor = short_runtime->allocate(4);
        pending = short_runtime->upload({survivor, 0, 4}, host);
    }
    pending.wait(); require(!survivor->host_data(), "surviving allocation lost device isolation");
    // Download completion retains host destination until destructor drains work.
    auto abandoned_output = std::make_shared<Buffer>(4);
    std::weak_ptr<Buffer> weak_output = abandoned_output;
    {
        auto abandoned = runtime->download(abandoned_output, {a, 3, 4});
        abandoned_output.reset(); require(!weak_output.expired(), "pending download lost destination");
    }
    require(weak_output.expired(), "download destructor did not release completed destination");
    require(runtime->kernel_statistics().pipeline_creations == 5, "pipelines were not prepared once");
    std::cout << "PASS: owned copies, offsets, overlap, zero bytes, ordering, publication, lifetime and errors\n";
}
}
int main(int argc, char**) {
    try {
        if (MetalDeviceRuntime::enumerate_devices().empty()) {
            std::cerr << "No visible Metal device\n"; return argc > 1 ? 1 : 77;
        }
        run(); return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
