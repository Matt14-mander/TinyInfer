#include "tinyinfer/backend/metal/metal_runtime.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
using namespace tinyinfer;
MetalTensorView upload(MetalDeviceRuntime& runtime, Shape shape, const std::vector<float>& values) {
    const auto bytes = values.size() * sizeof(float);
    auto host = std::make_shared<Buffer>(bytes);
    if (bytes) std::memcpy(host->data(), values.data(), bytes);
    auto allocation = runtime.allocate(bytes);
    runtime.upload({allocation, 0, bytes}, host).wait();
    return {{std::move(shape), DataType::Float32}, {allocation, 0, bytes}};
}
int main() {
    try {
        MetalDeviceRuntime runtime;
        auto a = upload(runtime, {2, 2}, {1, 2, 3, 4});
        auto b = upload(runtime, {2, 3}, {1, -1, 2, 1, 1, 2});
        auto bias = upload(runtime, {3}, {0, -2, 1});
        constexpr auto bytes = 6 * sizeof(float);
        MetalTensorView product({{2, 3}, DataType::Float32}, {runtime.allocate(bytes), 0, bytes});
        MetalTensorView result({{2, 3}, DataType::Float32}, {runtime.allocate(bytes), 0, bytes});
        // One ordered queue: no host copy or intermediate wait between these kernels.
        auto matrix = runtime.execute(OpType::Gemm, {a, b, bias}, product);
        auto activation = runtime.execute(OpType::ReLU, {product}, result);
        auto output = std::make_shared<Buffer>(bytes);
        runtime.download(output, result.storage).wait();
        matrix.wait(); activation.wait();
        const auto* values = static_cast<const float*>(output->data());
        const float expected[] = {3, 0, 7, 7, 0, 15};
        std::cout << runtime.device_info().name << '\n';
        for (int i = 0; i < 6; ++i) {
            if (!std::isfinite(values[i]) || std::abs(values[i] - expected[i]) > 1e-5F) throw std::runtime_error("incorrect GPU output");
            std::cout << values[i] << (i % 3 == 2 ? '\n' : ' ');
        }
        std::cout << "Checked GPU Gemm -> ReLU; pipelines=" << runtime.kernel_statistics().pipeline_creations
                  << ", dispatches=" << runtime.kernel_statistics().dispatches << '\n';
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
