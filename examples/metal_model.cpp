#include "tinyinfer/backend/metal/execution_plan.h"
#include "tinyinfer/model/onnx/onnx_importer.h"
#include <iostream>
int main(int argc, char** argv) {
    try {
        if (argc != 2) { std::cerr << "Usage: tinyinfer_metal_model model.onnx\n"; return 1; }
        tinyinfer::onnx::OnnxImporter importer;
        auto model = importer.load(argv[1]);
        tinyinfer::MetalExecutionPlan plan(model);
        auto context = plan.create_context();
        for (const auto& input : plan.inputs()) {
            tinyinfer::Tensor tensor(plan.input_spec(input.second).shape);
            for (std::size_t i = 0; i < tensor.numel(); ++i) tensor.at(i) = float(i) / 4;
            context.bind_input(input.first, std::move(tensor));
        }
        for (int i = 0; i < 3; ++i) context.run();
        for (const auto& output : plan.outputs()) std::cout << output.first << ": " << context.output(output.first) << '\n';
        auto stats = plan.statistics();
        std::cout << "constant uploads=" << stats.constant_uploads << ", bytes=" << stats.constant_bytes
                  << ", activation bytes=" << stats.activation_bytes << ", GPU runs=" << context.statistics().gpu_runs << '\n';
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
