#include "tinyinfer/tinyinfer.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <filesystem>

int main() {
    using tinyinfer::CpuBackend;
    using tinyinfer::ExecutionContext;
    using tinyinfer::Executor;
    using tinyinfer::OpType;
    using tinyinfer::Tensor;
    using tinyinfer::onnx::OnnxImporter;

    const auto fixture = std::filesystem::path(TINYINFER_TEST_SOURCE_DIR) /
                         "fixtures/phase3_mlp_gemm.onnx";
    OnnxImporter importer;
    const auto model = importer.load(fixture);

    assert(model.inputs().size() == 1);
    assert(model.outputs().size() == 1);
    assert(model.graph().size() == 4);
    assert(model.graph().values().size() == 9);

    std::size_t gemm_count = 0;
    for (const auto& node : model.graph().nodes()) {
        if (node.op != OpType::Gemm) continue;
        ++gemm_count;
        assert(node.attribute<std::int64_t>("transB") == 1);
    }
    assert(gemm_count == 2);

    ExecutionContext context(model.graph());
    context.bind_input(model.input_id("x"),
                       Tensor::from_vector({1, 2}, {1.0F, -2.0F}));
    CpuBackend backend;
    Executor executor(backend);
    executor.run(model.graph(), context);

    const auto& output = context.output(model.output_id("probabilities"));
    assert(output.shape() == tinyinfer::Shape({1, 2}));
    assert(std::fabs(output.at(0) - 0.26894143F) < 1e-5F);
    assert(std::fabs(output.at(1) - 0.73105860F) < 1e-5F);
}
