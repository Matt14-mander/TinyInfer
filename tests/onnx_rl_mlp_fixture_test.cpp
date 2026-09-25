#include "tinyinfer/tinyinfer.h"

#include <array>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

}  // namespace

int main() {
    using tinyinfer::OpType;
    const auto fixture = std::filesystem::path(TINYINFER_TEST_SOURCE_DIR) /
                         "fixtures/rl_actor_mlp_tanh.onnx";
    tinyinfer::onnx::OnnxImporter importer;
    const auto model = importer.load(fixture);

    require(model.inputs().size() == 1 && model.outputs().size() == 1,
            "actor fixture should have one observation input and one action output");
    const auto& nodes = model.graph().nodes();
    require(nodes.size() == 4, "actor fixture should have four exported nodes");
    constexpr std::array<OpType, 4> expected_ops{
        OpType::Gemm, OpType::Tanh, OpType::Gemm, OpType::Tanh};
    for (std::size_t index = 0; index < nodes.size(); ++index) {
        require(nodes[index].op == expected_ops[index],
                "unexpected actor fixture operator at node " +
                    std::to_string(index));
    }

    // Independent outputs from PyTorch 2.2.2 with the fixed exporter weights.
    constexpr std::array<std::array<float, 4>, 2> observations{{
        {0.25F, -0.5F, 1.0F, 0.75F},
        {-1.0F, 0.0F, 0.5F, -0.25F},
    }};
    constexpr std::array<std::array<float, 2>, 2> expected_actions{{
        {0.58088166F, -0.47540686F},
        {-0.17322102F, -0.59409916F},
    }};
    tinyinfer::CpuBackend backend;
    tinyinfer::Executor executor(backend);
    tinyinfer::ExecutionContext context(model.graph());
    for (std::size_t sample = 0; sample < observations.size(); ++sample) {
        const auto& x = observations[sample];
        context.bind_input(model.input_id("observations"),
                           tinyinfer::Tensor::from_vector(
                               {1, 4}, {x[0], x[1], x[2], x[3]}));
        executor.run(model.graph(), context);
        const auto& actions = context.output(model.output_id("actions"));
        require(actions.shape() == tinyinfer::Shape({1, 2}),
                "actor action shape should be [1, 2]");
        for (std::size_t index = 0; index < 2; ++index) {
            require(std::fabs(actions.at(index) - expected_actions[sample][index]) <
                        1e-5F,
                    "actor action differs from PyTorch reference");
        }
    }
}
