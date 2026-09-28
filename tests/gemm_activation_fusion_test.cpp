#include "tinyinfer/tinyinfer.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace tinyinfer;
using namespace tinyinfer::optimizer;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

struct Fixture {
    Model model;
    ValueId gemm_output;
    ValueId activation_output;
    ValueId orphan;
    Tensor bias;
    bool dynamic_bias;
};

Fixture make_fixture(bool with_bias = true, bool dynamic_bias = false,
                     bool transpose_rhs = true,
                     OpType activation = OpType::ReLU,
                     bool expose_gemm = false,
                     bool extra_consumer = false,
                     bool downstream = false) {
    Graph graph;
    const auto x = graph.add_input("x", {{2, 3}, DataType::Float32});
    const auto weight = transpose_rhs
        ? graph.add_constant(
              "weight", Tensor::from_vector(
                            {4, 3}, {1, -2, 3, 4, 5, -6,
                                     -7, 8, 9, 10, -11, 12}))
        : graph.add_constant(
              "weight", Tensor::from_vector(
                            {3, 4}, {1, 4, -7, 10, -2, 5,
                                     8, -11, 3, -6, 9, 12}));
    auto bias_tensor = Tensor::from_vector({4}, {-1, 2, -3, 4});
    std::vector<ValueId> inputs{x, weight};
    if (with_bias) {
        const auto bias = dynamic_bias
            ? graph.add_input("bias", {{4}, DataType::Float32})
            : graph.add_constant("bias", bias_tensor);
        inputs.push_back(bias);
    }
    const auto orphan = graph.add_constant(
        "orphan", Tensor::from_vector({1}, {42.0F}));
    NodeAttributes attributes{{"alpha", 0.5F},
                              {"beta", 1.5F},
                              {"transA", std::int64_t{0}},
                              {"transB", std::int64_t{transpose_rhs ? 1 : 0}}};
    const auto gemm = graph.add_node("affine", OpType::Gemm, inputs, attributes);
    const auto gemm_output = graph.node(gemm).outputs.front();
    const auto activated = graph.add_node("activation", activation, {gemm_output});
    const auto activation_output = graph.node(activated).outputs.front();
    auto model_output = activation_output;
    if (downstream) {
        const auto tail = graph.add_node("tail", OpType::Tanh, {activation_output});
        model_output = graph.node(tail).outputs.front();
    }
    graph.mark_output(model_output);
    std::vector<Model::NamedValue> outputs{{"y", model_output}};
    if (expose_gemm) {
        graph.mark_output(gemm_output);
        outputs.emplace_back("affine", gemm_output);
    }
    if (extra_consumer) {
        const auto extra = graph.add_node("extra", OpType::Tanh, {gemm_output});
        const auto extra_output = graph.node(extra).outputs.front();
        graph.mark_output(extra_output);
        outputs.emplace_back("extra", extra_output);
    }
    std::vector<Model::NamedValue> model_inputs{{"x", x}};
    if (with_bias && dynamic_bias) {
        model_inputs.emplace_back("bias", inputs.back());
    }
    return {Model(std::move(graph), std::move(model_inputs),
                  std::move(outputs)),
            gemm_output, activation_output, orphan,
            std::move(bias_tensor), with_bias && dynamic_bias};
}

std::vector<Tensor> execute(const Fixture& fixture, const Model& model,
                            bool planned = false) {
    const auto x = Tensor::from_vector({2, 3}, {1, -2, 3, -4, 5, -6});
    CpuBackend backend;
    Executor executor(backend);
    std::vector<Tensor> outputs;
    if (planned) {
        MemoryPlan plan(model.graph());
        ExecutionContext context(model.graph(), plan);
        context.bind_input(model.input_id("x"), x);
        if (fixture.dynamic_bias) {
            context.bind_input(model.input_id("bias"), fixture.bias);
        }
        executor.run(model.graph(), context);
        for (const auto& output : model.outputs()) {
            outputs.push_back(context.output(output.second));
        }
    } else {
        ExecutionContext context(model.graph());
        context.bind_input(model.input_id("x"), x);
        if (fixture.dynamic_bias) {
            context.bind_input(model.input_id("bias"), fixture.bias);
        }
        executor.run(model.graph(), context);
        for (const auto& output : model.outputs()) {
            outputs.push_back(context.output(output.second));
        }
    }
    return outputs;
}

void expect_scalar(float actual, float expected, const std::string& message) {
    if (std::isnan(expected)) {
        require(std::isnan(actual), message + ": expected NaN");
        return;
    }
    if (std::isinf(expected)) {
        require(std::isinf(actual) && std::signbit(actual) == std::signbit(expected),
                message + ": infinity changed");
        return;
    }
    require(std::fabs(actual - expected) <= 1e-5F, message);
}

void expect_same(const std::vector<Tensor>& actual,
                 const std::vector<Tensor>& expected) {
    require(actual.size() == expected.size(), "output count changed");
    for (std::size_t output = 0; output < actual.size(); ++output) {
        require(actual[output].shape() == expected[output].shape(),
                "output shape changed");
        require(actual[output].dtype() == expected[output].dtype(),
                "output dtype changed");
        for (std::size_t index = 0; index < actual[output].numel(); ++index) {
            expect_scalar(actual[output].at(index), expected[output].at(index),
                          "output value changed at index " +
                              std::to_string(index));
        }
    }
}

void check_fused(const Fixture& fixture) {
    GemmActivationFusionPass pass;
    const auto reference = execute(fixture, fixture.model);
    const MemoryPlan original_plan(fixture.model.graph());
    const auto result = pass.run(fixture.model);
    require(result.changed(), "eligible Gemm + ReLU was not fused");
    require(result.pass_statistics.size() == 1 &&
                result.pass_statistics.front().nodes_rewritten == 1,
            "fusion statistics are incorrect");
    require(result.model.graph().size() + 1 == fixture.model.graph().size(),
            "fusion should remove exactly one node");
    require(result.mapped_value(fixture.gemm_output) == kInvalidValueId,
            "removed Gemm output should have invalid mapping");
    require(result.mapped_value(fixture.activation_output) != kInvalidValueId,
            "ReLU output should map to the fused output");
    require(result.mapped_value(fixture.orphan) != kInvalidValueId &&
                result.model.graph().constant(
                    result.mapped_value(fixture.orphan)).at(0) == 42.0F,
            "fusion should preserve unrelated constants");

    const Node* fused = nullptr;
    for (const auto& node : result.model.graph().nodes()) {
        if (node.op == OpType::FusedGemmActivation) fused = &node;
    }
    require(fused != nullptr && fused->name == "activation",
            "fused node is missing or has the wrong name");
    require(fused->attribute<std::string>("activation") == "relu" &&
                fused->attribute<float>("alpha") == 0.5F &&
                fused->attribute<float>("beta") == 1.5F,
            "fused attributes changed");
    expect_same(execute(fixture, result.model), reference);
    expect_same(execute(fixture, result.model, true), reference);

    const MemoryPlan fused_plan(result.model.graph());
    require(fused_plan.naive_intermediate_bytes() <
                original_plan.naive_intermediate_bytes(),
            "fusion should remove one intermediate tensor");
    require(fixture.model.graph().node(0).op == OpType::Gemm,
            "source model was modified");

    const auto second = pass.run(result.model);
    require(!second.changed(), "fusion pass should be idempotent");
    expect_same(execute(fixture, second.model), reference);
}

void check_unchanged(const Fixture& fixture) {
    GemmActivationFusionPass pass;
    const auto reference = execute(fixture, fixture.model);
    const auto result = pass.run(fixture.model);
    require(!result.changed(), "ineligible graph was fused");
    require(result.model.graph().size() == fixture.model.graph().size(),
            "ineligible graph size changed");
    require(result.mapped_value(fixture.gemm_output) != kInvalidValueId,
            "retained Gemm output lost its mapping");
    expect_same(execute(fixture, result.model), reference);
}

void check_pipeline_composition() {
    Graph graph;
    const auto x = graph.add_input("x", {{2, 3}, DataType::Float32});
    const auto weight = graph.add_constant(
        "weight", Tensor::from_vector({3, 2}, {1, 2, 3, 4, 5, 6}));
    const auto bias = graph.add_constant(
        "bias", Tensor::from_vector({2}, {-20, 10}));
    const auto matmul = graph.add_node("matmul", OpType::MatMul, {x, weight});
    const auto product = graph.node(matmul).outputs.front();
    const auto add = graph.add_node("add", OpType::Add, {product, bias});
    const auto sum = graph.node(add).outputs.front();
    const auto relu = graph.add_node("relu", OpType::ReLU, {sum});
    const auto output = graph.node(relu).outputs.front();
    graph.mark_output(output);
    const Model model(std::move(graph), {{"x", x}}, {{"y", output}});
    const Fixture fixture{model, sum, output, kInvalidValueId, Tensor{}, false};
    const auto reference = execute(fixture, model);

    PassManager pipeline;
    pipeline.add_pass(std::make_unique<MatMulAddCanonicalizationPass>());
    pipeline.add_pass(std::make_unique<GemmActivationFusionPass>());
    pipeline.add_pass(std::make_unique<DeadCodeEliminationPass>());
    const auto result = pipeline.run(model);
    require(result.changed() && result.model.graph().size() == 1,
            "canonicalization and activation fusion did not compose");
    require(result.model.graph().node(0).op == OpType::FusedGemmActivation,
            "pipeline did not produce the fused operator");
    require(result.mapped_value(product) == kInvalidValueId &&
                result.mapped_value(sum) == kInvalidValueId &&
                result.mapped_value(output) != kInvalidValueId,
            "pipeline ValueId mapping is incorrect");
    expect_same(execute(fixture, result.model), reference);
}

void check_special_values() {
    const auto infinity = std::numeric_limits<float>::infinity();
    const auto nan = std::numeric_limits<float>::quiet_NaN();
    const auto lhs = Tensor::from_vector({1, 4}, {-infinity, nan, -0.0F, 2.0F});
    const auto rhs = Tensor::from_vector(
        {4, 4}, {1, 0, 0, 0, 0, 1, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 1});
    Tensor unfused({1, 4});
    Tensor fused({1, 4});
    ops::gemm_out(unfused, lhs, rhs);
    ops::relu_out(unfused, unfused);
    ops::gemm_relu_out(fused, lhs, rhs);
    for (std::size_t index = 0; index < fused.numel(); ++index) {
        expect_scalar(fused.at(index), unfused.at(index),
                      "special-value behavior changed");
    }
}

void check_schema_rejects_unknown_activation() {
    bool rejected = false;
    try {
        static_cast<void>(infer_output_specs(
            OpType::FusedGemmActivation,
            {{{2, 3}, DataType::Float32}, {{3, 4}, DataType::Float32}},
            {{"activation", std::string{"gelu"}}}));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "unsupported fused activation was accepted");
}

}  // namespace

int main() {
    check_fused(make_fixture());
    check_fused(make_fixture(false, false, false));
    check_fused(make_fixture(true, true));
    check_fused(make_fixture(true, false, true, OpType::ReLU,
                             false, false, true));

    check_unchanged(make_fixture(true, false, true, OpType::ReLU, true));
    check_unchanged(make_fixture(true, false, true, OpType::ReLU,
                                 false, true));
    check_unchanged(make_fixture(true, false, true, OpType::Tanh));

    check_pipeline_composition();
    check_special_values();
    check_schema_rejects_unknown_activation();
}
