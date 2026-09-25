#include "tinyinfer/tinyinfer.h"

#include <cmath>
#include <cstdint>
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
    ValueId product;
    ValueId sum;
    ValueId orphan;
    Tensor bias;
    bool dynamic_bias;
};

Fixture make_pair(Tensor bias, bool dynamic_bias = false,
                  bool reverse_add = false, bool expose_product = false,
                  bool extra_consumer = false, bool downstream = false) {
    Graph graph;
    const auto x = graph.add_input("x", {{2, 3}, DataType::Float32});
    const auto weight = graph.add_constant(
        "weight", Tensor::from_vector({3, 2}, {1, 2, 3, 4, 5, 6}));
    const auto c = dynamic_bias
                       ? graph.add_input("c", {bias.shape(), bias.dtype()})
                       : graph.add_constant("c", bias);
    const auto orphan = graph.add_constant(
        "orphan", Tensor::from_vector({1}, {42.0F}));
    const auto matmul = graph.add_node("matmul", OpType::MatMul, {x, weight});
    const auto product = graph.node(matmul).outputs.front();
    const auto add_inputs = reverse_add
                                ? std::vector<ValueId>{c, product}
                                : std::vector<ValueId>{product, c};
    const auto add = graph.add_node("add", OpType::Add, add_inputs);
    const auto sum = graph.node(add).outputs.front();
    auto model_output = sum;
    if (downstream) {
        const auto activation = graph.add_node("relu", OpType::ReLU, {sum});
        model_output = graph.node(activation).outputs.front();
    }
    graph.mark_output(model_output);
    std::vector<Model::NamedValue> outputs{{"y", model_output}};
    if (expose_product) {
        graph.mark_output(product);
        outputs.emplace_back("product", product);
    }
    if (extra_consumer) {
        const auto activation = graph.add_node(
            "extra_relu", OpType::ReLU, {product});
        const auto extra = graph.node(activation).outputs.front();
        graph.mark_output(extra);
        outputs.emplace_back("extra", extra);
    }
    std::vector<Model::NamedValue> inputs{{"x", x}};
    if (dynamic_bias) inputs.emplace_back("c", c);
    return {Model(std::move(graph), std::move(inputs), std::move(outputs)),
            product, sum, orphan, std::move(bias), dynamic_bias};
}

std::vector<Tensor> execute(const Fixture& fixture, const Model& model,
                            bool planned = false) {
    CpuBackend backend;
    Executor executor(backend);
    std::vector<Tensor> outputs;
    const auto x = Tensor::from_vector({2, 3}, {1, 2, 3, 4, 5, 6});
    if (planned) {
        MemoryPlan plan(model.graph());
        ExecutionContext context(model.graph(), plan);
        context.bind_input(model.input_id("x"), x);
        if (fixture.dynamic_bias) {
            context.bind_input(model.input_id("c"), fixture.bias);
        }
        executor.run(model.graph(), context);
        for (const auto& binding : model.outputs()) {
            outputs.push_back(context.output(binding.second));
        }
    } else {
        ExecutionContext context(model.graph());
        context.bind_input(model.input_id("x"), x);
        if (fixture.dynamic_bias) {
            context.bind_input(model.input_id("c"), fixture.bias);
        }
        executor.run(model.graph(), context);
        for (const auto& binding : model.outputs()) {
            outputs.push_back(context.output(binding.second));
        }
    }
    return outputs;
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
            const auto lhs = actual[output].at(index);
            const auto rhs = expected[output].at(index);
            require(std::fabs(lhs - rhs) <= 1e-5F,
                    "output value changed at index " + std::to_string(index));
        }
    }
}

void check_fused(const Fixture& fixture) {
    MatMulAddCanonicalizationPass pass;
    const auto reference = execute(fixture, fixture.model);
    const auto original_plan = MemoryPlan(fixture.model.graph());
    const auto result = pass.run(fixture.model);
    require(result.changed(), "eligible pair was not canonicalized");
    require(result.pass_statistics.size() == 1 &&
                result.pass_statistics.front().nodes_rewritten == 1,
            "fusion statistics are incorrect");
    require(result.model.graph().size() + 1 == fixture.model.graph().size(),
            "fusion should remove exactly one node");
    require(result.mapped_value(fixture.product) == kInvalidValueId,
            "removed MatMul output should have invalid mapping");
    require(result.mapped_value(fixture.sum) != kInvalidValueId,
            "Add output should map to Gemm output");
    require(result.mapped_value(fixture.orphan) != kInvalidValueId &&
                result.model.graph().constant(
                    result.mapped_value(fixture.orphan)).at(0) == 42.0F,
            "canonicalization should preserve unrelated constants");
    require(result.model.input_id("x") ==
                result.mapped_value(fixture.model.input_id("x")),
            "external input binding changed");
    if (fixture.dynamic_bias) {
        require(result.model.input_id("c") ==
                    result.mapped_value(fixture.model.input_id("c")),
                "dynamic C input binding changed");
    }
    require(result.model.output_id("y") ==
                result.mapped_value(fixture.model.output_id("y")),
            "external output binding changed");
    const auto& gemm = result.model.graph().node(0);
    require(gemm.op == OpType::Gemm && gemm.name == "add",
            "Add should be replaced by a Gemm with the same node name");
    require(gemm.attribute<float>("alpha") == 1.0F &&
                gemm.attribute<float>("beta") == 1.0F &&
                gemm.attribute<std::int64_t>("transA") == 0 &&
                gemm.attribute<std::int64_t>("transB") == 0,
            "canonical Gemm attributes are incorrect");
    expect_same(execute(fixture, result.model), reference);
    expect_same(execute(fixture, result.model, true), reference);
    const auto optimized_plan = MemoryPlan(result.model.graph());
    require(optimized_plan.naive_intermediate_bytes() <
                original_plan.naive_intermediate_bytes(),
            "fusion should remove the product intermediate");
    require(fixture.model.graph().node(0).op == OpType::MatMul,
            "source graph was modified");

    const auto second = pass.run(result.model);
    require(!second.changed(), "canonicalization should be idempotent");
    require(second.model.graph().size() == result.model.graph().size(),
            "second run changed graph size");
    expect_same(execute(fixture, second.model), reference);

    PassManager pipeline;
    pipeline.add_pass(std::make_unique<DeadCodeEliminationPass>());
    pipeline.add_pass(std::make_unique<MatMulAddCanonicalizationPass>());
    pipeline.add_pass(std::make_unique<DeadCodeEliminationPass>());
    const auto pipelined = pipeline.run(fixture.model);
    require(pipelined.changed() && pipelined.pass_statistics.size() == 3,
            "PassManager did not record canonicalization");
    require(pipelined.mapped_value(fixture.product) == kInvalidValueId,
            "PassManager did not compose removed-value mapping");
    expect_same(execute(fixture, pipelined.model), reference);
}

void check_unchanged(const Fixture& fixture) {
    MatMulAddCanonicalizationPass pass;
    const auto reference = execute(fixture, fixture.model);
    const auto result = pass.run(fixture.model);
    require(!result.changed(), "ineligible pair was canonicalized");
    require(result.model.graph().size() == fixture.model.graph().size(),
            "ineligible graph size changed");
    require(result.mapped_value(fixture.product) != kInvalidValueId,
            "retained MatMul output lost its mapping");
    require(result.mapped_value(fixture.orphan) == fixture.orphan,
            "no-op result should preserve ValueIds");
    expect_same(execute(fixture, result.model), reference);
}

}  // namespace

int main() {
    check_fused(make_pair(Tensor::from_vector({}, {0.5F})));
    check_fused(make_pair(Tensor::from_vector({2}, {1, 2}), true, true));
    check_fused(make_pair(Tensor::from_vector({1, 2}, {1, 2})));
    check_fused(make_pair(Tensor::from_vector({2, 1}, {1, 2})));
    check_fused(make_pair(Tensor::from_vector({2, 2}, {1, 2, 3, 4}),
                          false, false, false, false, true));

    check_unchanged(make_pair(Tensor::from_vector({2}, {1, 2}),
                              false, false, true));
    check_unchanged(make_pair(Tensor::from_vector({2}, {1, 2}),
                              false, false, false, true));
    check_unchanged(make_pair(Tensor::from_vector({1, 2, 2}, {1, 2, 3, 4})));

    Graph graph;
    const auto x = graph.add_input("x", {{2, 3}, DataType::Float32});
    const auto weight = graph.add_constant(
        "weight", Tensor::from_vector({3, 2}, {1, 2, 3, 4, 5, 6}));
    const auto first = graph.add_node("first", OpType::MatMul, {x, weight});
    const auto second = graph.add_node("second", OpType::MatMul, {x, weight});
    const auto sum = graph.add_node(
        "sum", OpType::Add,
        {graph.node(first).outputs.front(), graph.node(second).outputs.front()});
    const auto output = graph.node(sum).outputs.front();
    graph.mark_output(output);
    const Model two_products(std::move(graph), {{"x", x}}, {{"y", output}});
    MatMulAddCanonicalizationPass pass;
    const auto unchanged = pass.run(two_products);
    require(!unchanged.changed() && unchanged.model.graph().size() == 3,
            "Add of two MatMul outputs should not canonicalize");

    Graph standalone_graph;
    const auto input = standalone_graph.add_input("x", {{2, 3}, DataType::Float32});
    const auto rhs = standalone_graph.add_constant(
        "weight", Tensor::from_vector({3, 2}, {1, 2, 3, 4, 5, 6}));
    const auto matmul = standalone_graph.add_node(
        "matmul", OpType::MatMul, {input, rhs});
    const auto standalone_output = standalone_graph.node(matmul).outputs.front();
    standalone_graph.mark_output(standalone_output);
    const Model standalone(std::move(standalone_graph), {{"x", input}},
                           {{"y", standalone_output}});
    const auto standalone_result = pass.run(standalone);
    require(!standalone_result.changed() &&
                standalone_result.model.graph().node(0).op == OpType::MatMul,
            "standalone MatMul should remain unchanged");

    Graph gemm_graph;
    const auto gemm_input = gemm_graph.add_input("x", {{2, 3}, DataType::Float32});
    const auto gemm_weight = gemm_graph.add_constant(
        "weight", Tensor::from_vector({2, 3}, {1, 3, 5, 2, 4, 6}));
    const auto gemm_node = gemm_graph.add_node(
        "gemm", OpType::Gemm, {gemm_input, gemm_weight},
        {{"transB", std::int64_t{1}}, {"alpha", 0.5F}});
    const auto gemm_output = gemm_graph.node(gemm_node).outputs.front();
    gemm_graph.mark_output(gemm_output);
    const Model existing_gemm(std::move(gemm_graph), {{"x", gemm_input}},
                               {{"y", gemm_output}});
    const auto gemm_result = pass.run(existing_gemm);
    require(!gemm_result.changed() &&
                gemm_result.model.graph().node(0).attributes ==
                    existing_gemm.graph().node(0).attributes,
            "existing Gemm attributes should remain unchanged");

    Graph chain_graph;
    const auto chain_input = chain_graph.add_input(
        "x", {{2, 2}, DataType::Float32});
    const auto chain_weight = chain_graph.add_constant(
        "weight", Tensor::from_vector({2, 2}, {1, 2, 3, 4}));
    const auto chain_bias = chain_graph.add_constant(
        "bias", Tensor::from_vector({2}, {1, -1}));
    const auto first_matmul = chain_graph.add_node(
        "first_matmul", OpType::MatMul, {chain_input, chain_weight});
    const auto first_product = chain_graph.node(first_matmul).outputs.front();
    const auto first_add = chain_graph.add_node(
        "first_add", OpType::Add, {first_product, chain_bias});
    const auto first_sum = chain_graph.node(first_add).outputs.front();
    const auto second_matmul = chain_graph.add_node(
        "second_matmul", OpType::MatMul, {first_sum, chain_weight});
    const auto second_product = chain_graph.node(second_matmul).outputs.front();
    const auto second_add = chain_graph.add_node(
        "second_add", OpType::Add, {second_product, chain_bias});
    const auto chain_output = chain_graph.node(second_add).outputs.front();
    chain_graph.mark_output(chain_output);
    const Model chain(std::move(chain_graph), {{"x", chain_input}},
                      {{"y", chain_output}});
    const auto chain_result = pass.run(chain);
    require(chain_result.changed() &&
                chain_result.pass_statistics.front().nodes_rewritten == 2 &&
                chain_result.model.graph().size() == 2 &&
                chain_result.mapped_value(first_product) == kInvalidValueId &&
                chain_result.mapped_value(second_product) == kInvalidValueId,
            "two dependent pairs should both canonicalize");
    for (const auto& node : chain_result.model.graph().nodes()) {
        require(node.op == OpType::Gemm,
                "canonicalized chain should contain only Gemm nodes");
    }
    const auto run_chain = [](const Model& model) {
        ExecutionContext context(model.graph());
        context.bind_input(model.input_id("x"),
                           Tensor::from_vector({2, 2}, {1, 2, 3, 4}));
        CpuBackend backend;
        Executor executor(backend);
        executor.run(model.graph(), context);
        return context.output(model.output_id("y"));
    };
    const auto chain_reference = run_chain(chain);
    const auto chain_actual = run_chain(chain_result.model);
    for (std::size_t index = 0; index < chain_reference.numel(); ++index) {
        require(std::fabs(chain_actual.at(index) - chain_reference.at(index)) <=
                    1e-5F,
                "dependent pair numerical result changed");
    }
}
