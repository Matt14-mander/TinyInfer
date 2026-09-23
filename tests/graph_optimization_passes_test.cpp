#include "tinyinfer/tinyinfer.h"

#include <cassert>
#include <cmath>
#include <memory>
#include <utility>

namespace {

struct Fixture {
    tinyinfer::Model model;
    tinyinfer::ValueId first_folded;
    tinyinfer::ValueId final_folded;
    tinyinfer::ValueId output;
    tinyinfer::ValueId dead_output;
};

Fixture make_fixture() {
    using namespace tinyinfer;
    Graph graph;
    const auto input = graph.add_input(
        "input", TensorSpec{{2}, DataType::Float32});
    const auto lhs = graph.add_constant(
        "lhs", Tensor::from_vector({2}, {1.0F, 2.0F}));
    const auto rhs = graph.add_constant(
        "rhs", Tensor::from_vector({2}, {3.0F, 4.0F}));
    const auto sum = graph.add_node("constant_sum", OpType::Add, {lhs, rhs});
    const auto first_folded = graph.node(sum).outputs.front();
    const auto activation = graph.add_node(
        "constant_relu", OpType::ReLU, {first_folded});
    const auto final_folded = graph.node(activation).outputs.front();
    const auto runtime = graph.add_node(
        "runtime_add", OpType::Add, {input, final_folded});
    const auto output = graph.node(runtime).outputs.front();
    const auto dead = graph.add_node("dead_relu", OpType::ReLU, {input});
    const auto dead_output = graph.node(dead).outputs.front();
    graph.mark_output(output);
    return {Model(std::move(graph), {{"input", input}}, {{"output", output}}),
            first_folded, final_folded, output, dead_output};
}

tinyinfer::Tensor execute(const tinyinfer::Model& model) {
    tinyinfer::ExecutionContext context(model.graph());
    context.bind_input(model.input_id("input"),
                       tinyinfer::Tensor::from_vector({2}, {-5.0F, 1.0F}));
    tinyinfer::CpuBackend backend;
    tinyinfer::Executor executor(backend);
    executor.run(model.graph(), context);
    return context.output(model.output_id("output"));
}

void expect_output(const tinyinfer::Model& model) {
    const auto output = execute(model);
    assert(output.shape() == tinyinfer::Shape({2}));
    assert(std::fabs(output.at(0) - (-1.0F)) <= 1e-6F);
    assert(std::fabs(output.at(1) - 7.0F) <= 1e-6F);
}

}  // namespace

int main() {
    using namespace tinyinfer;
    using namespace tinyinfer::optimizer;

    const auto fixture = make_fixture();
    expect_output(fixture.model);

    ConstantFoldingPass folding;
    const auto folded = folding.run(fixture.model);
    assert(folded.changed());
    assert(folded.pass_statistics.front().nodes_rewritten == 2);
    assert(folded.model.graph().size() == 2);
    assert(folded.model.graph().is_constant(
        folded.mapped_value(fixture.final_folded)));
    assert(folded.mapped_value(1) == kInvalidValueId);
    assert(folded.mapped_value(2) == kInvalidValueId);
    expect_output(folded.model);

    DeadCodeEliminationPass elimination;
    const auto eliminated = elimination.run(fixture.model);
    assert(eliminated.changed());
    assert(eliminated.pass_statistics.front().nodes_rewritten == 1);
    assert(eliminated.model.graph().size() == 3);
    assert(eliminated.mapped_value(fixture.dead_output) == kInvalidValueId);
    expect_output(eliminated.model);

    PassManager pipeline;
    pipeline.add_pass(std::make_unique<ConstantFoldingPass>());
    pipeline.add_pass(std::make_unique<DeadCodeEliminationPass>());
    const auto optimized = pipeline.run(fixture.model);
    assert(optimized.changed());
    assert(optimized.pass_statistics.size() == 2);
    assert(optimized.model.graph().size() == 1);
    assert(optimized.model.graph().value_count() == 3);
    assert(optimized.mapped_value(fixture.first_folded) == kInvalidValueId);
    assert(optimized.mapped_value(fixture.final_folded) != kInvalidValueId);
    assert(optimized.mapped_value(fixture.dead_output) == kInvalidValueId);
    assert(optimized.mapped_value(fixture.output) ==
           optimized.model.output_id("output"));
    expect_output(optimized.model);

    const auto second_run = pipeline.run(optimized.model);
    assert(!second_run.changed());
    assert(second_run.model.graph().size() == 1);
    assert(second_run.model.graph().value_count() == 3);
    expect_output(second_run.model);

    Graph constant_graph;
    const auto passthrough = constant_graph.add_constant(
        "answer", Tensor::from_vector({1}, {42.0F}));
    constant_graph.mark_output(passthrough);
    const Model constant_model(std::move(constant_graph), {},
                               {{"answer", passthrough}});
    const auto constant_result = elimination.run(constant_model);
    assert(!constant_result.changed());
    assert(constant_result.model.graph().is_constant(
        constant_result.model.output_id("answer")));

    Graph repeated_input_graph;
    const auto repeated_constant = repeated_input_graph.add_constant(
        "repeated", Tensor::from_vector({2}, {2.0F, 3.0F}));
    const auto doubled = repeated_input_graph.add_node(
        "doubled", OpType::Add, {repeated_constant, repeated_constant});
    const auto doubled_output = repeated_input_graph.node(doubled).outputs.front();
    repeated_input_graph.mark_output(doubled_output);
    const Model repeated_input_model(std::move(repeated_input_graph), {},
                                     {{"doubled", doubled_output}});
    const auto repeated_result = folding.run(repeated_input_model);
    assert(repeated_result.model.graph().size() == 0);
    const auto& doubled_value = repeated_result.model.graph().constant(
        repeated_result.model.output_id("doubled"));
    assert(doubled_value.at(0) == 4.0F);
    assert(doubled_value.at(1) == 6.0F);
}
