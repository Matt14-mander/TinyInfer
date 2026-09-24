#include "tinyinfer/tinyinfer.h"

#include <cassert>
#include <cmath>
#include <memory>
#include <utility>

namespace {

struct Fixture {
    tinyinfer::Model model;
    tinyinfer::ValueId first;
    tinyinfer::ValueId second;
    tinyinfer::ValueId third;
    tinyinfer::ValueId output;
    tinyinfer::ValueId dead;
};

Fixture make_fixture() {
    using namespace tinyinfer;
    Graph graph;
    const auto input = graph.add_input(
        "input", TensorSpec{{4}, DataType::Float32});
    const auto bias = graph.add_constant(
        "bias", Tensor::from_vector({4}, {1.0F, -2.0F, 3.0F, -4.0F}));
    const auto scale = graph.add_constant(
        "scale", Tensor::from_vector({4}, {2.0F, 2.0F, 2.0F, 2.0F}));

    const auto add = graph.add_node("add", OpType::Add, {input, bias});
    const auto first = graph.node(add).outputs.front();
    const auto relu1 = graph.add_node("relu1", OpType::ReLU, {first});
    const auto second = graph.node(relu1).outputs.front();
    const auto mul = graph.add_node("mul", OpType::Multiply, {second, scale});
    const auto third = graph.node(mul).outputs.front();
    const auto relu2 = graph.add_node("relu2", OpType::ReLU, {third});
    const auto output = graph.node(relu2).outputs.front();
    const auto dead_node = graph.add_node("dead", OpType::GELU, {input});
    const auto dead = graph.node(dead_node).outputs.front();
    graph.mark_output(output);

    return {Model(std::move(graph), {{"input", input}}, {{"output", output}}),
            first, second, third, output, dead};
}

void bind_input(tinyinfer::ExecutionContext& context,
                const tinyinfer::Model& model) {
    context.bind_input(model.input_id("input"),
                       tinyinfer::Tensor::from_vector(
                           {4}, {-2.0F, 5.0F, -1.0F, 8.0F}));
}

void expect_output(const tinyinfer::Tensor& output) {
    const float expected[] = {0.0F, 6.0F, 4.0F, 8.0F};
    for (std::size_t index = 0; index < output.numel(); ++index) {
        assert(std::fabs(output.at(index) - expected[index]) <= 1e-6F);
    }
}

}  // namespace

int main() {
    using namespace tinyinfer;
    using namespace tinyinfer::optimizer;

    const auto fixture = make_fixture();
    PassManager optimizer;
    optimizer.add_pass(std::make_unique<DeadCodeEliminationPass>());
    const auto optimized = optimizer.run(fixture.model);
    assert(optimized.model.graph().size() == 4);
    assert(optimized.mapped_value(fixture.dead) == kInvalidValueId);

    const auto first = optimized.mapped_value(fixture.first);
    const auto second = optimized.mapped_value(fixture.second);
    const auto third = optimized.mapped_value(fixture.third);
    const auto output = optimized.mapped_value(fixture.output);
    MemoryPlan plan(optimized.model.graph());

    assert(plan.naive_intermediate_bytes() == 64);
    assert(plan.slot_count() == 2);
    assert(plan.buffer_size_bytes() == 32);
    assert(plan.allocation(first).slot != plan.allocation(second).slot);
    assert(plan.allocation(first).slot == plan.allocation(third).slot);
    assert(plan.allocation(second).slot == plan.allocation(output).slot);
    assert(plan.lifetime(first).first_step == 0);
    assert(plan.lifetime(first).last_step == 1);
    assert(plan.lifetime(output).last_step == optimized.model.graph().size());

    CpuBackend backend;
    Executor executor(backend);

    ExecutionContext reference(optimized.model.graph());
    bind_input(reference, optimized.model);
    executor.run(optimized.model.graph(), reference);
    expect_output(reference.output(optimized.model.output_id("output")));
    assert(reference.has_value(first));

    ExecutionContext planned(optimized.model.graph(), plan);
    assert(planned.uses_memory_plan());
    bind_input(planned, optimized.model);
    executor.run(optimized.model.graph(), planned);
    const auto output_id = optimized.model.output_id("output");
    expect_output(planned.output(output_id));
    assert(!planned.has_value(first));
    assert(!planned.has_value(second));
    assert(!planned.has_value(third));
    assert(planned.has_value(output_id));

    const auto* first_run_buffer =
        planned.output(output_id).storage().buffer().get();
    bind_input(planned, optimized.model);
    executor.run(optimized.model.graph(), planned);
    expect_output(planned.output(output_id));
    assert(planned.output(output_id).storage().buffer().get() ==
           first_run_buffer);

    bool threw = false;
    try {
        static_cast<void>(plan.allocation(optimized.model.input_id("input")));
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}
