#include "tinyinfer/tinyinfer.h"

#include <cassert>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>

namespace {

tinyinfer::Model make_model() {
    using tinyinfer::DataType;
    using tinyinfer::Graph;
    using tinyinfer::Model;
    using tinyinfer::OpType;
    using tinyinfer::Tensor;
    using tinyinfer::TensorSpec;

    Graph graph;
    const auto input = graph.add_input(
        "observation", TensorSpec{{1, 2}, DataType::Float32});
    const auto weight = graph.add_constant(
        "weight", Tensor::from_vector(
                      {2, 2}, {1.0F, -1.0F, 0.5F, 2.0F}));
    const auto bias = graph.add_constant(
        "bias", Tensor::from_vector({2}, {0.25F, -0.5F}));
    const auto gemm = graph.add_node(
        "policy", OpType::Gemm, {input, weight, bias});
    const auto relu = graph.add_node(
        "activation", OpType::ReLU,
        {graph.node(gemm).outputs.front()});
    const auto softmax = graph.add_node(
        "probabilities", OpType::Softmax,
        {graph.node(relu).outputs.front()},
        {{"axis", std::int64_t{-1}}});
    const auto output = graph.node(softmax).outputs.front();
    graph.mark_output(output);

    return Model(std::move(graph), {{"observation", input}},
                 {{"probabilities", output}});
}

tinyinfer::Tensor execute(const tinyinfer::Model& model) {
    tinyinfer::ExecutionContext context(model.graph());
    context.bind_input(
        model.input_id("observation"),
        tinyinfer::Tensor::from_vector({1, 2}, {1.0F, -2.0F}));
    tinyinfer::CpuBackend backend;
    tinyinfer::Executor executor(backend);
    executor.run(model.graph(), context);
    return context.output(model.output_id("probabilities"));
}

void expect_close(const tinyinfer::Tensor& lhs,
                  const tinyinfer::Tensor& rhs) {
    assert(lhs.shape() == rhs.shape());
    assert(lhs.dtype() == rhs.dtype());
    for (std::size_t index = 0; index < lhs.numel(); ++index) {
        assert(std::fabs(lhs.at(index) - rhs.at(index)) <= 1e-6F);
    }
}

template <typename Exception, typename Function>
void expect_throw(Function&& function) {
    bool thrown = false;
    try {
        std::forward<Function>(function)();
    } catch (const Exception&) {
        thrown = true;
    }
    assert(thrown);
}

}  // namespace

int main() {
    using tinyinfer::kInvalidValueId;
    using tinyinfer::optimizer::NoOpPass;
    using tinyinfer::optimizer::PassManager;

    const auto model = make_model();
    const auto expected = execute(model);

    PassManager manager;
    assert(manager.empty());
    expect_throw<std::invalid_argument>([&] { manager.add_pass(nullptr); });
    manager.add_pass(std::make_unique<NoOpPass>());
    assert(manager.size() == 1);

    const auto optimized = manager.run(model);
    assert(!optimized.changed());
    assert(optimized.pass_statistics.size() == 1);
    const auto& statistics = optimized.pass_statistics.front();
    assert(statistics.pass_name == "NoOpPass");
    assert(statistics.nodes_before == model.graph().size());
    assert(statistics.nodes_after == model.graph().size());
    assert(statistics.values_before == model.graph().value_count());
    assert(statistics.values_after == model.graph().value_count());
    assert(statistics.nodes_rewritten == 0);
    assert(!statistics.changed);

    assert(optimized.model.inputs() == model.inputs());
    assert(optimized.model.outputs() == model.outputs());
    assert(optimized.model.input_id("observation") ==
           model.input_id("observation"));
    assert(optimized.model.output_id("probabilities") ==
           model.output_id("probabilities"));
    assert(optimized.model.graph().size() == model.graph().size());
    assert(optimized.model.graph().value_count() ==
           model.graph().value_count());

    for (tinyinfer::ValueId id = 0; id < model.graph().value_count(); ++id) {
        assert(optimized.mapped_value(id) == id);
        const auto& before = model.graph().value(id);
        const auto& after = optimized.model.graph().value(id);
        assert(before.id == after.id);
        assert(before.name == after.name);
        assert(before.spec == after.spec);
        assert(before.kind == after.kind);
        assert(before.producer == after.producer);
    }
    for (tinyinfer::NodeId id = 0; id < model.graph().size(); ++id) {
        const auto& before = model.graph().node(id);
        const auto& after = optimized.model.graph().node(id);
        assert(before.id == after.id);
        assert(before.name == after.name);
        assert(before.op == after.op);
        assert(before.inputs == after.inputs);
        assert(before.outputs == after.outputs);
        assert(before.attributes == after.attributes);
    }

    const auto weight = tinyinfer::ValueId{1};
    assert(model.graph().is_constant(weight));
    assert(optimized.model.graph().constant(weight).storage().buffer() !=
           model.graph().constant(weight).storage().buffer());
    expect_close(execute(optimized.model), expected);

    expect_throw<std::out_of_range>([&] {
        static_cast<void>(optimized.mapped_value(
            optimized.value_mapping.size()));
    });
    assert(kInvalidValueId != optimized.mapped_value(0));

    PassManager empty_manager;
    const auto unchanged = empty_manager.run(model);
    assert(!unchanged.changed());
    assert(unchanged.pass_statistics.empty());
    expect_close(execute(unchanged.model), expected);

    PassManager pipeline;
    pipeline.add_pass(std::make_unique<NoOpPass>());
    pipeline.add_pass(std::make_unique<NoOpPass>());
    const auto twice = pipeline.run(model);
    assert(twice.pass_statistics.size() == 2);
    assert(!twice.changed());
    for (tinyinfer::ValueId id = 0; id < model.graph().value_count(); ++id) {
        assert(twice.mapped_value(id) == id);
    }
    expect_close(execute(twice.model), expected);
}
