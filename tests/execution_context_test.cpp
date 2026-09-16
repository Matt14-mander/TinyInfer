#include "tinyinfer/runtime/execution_context.h"

#include <cassert>
#include <stdexcept>

namespace {

template <typename Exception, typename Function>
void expect_throw(Function&& function) {
    bool thrown = false;
    try {
        function();
    } catch (const Exception&) {
        thrown = true;
    }
    assert(thrown);
}

}  // namespace

int main() {
    using tinyinfer::DataType;
    using tinyinfer::ExecutionContext;
    using tinyinfer::Graph;
    using tinyinfer::OpType;
    using tinyinfer::Tensor;
    using tinyinfer::TensorSpec;

    Graph graph;
    const TensorSpec spec{{2}, DataType::Float32};
    const auto input = graph.add_input("input", spec);
    const auto bias = graph.add_constant(
        "bias", Tensor::from_vector({2}, {0.5F, -0.5F}));
    const auto add = graph.add_node("add", OpType::Add, {input, bias});
    const auto sum = graph.node(add).outputs.front();
    const auto relu = graph.add_node("relu", OpType::ReLU, {sum});
    const auto result = graph.node(relu).outputs.front();
    graph.mark_output(result);

    ExecutionContext context(graph);
    assert(!context.has_value(input));
    assert(context.has_value(bias));
    assert(!context.has_value(sum));
    assert(!context.all_inputs_bound());
    assert(context.value(bias).at(0) == 0.5F);
    assert(context.value(bias).data() != graph.constant(bias).data());
    expect_throw<std::logic_error>([&] { context.require_all_inputs_bound(); });
    expect_throw<std::logic_error>([&] { context.value(input); });
    expect_throw<std::logic_error>([&] { context.output(result); });

    context.bind_input(input, Tensor::from_vector({2}, {1.0F, 2.0F}));
    assert(context.all_inputs_bound());
    context.require_all_inputs_bound();
    assert(context.value(input).at(1) == 2.0F);

    context.set_value(sum, Tensor::from_vector({2}, {1.5F, 1.5F}));
    context.set_value(result, Tensor::from_vector({2}, {1.5F, 1.5F}));
    assert(context.output(result).at(0) == 1.5F);
    expect_throw<std::invalid_argument>([&] { context.output(sum); });

    expect_throw<std::invalid_argument>([&] {
        context.bind_input(input, Tensor({3}));
    });
    expect_throw<std::invalid_argument>([&] {
        context.bind_input(input, Tensor({2}, DataType::Int32));
    });
    expect_throw<std::invalid_argument>([&] {
        context.bind_input(bias, Tensor({2}));
    });
    expect_throw<std::invalid_argument>([&] {
        context.set_value(input, Tensor({2}));
    });
    expect_throw<std::out_of_range>([&] { context.has_value(999); });

    context.clear_intermediates();
    assert(context.has_value(input));
    assert(context.has_value(bias));
    assert(!context.has_value(sum));
    assert(!context.has_value(result));
    assert(context.all_inputs_bound());

    graph.add_input("late_input", spec);
    expect_throw<std::logic_error>([&] { context.has_value(input); });
}
