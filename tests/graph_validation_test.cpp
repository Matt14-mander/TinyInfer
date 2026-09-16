#include "tinyinfer/graph/graph.h"

#include <cassert>
#include <stdexcept>
#include <vector>

namespace {
template <typename Exception, typename Function>
void expect_throw(Function&& function) {
    bool thrown = false;
    try { function(); } catch (const Exception&) { thrown = true; }
    assert(thrown);
}
}  // namespace

int main() {
    using tinyinfer::DataType;
    using tinyinfer::Graph;
    using tinyinfer::NodeId;
    using tinyinfer::OpType;
    using tinyinfer::Tensor;
    using tinyinfer::TensorSpec;

    const TensorSpec spec{{2, 3}, DataType::Float32};
    Graph graph;
    const auto lhs = graph.add_input("lhs", spec);
    const auto rhs = graph.add_input("rhs", spec);
    const auto add = graph.add_node("add", OpType::Add, {lhs, rhs});
    const auto sum = graph.node(add).outputs.front();
    const auto relu = graph.add_node("relu", OpType::ReLU, {sum});
    const auto gelu = graph.add_node("gelu", OpType::GELU, {sum});
    const auto merge = graph.add_node(
        "merge", OpType::Add,
        {graph.node(relu).outputs.front(), graph.node(gelu).outputs.front()});
    graph.mark_output(graph.node(merge).outputs.front());

    graph.validate();
    assert(graph.topological_order() ==
           std::vector<NodeId>({add, relu, gelu, merge}));

    Graph repeated_dependency;
    const auto input = repeated_dependency.add_input("x", spec);
    const auto unary = repeated_dependency.add_node("relu", OpType::ReLU,
                                                     {input});
    const auto value = repeated_dependency.node(unary).outputs.front();
    const auto doubled = repeated_dependency.add_node(
        "double", OpType::Add, {value, value});
    repeated_dependency.mark_output(
        repeated_dependency.node(doubled).outputs.front());
    repeated_dependency.validate();
    assert(repeated_dependency.topological_order() ==
           std::vector<NodeId>({unary, doubled}));

    Graph constant_graph;
    const auto constant = constant_graph.add_constant(
        "constant", Tensor::from_vector({2}, {1.0F, 2.0F}));
    constant_graph.mark_output(constant);
    constant_graph.validate();
    assert(constant_graph.topological_order().empty());

    Graph missing_output;
    missing_output.add_input("input", spec);
    expect_throw<std::invalid_argument>([&] { missing_output.validate(); });

    Graph invalid_schema;
    const auto matrix = invalid_schema.add_input("matrix", spec);
    expect_throw<std::invalid_argument>([&] {
        invalid_schema.add_node("bad_relu", OpType::ReLU, {matrix, matrix});
    });
}
