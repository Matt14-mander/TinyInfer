#include "tinyinfer/graph/graph.h"

#include <cassert>
#include <cstdint>
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
    using tinyinfer::Graph;
    using tinyinfer::OpType;
    using tinyinfer::Shape;
    using tinyinfer::Tensor;
    using tinyinfer::TensorSpec;
    using tinyinfer::ValueKind;

    Graph graph;
    const TensorSpec input_spec{{2, 3}, DataType::Float32};
    const auto input = graph.add_input("x", input_spec);
    const auto weight = graph.add_constant(
        "weight", Tensor::from_vector(
                      {3, 2}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}));
    const TensorSpec output_spec{{2, 2}, DataType::Float32};
    const auto matmul = graph.add_node("matmul", OpType::MatMul,
                                       {input, weight});
    const auto output = graph.node(matmul).outputs.front();
    const auto softmax = graph.add_node(
        "softmax", OpType::Softmax, {output},
        {{"axis", std::int64_t{-1}}});
    const auto probabilities = graph.node(softmax).outputs.front();
    graph.mark_output(probabilities);

    assert(graph.size() == 2);
    assert(graph.value_count() == 4);
    assert(graph.inputs() == std::vector<tinyinfer::ValueId>({input}));
    assert(graph.outputs() ==
           std::vector<tinyinfer::ValueId>({probabilities}));
    assert(graph.value(input).kind == ValueKind::Input);
    assert(graph.value(input).spec == input_spec);
    assert(!graph.value(input).producer.has_value());
    assert(graph.value(weight).kind == ValueKind::Constant);
    assert(graph.is_constant(weight));
    assert(graph.constant(weight).at({2, 1}) == 6.0F);
    assert(graph.value(output).kind == ValueKind::Intermediate);
    assert(graph.value(output).name == "matmul:0");
    assert(graph.value(output).producer == matmul);
    assert(graph.value(output).spec == output_spec);
    assert(graph.node(matmul).inputs ==
           std::vector<tinyinfer::ValueId>({input, weight}));
    assert(!graph.node(matmul).has_attribute("axis"));
    assert(graph.node(softmax).attribute<std::int64_t>("axis") == -1);
    assert(graph.value(probabilities).spec == output_spec);

    expect_throw<std::invalid_argument>([&] {
        graph.add_input("x", input_spec);
    });
    expect_throw<std::invalid_argument>([&] {
        graph.add_input("", input_spec);
    });
    expect_throw<std::invalid_argument>([&] {
        graph.add_input("bad_shape", TensorSpec{{2, -1}, DataType::Float32});
    });
    expect_throw<std::invalid_argument>([&] {
        graph.add_node("missing_input", OpType::Add, {999});
    });
    expect_throw<std::invalid_argument>([&] {
        graph.add_node("wrong_arity", OpType::ReLU, {});
    });
    expect_throw<std::invalid_argument>([&] {
        graph.add_node("unknown_attribute", OpType::ReLU, {input},
                       {{"axis", std::int64_t{0}}});
    });
    expect_throw<std::invalid_argument>([&] {
        graph.mark_output(probabilities);
    });
    expect_throw<std::invalid_argument>([&] { graph.constant(input); });
    expect_throw<std::out_of_range>([&] { graph.value(999); });
    expect_throw<std::out_of_range>([&] { graph.node(999); });
    expect_throw<std::out_of_range>([&] {
        graph.node(matmul).attribute<std::int64_t>("axis");
    });
}
