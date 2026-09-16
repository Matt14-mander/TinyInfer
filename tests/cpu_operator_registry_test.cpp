#include "tinyinfer/tinyinfer.h"

#include <cassert>
#include <cmath>
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

void expect_close(const tinyinfer::Tensor& actual,
                  const tinyinfer::Tensor& expected) {
    assert(actual.shape() == expected.shape());
    for (std::size_t index = 0; index < actual.numel(); ++index) {
        assert(std::fabs(actual.at(index) - expected.at(index)) < 1e-5F);
    }
}

}  // namespace

int main() {
    using tinyinfer::CpuBackend;
    using tinyinfer::DataType;
    using tinyinfer::ExecutionContext;
    using tinyinfer::Executor;
    using tinyinfer::Graph;
    using tinyinfer::Node;
    using tinyinfer::OpType;
    using tinyinfer::Tensor;
    using tinyinfer::TensorSpec;

    Graph graph;
    const TensorSpec matrix_spec{{2, 2}, DataType::Float32};
    const auto input_id = graph.add_input("input", matrix_spec);
    const auto rhs_id = graph.add_constant(
        "rhs", Tensor::from_vector({2, 2}, {1.0F, 2.0F, 3.0F, 4.0F}));
    const auto bias_id = graph.add_constant(
        "bias", Tensor::from_vector({2}, {0.5F, -0.5F}));
    const auto weight_id = graph.add_constant(
        "weight", Tensor::from_vector({2}, {1.5F, 0.5F}));

    const auto add = graph.add_node("add", OpType::Add, {input_id, rhs_id});
    const auto sum = graph.node(add).outputs.front();
    const auto subtract = graph.add_node(
        "subtract", OpType::Subtract, {sum, rhs_id});
    const auto difference = graph.node(subtract).outputs.front();
    const auto multiply = graph.add_node(
        "multiply", OpType::Multiply, {difference, rhs_id});
    const auto product = graph.node(multiply).outputs.front();
    const auto matmul = graph.add_node("matmul", OpType::MatMul,
                                       {input_id, rhs_id});
    const auto matrix_product = graph.node(matmul).outputs.front();
    const auto relu = graph.add_node("relu", OpType::ReLU, {product});
    const auto activated = graph.node(relu).outputs.front();
    const auto gelu = graph.add_node("gelu", OpType::GELU, {product});
    const auto gelu_output = graph.node(gelu).outputs.front();
    const auto softmax = graph.add_node(
        "softmax", OpType::Softmax, {activated}, {{"axis", std::int64_t{0}}});
    const auto probabilities = graph.node(softmax).outputs.front();
    const auto layer_norm = graph.add_node(
        "layer_norm", OpType::LayerNorm,
        {matrix_product, weight_id, bias_id}, {{"epsilon", 1e-4F}});
    const auto normalized = graph.node(layer_norm).outputs.front();

    for (const auto output :
         {difference, matrix_product, gelu_output, probabilities, normalized}) {
        graph.mark_output(output);
    }

    const auto input = Tensor::from_vector(
        {2, 2}, {-1.0F, 2.0F, -3.0F, 4.0F});
    ExecutionContext context(graph);
    context.bind_input(input_id, input);

    CpuBackend backend;
    for (const auto op : {OpType::Add, OpType::Subtract, OpType::Multiply,
                          OpType::MatMul, OpType::ReLU, OpType::GELU,
                          OpType::Softmax, OpType::LayerNorm}) {
        assert(backend.supports(op));
    }

    Executor executor(backend);
    executor.run(graph, context);

    expect_close(context.output(difference), input);
    expect_close(context.output(matrix_product),
                 tinyinfer::ops::matmul(input, context.value(rhs_id)));
    const auto expected_product = tinyinfer::ops::mul(input, context.value(rhs_id));
    expect_close(context.output(gelu_output),
                 tinyinfer::ops::gelu(expected_product));
    expect_close(context.output(probabilities),
                 tinyinfer::ops::softmax(
                     tinyinfer::ops::relu(expected_product), 0));
    expect_close(context.output(normalized),
                 tinyinfer::ops::layer_norm(
                     context.output(matrix_product), context.value(weight_id),
                     context.value(bias_id), 1e-4F));

    tinyinfer::cpu::OperatorRegistry registry;
    expect_throw<std::invalid_argument>([&] {
        registry.register_kernel(OpType::Add, {});
    });
    expect_throw<std::invalid_argument>([&] {
        registry.register_kernel(
            OpType::Add, [](const Node&, ExecutionContext&) {});
    });

    Graph another_graph;
    const auto another_input = another_graph.add_input("input", matrix_spec);
    const auto another_relu = another_graph.add_node(
        "relu", OpType::ReLU, {another_input});
    another_graph.mark_output(another_graph.node(another_relu).outputs.front());
    ExecutionContext another_context(another_graph);
    another_context.bind_input(another_input, Tensor({2, 2}));
    expect_throw<std::invalid_argument>([&] {
        executor.run(graph, another_context);
    });

    Node unknown;
    unknown.op = static_cast<OpType>(999);
    assert(!backend.supports(unknown.op));
    expect_throw<std::runtime_error>([&] {
        backend.execute(unknown, context);
    });
}
