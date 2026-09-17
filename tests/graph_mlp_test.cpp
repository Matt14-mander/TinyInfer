#include "tinyinfer/tinyinfer.h"

#include <cassert>
#include <cmath>

namespace {

bool close(float actual, float expected, float tolerance = 1e-5F) {
    return std::fabs(actual - expected) <= tolerance;
}

void expect_close(const tinyinfer::Tensor& actual,
                  const tinyinfer::Tensor& expected) {
    assert(actual.shape() == expected.shape());
    for (std::size_t index = 0; index < actual.numel(); ++index) {
        assert(close(actual.at(index), expected.at(index)));
    }
}

}  // namespace

int main() {
    using tinyinfer::CpuBackend;
    using tinyinfer::DataType;
    using tinyinfer::ExecutionContext;
    using tinyinfer::Executor;
    using tinyinfer::Graph;
    using tinyinfer::OpType;
    using tinyinfer::Tensor;
    using tinyinfer::TensorSpec;

    const auto w1_tensor = Tensor::from_vector(
        {2, 3}, {1.0F, 0.0F, -1.0F, 1.0F, -1.0F, 1.0F});
    const auto b1_tensor = Tensor::from_vector(
        {3}, {0.0F, 0.0F, 1.0F});
    const auto w2_tensor = Tensor::from_vector(
        {3, 2}, {1.0F, 0.0F, 1.0F, 2.0F, 0.0F, 1.0F});
    const auto b2_tensor = Tensor::from_vector({2}, {0.0F, -1.0F});

    Graph graph;
    const auto x = graph.add_input(
        "x", TensorSpec{{1, 2}, DataType::Float32});
    const auto w1 = graph.add_constant("w1", w1_tensor);
    const auto b1 = graph.add_constant("b1", b1_tensor);
    const auto w2 = graph.add_constant("w2", w2_tensor);
    const auto b2 = graph.add_constant("b2", b2_tensor);

    const auto matmul1 = graph.add_node("matmul1", OpType::MatMul, {x, w1});
    const auto add1 = graph.add_node(
        "add1", OpType::Add, {graph.node(matmul1).outputs.front(), b1});
    const auto relu = graph.add_node(
        "relu", OpType::ReLU, {graph.node(add1).outputs.front()});
    const auto matmul2 = graph.add_node(
        "matmul2", OpType::MatMul,
        {graph.node(relu).outputs.front(), w2});
    const auto add2 = graph.add_node(
        "add2", OpType::Add, {graph.node(matmul2).outputs.front(), b2});
    const auto logits = graph.node(add2).outputs.front();
    const auto softmax = graph.add_node("softmax", OpType::Softmax, {logits});
    const auto probabilities = graph.node(softmax).outputs.front();
    graph.mark_output(probabilities);

    CpuBackend backend;
    Executor executor(backend);
    ExecutionContext context(graph);

    const auto first_input = Tensor::from_vector({1, 2}, {1.0F, -2.0F});
    context.bind_input(x, first_input);
    executor.run(graph, context);

    const auto eager_hidden = tinyinfer::ops::relu(
        tinyinfer::ops::linear(first_input, w1_tensor, b1_tensor));
    const auto eager_logits = tinyinfer::ops::linear(
        eager_hidden, w2_tensor, b2_tensor);
    const auto eager_probabilities = tinyinfer::ops::softmax(eager_logits);

    expect_close(context.value(logits), eager_logits);
    expect_close(context.output(probabilities), eager_probabilities);
    assert(close(context.value(logits).at(0), 2.0F));
    assert(close(context.value(logits).at(1), 3.0F));
    assert(close(context.output(probabilities).at(0), 0.26894143F));
    assert(close(context.output(probabilities).at(1), 0.73105860F));

    const auto second_input = Tensor::from_vector({1, 2}, {2.0F, 1.0F});
    context.bind_input(x, second_input);
    executor.run(graph, context);
    const auto second_eager_output = tinyinfer::ops::softmax(
        tinyinfer::ops::linear(
            tinyinfer::ops::relu(
                tinyinfer::ops::linear(second_input, w1_tensor, b1_tensor)),
            w2_tensor, b2_tensor));
    expect_close(context.output(probabilities), second_eager_output);
}
