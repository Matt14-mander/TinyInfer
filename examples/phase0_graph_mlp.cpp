#include <iostream>

#include "tinyinfer/tinyinfer.h"

int main() {
    using tinyinfer::CpuBackend;
    using tinyinfer::DataType;
    using tinyinfer::ExecutionContext;
    using tinyinfer::Executor;
    using tinyinfer::Graph;
    using tinyinfer::OpType;
    using tinyinfer::Tensor;
    using tinyinfer::TensorSpec;

    Graph graph;
    const auto x = graph.add_input(
        "x", TensorSpec{{1, 2}, DataType::Float32});
    const auto w1 = graph.add_constant(
        "w1", Tensor::from_vector({2, 3}, {1.0F, 0.0F, -1.0F,
                                             1.0F, -1.0F, 1.0F}));
    const auto b1 = graph.add_constant(
        "b1", Tensor::from_vector({3}, {0.0F, 0.0F, 1.0F}));
    const auto w2 = graph.add_constant(
        "w2", Tensor::from_vector({3, 2}, {1.0F, 0.0F,
                                             1.0F, 2.0F,
                                             0.0F, 1.0F}));
    const auto b2 = graph.add_constant(
        "b2", Tensor::from_vector({2}, {0.0F, -1.0F}));

    const auto linear1 = graph.add_node("linear1.matmul", OpType::MatMul,
                                        {x, w1});
    const auto biased1 = graph.add_node(
        "linear1.bias", OpType::Add,
        {graph.node(linear1).outputs.front(), b1});
    const auto relu = graph.add_node(
        "relu", OpType::ReLU, {graph.node(biased1).outputs.front()});
    const auto linear2 = graph.add_node(
        "linear2.matmul", OpType::MatMul,
        {graph.node(relu).outputs.front(), w2});
    const auto biased2 = graph.add_node(
        "linear2.bias", OpType::Add,
        {graph.node(linear2).outputs.front(), b2});
    const auto logits = graph.node(biased2).outputs.front();
    const auto softmax = graph.add_node(
        "softmax", OpType::Softmax, {logits});
    const auto probabilities = graph.node(softmax).outputs.front();
    graph.mark_output(probabilities);

    ExecutionContext context(graph);
    context.bind_input(
        x, Tensor::from_vector({1, 2}, {1.0F, -2.0F}));

    CpuBackend backend;
    Executor executor(backend);
    executor.run(graph, context);

    std::cout << "logits: " << context.value(logits) << '\n'
              << "probabilities: " << context.output(probabilities) << '\n';
}
