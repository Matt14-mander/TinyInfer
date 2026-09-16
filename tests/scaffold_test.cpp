#include <cassert>
#include <stdexcept>
#include "tinyinfer/tinyinfer.h"

int main() {
    const tinyinfer::Tensor tensor({2, 3});
    assert((tensor.shape() == tinyinfer::Shape{2, 3}));
    assert((tensor.strides() == tinyinfer::Strides{3, 1}));
    assert(tensor.numel() == 6);
    assert(tensor.size_bytes() == 24);
    assert(tensor.is_contiguous());
    tinyinfer::Graph graph;
    const tinyinfer::TensorSpec spec{{2, 3}, tinyinfer::DataType::Float32};
    const auto input = graph.add_input("input", spec);
    const auto relu = graph.add_node("relu", tinyinfer::OpType::ReLU, {input});
    graph.mark_output(graph.node(relu).outputs.front());
    assert(graph.size() == 1);
    assert(graph.node(relu).inputs.front() == input);

    tinyinfer::CpuBackend cpu;
    tinyinfer::Executor executor(cpu);
    tinyinfer::ExecutionContext context(graph);
    context.bind_input(input, tinyinfer::Tensor::from_vector(
                                  {2, 3}, {-1.0F, 2.0F, -3.0F,
                                           4.0F, -5.0F, 6.0F}));
    executor.run(graph, context);
    const auto output = graph.node(relu).outputs.front();
    assert(context.output(output).at(0) == 0.0F);
    assert(context.output(output).at(5) == 6.0F);

    bool rejected = false;
    try { graph.add_node("invalid", tinyinfer::OpType::Add, {99}); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
    return 0;
}
