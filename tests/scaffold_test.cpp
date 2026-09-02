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
    const auto input = graph.add_node("input", tinyinfer::OpType::Input);
    const auto relu = graph.add_node("relu", tinyinfer::OpType::ReLU, {input});
    assert(graph.size() == 2);
    assert(graph.node(relu).inputs.front() == input);

    tinyinfer::CpuBackend cpu;
    tinyinfer::Executor executor(cpu);
    bool rejected_unimplemented_op = false;
    try { executor.run(graph); }
    catch (const std::runtime_error&) { rejected_unimplemented_op = true; }
    assert(rejected_unimplemented_op);

    bool rejected = false;
    try { graph.add_node("invalid", tinyinfer::OpType::Add, {99}); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
    return 0;
}
