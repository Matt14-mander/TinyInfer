#include <iostream>
#include "tinyinfer/tinyinfer.h"

int main() {
    tinyinfer::Tensor input({1, 4});
    tinyinfer::Graph graph;
    const tinyinfer::TensorSpec spec{{1, 4}, tinyinfer::DataType::Float32};
    const auto x = graph.add_input("input", spec);
    graph.add_node("relu", tinyinfer::OpType::ReLU, {x});
    tinyinfer::CpuBackend cpu;
    std::cout << "TinyInfer scaffold: tensor has " << input.numel() << " elements, graph has "
              << graph.size() << " nodes, backend is " << cpu.name() << ".\n";
    std::cout << "See tinyinfer_phase0_graph_mlp for numerical graph execution.\n";
    return 0;
}
