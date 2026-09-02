#include <iostream>
#include "tinyinfer/tinyinfer.h"

int main() {
    tinyinfer::Tensor input({1, 4});
    tinyinfer::Graph graph;
    const auto x = graph.add_node("input", tinyinfer::OpType::Input);
    graph.add_node("relu", tinyinfer::OpType::ReLU, {x});
    tinyinfer::CpuBackend cpu;
    std::cout << "TinyInfer scaffold: tensor has " << input.numel() << " elements, graph has "
              << graph.size() << " nodes, backend is " << cpu.name() << ".\n";
    std::cout << "Numerical execution will be added in the v0.1 Tensor Engine milestone.\n";
    return 0;
}
