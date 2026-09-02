#pragma once
#include <cstddef>
#include <string>
#include <vector>

namespace tinyinfer {
using NodeId = std::size_t;
enum class OpType { Input, Constant, Add, Multiply, Subtract, MatMul, ReLU, GELU, Softmax, LayerNorm };
struct Node {
    NodeId id{};
    std::string name;
    OpType op{OpType::Input};
    std::vector<NodeId> inputs;
};
}  // namespace tinyinfer
