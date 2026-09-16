#pragma once

#include <functional>
#include <unordered_map>

#include "tinyinfer/graph/node.h"

namespace tinyinfer {

class ExecutionContext;

namespace cpu {

using OperatorKernel = std::function<void(const Node&, ExecutionContext&)>;

class OperatorRegistry {
public:
    OperatorRegistry();

    void register_kernel(OpType op, OperatorKernel kernel);
    bool supports(OpType op) const noexcept;
    void execute(const Node& node, ExecutionContext& context) const;

private:
    std::unordered_map<OpType, OperatorKernel> kernels_;
};

}  // namespace cpu
}  // namespace tinyinfer
