#pragma once
#include <string_view>
#include "tinyinfer/graph/node.h"

namespace tinyinfer {
class ExecutionContext;

class Backend {
public:
    virtual ~Backend() = default;
    virtual std::string_view name() const noexcept = 0;
    virtual bool supports(OpType op) const noexcept = 0;
    virtual void execute(const Node& node, ExecutionContext& context) = 0;
};
}  // namespace tinyinfer
