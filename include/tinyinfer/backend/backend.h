#pragma once
#include <string_view>
#include "tinyinfer/graph/node.h"
#include "tinyinfer/backend/capability.h"

namespace tinyinfer {
class ExecutionContext;

class Backend {
public:
    virtual ~Backend() = default;
    virtual std::string_view name() const noexcept = 0;
    virtual bool supports(OpType op) const noexcept = 0;
    // The legacy execute path uses host ExecutionContext. GPU plans own separate
    // device state; exposing device memory through host Tensor is not supported.
    virtual DeviceInfo device_info() const;
    virtual NodeCapability query_capability(const CapabilityQuery& query) const;
    virtual void execute(const Node& node, ExecutionContext& context) = 0;
};
}  // namespace tinyinfer
