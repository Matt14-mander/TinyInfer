#pragma once
#include "tinyinfer/backend/backend.h"
#include "tinyinfer/backend/cpu/operator_registry.h"

namespace tinyinfer {
class CpuBackend final : public Backend {
public:
    std::string_view name() const noexcept override { return "cpu"; }
    bool supports(OpType op) const noexcept override;
    void execute(const Node& node, ExecutionContext& context) override;

    const cpu::OperatorRegistry& registry() const noexcept { return registry_; }

private:
    cpu::OperatorRegistry registry_;
};
}  // namespace tinyinfer
