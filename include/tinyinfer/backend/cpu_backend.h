#pragma once
#include "tinyinfer/backend/backend.h"

namespace tinyinfer {
class CpuBackend final : public Backend {
public:
    std::string_view name() const noexcept override { return "cpu"; }
    bool supports(OpType op) const noexcept override;
    void execute(const Node& node) override;
};
}  // namespace tinyinfer
