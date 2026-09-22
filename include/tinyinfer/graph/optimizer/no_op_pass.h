#pragma once

#include "tinyinfer/graph/optimizer/graph_pass.h"

namespace tinyinfer::optimizer {

class NoOpPass final : public GraphPass {
public:
    std::string_view name() const noexcept override { return "NoOpPass"; }
    OptimizationResult run(const Model& model) const override;
};

}  // namespace tinyinfer::optimizer
