#pragma once

#include "tinyinfer/graph/optimizer/graph_pass.h"

namespace tinyinfer::optimizer {

class ConstantFoldingPass final : public GraphPass {
public:
    std::string_view name() const noexcept override {
        return "ConstantFoldingPass";
    }

    OptimizationResult run(const Model& model) const override;
};

}  // namespace tinyinfer::optimizer
