#pragma once

#include "tinyinfer/graph/optimizer/graph_pass.h"

namespace tinyinfer::optimizer {

// Rewrites a single-use Gemm followed by ReLU into one internal fused op.
class GemmActivationFusionPass final : public GraphPass {
public:
    std::string_view name() const noexcept override {
        return "GemmActivationFusionPass";
    }

    OptimizationResult run(const Model& model) const override;
};

}  // namespace tinyinfer::optimizer
