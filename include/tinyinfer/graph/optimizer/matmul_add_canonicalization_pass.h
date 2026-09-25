#pragma once

#include "tinyinfer/graph/optimizer/graph_pass.h"

namespace tinyinfer::optimizer {

// Rewrites a single-use MatMul followed by shape-compatible Add into Gemm.
class MatMulAddCanonicalizationPass final : public GraphPass {
public:
    std::string_view name() const noexcept override {
        return "MatMulAddCanonicalizationPass";
    }

    OptimizationResult run(const Model& model) const override;
};

}  // namespace tinyinfer::optimizer
