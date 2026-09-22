#pragma once

#include <string_view>

#include "tinyinfer/graph/optimizer/optimization_result.h"

namespace tinyinfer::optimizer {

class GraphPass {
public:
    virtual ~GraphPass() = default;

    virtual std::string_view name() const noexcept = 0;
    virtual OptimizationResult run(const Model& model) const = 0;
};

}  // namespace tinyinfer::optimizer
