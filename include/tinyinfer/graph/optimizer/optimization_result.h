#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "tinyinfer/model/model.h"

namespace tinyinfer::optimizer {

struct PassStatistics {
    std::string pass_name;
    std::size_t nodes_before{};
    std::size_t nodes_after{};
    std::size_t values_before{};
    std::size_t values_after{};
    std::size_t nodes_rewritten{};
    bool changed{};
};

struct OptimizationResult {
    Model model;

    // Maps ValueIds from the input of the complete optimization pipeline to
    // ValueIds in model. Removed values use kInvalidValueId.
    std::vector<ValueId> value_mapping;
    std::vector<PassStatistics> pass_statistics;

    ValueId mapped_value(ValueId original) const;
    bool changed() const noexcept;
};

}  // namespace tinyinfer::optimizer
