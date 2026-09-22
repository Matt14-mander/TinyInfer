#include "tinyinfer/graph/optimizer/optimization_result.h"

#include <algorithm>
#include <stdexcept>

namespace tinyinfer::optimizer {

ValueId OptimizationResult::mapped_value(ValueId original) const {
    if (original >= value_mapping.size()) {
        throw std::out_of_range("optimization source ValueId is out of range");
    }
    return value_mapping[original];
}

bool OptimizationResult::changed() const noexcept {
    return std::any_of(
        pass_statistics.begin(), pass_statistics.end(),
        [](const PassStatistics& statistics) { return statistics.changed; });
}

}  // namespace tinyinfer::optimizer
