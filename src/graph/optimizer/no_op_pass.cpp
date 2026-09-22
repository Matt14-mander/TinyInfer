#include "tinyinfer/graph/optimizer/no_op_pass.h"

#include <numeric>
#include <utility>

namespace tinyinfer::optimizer {

OptimizationResult NoOpPass::run(const Model& model) const {
    std::vector<ValueId> mapping(model.graph().value_count());
    std::iota(mapping.begin(), mapping.end(), ValueId{0});

    PassStatistics statistics;
    statistics.pass_name = std::string(name());
    statistics.nodes_before = model.graph().size();
    statistics.nodes_after = model.graph().size();
    statistics.values_before = model.graph().value_count();
    statistics.values_after = model.graph().value_count();

    return OptimizationResult{model, std::move(mapping),
                              {std::move(statistics)}};
}

}  // namespace tinyinfer::optimizer
