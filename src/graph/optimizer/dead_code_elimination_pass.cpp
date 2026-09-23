#include "tinyinfer/graph/optimizer/dead_code_elimination_pass.h"

#include <utility>
#include <vector>

#include "tinyinfer/graph/optimizer/graph_rewriter.h"

namespace tinyinfer::optimizer {

OptimizationResult DeadCodeEliminationPass::run(const Model& model) const {
    const auto& source = model.graph();
    std::vector<bool> live_values(source.value_count(), false);
    std::vector<bool> live_nodes(source.size(), false);
    std::vector<ValueId> pending(source.outputs().begin(), source.outputs().end());

    while (!pending.empty()) {
        const auto value_id = pending.back();
        pending.pop_back();
        if (live_values[value_id]) {
            continue;
        }
        live_values[value_id] = true;
        const auto& value = source.value(value_id);
        if (!value.producer) {
            continue;
        }
        const auto producer = *value.producer;
        if (!live_nodes[producer]) {
            live_nodes[producer] = true;
            const auto& node = source.node(producer);
            pending.insert(pending.end(), node.inputs.begin(), node.inputs.end());
        }
    }

    GraphRewriter rewriter(model);
    for (const auto node_id : source.topological_order()) {
        if (live_nodes[node_id]) {
            rewriter.copy_node(node_id);
        }
    }
    for (const auto output : source.outputs()) {
        if (rewriter.value_mapping()[output] == kInvalidValueId) {
            rewriter.copy_value(output);
        }
    }
    auto mapping = rewriter.value_mapping();
    auto optimized = rewriter.finish();

    PassStatistics statistics;
    statistics.pass_name = std::string(name());
    statistics.nodes_before = source.size();
    statistics.nodes_after = optimized.graph().size();
    statistics.values_before = source.value_count();
    statistics.values_after = optimized.graph().value_count();
    statistics.nodes_rewritten = source.size() - optimized.graph().size();
    statistics.changed = statistics.nodes_before != statistics.nodes_after ||
                         statistics.values_before != statistics.values_after;
    return OptimizationResult{std::move(optimized), std::move(mapping),
                              {std::move(statistics)}};
}

}  // namespace tinyinfer::optimizer
