#include "tinyinfer/runtime/executor.h"
#include <stdexcept>

namespace tinyinfer {
void Executor::run(const Graph& graph, ExecutionContext& context) {
    if (&context.graph() != &graph) {
        throw std::invalid_argument("ExecutionContext belongs to another graph");
    }
    context.require_all_inputs_bound();
    context.clear_intermediates();
    for (const auto node_id : graph.topological_order()) {
        const auto& node = graph.node(node_id);
        if (!backend_.supports(node.op)) throw std::runtime_error("operator is not supported by the selected backend");
        backend_.execute(node, context);
    }
}
}  // namespace tinyinfer
