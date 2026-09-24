#include "tinyinfer/runtime/executor.h"
#include <stdexcept>

namespace tinyinfer {
void Executor::run(const Graph& graph, ExecutionContext& context) {
    if (&context.graph() != &graph) {
        throw std::invalid_argument("ExecutionContext belongs to another graph");
    }
    context.require_all_inputs_bound();
    context.clear_intermediates();
    const auto order = graph.topological_order();
    for (std::size_t step = 0; step < order.size(); ++step) {
        const auto node_id = order[step];
        const auto& node = graph.node(node_id);
        if (!backend_.supports(node.op)) throw std::runtime_error("operator is not supported by the selected backend");
        backend_.execute(node, context);
        context.release_after_step(step);
    }
}
}  // namespace tinyinfer
