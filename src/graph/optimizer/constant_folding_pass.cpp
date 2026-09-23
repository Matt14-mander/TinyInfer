#include "tinyinfer/graph/optimizer/constant_folding_pass.h"

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "tinyinfer/backend/cpu_backend.h"
#include "tinyinfer/graph/optimizer/graph_rewriter.h"
#include "tinyinfer/runtime/execution_context.h"
#include "tinyinfer/runtime/executor.h"

namespace tinyinfer::optimizer {
namespace {

Tensor evaluate_constant_node(const Node& node,
                              const std::vector<const Tensor*>& inputs) {
    Graph graph;
    std::vector<ValueId> input_ids;
    input_ids.reserve(inputs.size());
    for (std::size_t index = 0; index < inputs.size(); ++index) {
        input_ids.push_back(graph.add_constant(
            node.name + "/constant_input/" + std::to_string(index),
            *inputs[index]));
    }
    const auto node_id = graph.add_node(
        node.name, node.op, std::move(input_ids), node.attributes);
    const auto output = graph.node(node_id).outputs.front();
    graph.mark_output(output);

    CpuBackend backend;
    ExecutionContext context(graph);
    Executor executor(backend);
    executor.run(graph, context);
    return context.output(output);
}

}  // namespace

OptimizationResult ConstantFoldingPass::run(const Model& model) const {
    const auto& source = model.graph();
    GraphRewriter rewriter(model);
    CpuBackend backend;

    std::vector<const Tensor*> known_constants(source.value_count(), nullptr);
    std::vector<std::optional<Tensor>> folded_constants(source.value_count());
    for (ValueId id = 0; id < source.value_count(); ++id) {
        if (source.is_constant(id)) {
            known_constants[id] = &source.constant(id);
        }
    }

    std::size_t folded_nodes = 0;
    for (const auto node_id : source.topological_order()) {
        const auto& node = source.node(node_id);
        bool foldable = node.outputs.size() == 1 && backend.supports(node.op);
        std::vector<const Tensor*> inputs;
        inputs.reserve(node.inputs.size());
        for (const auto input : node.inputs) {
            inputs.push_back(known_constants[input]);
            foldable = foldable && known_constants[input] != nullptr;
        }

        if (!foldable) {
            rewriter.copy_node(node_id);
            continue;
        }

        auto folded = evaluate_constant_node(node, inputs);
        const auto output = node.outputs.front();
        folded_constants[output].emplace(folded);
        known_constants[output] = &*folded_constants[output];
        rewriter.replace_with_constant(output, std::move(folded));
        ++folded_nodes;
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
    statistics.nodes_rewritten = folded_nodes;
    statistics.changed = folded_nodes != 0;
    return OptimizationResult{std::move(optimized), std::move(mapping),
                              {std::move(statistics)}};
}

}  // namespace tinyinfer::optimizer
