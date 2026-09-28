#include "tinyinfer/graph/optimizer/gemm_activation_fusion_pass.h"

#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "tinyinfer/graph/graph_analysis.h"
#include "tinyinfer/graph/optimizer/graph_rewriter.h"
#include "tinyinfer/ops/operator_schema.h"

namespace tinyinfer::optimizer {
namespace {

struct Candidate {
    NodeId gemm;
    std::vector<ValueId> inputs;
    NodeAttributes attributes;
};

std::optional<Candidate> match(const Graph& graph,
                               const GraphAnalysis& analysis,
                               const Node& activation) {
    if (activation.op != OpType::ReLU || activation.inputs.size() != 1 ||
        activation.outputs.size() != 1) {
        return std::nullopt;
    }
    const auto gemm_output = activation.inputs.front();
    const auto producer = graph.value(gemm_output).producer;
    if (!producer || graph.node(*producer).op != OpType::Gemm ||
        analysis.use_count(gemm_output) != 1 ||
        analysis.is_graph_output(gemm_output)) {
        return std::nullopt;
    }
    const auto& consumers = analysis.consumers(gemm_output);
    if (consumers.size() != 1 || consumers.front() != activation.id) {
        return std::nullopt;
    }

    const auto& gemm = graph.node(*producer);
    if (gemm.outputs.size() != 1 || gemm.outputs.front() != gemm_output) {
        return std::nullopt;
    }
    auto attributes = gemm.attributes;
    attributes["activation"] = std::string{"relu"};
    std::vector<TensorSpec> specs;
    specs.reserve(gemm.inputs.size());
    for (const auto input : gemm.inputs) specs.push_back(graph.value(input).spec);
    try {
        const auto inferred = infer_output_specs(
            OpType::FusedGemmActivation, specs, attributes);
        if (inferred.size() != 1 ||
            inferred.front() != graph.value(activation.outputs.front()).spec) {
            return std::nullopt;
        }
    } catch (const std::invalid_argument&) {
        return std::nullopt;
    } catch (const std::out_of_range&) {
        return std::nullopt;
    } catch (const std::overflow_error&) {
        return std::nullopt;
    }
    return Candidate{*producer, gemm.inputs, std::move(attributes)};
}

}  // namespace

OptimizationResult GemmActivationFusionPass::run(const Model& model) const {
    const auto& source = model.graph();
    const GraphAnalysis analysis(source);
    std::vector<std::optional<Candidate>> candidates(source.size());
    std::vector<bool> removed_gemm(source.size(), false);
    std::size_t matched = 0;
    for (const auto& node : source.nodes()) {
        auto candidate = match(source, analysis, node);
        if (!candidate) continue;
        removed_gemm[candidate->gemm] = true;
        candidates[node.id] = std::move(candidate);
        ++matched;
    }

    if (matched == 0) {
        std::vector<ValueId> identity(source.value_count());
        for (ValueId id = 0; id < identity.size(); ++id) identity[id] = id;
        PassStatistics statistics;
        statistics.pass_name = std::string(name());
        statistics.nodes_before = statistics.nodes_after = source.size();
        statistics.values_before = statistics.values_after = source.value_count();
        return OptimizationResult{model, std::move(identity),
                                  {std::move(statistics)}};
    }

    GraphRewriter rewriter(model);
    std::size_t fused_nodes = 0;
    for (const auto node_id : source.topological_order()) {
        if (removed_gemm[node_id]) continue;
        if (candidates[node_id]) {
            rewriter.replace_node(
                node_id, OpType::FusedGemmActivation,
                candidates[node_id]->inputs, candidates[node_id]->attributes);
            ++fused_nodes;
        } else {
            rewriter.copy_node(node_id);
        }
    }
    for (const auto& value : source.values()) {
        if (value.kind == ValueKind::Constant &&
            rewriter.value_mapping()[value.id] == kInvalidValueId) {
            rewriter.copy_value(value.id);
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
    statistics.nodes_rewritten = fused_nodes;
    statistics.changed = fused_nodes != 0;
    return OptimizationResult{std::move(optimized), std::move(mapping),
                              {std::move(statistics)}};
}

}  // namespace tinyinfer::optimizer
