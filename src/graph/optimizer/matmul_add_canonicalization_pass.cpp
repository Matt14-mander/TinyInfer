#include "tinyinfer/graph/optimizer/matmul_add_canonicalization_pass.h"

#include <cstdint>
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
    NodeId matmul;
    std::vector<ValueId> gemm_inputs;
};

NodeAttributes canonical_attributes() {
    return {{"alpha", 1.0F}, {"beta", 1.0F},
            {"transA", std::int64_t{0}}, {"transB", std::int64_t{0}}};
}

std::optional<Candidate> match(const Graph& graph,
                               const GraphAnalysis& analysis,
                               const Node& add) {
    if (add.op != OpType::Add || add.inputs.size() != 2 ||
        add.outputs.size() != 1) {
        return std::nullopt;
    }

    std::optional<std::size_t> product_index;
    NodeId matmul_id = kInvalidNodeId;
    for (std::size_t index = 0; index < add.inputs.size(); ++index) {
        const auto producer = graph.value(add.inputs[index]).producer;
        if (!producer || graph.node(*producer).op != OpType::MatMul) continue;
        if (product_index) return std::nullopt;  // Both inputs are MatMul values.
        product_index = index;
        matmul_id = *producer;
    }
    if (!product_index) return std::nullopt;

    const auto product = add.inputs[*product_index];
    const auto& matmul = graph.node(matmul_id);
    if (matmul.inputs.size() != 2 || matmul.outputs.size() != 1 ||
        matmul.outputs.front() != product ||
        analysis.use_count(product) != 1 ||
        analysis.is_graph_output(product)) {
        return std::nullopt;
    }
    const auto& consumers = analysis.consumers(product);
    if (consumers.size() != 1 || consumers.front() != add.id) {
        return std::nullopt;
    }

    const auto bias = add.inputs[1 - *product_index];
    std::vector<TensorSpec> specs;
    for (const auto value : {matmul.inputs[0], matmul.inputs[1], bias}) {
        specs.push_back(graph.value(value).spec);
    }
    try {
        const auto inferred = infer_output_specs(
            OpType::Gemm, specs, canonical_attributes());
        if (inferred.size() != 1 ||
            inferred.front() != graph.value(add.outputs.front()).spec) {
            return std::nullopt;
        }
    } catch (const std::invalid_argument&) {
        return std::nullopt;
    } catch (const std::out_of_range&) {
        return std::nullopt;
    } catch (const std::overflow_error&) {
        return std::nullopt;
    }
    return Candidate{matmul_id, {matmul.inputs[0], matmul.inputs[1], bias}};
}

}  // namespace

OptimizationResult MatMulAddCanonicalizationPass::run(
    const Model& model) const {
    const auto& source = model.graph();
    const GraphAnalysis analysis(source);
    std::vector<std::optional<Candidate>> candidates(source.size());
    std::vector<bool> removed_matmul(source.size(), false);
    std::size_t matched = 0;
    for (const auto& node : source.nodes()) {
        auto candidate = match(source, analysis, node);
        if (!candidate) continue;
        removed_matmul[candidate->matmul] = true;
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
        if (removed_matmul[node_id]) continue;
        if (candidates[node_id]) {
            rewriter.replace_node(node_id, OpType::Gemm,
                                  candidates[node_id]->gemm_inputs,
                                  canonical_attributes());
            ++fused_nodes;
        } else {
            rewriter.copy_node(node_id);
        }
    }
    // Preserve unrelated constants; removing dead values belongs to DCE.
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
