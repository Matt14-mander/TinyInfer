#pragma once

#include <cstddef>
#include <vector>

#include "tinyinfer/graph/graph.h"

namespace tinyinfer {

// A read-only snapshot of value uses and outputs. Rebuild after changing Graph.
class GraphAnalysis {
public:
    explicit GraphAnalysis(const Graph& graph);

    // Unique consuming nodes, ordered by NodeId.
    const std::vector<NodeId>& consumers(ValueId value) const;
    // Number of input positions that use the value (duplicates count).
    std::size_t use_count(ValueId value) const;
    bool is_graph_output(ValueId value) const;
    // Outputs of one node, in its declared order.
    const std::vector<ValueId>& outputs(NodeId node) const;
    // Registered graph outputs, in registration order.
    const std::vector<ValueId>& graph_outputs() const noexcept {
        return graph_outputs_;
    }

private:
    std::vector<std::vector<NodeId>> consumers_;
    std::vector<std::size_t> use_counts_;
    std::vector<bool> graph_output_flags_;
    std::vector<std::vector<ValueId>> node_outputs_;
    std::vector<ValueId> graph_outputs_;
};

}  // namespace tinyinfer
