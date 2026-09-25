#include "tinyinfer/graph/graph_analysis.h"

#include <stdexcept>

namespace tinyinfer {

GraphAnalysis::GraphAnalysis(const Graph& graph)
    : consumers_(graph.value_count()),
      use_counts_(graph.value_count(), 0),
      graph_output_flags_(graph.value_count(), false),
      node_outputs_(graph.size()),
      graph_outputs_(graph.outputs()) {
    graph.validate();
    for (const auto& node : graph.nodes()) {
        node_outputs_[node.id] = node.outputs;
        for (const auto input : node.inputs) {
            ++use_counts_[input];
            auto& consumers = consumers_[input];
            if (consumers.empty() || consumers.back() != node.id) {
                consumers.push_back(node.id);
            }
        }
    }
    for (const auto output : graph_outputs_) {
        graph_output_flags_[output] = true;
    }
}

const std::vector<NodeId>& GraphAnalysis::consumers(ValueId value) const {
    if (value >= consumers_.size()) {
        throw std::out_of_range("GraphAnalysis ValueId is out of range");
    }
    return consumers_[value];
}

std::size_t GraphAnalysis::use_count(ValueId value) const {
    if (value >= use_counts_.size()) {
        throw std::out_of_range("GraphAnalysis ValueId is out of range");
    }
    return use_counts_[value];
}

bool GraphAnalysis::is_graph_output(ValueId value) const {
    if (value >= graph_output_flags_.size()) {
        throw std::out_of_range("GraphAnalysis ValueId is out of range");
    }
    return graph_output_flags_[value];
}

const std::vector<ValueId>& GraphAnalysis::outputs(NodeId node) const {
    if (node >= node_outputs_.size()) {
        throw std::out_of_range("GraphAnalysis NodeId is out of range");
    }
    return node_outputs_[node];
}

}  // namespace tinyinfer
