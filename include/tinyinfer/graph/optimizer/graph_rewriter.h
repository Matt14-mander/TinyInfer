#pragma once

#include <vector>

#include "tinyinfer/model/model.h"

namespace tinyinfer::optimizer {

// Rebuilds a graph while recording how source ValueIds map to the new graph.
class GraphRewriter {
public:
    explicit GraphRewriter(const Model& source);

    ValueId copy_value(ValueId source_value);
    NodeId copy_node(NodeId source_node);
    NodeId replace_node(NodeId source_node, OpType replacement_op,
                        std::vector<ValueId> source_inputs,
                        NodeAttributes attributes = {});
    ValueId replace_with_constant(ValueId source_value, Tensor value);

    const std::vector<ValueId>& value_mapping() const noexcept {
        return value_mapping_;
    }

    Model finish();

private:
    const Model& source_;
    Graph graph_;
    std::vector<ValueId> value_mapping_;
    bool finished_{false};
};

}  // namespace tinyinfer::optimizer
