#pragma once
#include <optional>
#include <string>
#include <vector>

#include "tinyinfer/core/tensor.h"
#include "tinyinfer/graph/node.h"

namespace tinyinfer {
class Graph {
public:
    ValueId add_input(std::string name, TensorSpec spec);
    ValueId add_constant(std::string name, Tensor value);
    NodeId add_node(std::string name, OpType op,
                    std::vector<ValueId> inputs,
                    std::vector<TensorSpec> output_specs,
                    NodeAttributes attributes = {});

    const Node& node(NodeId id) const;
    const GraphValue& value(ValueId id) const;
    const Tensor& constant(ValueId id) const;
    bool is_constant(ValueId id) const;

    const std::vector<Node>& nodes() const noexcept { return nodes_; }
    const std::vector<GraphValue>& values() const noexcept { return values_; }
    const std::vector<ValueId>& inputs() const noexcept { return input_ids_; }
    const std::vector<ValueId>& outputs() const noexcept { return output_ids_; }
    void mark_output(ValueId id);

    bool empty() const noexcept { return nodes_.empty(); }
    std::size_t size() const noexcept { return nodes_.size(); }
    std::size_t value_count() const noexcept { return values_.size(); }

private:
    static void validate_spec(const TensorSpec& spec);
    void require_unique_name(const std::string& name) const;

    std::vector<Node> nodes_;
    std::vector<GraphValue> values_;
    std::vector<std::optional<Tensor>> constants_;
    std::vector<ValueId> input_ids_;
    std::vector<ValueId> output_ids_;
};
}  // namespace tinyinfer
