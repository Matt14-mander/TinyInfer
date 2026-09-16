#include "tinyinfer/graph/graph.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include "tinyinfer/ops/operator_schema.h"

namespace tinyinfer {

ValueId Graph::add_input(std::string name, TensorSpec spec) {
    require_unique_name(name);
    validate_spec(spec);
    const auto id = values_.size();
    values_.push_back(GraphValue{id, std::move(name), std::move(spec),
                                 ValueKind::Input, std::nullopt});
    constants_.emplace_back(std::nullopt);
    input_ids_.push_back(id);
    return id;
}

ValueId Graph::add_constant(std::string name, Tensor tensor) {
    require_unique_name(name);
    TensorSpec spec{tensor.shape(), tensor.dtype()};
    validate_spec(spec);
    const auto id = values_.size();
    values_.push_back(GraphValue{id, std::move(name), std::move(spec),
                                 ValueKind::Constant, std::nullopt});
    constants_.emplace_back(std::move(tensor));
    return id;
}

NodeId Graph::add_node(std::string name, OpType op,
                       std::vector<ValueId> inputs,
                       NodeAttributes attributes) {
    require_unique_name(name);
    std::vector<TensorSpec> input_specs;
    input_specs.reserve(inputs.size());
    for (const auto input : inputs) {
        if (input >= values_.size()) {
            throw std::invalid_argument("graph input value does not exist");
        }
        input_specs.push_back(values_[input].spec);
    }
    auto output_specs = infer_output_specs(op, input_specs, attributes);
    for (const auto& spec : output_specs) validate_spec(spec);
    for (std::size_t index = 0; index < output_specs.size(); ++index) {
        require_unique_name(name + ":" + std::to_string(index));
    }

    const auto node_id = nodes_.size();
    std::vector<ValueId> outputs;
    outputs.reserve(output_specs.size());
    values_.reserve(values_.size() + output_specs.size());
    constants_.reserve(constants_.size() + output_specs.size());
    nodes_.reserve(nodes_.size() + 1);
    for (std::size_t index = 0; index < output_specs.size(); ++index) {
        const auto value_id = values_.size();
        outputs.push_back(value_id);
        values_.push_back(GraphValue{
            value_id, name + ":" + std::to_string(index),
            std::move(output_specs[index]), ValueKind::Intermediate, node_id});
        constants_.emplace_back(std::nullopt);
    }
    nodes_.push_back(Node{node_id, std::move(name), op, std::move(inputs),
                          std::move(outputs), std::move(attributes)});
    return node_id;
}

const Node& Graph::node(NodeId id) const {
    if (id >= nodes_.size()) {
        throw std::out_of_range("graph node id is out of range");
    }
    return nodes_[id];
}

const GraphValue& Graph::value(ValueId id) const {
    if (id >= values_.size()) {
        throw std::out_of_range("graph value id is out of range");
    }
    return values_[id];
}

const Tensor& Graph::constant(ValueId id) const {
    value(id);
    if (!constants_[id]) {
        throw std::invalid_argument("graph value is not a constant");
    }
    return *constants_[id];
}

bool Graph::is_constant(ValueId id) const {
    value(id);
    return constants_[id].has_value();
}

void Graph::mark_output(ValueId id) {
    value(id);
    if (std::find(output_ids_.begin(), output_ids_.end(), id) !=
        output_ids_.end()) {
        throw std::invalid_argument("graph output is already registered");
    }
    output_ids_.push_back(id);
}

void Graph::validate_spec(const TensorSpec& spec) {
    const TensorLayout layout(spec.shape);
    layout.size_bytes(spec.dtype);
}

void Graph::require_unique_name(const std::string& name) const {
    if (name.empty()) throw std::invalid_argument("graph names must not be empty");
    const auto value_match = std::find_if(
        values_.begin(), values_.end(),
        [&](const GraphValue& value) { return value.name == name; });
    const auto node_match = std::find_if(
        nodes_.begin(), nodes_.end(),
        [&](const Node& node) { return node.name == name; });
    if (value_match != values_.end() || node_match != nodes_.end()) {
        throw std::invalid_argument("graph names must be unique");
    }
}

}  // namespace tinyinfer
