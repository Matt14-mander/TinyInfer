#include "tinyinfer/graph/graph.h"

#include <algorithm>
#include <functional>
#include <queue>
#include <set>
#include <stdexcept>
#include <unordered_set>
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

void Graph::validate() const {
    static_cast<void>(topological_order());
}

std::vector<NodeId> Graph::topological_order() const {
    if (constants_.size() != values_.size()) {
        throw std::logic_error("graph value and constant tables are inconsistent");
    }
    if (output_ids_.empty()) {
        throw std::invalid_argument("graph must register at least one output");
    }

    std::unordered_set<std::string> names;
    std::vector<bool> registered_input(values_.size(), false);
    for (const auto input : input_ids_) {
        if (input >= values_.size()) {
            throw std::logic_error("graph input id is out of range");
        }
        if (registered_input[input]) {
            throw std::logic_error("graph input is registered more than once");
        }
        registered_input[input] = true;
    }

    std::vector<bool> registered_output(values_.size(), false);
    for (const auto output : output_ids_) {
        if (output >= values_.size()) {
            throw std::logic_error("graph output id is out of range");
        }
        if (registered_output[output]) {
            throw std::logic_error("graph output is registered more than once");
        }
        registered_output[output] = true;
    }

    for (std::size_t id = 0; id < values_.size(); ++id) {
        const auto& graph_value = values_[id];
        if (graph_value.id != id) {
            throw std::logic_error("graph value id does not match its table index");
        }
        validate_spec(graph_value.spec);
        if (!names.insert(graph_value.name).second) {
            throw std::logic_error("graph contains duplicate names");
        }

        switch (graph_value.kind) {
            case ValueKind::Input:
                if (!registered_input[id] || graph_value.producer || constants_[id]) {
                    throw std::logic_error("graph input value metadata is inconsistent");
                }
                break;
            case ValueKind::Constant:
                if (registered_input[id] || graph_value.producer || !constants_[id]) {
                    throw std::logic_error("graph constant value metadata is inconsistent");
                }
                if (constants_[id]->shape() != graph_value.spec.shape ||
                    constants_[id]->dtype() != graph_value.spec.dtype) {
                    throw std::logic_error("graph constant TensorSpec is inconsistent");
                }
                break;
            case ValueKind::Intermediate:
                if (registered_input[id] || !graph_value.producer || constants_[id]) {
                    throw std::logic_error(
                        "graph intermediate value metadata is inconsistent");
                }
                if (*graph_value.producer >= nodes_.size()) {
                    throw std::logic_error("graph value producer is out of range");
                }
                break;
        }
    }

    std::vector<std::size_t> indegree(nodes_.size(), 0);
    std::vector<std::vector<NodeId>> consumers(nodes_.size());
    std::vector<bool> produced_value(values_.size(), false);
    for (std::size_t id = 0; id < nodes_.size(); ++id) {
        const auto& graph_node = nodes_[id];
        if (graph_node.id != id) {
            throw std::logic_error("graph node id does not match its table index");
        }
        if (!names.insert(graph_node.name).second) {
            throw std::logic_error("graph contains duplicate names");
        }
        if (graph_node.outputs.empty()) {
            throw std::logic_error("graph node must produce at least one value");
        }

        std::vector<TensorSpec> input_specs;
        input_specs.reserve(graph_node.inputs.size());
        std::set<NodeId> dependencies;
        for (const auto input : graph_node.inputs) {
            if (input >= values_.size()) {
                throw std::logic_error("graph node input value is out of range");
            }
            input_specs.push_back(values_[input].spec);
            if (values_[input].producer) {
                dependencies.insert(*values_[input].producer);
            }
        }

        const auto inferred = infer_output_specs(
            graph_node.op, input_specs, graph_node.attributes);
        if (inferred.size() != graph_node.outputs.size()) {
            throw std::logic_error("graph node output count violates its schema");
        }
        for (std::size_t index = 0; index < graph_node.outputs.size(); ++index) {
            const auto output = graph_node.outputs[index];
            if (output >= values_.size()) {
                throw std::logic_error("graph node output value is out of range");
            }
            if (produced_value[output]) {
                throw std::logic_error("graph value has multiple producers");
            }
            produced_value[output] = true;
            if (values_[output].kind != ValueKind::Intermediate ||
                values_[output].producer != graph_node.id ||
                values_[output].spec != inferred[index]) {
                throw std::logic_error(
                    "graph node output metadata is inconsistent");
            }
        }

        indegree[id] = dependencies.size();
        for (const auto dependency : dependencies) {
            if (dependency >= nodes_.size()) {
                throw std::logic_error("graph node dependency is out of range");
            }
            consumers[dependency].push_back(id);
        }
    }

    for (std::size_t id = 0; id < values_.size(); ++id) {
        if (values_[id].kind == ValueKind::Intermediate && !produced_value[id]) {
            throw std::logic_error("graph intermediate value has no producer");
        }
    }

    std::priority_queue<NodeId, std::vector<NodeId>, std::greater<NodeId>> ready;
    for (NodeId id = 0; id < nodes_.size(); ++id) {
        if (indegree[id] == 0) ready.push(id);
    }

    std::vector<NodeId> order;
    order.reserve(nodes_.size());
    while (!ready.empty()) {
        const auto id = ready.top();
        ready.pop();
        order.push_back(id);
        for (const auto consumer : consumers[id]) {
            if (--indegree[consumer] == 0) ready.push(consumer);
        }
    }
    if (order.size() != nodes_.size()) {
        throw std::invalid_argument("graph contains a cycle");
    }
    return order;
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
