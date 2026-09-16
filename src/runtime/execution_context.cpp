#include "tinyinfer/runtime/execution_context.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace tinyinfer {

ExecutionContext::ExecutionContext(const Graph& graph)
    : graph_(graph),
      graph_value_count_(graph.value_count()),
      values_(graph_value_count_) {
    graph_.validate();
    for (const auto& graph_value : graph_.values()) {
        if (graph_value.kind == ValueKind::Constant) {
            values_[graph_value.id].emplace(graph_.constant(graph_value.id));
        }
    }
}

void ExecutionContext::bind_input(ValueId id, Tensor tensor) {
    require_current_graph();
    if (graph_.value(id).kind != ValueKind::Input) {
        throw std::invalid_argument("only graph inputs can be bound as inputs");
    }
    require_matching_spec(id, tensor);
    values_[id] = std::move(tensor);
}

void ExecutionContext::set_value(ValueId id, Tensor tensor) {
    require_current_graph();
    if (graph_.value(id).kind != ValueKind::Intermediate) {
        throw std::invalid_argument("only intermediate graph values can be written");
    }
    require_matching_spec(id, tensor);
    values_[id] = std::move(tensor);
}

bool ExecutionContext::has_value(ValueId id) const {
    return slot(id).has_value();
}

bool ExecutionContext::all_inputs_bound() const {
    require_current_graph();
    return std::all_of(graph_.inputs().begin(), graph_.inputs().end(),
                       [this](ValueId id) { return values_[id].has_value(); });
}

void ExecutionContext::require_all_inputs_bound() const {
    if (!all_inputs_bound()) {
        throw std::logic_error("not all graph inputs have been bound");
    }
}

Tensor& ExecutionContext::value(ValueId id) {
    auto& stored = slot(id);
    if (!stored) throw std::logic_error("graph value is not materialized");
    return *stored;
}

const Tensor& ExecutionContext::value(ValueId id) const {
    const auto& stored = slot(id);
    if (!stored) throw std::logic_error("graph value is not materialized");
    return *stored;
}

const Tensor& ExecutionContext::output(ValueId id) const {
    require_current_graph();
    const auto& outputs = graph_.outputs();
    if (std::find(outputs.begin(), outputs.end(), id) == outputs.end()) {
        throw std::invalid_argument("value is not a registered graph output");
    }
    return value(id);
}

void ExecutionContext::clear_intermediates() {
    require_current_graph();
    for (const auto& graph_value : graph_.values()) {
        if (graph_value.kind == ValueKind::Intermediate) {
            values_[graph_value.id].reset();
        }
    }
}

void ExecutionContext::require_current_graph() const {
    if (graph_.value_count() != graph_value_count_) {
        throw std::logic_error(
            "graph structure changed after ExecutionContext construction");
    }
}

void ExecutionContext::require_matching_spec(ValueId id,
                                             const Tensor& tensor) const {
    const auto& expected = graph_.value(id).spec;
    if (tensor.shape() != expected.shape || tensor.dtype() != expected.dtype) {
        throw std::invalid_argument("tensor does not match graph value spec");
    }
}

std::optional<Tensor>& ExecutionContext::slot(ValueId id) {
    require_current_graph();
    if (id >= values_.size()) throw std::out_of_range("graph value id is out of range");
    return values_[id];
}

const std::optional<Tensor>& ExecutionContext::slot(ValueId id) const {
    require_current_graph();
    if (id >= values_.size()) throw std::out_of_range("graph value id is out of range");
    return values_[id];
}

}  // namespace tinyinfer
