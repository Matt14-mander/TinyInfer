#include "tinyinfer/model/model.h"

#include <algorithm>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace tinyinfer {
namespace {

ValueId find_named_value(const std::vector<Model::NamedValue>& values,
                         const std::string& name, const char* kind) {
    const auto iterator = std::find_if(
        values.begin(), values.end(),
        [&](const Model::NamedValue& value) { return value.first == name; });
    if (iterator == values.end()) {
        throw std::out_of_range(std::string("model ") + kind +
                                " does not exist: " + name);
    }
    return iterator->second;
}

void require_unique_names(const std::vector<Model::NamedValue>& values,
                          const char* kind) {
    std::unordered_set<std::string> names;
    for (const auto& value : values) {
        if (value.first.empty() || !names.insert(value.first).second) {
            throw std::invalid_argument(std::string("model ") + kind +
                                        " names must be non-empty and unique");
        }
    }
}

}  // namespace

Model::Model(Graph graph, std::vector<NamedValue> inputs,
             std::vector<NamedValue> outputs)
    : graph_(std::move(graph)),
      inputs_(std::move(inputs)),
      outputs_(std::move(outputs)) {
    graph_.validate();
    require_unique_names(inputs_, "input");
    require_unique_names(outputs_, "output");
    if (inputs_.size() != graph_.inputs().size() ||
        outputs_.size() != graph_.outputs().size()) {
        throw std::invalid_argument(
            "model input/output bindings must cover the complete graph interface");
    }

    std::unordered_set<ValueId> bound_inputs;
    for (const auto& input : inputs_) {
        if (graph_.value(input.second).kind != ValueKind::Input ||
            !bound_inputs.insert(input.second).second) {
            throw std::invalid_argument("model input does not reference a graph input");
        }
    }
    std::unordered_set<ValueId> bound_outputs;
    for (const auto& output : outputs_) {
        const auto& graph_outputs = graph_.outputs();
        if (std::find(graph_outputs.begin(), graph_outputs.end(), output.second) ==
                graph_outputs.end() ||
            !bound_outputs.insert(output.second).second) {
            throw std::invalid_argument(
                "model output does not reference a registered graph output");
        }
    }
}

ValueId Model::input_id(const std::string& name) const {
    return find_named_value(inputs_, name, "input");
}

ValueId Model::output_id(const std::string& name) const {
    return find_named_value(outputs_, name, "output");
}

}  // namespace tinyinfer
