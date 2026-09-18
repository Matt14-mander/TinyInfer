#pragma once

#include <string>
#include <utility>
#include <vector>

#include "tinyinfer/graph/graph.h"

namespace tinyinfer {

class Model {
public:
    using NamedValue = std::pair<std::string, ValueId>;

    Model(Graph graph, std::vector<NamedValue> inputs,
          std::vector<NamedValue> outputs);

    const Graph& graph() const noexcept { return graph_; }
    ValueId input_id(const std::string& name) const;
    ValueId output_id(const std::string& name) const;
    const std::vector<NamedValue>& inputs() const noexcept { return inputs_; }
    const std::vector<NamedValue>& outputs() const noexcept { return outputs_; }

private:
    Graph graph_;
    std::vector<NamedValue> inputs_;
    std::vector<NamedValue> outputs_;
};

}  // namespace tinyinfer
