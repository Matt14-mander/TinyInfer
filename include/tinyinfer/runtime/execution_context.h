#pragma once

#include <cstddef>
#include <optional>
#include <vector>

#include "tinyinfer/core/tensor.h"
#include "tinyinfer/graph/graph.h"

namespace tinyinfer {

class ExecutionContext {
public:
    explicit ExecutionContext(const Graph& graph);

    const Graph& graph() const noexcept { return graph_; }

    void bind_input(ValueId id, Tensor tensor);
    void set_value(ValueId id, Tensor tensor);

    bool has_value(ValueId id) const;
    bool all_inputs_bound() const;
    void require_all_inputs_bound() const;

    Tensor& value(ValueId id);
    const Tensor& value(ValueId id) const;
    const Tensor& output(ValueId id) const;

    void clear_intermediates();

private:
    void require_current_graph() const;
    void require_matching_spec(ValueId id, const Tensor& tensor) const;
    std::optional<Tensor>& slot(ValueId id);
    const std::optional<Tensor>& slot(ValueId id) const;

    const Graph& graph_;
    std::size_t graph_value_count_;
    std::vector<std::optional<Tensor>> values_;
};

}  // namespace tinyinfer
