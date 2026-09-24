#pragma once

#include <cstddef>
#include <optional>
#include <vector>

#include "tinyinfer/graph/graph.h"

namespace tinyinfer {

struct ValueLifetime {
    ValueId value{kInvalidValueId};
    std::size_t first_step{};
    std::size_t last_step{};
    std::size_t size_bytes{};
};

struct BufferAllocation {
    std::size_t slot{};
    std::size_t byte_offset{};
    std::size_t size_bytes{};
};

// Static execution-memory plan for Graph intermediate values.
class MemoryPlan {
public:
    explicit MemoryPlan(const Graph& graph);

    bool has_allocation(ValueId value) const;
    const BufferAllocation& allocation(ValueId value) const;
    const ValueLifetime& lifetime(ValueId value) const;

    std::size_t value_count() const noexcept { return allocations_.size(); }
    std::size_t node_count() const noexcept { return node_count_; }
    std::size_t slot_count() const noexcept { return slot_capacities_.size(); }
    std::size_t buffer_size_bytes() const noexcept { return buffer_size_bytes_; }
    std::size_t naive_intermediate_bytes() const noexcept {
        return naive_intermediate_bytes_;
    }
    bool matches(const Graph& graph) const noexcept {
        return graph_identity_ == &graph && value_count() == graph.value_count() &&
               node_count_ == graph.size();
    }
    const std::vector<std::size_t>& slot_capacities() const noexcept {
        return slot_capacities_;
    }
    const std::vector<ValueId>& values_released_after_step(
        std::size_t execution_step) const;

private:
    const Graph* graph_identity_{};
    std::size_t node_count_{};
    std::size_t buffer_size_bytes_{};
    std::size_t naive_intermediate_bytes_{};
    std::vector<std::optional<ValueLifetime>> lifetimes_;
    std::vector<std::optional<BufferAllocation>> allocations_;
    std::vector<std::size_t> slot_capacities_;
    std::vector<std::vector<ValueId>> releases_;
};

}  // namespace tinyinfer
