#include "tinyinfer/runtime/memory_plan.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

#include "tinyinfer/core/memory/buffer.h"

namespace tinyinfer {
namespace {

std::size_t align_up(std::size_t value, std::size_t alignment) {
    if (value > std::numeric_limits<std::size_t>::max() - (alignment - 1)) {
        throw std::overflow_error("memory plan size overflow");
    }
    return (value + alignment - 1) & ~(alignment - 1);
}

}  // namespace

MemoryPlan::MemoryPlan(const Graph& graph)
    : graph_identity_(&graph),
      node_count_(graph.size()),
      lifetimes_(graph.value_count()),
      allocations_(graph.value_count()),
      releases_(graph.size()) {
    graph.validate();
    const auto order = graph.topological_order();
    std::vector<std::size_t> node_steps(graph.size());
    for (std::size_t step = 0; step < order.size(); ++step) {
        node_steps[order[step]] = step;
    }

    for (const auto& value : graph.values()) {
        if (value.kind != ValueKind::Intermediate) continue;
        const auto start = node_steps.at(*value.producer);
        const auto bytes = TensorLayout(value.spec.shape).size_bytes(value.spec.dtype);
        lifetimes_[value.id] = ValueLifetime{value.id, start, start, bytes};
        if (bytes > std::numeric_limits<std::size_t>::max() -
                        naive_intermediate_bytes_) {
            throw std::overflow_error("intermediate memory size overflow");
        }
        naive_intermediate_bytes_ += bytes;
    }

    for (std::size_t step = 0; step < order.size(); ++step) {
        const auto& node = graph.node(order[step]);
        for (const auto input : node.inputs) {
            if (lifetimes_[input]) {
                lifetimes_[input]->last_step =
                    std::max(lifetimes_[input]->last_step, step);
            }
        }
    }
    for (const auto output : graph.outputs()) {
        if (lifetimes_[output]) lifetimes_[output]->last_step = order.size();
    }
    for (const auto& lifetime : lifetimes_) {
        if (lifetime && lifetime->last_step < order.size()) {
            releases_[lifetime->last_step].push_back(lifetime->value);
        }
    }

    std::vector<ValueLifetime> intervals;
    for (const auto& lifetime : lifetimes_) {
        if (lifetime) intervals.push_back(*lifetime);
    }
    std::sort(intervals.begin(), intervals.end(),
              [](const ValueLifetime& lhs, const ValueLifetime& rhs) {
                  if (lhs.first_step != rhs.first_step) {
                      return lhs.first_step < rhs.first_step;
                  }
                  return lhs.value < rhs.value;
              });

    std::vector<std::size_t> available_after;
    std::vector<std::size_t> value_slots(graph.value_count());
    for (const auto& interval : intervals) {
        std::size_t selected = slot_capacities_.size();
        std::size_t selected_growth = std::numeric_limits<std::size_t>::max();
        for (std::size_t slot = 0; slot < slot_capacities_.size(); ++slot) {
            if (available_after[slot] >= interval.first_step) continue;
            const auto growth = interval.size_bytes > slot_capacities_[slot]
                                    ? interval.size_bytes - slot_capacities_[slot]
                                    : 0;
            if (growth < selected_growth) {
                selected = slot;
                selected_growth = growth;
            }
        }
        if (selected == slot_capacities_.size()) {
            slot_capacities_.push_back(interval.size_bytes);
            available_after.push_back(interval.last_step);
        } else {
            slot_capacities_[selected] =
                std::max(slot_capacities_[selected], interval.size_bytes);
            available_after[selected] = interval.last_step;
        }
        value_slots[interval.value] = selected;
    }

    std::vector<std::size_t> slot_offsets(slot_capacities_.size());
    for (std::size_t slot = 0; slot < slot_capacities_.size(); ++slot) {
        buffer_size_bytes_ = align_up(buffer_size_bytes_, kDefaultBufferAlignment);
        slot_offsets[slot] = buffer_size_bytes_;
        if (slot_capacities_[slot] >
            std::numeric_limits<std::size_t>::max() - buffer_size_bytes_) {
            throw std::overflow_error("memory plan buffer size overflow");
        }
        buffer_size_bytes_ += slot_capacities_[slot];
    }
    for (const auto& interval : intervals) {
        const auto slot = value_slots[interval.value];
        allocations_[interval.value] = BufferAllocation{
            slot, slot_offsets[slot], interval.size_bytes};
    }
}

bool MemoryPlan::has_allocation(ValueId value) const {
    if (value >= allocations_.size()) {
        throw std::out_of_range("memory plan ValueId is out of range");
    }
    return allocations_[value].has_value();
}

const BufferAllocation& MemoryPlan::allocation(ValueId value) const {
    if (!has_allocation(value)) {
        throw std::invalid_argument("graph value has no planned allocation");
    }
    return *allocations_[value];
}

const ValueLifetime& MemoryPlan::lifetime(ValueId value) const {
    if (value >= lifetimes_.size()) {
        throw std::out_of_range("memory plan ValueId is out of range");
    }
    if (!lifetimes_[value]) {
        throw std::invalid_argument("graph value has no intermediate lifetime");
    }
    return *lifetimes_[value];
}

const std::vector<ValueId>& MemoryPlan::values_released_after_step(
    std::size_t execution_step) const {
    if (execution_step >= releases_.size()) {
        throw std::out_of_range("execution step is out of range");
    }
    return releases_[execution_step];
}

}  // namespace tinyinfer
