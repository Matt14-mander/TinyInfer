#include "tinyinfer/core/tensor_iterator.h"

#include <limits>
#include <stdexcept>
#include <utility>

namespace tinyinfer {

TensorIterator::TensorIterator(std::vector<TensorLayout> operands) {
    if (operands.empty()) {
        throw std::invalid_argument("TensorIterator requires at least one operand");
    }

    shape_ = broadcast_shape(operands);
    numel_ = TensorLayout(shape_).numel();
    operand_strides_.reserve(operands.size());
    for (const auto& operand : operands) {
        operand_strides_.push_back(broadcast_strides(operand, shape_));
    }
    coalesce_dimensions();
    detect_contiguous_operands();
}

TensorIterator::TensorIterator(
    std::initializer_list<TensorLayout> operands)
    : TensorIterator(std::vector<TensorLayout>(operands)) {}

const Strides& TensorIterator::operand_strides(std::size_t operand) const {
    if (operand >= operand_count()) {
        throw std::out_of_range("TensorIterator operand is out of range");
    }
    return operand_strides_[operand];
}

const Strides& TensorIterator::iteration_strides(std::size_t operand) const {
    if (operand >= operand_count()) {
        throw std::out_of_range("TensorIterator operand is out of range");
    }
    return iteration_strides_[operand];
}

bool TensorIterator::operand_is_contiguous(std::size_t operand) const {
    if (operand >= operand_count()) {
        throw std::out_of_range("TensorIterator operand is out of range");
    }
    return contiguous_operands_[operand];
}

std::size_t TensorIterator::operand_offset(
    std::size_t operand, std::size_t logical_index) const {
    const auto& strides = iteration_strides(operand);
    if (logical_index >= numel_) {
        throw std::out_of_range("TensorIterator logical index is out of range");
    }

    std::size_t offset = 0;
    for (std::size_t dimension = iteration_rank(); dimension > 0; --dimension) {
        const auto extent = static_cast<std::size_t>(iteration_shape_[dimension - 1]);
        const auto coordinate = logical_index % extent;
        logical_index /= extent;
        offset += coordinate * static_cast<std::size_t>(strides[dimension - 1]);
    }
    return offset;
}

void TensorIterator::coalesce_dimensions() {
    iteration_shape_ = shape_;
    iteration_strides_ = operand_strides_;

    // Size-one dimensions never change an offset, so removing them is safe.
    for (std::size_t dimension = iteration_shape_.size(); dimension > 0; --dimension) {
        const auto index = dimension - 1;
        if (iteration_shape_[index] != 1) continue;
        iteration_shape_.erase(iteration_shape_.begin() + index);
        for (auto& strides : iteration_strides_) {
            strides.erase(strides.begin() + index);
        }
    }

    if (iteration_shape_.empty()) {
        iteration_shape_.push_back(1);
        for (auto& strides : iteration_strides_) strides.push_back(0);
        return;
    }

    // Merge adjacent dimensions only when every operand advances linearly
    // across their boundary. Broadcast strides (zero) naturally participate.
    for (std::size_t inner = iteration_shape_.size(); inner > 1;) {
        const auto inner_index = inner - 1;
        const auto outer_index = inner_index - 1;
        const auto inner_extent = static_cast<std::size_t>(iteration_shape_[inner_index]);
        bool mergeable = true;
        for (const auto& strides : iteration_strides_) {
            const auto inner_stride = static_cast<std::size_t>(strides[inner_index]);
            const auto outer_stride = static_cast<std::size_t>(strides[outer_index]);
            if (inner_stride != 0 &&
                inner_extent > std::numeric_limits<std::size_t>::max() / inner_stride) {
                mergeable = false;
                break;
            }
            if (outer_stride != inner_stride * inner_extent) {
                mergeable = false;
                break;
            }
        }

        const auto outer_extent = static_cast<std::size_t>(iteration_shape_[outer_index]);
        if (mergeable && inner_extent != 0 &&
            outer_extent > static_cast<std::size_t>(
                               std::numeric_limits<std::int64_t>::max()) /
                               inner_extent) {
            mergeable = false;
        }
        if (!mergeable) {
            --inner;
            continue;
        }

        iteration_shape_[outer_index] = static_cast<std::int64_t>(
            outer_extent * inner_extent);
        iteration_shape_.erase(iteration_shape_.begin() + inner_index);
        for (auto& strides : iteration_strides_) {
            strides[outer_index] = strides[inner_index];
            strides.erase(strides.begin() + inner_index);
        }
        inner = iteration_shape_.size();
    }
}

void TensorIterator::detect_contiguous_operands() {
    contiguous_operands_.assign(operand_count(), true);
    contiguous_operand_count_ = 0;
    if (numel_ == 0) {
        contiguous_operand_count_ = operand_count();
        return;
    }
    for (std::size_t operand = 0; operand < operand_count(); ++operand) {
        std::size_t expected_stride = 1;
        for (std::size_t dimension = rank(); dimension > 0; --dimension) {
            const auto extent = static_cast<std::size_t>(shape_[dimension - 1]);
            if (extent <= 1) continue;
            const auto stride = static_cast<std::size_t>(
                operand_strides_[operand][dimension - 1]);
            if (stride != expected_stride) {
                contiguous_operands_[operand] = false;
                break;
            }
            expected_stride *= extent;
        }
        if (contiguous_operands_[operand]) ++contiguous_operand_count_;
    }
}

Shape TensorIterator::broadcast_shape(
    const std::vector<TensorLayout>& operands) {
    std::size_t rank = 0;
    for (const auto& operand : operands) {
        if (operand.rank() > rank) rank = operand.rank();
    }

    Shape result(rank, 1);
    for (const auto& operand : operands) {
        const auto leading = rank - operand.rank();
        for (std::size_t dimension = 0; dimension < operand.rank(); ++dimension) {
            const auto output_dimension = leading + dimension;
            const auto operand_extent = operand.shape()[dimension];
            auto& result_extent = result[output_dimension];
            if (result_extent == 1) {
                result_extent = operand_extent;
            } else if (operand_extent != 1 && operand_extent != result_extent) {
                throw std::invalid_argument("TensorIterator operands cannot be broadcast together");
            }
        }
    }
    return result;
}

Strides TensorIterator::broadcast_strides(
    const TensorLayout& operand, const Shape& output_shape) {
    Strides result(output_shape.size(), 0);
    const auto leading = output_shape.size() - operand.rank();
    for (std::size_t dimension = 0; dimension < operand.rank(); ++dimension) {
        const auto output_dimension = leading + dimension;
        const auto operand_extent = operand.shape()[dimension];
        const auto output_extent = output_shape[output_dimension];
        if (operand_extent == output_extent) {
            result[output_dimension] = operand.strides()[dimension];
        } else if (operand_extent != 1) {
            throw std::invalid_argument("TensorIterator operands cannot be broadcast together");
        }
    }
    return result;
}

}  // namespace tinyinfer
