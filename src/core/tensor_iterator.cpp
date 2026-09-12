#include "tinyinfer/core/tensor_iterator.h"

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

std::size_t TensorIterator::operand_offset(
    std::size_t operand, std::size_t logical_index) const {
    const auto& strides = operand_strides(operand);
    if (logical_index >= numel_) {
        throw std::out_of_range("TensorIterator logical index is out of range");
    }

    std::size_t offset = 0;
    for (std::size_t dimension = rank(); dimension > 0; --dimension) {
        const auto extent = static_cast<std::size_t>(shape_[dimension - 1]);
        const auto coordinate = logical_index % extent;
        logical_index /= extent;
        offset += coordinate * static_cast<std::size_t>(strides[dimension - 1]);
    }
    return offset;
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
