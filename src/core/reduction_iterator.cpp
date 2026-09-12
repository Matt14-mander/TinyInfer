#include "tinyinfer/core/reduction_iterator.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace tinyinfer {

ReductionIterator::ReductionIterator(TensorLayout input_layout,
                                     std::vector<std::int64_t> axes,
                                     bool keepdim)
    : input_layout_(std::move(input_layout)), keepdim_(keepdim) {
    if (axes.empty()) {
        throw std::invalid_argument("reduction requires at least one axis");
    }

    reduced_.assign(input_layout_.rank(), false);
    axes_.reserve(axes.size());
    for (auto axis : axes) {
        if (axis < 0) axis += static_cast<std::int64_t>(input_layout_.rank());
        if (axis < 0 || axis >= static_cast<std::int64_t>(input_layout_.rank())) {
            throw std::out_of_range("reduction axis is out of range");
        }
        const auto normalized = static_cast<std::size_t>(axis);
        if (reduced_[normalized]) {
            throw std::invalid_argument("reduction axes must be unique");
        }
        reduced_[normalized] = true;
        axes_.push_back(normalized);
    }
    std::sort(axes_.begin(), axes_.end());

    for (std::size_t dimension = 0; dimension < input_layout_.rank(); ++dimension) {
        if (reduced_[dimension]) {
            if (keepdim_) output_shape_.push_back(1);
        } else {
            output_shape_.push_back(input_layout_.shape()[dimension]);
        }
    }
    output_numel_ = TensorLayout(output_shape_).numel();

    Shape reduction_shape;
    reduction_shape.reserve(axes_.size());
    for (const auto axis : axes_) {
        reduction_shape.push_back(input_layout_.shape()[axis]);
    }
    reduction_numel_ = TensorLayout(reduction_shape).numel();
    logical_strides_ = TensorLayout(input_layout_.shape()).strides();

    contiguous_reduction_ = input_layout_.is_contiguous();
    const auto first_reduced = input_layout_.rank() - axes_.size();
    for (std::size_t index = 0; index < axes_.size(); ++index) {
        if (axes_[index] != first_reduced + index) {
            contiguous_reduction_ = false;
            break;
        }
    }
}

std::size_t ReductionIterator::input_offset(
    std::size_t output_index, std::size_t reduction_index) const {
    return offset_with_strides(input_layout_.strides(), output_index,
                               reduction_index);
}

std::size_t ReductionIterator::input_logical_index(
    std::size_t output_index, std::size_t reduction_index) const {
    return offset_with_strides(logical_strides_, output_index,
                               reduction_index);
}

std::size_t ReductionIterator::offset_with_strides(
    const Strides& strides, std::size_t output_index,
    std::size_t reduction_index) const {
    if (output_index >= output_numel_) {
        throw std::out_of_range("reduction output index is out of range");
    }
    if (reduction_index >= reduction_numel_) {
        throw std::out_of_range("reduction index is out of range");
    }

    std::size_t result = 0;
    std::size_t output_dimension = output_shape_.size();
    for (std::size_t dimension = input_layout_.rank(); dimension > 0; --dimension) {
        const auto input_dimension = dimension - 1;
        if (reduced_[input_dimension]) continue;
        const auto mapped_output_dimension = keepdim_
                                                 ? input_dimension
                                                 : --output_dimension;
        const auto extent = static_cast<std::size_t>(
            output_shape_[mapped_output_dimension]);
        const auto coordinate = output_index % extent;
        output_index /= extent;
        result += coordinate * static_cast<std::size_t>(strides[input_dimension]);
    }

    for (std::size_t axis_index = axes_.size(); axis_index > 0; --axis_index) {
        const auto axis = axes_[axis_index - 1];
        const auto extent = static_cast<std::size_t>(input_layout_.shape()[axis]);
        const auto coordinate = reduction_index % extent;
        reduction_index /= extent;
        result += coordinate * static_cast<std::size_t>(strides[axis]);
    }
    return result;
}

}  // namespace tinyinfer
