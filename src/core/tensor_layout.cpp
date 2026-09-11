#include "tinyinfer/core/tensor_layout.h"

#include <limits>
#include <stdexcept>
#include <utility>

namespace tinyinfer {

TensorLayout::TensorLayout() : TensorLayout(Shape{}) {}

TensorLayout::TensorLayout(Shape shape)
    : shape_(std::move(shape)) {
    validate_shape(shape_);
    numel_ = checked_numel(shape_);
    strides_ = make_contiguous_strides(shape_);
    storage_span_ = numel_;
    contiguous_ = true;
}

TensorLayout::TensorLayout(Shape shape, Strides strides)
    : shape_(std::move(shape)), strides_(std::move(strides)) {
    validate_shape(shape_);
    if (shape_.size() != strides_.size()) {
        throw std::invalid_argument("shape and strides must have the same rank");
    }
    for (const auto stride : strides_) {
        if (stride < 0) throw std::invalid_argument("strides must be non-negative");
    }
    numel_ = checked_numel(shape_);
    storage_span_ = checked_storage_span(shape_, strides_);
    contiguous_ = strides_ == make_contiguous_strides(shape_);
}

std::size_t TensorLayout::size_bytes(DataType dtype) const {
    const auto element_size = size_of(dtype);
    if (numel_ > std::numeric_limits<std::size_t>::max() / element_size) {
        throw std::overflow_error("tensor byte size overflows size_t");
    }
    return numel_ * element_size;
}

std::size_t TensorLayout::offset(const Shape& indices) const {
    if (indices.size() != rank()) {
        throw std::invalid_argument("number of indices must match tensor rank");
    }

    std::size_t result = 0;
    for (std::size_t dimension = 0; dimension < rank(); ++dimension) {
        const auto index = indices[dimension];
        if (index < 0 || index >= shape_[dimension]) {
            throw std::out_of_range("tensor index is out of range");
        }
        result += static_cast<std::size_t>(index) *
                  static_cast<std::size_t>(strides_[dimension]);
    }
    return result;
}

std::size_t TensorLayout::offset(
    std::initializer_list<std::int64_t> indices) const {
    return offset(Shape(indices));
}

std::size_t TensorLayout::storage_offset(std::size_t logical_index) const {
    if (logical_index >= numel_) {
        throw std::out_of_range("logical tensor index is out of range");
    }

    std::size_t result = 0;
    for (std::size_t dimension = rank(); dimension > 0; --dimension) {
        const auto size = static_cast<std::size_t>(shape_[dimension - 1]);
        const auto coordinate = logical_index % size;
        logical_index /= size;
        result += coordinate * static_cast<std::size_t>(strides_[dimension - 1]);
    }
    return result;
}

TensorLayout TensorLayout::reshaped(Shape new_shape) const {
    if (!is_contiguous()) throw std::logic_error("reshape requires a contiguous layout");
    return TensorLayout(resolve_reshape_shape(std::move(new_shape), numel_));
}

TensorLayout TensorLayout::transposed(std::size_t dimension0,
                                      std::size_t dimension1) const {
    if (dimension0 >= rank() || dimension1 >= rank()) {
        throw std::out_of_range("transpose dimension is out of range");
    }
    auto shape = shape_;
    auto strides = strides_;
    std::swap(shape[dimension0], shape[dimension1]);
    std::swap(strides[dimension0], strides[dimension1]);
    return TensorLayout(std::move(shape), std::move(strides));
}

void TensorLayout::validate_shape(const Shape& shape) {
    for (const auto dimension : shape) {
        if (dimension < 0) throw std::invalid_argument("tensor dimensions must be non-negative");
    }
}

std::size_t TensorLayout::checked_numel(const Shape& shape) {
    if (shape.empty()) return 1;
    for (const auto dimension : shape) {
        if (dimension == 0) return 0;
    }

    std::size_t result = 1;
    for (const auto dimension : shape) {
        const auto size = static_cast<std::size_t>(dimension);
        if (result > std::numeric_limits<std::size_t>::max() / size) {
            throw std::overflow_error("tensor element count overflows size_t");
        }
        result *= size;
    }
    return result;
}

Strides TensorLayout::make_contiguous_strides(const Shape& shape) {
    Strides strides(shape.size(), 1);
    std::size_t running_stride = 1;
    for (std::size_t dimension = shape.size(); dimension > 0; --dimension) {
        if (running_stride > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
            throw std::overflow_error("contiguous stride overflows int64_t");
        }
        strides[dimension - 1] = static_cast<std::int64_t>(running_stride);
        if (dimension > 1) {
            const auto size = static_cast<std::size_t>(shape[dimension - 1]);
            if (size != 0 && running_stride > std::numeric_limits<std::size_t>::max() / size) {
                throw std::overflow_error("contiguous stride overflows size_t");
            }
            running_stride *= size;
        }
    }
    return strides;
}

std::size_t TensorLayout::checked_storage_span(const Shape& shape,
                                               const Strides& strides) {
    if (shape.empty()) return 1;
    for (const auto dimension : shape) {
        if (dimension == 0) return 0;
    }

    std::size_t maximum_offset = 0;
    for (std::size_t dimension = 0; dimension < shape.size(); ++dimension) {
        const auto extent = static_cast<std::size_t>(shape[dimension] - 1);
        const auto stride = static_cast<std::size_t>(strides[dimension]);
        if (extent != 0 && stride > std::numeric_limits<std::size_t>::max() / extent) {
            throw std::overflow_error("tensor storage span overflows size_t");
        }
        const auto contribution = extent * stride;
        if (maximum_offset > std::numeric_limits<std::size_t>::max() - contribution) {
            throw std::overflow_error("tensor storage span overflows size_t");
        }
        maximum_offset += contribution;
    }
    if (maximum_offset == std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error("tensor storage span overflows size_t");
    }
    return maximum_offset + 1;
}

Shape TensorLayout::resolve_reshape_shape(Shape shape,
                                          std::size_t current_elements) {
    std::size_t known_elements = 1;
    std::size_t inferred_dimension = shape.size();
    bool has_zero = false;

    for (std::size_t dimension = 0; dimension < shape.size(); ++dimension) {
        const auto size = shape[dimension];
        if (size == -1) {
            if (inferred_dimension != shape.size()) {
                throw std::invalid_argument("reshape allows at most one inferred dimension");
            }
            inferred_dimension = dimension;
            continue;
        }
        if (size < 0) {
            throw std::invalid_argument("reshape dimensions must be non-negative or -1");
        }
        if (size == 0) {
            has_zero = true;
            known_elements = 0;
            continue;
        }
        if (has_zero) continue;

        const auto unsigned_size = static_cast<std::size_t>(size);
        if (known_elements > std::numeric_limits<std::size_t>::max() / unsigned_size) {
            throw std::overflow_error("reshape element count overflows size_t");
        }
        known_elements *= unsigned_size;
    }

    if (inferred_dimension != shape.size()) {
        if (known_elements == 0) {
            throw std::invalid_argument("cannot infer a reshape dimension when known dimensions multiply to zero");
        }
        if (current_elements % known_elements != 0) {
            throw std::invalid_argument("reshape cannot infer an integral dimension");
        }
        const auto inferred_size = current_elements / known_elements;
        if (inferred_size > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
            throw std::overflow_error("inferred reshape dimension overflows int64_t");
        }
        shape[inferred_dimension] = static_cast<std::int64_t>(inferred_size);
        known_elements *= inferred_size;
    }

    if (known_elements != current_elements) {
        throw std::invalid_argument("reshape must preserve the number of elements");
    }
    return shape;
}

}  // namespace tinyinfer
