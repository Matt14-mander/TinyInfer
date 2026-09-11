#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <vector>

#include "tinyinfer/core/dtype.h"

namespace tinyinfer {

using Shape = std::vector<std::int64_t>;
using Strides = std::vector<std::int64_t>;

class TensorLayout {
public:
    TensorLayout();
    explicit TensorLayout(Shape shape);
    TensorLayout(Shape shape, Strides strides);

    const Shape& shape() const noexcept { return shape_; }
    const Strides& strides() const noexcept { return strides_; }
    std::size_t rank() const noexcept { return shape_.size(); }
    std::size_t numel() const noexcept { return numel_; }
    std::size_t storage_span() const noexcept { return storage_span_; }
    std::size_t size_bytes(DataType dtype) const;
    bool is_contiguous() const noexcept { return contiguous_; }

    std::size_t offset(const Shape& indices) const;
    std::size_t offset(std::initializer_list<std::int64_t> indices) const;
    std::size_t storage_offset(std::size_t logical_index) const;

    TensorLayout reshaped(Shape new_shape) const;
    TensorLayout transposed(std::size_t dimension0,
                            std::size_t dimension1) const;

private:
    static void validate_shape(const Shape& shape);
    static std::size_t checked_numel(const Shape& shape);
    static Strides make_contiguous_strides(const Shape& shape);
    static std::size_t checked_storage_span(const Shape& shape,
                                            const Strides& strides);
    static Shape resolve_reshape_shape(Shape shape,
                                       std::size_t current_elements);

    Shape shape_;
    Strides strides_;
    std::size_t numel_{1};
    std::size_t storage_span_{1};
    bool contiguous_{true};
};

}  // namespace tinyinfer
