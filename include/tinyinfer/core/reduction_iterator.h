#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "tinyinfer/core/tensor_layout.h"

namespace tinyinfer {

class ReductionIterator {
public:
    ReductionIterator(TensorLayout input_layout,
                      std::vector<std::int64_t> axes,
                      bool keepdim = false);

    const Shape& output_shape() const noexcept { return output_shape_; }
    const std::vector<std::size_t>& axes() const noexcept { return axes_; }
    std::size_t output_numel() const noexcept { return output_numel_; }
    std::size_t reduction_numel() const noexcept { return reduction_numel_; }
    bool keepdim() const noexcept { return keepdim_; }
    bool has_contiguous_reduction() const noexcept {
        return contiguous_reduction_;
    }

    std::size_t input_offset(std::size_t output_index,
                             std::size_t reduction_index) const;
    std::size_t input_logical_index(std::size_t output_index,
                                    std::size_t reduction_index) const;

private:
    std::size_t offset_with_strides(const Strides& strides,
                                    std::size_t output_index,
                                    std::size_t reduction_index) const;

    TensorLayout input_layout_;
    Strides logical_strides_;
    std::vector<std::size_t> axes_;
    std::vector<bool> reduced_;
    Shape output_shape_;
    std::size_t output_numel_{0};
    std::size_t reduction_numel_{0};
    bool keepdim_{false};
    bool contiguous_reduction_{false};
};

}  // namespace tinyinfer
