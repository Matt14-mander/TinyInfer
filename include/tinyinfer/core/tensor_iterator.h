#pragma once

#include <cstddef>
#include <initializer_list>
#include <vector>

#include "tinyinfer/core/tensor_layout.h"

namespace tinyinfer {

// Iterates several logical tensor layouts in one common, broadcasted order.
// It owns layout metadata only; tensor memory remains owned by Tensor/Storage.
class TensorIterator {
public:
    explicit TensorIterator(std::vector<TensorLayout> operands);
    TensorIterator(std::initializer_list<TensorLayout> operands);

    const Shape& shape() const noexcept { return shape_; }
    std::size_t rank() const noexcept { return shape_.size(); }
    std::size_t numel() const noexcept { return numel_; }
    std::size_t operand_count() const noexcept { return operand_strides_.size(); }
    const Strides& operand_strides(std::size_t operand) const;
    const Shape& iteration_shape() const noexcept { return iteration_shape_; }
    std::size_t iteration_rank() const noexcept { return iteration_shape_.size(); }
    const Strides& iteration_strides(std::size_t operand) const;
    bool operand_is_contiguous(std::size_t operand) const;
    bool has_contiguous_fast_path() const noexcept {
        return contiguous_operand_count_ == operand_count();
    }

    std::size_t operand_offset(std::size_t operand,
                               std::size_t logical_index) const;

private:
    static Shape broadcast_shape(const std::vector<TensorLayout>& operands);
    static Strides broadcast_strides(const TensorLayout& operand,
                                     const Shape& output_shape);
    void coalesce_dimensions();
    void detect_contiguous_operands();

    Shape shape_;
    std::size_t numel_{0};
    std::vector<Strides> operand_strides_;
    Shape iteration_shape_;
    std::vector<Strides> iteration_strides_;
    std::vector<bool> contiguous_operands_;
    std::size_t contiguous_operand_count_{0};
};

}  // namespace tinyinfer
