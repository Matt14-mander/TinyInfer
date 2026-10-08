#pragma once

#include <cstddef>
#include "tinyinfer/core/tensor.h"

namespace tinyinfer::cpu {

enum class GemmBiasLayout { None, Scalar, Row, Column, Full };

// Shape/attributes are decoded once; bind() reads the current bias storage/strides.
// A prototype contains no Tensor or storage pointer and is safe to expose in plan diagnostics.
// A bound descriptor borrows the bias Tensor/storage; keep both alive and separate
// from the output until the kernel returns. Fusion eligibility is runtime layout dependent.
class GemmEpilogue {
public:
    GemmEpilogue() = default;
    GemmEpilogue(const Shape& output_shape, const Shape* bias_shape,
                 float alpha, float beta, bool relu);
    GemmEpilogue bind(const Tensor* bias) const;
    void validate(const Tensor& output) const;
    GemmBiasLayout bias_layout() const noexcept { return layout_; }
    bool supports_fusion() const noexcept {
        return column_stride_ <= 1 && supports_specialization();
    }
    bool supports_specialization() const noexcept {
        return layout_ == GemmBiasLayout::None || (alpha_ == 1.0F && beta_ == 1.0F);
    }
    const Tensor* bias_tensor() const noexcept { return bias_; }
    std::size_t row_stride() const noexcept { return row_stride_; }
    std::size_t column_stride() const noexcept { return column_stride_; }
    const float* bias_data() const noexcept { return data_; }
    float alpha() const noexcept { return alpha_; }
    float beta() const noexcept { return beta_; }
    bool relu() const noexcept { return relu_; }
    float finish(float value, std::size_t row, std::size_t column) const;

private:
    GemmBiasLayout layout_{GemmBiasLayout::None};
    std::size_t rows_{0}, columns_{0}, bias_rank_{0};
    std::int64_t bias_rows_{1}, bias_columns_{1};
    std::size_t row_stride_{0}, column_stride_{0};
    const float* data_{nullptr};
    const Tensor* bias_{nullptr};
    float alpha_{1.0F}, beta_{1.0F};
    bool relu_{false};
};

// Direct broadcast strides avoid TensorIterator construction and per-element division.
void apply_gemm_epilogue(Tensor& output, const GemmEpilogue& epilogue);

} // namespace tinyinfer::cpu
