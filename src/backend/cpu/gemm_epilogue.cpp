#include "tinyinfer/backend/cpu/gemm_epilogue.h"
#include <algorithm>
#include "ops/gemm_internal.h"
#include <stdexcept>

namespace tinyinfer::cpu {
GemmEpilogue::GemmEpilogue(const Shape& output, const Shape* bias,
                         float alpha, float beta, bool relu)
    : alpha_(alpha), beta_(beta), relu_(relu) {
    if (output.size() != 2 || output[0] < 0 || output[1] < 0)
        throw std::invalid_argument("Gemm epilogue requires a rank-2 output");
    rows_ = static_cast<std::size_t>(output[0]);
    columns_ = static_cast<std::size_t>(output[1]);
    if (!bias) return;
    bias_rank_ = bias->size();
    if (bias_rank_ > 2) throw std::invalid_argument("Gemm bias rank exceeds output rank");
    if (bias_rank_ == 2) bias_rows_ = (*bias)[0];
    if (bias_rank_ > 0) bias_columns_ = bias->back();
    if ((bias_rows_ != 1 && bias_rows_ != output[0]) ||
        (bias_columns_ != 1 && bias_columns_ != output[1]))
        throw std::invalid_argument("Gemm bias does not broadcast to output");
    if (bias_rows_ == 1 && bias_columns_ == 1) layout_ = GemmBiasLayout::Scalar;
    else if (bias_rows_ == 1) layout_ = GemmBiasLayout::Row;
    else if (bias_columns_ == 1) layout_ = GemmBiasLayout::Column;
    else layout_ = GemmBiasLayout::Full;
}

GemmEpilogue GemmEpilogue::bind(const Tensor* bias) const {
    auto result = *this;
    if (layout_ == GemmBiasLayout::None) {
        if (bias) throw std::invalid_argument("unexpected Gemm bias");
        return result;
    }
    if (!bias || bias->dtype() != DataType::Float32 || bias->rank() != bias_rank_ ||
        (bias_rank_ == 2 && bias->shape()[0] != bias_rows_) ||
        (bias_rank_ > 0 && bias->shape().back() != bias_columns_))
        throw std::invalid_argument("Gemm bias metadata differs from descriptor");
    result.data_ = bias->data<float>();
    result.bias_ = bias;
    result.row_stride_ = bias_rank_ == 2 && bias_rows_ != 1
        ? static_cast<std::size_t>(bias->strides()[0]) : 0;
    result.column_stride_ = bias_rank_ > 0 && bias_columns_ != 1
        ? static_cast<std::size_t>(bias->strides().back()) : 0;
    return result;
}

void GemmEpilogue::validate(const Tensor& output) const {
    if (output.dtype() != DataType::Float32 || output.rank() != 2 ||
        !output.is_contiguous() || static_cast<std::size_t>(output.shape()[0]) != rows_ ||
        static_cast<std::size_t>(output.shape()[1]) != columns_)
        throw std::invalid_argument("Gemm epilogue output differs from descriptor");
    if (layout_ != GemmBiasLayout::None && !data_ && output.numel())
        throw std::logic_error("Gemm epilogue bias is not bound");
}

float GemmEpilogue::finish(float value, std::size_t row, std::size_t column) const {
    if (layout_ != GemmBiasLayout::None)
        value = alpha_ * value + beta_ * data_[row * row_stride_ + column * column_stride_];
    else if (alpha_ != 1.0F || relu_) value = alpha_ * value;
    return relu_ ? std::max(0.0F, value) : value;
}

void apply_gemm_epilogue(Tensor& output, const GemmEpilogue& epilogue) {
    epilogue.validate(output);
    // Without bias there is no broadcast traversal to specialize. The shared
    // linear helper avoids per-element descriptor/layout tests and preserves
    // alpha multiplication, NaN and signed-zero behavior exactly as Legacy.
    if (epilogue.bias_layout() == GemmBiasLayout::None ||
        !epilogue.supports_specialization()) {
        ops::detail::apply_gemm_epilogue(output, epilogue.bias_tensor(), epilogue.alpha(),
                                        epilogue.beta(), epilogue.relu());
        return;
    }
    auto* data = output.data<float>();
    const auto columns = static_cast<std::size_t>(output.shape()[1]);
    for (std::size_t row = 0; row < static_cast<std::size_t>(output.shape()[0]); ++row)
        for (std::size_t column = 0; column < columns; ++column)
            data[row * columns + column] = epilogue.finish(data[row * columns + column], row, column);
}
} // namespace tinyinfer::cpu
