#pragma once

#include "tinyinfer/core/tensor.h"

namespace tinyinfer::ops::detail {

// Shared by ordinary and prepared CPU Gemm; keep the arithmetic/ReLU order.
void apply_gemm_epilogue(Tensor& output, const Tensor* bias,
                         float alpha, float beta, bool apply_relu);

}  // namespace tinyinfer::ops::detail
