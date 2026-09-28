#pragma once

#include <cstdint>

#include "tinyinfer/core/tensor.h"

namespace tinyinfer::ops {

// Computes alpha * (A' @ B') + beta * C. C may be null.
void gemm_out(Tensor& output, const Tensor& lhs, const Tensor& rhs,
              const Tensor* bias = nullptr, float alpha = 1.0F,
              float beta = 1.0F, std::int64_t trans_a = 0,
              std::int64_t trans_b = 0);

// Uses the same Gemm contract and applies ReLU in the Gemm epilogue.
void gemm_relu_out(Tensor& output, const Tensor& lhs, const Tensor& rhs,
                   const Tensor* bias = nullptr, float alpha = 1.0F,
                   float beta = 1.0F, std::int64_t trans_a = 0,
                   std::int64_t trans_b = 0);

}  // namespace tinyinfer::ops
