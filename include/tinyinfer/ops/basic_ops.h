#pragma once

#include <cstdint>
#include <vector>

#include "tinyinfer/core/tensor.h"
#include "tinyinfer/ops/matmul.h"

namespace tinyinfer::ops {

// Phase 0 reference operators. Elementwise operators use TensorIterator for
// stride-aware traversal and NumPy-style broadcasting.
Tensor add(const Tensor& lhs, const Tensor& rhs);
void add_out(Tensor& output, const Tensor& lhs, const Tensor& rhs);
Tensor sub(const Tensor& lhs, const Tensor& rhs);
void sub_out(Tensor& output, const Tensor& lhs, const Tensor& rhs);
Tensor mul(const Tensor& lhs, const Tensor& rhs);
void mul_out(Tensor& output, const Tensor& lhs, const Tensor& rhs);
Tensor relu(const Tensor& input);
void relu_out(Tensor& output, const Tensor& input);
Tensor gelu(const Tensor& input);
void gelu_out(Tensor& output, const Tensor& input);
Tensor reduce_sum(const Tensor& input, const std::vector<std::int64_t>& axes,
                  bool keepdim = false);
Tensor reduce_max(const Tensor& input, const std::vector<std::int64_t>& axes,
                  bool keepdim = false);
Tensor softmax(const Tensor& input, std::int64_t axis = -1);
void softmax_out(Tensor& output, const Tensor& input,
                 std::int64_t axis = -1);
Tensor layer_norm(const Tensor& input, float epsilon = 1e-5F);
void layer_norm_out(Tensor& output, const Tensor& input,
                    float epsilon = 1e-5F);
Tensor layer_norm(const Tensor& input, const Tensor& weight,
                  const Tensor& bias, float epsilon = 1e-5F);
void layer_norm_out(Tensor& output, const Tensor& input,
                    const Tensor& weight, const Tensor& bias,
                    float epsilon = 1e-5F);
Tensor linear(const Tensor& input, const Tensor& weight, const Tensor& bias);

}  // namespace tinyinfer::ops
