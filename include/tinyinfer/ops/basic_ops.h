#pragma once

#include "tinyinfer/core/tensor.h"

namespace tinyinfer::ops {

// Phase 0 reference operators. These prioritize clarity and correctness over speed.
Tensor add(const Tensor& lhs, const Tensor& rhs);
Tensor matmul(const Tensor& lhs, const Tensor& rhs);
Tensor relu(const Tensor& input);
Tensor softmax(const Tensor& input);
Tensor linear(const Tensor& input, const Tensor& weight, const Tensor& bias);

}  // namespace tinyinfer::ops
