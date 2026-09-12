#pragma once

#include "tinyinfer/core/tensor.h"

namespace tinyinfer::ops {

Tensor matmul(const Tensor& lhs, const Tensor& rhs);

}  // namespace tinyinfer::ops
