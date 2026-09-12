#pragma once

#include <cstddef>

#include "tinyinfer/core/tensor.h"

namespace tinyinfer::cpu {

struct MatMulBlockSize {
    std::size_t rows{32};
    std::size_t columns{32};
    std::size_t inner{32};
};

// Clear correctness oracle. It deliberately uses Tensor's indexed API.
void matmul_reference(const Tensor& lhs, const Tensor& rhs, Tensor& output);

// Direct pointer implementation that respects arbitrary positive strides.
void matmul_strided(const Tensor& lhs, const Tensor& rhs, Tensor& output);

// Cache-blocked implementation used by the public MatMul operator.
void matmul_blocked(const Tensor& lhs, const Tensor& rhs, Tensor& output,
                    MatMulBlockSize block_size = {});

}  // namespace tinyinfer::cpu
