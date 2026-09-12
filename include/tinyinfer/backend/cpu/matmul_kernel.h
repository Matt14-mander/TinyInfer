#pragma once

#include <cstddef>
#include <vector>

#include "tinyinfer/core/tensor.h"

namespace tinyinfer::cpu {

struct MatMulBlockSize {
    std::size_t rows{32};
    std::size_t columns{32};
    std::size_t inner{32};
};

class PackedMatMulRhs {
public:
    explicit PackedMatMulRhs(const Tensor& rhs);

    std::size_t inner() const noexcept { return inner_; }
    std::size_t columns() const noexcept { return columns_; }
    const float* data() const noexcept { return data_.data(); }

private:
    std::size_t inner_{0};
    std::size_t columns_{0};
    std::vector<float> data_;
};

// Clear correctness oracle. It deliberately uses Tensor's indexed API.
void matmul_reference(const Tensor& lhs, const Tensor& rhs, Tensor& output);

// Direct pointer implementation that respects arbitrary positive strides.
void matmul_strided(const Tensor& lhs, const Tensor& rhs, Tensor& output);

// Cache-blocked implementation used by the public MatMul operator.
void matmul_blocked(const Tensor& lhs, const Tensor& rhs, Tensor& output,
                    MatMulBlockSize block_size = {});

// Uses reusable contiguous RHS packing and a platform SIMD micro-kernel.
void matmul_packed_simd(const Tensor& lhs, const PackedMatMulRhs& rhs,
                        Tensor& output, MatMulBlockSize block_size = {});

std::size_t matmul_simd_width() noexcept;

}  // namespace tinyinfer::cpu
