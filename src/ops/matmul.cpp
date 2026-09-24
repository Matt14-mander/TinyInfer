#include "tinyinfer/ops/matmul.h"

#include <stdexcept>

#include "tinyinfer/backend/cpu/matmul_kernel.h"

namespace tinyinfer::ops {

Tensor matmul(const Tensor& lhs, const Tensor& rhs) {
    if (lhs.dtype() != DataType::Float32 || rhs.dtype() != DataType::Float32) {
        throw std::invalid_argument("matmul currently supports only float32 tensors");
    }
    if (lhs.rank() != 2 || rhs.rank() != 2) {
        throw std::invalid_argument("matmul expects two rank-2 tensors");
    }
    if (lhs.shape()[1] != rhs.shape()[0]) {
        throw std::invalid_argument("matmul inner dimensions must match");
    }

    Tensor output({lhs.shape()[0], rhs.shape()[1]});
    matmul_out(output, lhs, rhs);
    return output;
}

void matmul_out(Tensor& output, const Tensor& lhs, const Tensor& rhs) {
    if (lhs.dtype() != DataType::Float32 || rhs.dtype() != DataType::Float32) {
        throw std::invalid_argument("matmul currently supports only float32 tensors");
    }
    if (lhs.rank() != 2 || rhs.rank() != 2) {
        throw std::invalid_argument("matmul expects two rank-2 tensors");
    }
    if (lhs.shape()[1] != rhs.shape()[0]) {
        throw std::invalid_argument("matmul inner dimensions must match");
    }

    const cpu::PackedMatMulRhs packed_rhs(rhs);
    cpu::matmul_packed_simd(lhs, packed_rhs, output);
}

}  // namespace tinyinfer::ops
