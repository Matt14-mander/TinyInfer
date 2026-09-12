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
    cpu::matmul_blocked(lhs, rhs, output);
    return output;
}

}  // namespace tinyinfer::ops
