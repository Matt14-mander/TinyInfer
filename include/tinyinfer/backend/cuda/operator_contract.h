#pragma once
#include <cstdint>
#include "tinyinfer/backend/capability.h"

namespace tinyinfer::cuda {
// Portable preparation contracts only: success does not mean a CUDA runtime,
// device or kernel is available. Never use this as executable Backend placement.
NodeCapability validate_operator_contract(const CapabilityQuery&);
enum class GemmWork { EmptyOutput, ZeroK, MatrixProduct };
struct GemmDescriptor {
    std::int32_t rows{}, columns{}, reduction{};
    // Row-major C is column-major C^T: the native call uses B first, A second.
    std::int32_t native_m{}, native_n{}, native_k{}, lda{}, ldb{}, ldc{};
    std::size_t left_input{1}, right_input{0};
    bool transpose_left{false}, transpose_right{false};
    float product_alpha{1}, product_beta{0};
    // Apply user alpha/beta/bias/activation in a separate FP32 epilogue.
    // In particular alpha=0 must not allow cuBLAS to skip NaN-valued operands.
    float epilogue_alpha{1}, epilogue_beta{1};
    bool has_bias{false}, relu{false};
    std::int32_t bias_rows{1}, bias_columns{1};
    GemmWork work{GemmWork::EmptyOutput};
};
// Throws for rejected schemas/layouts or non-matrix operators. Contains no
// pointers, Tensor storage, vendor enums or CUDA SDK dependencies.
GemmDescriptor make_gemm_descriptor(const CapabilityQuery&);
} // namespace tinyinfer::cuda
