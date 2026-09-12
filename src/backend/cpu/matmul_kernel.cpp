#include "tinyinfer/backend/cpu/matmul_kernel.h"

#include <algorithm>
#include <stdexcept>

#if defined(__AVX2__) || defined(__SSE2__)
#include <immintrin.h>
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

namespace tinyinfer::cpu {
namespace {

void validate_matmul(const Tensor& lhs, const Tensor& rhs,
                     const Tensor& output) {
    if (lhs.dtype() != DataType::Float32 || rhs.dtype() != DataType::Float32 ||
        output.dtype() != DataType::Float32) {
        throw std::invalid_argument("CPU MatMul currently supports only float32 tensors");
    }
    if (lhs.rank() != 2 || rhs.rank() != 2 || output.rank() != 2) {
        throw std::invalid_argument("CPU MatMul expects rank-2 tensors");
    }
    if (lhs.shape()[1] != rhs.shape()[0]) {
        throw std::invalid_argument("CPU MatMul inner dimensions must match");
    }
    if (output.shape()[0] != lhs.shape()[0] ||
        output.shape()[1] != rhs.shape()[1]) {
        throw std::invalid_argument("CPU MatMul output shape is incorrect");
    }
    if (!output.is_contiguous()) {
        throw std::invalid_argument("CPU MatMul output must be contiguous");
    }
}

void validate_packed_matmul(const Tensor& lhs, const PackedMatMulRhs& rhs,
                            const Tensor& output) {
    if (lhs.dtype() != DataType::Float32 || output.dtype() != DataType::Float32) {
        throw std::invalid_argument("CPU packed MatMul supports only float32 tensors");
    }
    if (lhs.rank() != 2 || output.rank() != 2) {
        throw std::invalid_argument("CPU packed MatMul expects rank-2 tensors");
    }
    if (static_cast<std::size_t>(lhs.shape()[1]) != rhs.inner()) {
        throw std::invalid_argument("CPU packed MatMul inner dimensions must match");
    }
    if (output.shape()[0] != lhs.shape()[0] ||
        static_cast<std::size_t>(output.shape()[1]) != rhs.columns()) {
        throw std::invalid_argument("CPU packed MatMul output shape is incorrect");
    }
    if (!output.is_contiguous()) {
        throw std::invalid_argument("CPU packed MatMul output must be contiguous");
    }
}

std::size_t simd_width() noexcept {
#if defined(__AVX2__)
    return 8;
#elif defined(__SSE2__) || defined(__ARM_NEON) || defined(__ARM_NEON__)
    return 4;
#else
    return 1;
#endif
}

void accumulate_simd_dot(float* output, const float* lhs_row,
                         std::size_t lhs_inner_stride,
                         const float* packed_rhs, std::size_t columns,
                         std::size_t column, std::size_t inner_begin,
                         std::size_t inner_end) {
#if defined(__AVX2__)
    auto accumulator = _mm256_loadu_ps(output + column);
    for (auto k = inner_begin; k < inner_end; ++k) {
        const auto lhs_vector = _mm256_set1_ps(lhs_row[k * lhs_inner_stride]);
        const auto rhs_vector =
            _mm256_loadu_ps(packed_rhs + k * columns + column);
#if defined(__FMA__)
        accumulator = _mm256_fmadd_ps(lhs_vector, rhs_vector, accumulator);
#else
        accumulator = _mm256_add_ps(
            accumulator, _mm256_mul_ps(lhs_vector, rhs_vector));
#endif
    }
    _mm256_storeu_ps(output + column, accumulator);
#elif defined(__SSE2__)
    auto accumulator = _mm_loadu_ps(output + column);
    for (auto k = inner_begin; k < inner_end; ++k) {
        const auto lhs_vector = _mm_set1_ps(lhs_row[k * lhs_inner_stride]);
        const auto rhs_vector =
            _mm_loadu_ps(packed_rhs + k * columns + column);
        accumulator = _mm_add_ps(accumulator,
                                 _mm_mul_ps(lhs_vector, rhs_vector));
    }
    _mm_storeu_ps(output + column, accumulator);
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    auto accumulator = vld1q_f32(output + column);
    for (auto k = inner_begin; k < inner_end; ++k) {
        const auto lhs_vector = vdupq_n_f32(lhs_row[k * lhs_inner_stride]);
        const auto rhs_vector =
            vld1q_f32(packed_rhs + k * columns + column);
#if defined(__aarch64__)
        accumulator = vfmaq_f32(accumulator, rhs_vector, lhs_vector);
#else
        accumulator = vmlaq_f32(accumulator, rhs_vector, lhs_vector);
#endif
    }
    vst1q_f32(output + column, accumulator);
#else
    (void)output;
    (void)lhs_row;
    (void)lhs_inner_stride;
    (void)packed_rhs;
    (void)columns;
    (void)column;
    (void)inner_begin;
    (void)inner_end;
#endif
}

}  // namespace

PackedMatMulRhs::PackedMatMulRhs(const Tensor& rhs) {
    if (rhs.dtype() != DataType::Float32 || rhs.rank() != 2) {
        throw std::invalid_argument(
            "PackedMatMulRhs expects a rank-2 float32 tensor");
    }
    inner_ = static_cast<std::size_t>(rhs.shape()[0]);
    columns_ = static_cast<std::size_t>(rhs.shape()[1]);
    data_.resize(inner_ * columns_);
    const auto* source = rhs.data<float>();
    const auto inner_stride = static_cast<std::size_t>(rhs.strides()[0]);
    const auto column_stride = static_cast<std::size_t>(rhs.strides()[1]);
    for (std::size_t inner = 0; inner < inner_; ++inner) {
        for (std::size_t column = 0; column < columns_; ++column) {
            data_[inner * columns_ + column] =
                source[inner * inner_stride + column * column_stride];
        }
    }
}

void matmul_reference(const Tensor& lhs, const Tensor& rhs, Tensor& output) {
    validate_matmul(lhs, rhs, output);
    const auto rows = static_cast<std::size_t>(lhs.shape()[0]);
    const auto inner = static_cast<std::size_t>(lhs.shape()[1]);
    const auto columns = static_cast<std::size_t>(rhs.shape()[1]);

    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < columns; ++column) {
            float sum = 0.0F;
            for (std::size_t k = 0; k < inner; ++k) {
                sum += lhs.at({static_cast<std::int64_t>(row),
                               static_cast<std::int64_t>(k)}) *
                       rhs.at({static_cast<std::int64_t>(k),
                               static_cast<std::int64_t>(column)});
            }
            output.at(row * columns + column) = sum;
        }
    }
}

void matmul_strided(const Tensor& lhs, const Tensor& rhs, Tensor& output) {
    validate_matmul(lhs, rhs, output);
    const auto rows = static_cast<std::size_t>(lhs.shape()[0]);
    const auto inner = static_cast<std::size_t>(lhs.shape()[1]);
    const auto columns = static_cast<std::size_t>(rhs.shape()[1]);
    const auto lhs_row_stride = static_cast<std::size_t>(lhs.strides()[0]);
    const auto lhs_inner_stride = static_cast<std::size_t>(lhs.strides()[1]);
    const auto rhs_inner_stride = static_cast<std::size_t>(rhs.strides()[0]);
    const auto rhs_column_stride = static_cast<std::size_t>(rhs.strides()[1]);
    const auto* lhs_data = lhs.data<float>();
    const auto* rhs_data = rhs.data<float>();
    auto* output_data = output.data<float>();

    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < columns; ++column) {
            float sum = 0.0F;
            for (std::size_t k = 0; k < inner; ++k) {
                sum += lhs_data[row * lhs_row_stride + k * lhs_inner_stride] *
                       rhs_data[k * rhs_inner_stride +
                                column * rhs_column_stride];
            }
            output_data[row * columns + column] = sum;
        }
    }
}

void matmul_blocked(const Tensor& lhs, const Tensor& rhs, Tensor& output,
                    MatMulBlockSize block_size) {
    validate_matmul(lhs, rhs, output);
    if (block_size.rows == 0 || block_size.columns == 0 ||
        block_size.inner == 0) {
        throw std::invalid_argument("CPU MatMul block dimensions must be positive");
    }

    const auto rows = static_cast<std::size_t>(lhs.shape()[0]);
    const auto inner = static_cast<std::size_t>(lhs.shape()[1]);
    const auto columns = static_cast<std::size_t>(rhs.shape()[1]);
    const auto lhs_row_stride = static_cast<std::size_t>(lhs.strides()[0]);
    const auto lhs_inner_stride = static_cast<std::size_t>(lhs.strides()[1]);
    const auto rhs_inner_stride = static_cast<std::size_t>(rhs.strides()[0]);
    const auto rhs_column_stride = static_cast<std::size_t>(rhs.strides()[1]);
    const auto* lhs_data = lhs.data<float>();
    const auto* rhs_data = rhs.data<float>();
    auto* output_data = output.data<float>();
    if (output.numel() > 0) {
        std::fill(output_data, output_data + output.numel(), 0.0F);
    }

    for (std::size_t row_block = 0; row_block < rows;
         row_block += block_size.rows) {
        const auto row_end = std::min(row_block + block_size.rows, rows);
        for (std::size_t column_block = 0; column_block < columns;
             column_block += block_size.columns) {
            const auto column_end =
                std::min(column_block + block_size.columns, columns);
            for (std::size_t inner_block = 0; inner_block < inner;
                 inner_block += block_size.inner) {
                const auto inner_end =
                    std::min(inner_block + block_size.inner, inner);
                for (std::size_t row = row_block; row < row_end; ++row) {
                    for (std::size_t k = inner_block; k < inner_end; ++k) {
                        const auto lhs_value =
                            lhs_data[row * lhs_row_stride + k * lhs_inner_stride];
                        for (std::size_t column = column_block;
                             column < column_end; ++column) {
                            output_data[row * columns + column] +=
                                lhs_value *
                                rhs_data[k * rhs_inner_stride +
                                         column * rhs_column_stride];
                        }
                    }
                }
            }
        }
    }
}

void matmul_packed_simd(const Tensor& lhs, const PackedMatMulRhs& rhs,
                        Tensor& output, MatMulBlockSize block_size) {
    validate_packed_matmul(lhs, rhs, output);
    if (block_size.rows == 0 || block_size.columns == 0 ||
        block_size.inner == 0) {
        throw std::invalid_argument("CPU MatMul block dimensions must be positive");
    }

    const auto rows = static_cast<std::size_t>(lhs.shape()[0]);
    const auto inner = rhs.inner();
    const auto columns = rhs.columns();
    const auto lhs_row_stride = static_cast<std::size_t>(lhs.strides()[0]);
    const auto lhs_inner_stride = static_cast<std::size_t>(lhs.strides()[1]);
    const auto* lhs_data = lhs.data<float>();
    const auto* rhs_data = rhs.data();
    auto* output_data = output.data<float>();
    if (output.numel() > 0) {
        std::fill(output_data, output_data + output.numel(), 0.0F);
    }

    for (std::size_t row_block = 0; row_block < rows;
         row_block += block_size.rows) {
        const auto row_end = std::min(row_block + block_size.rows, rows);
        for (std::size_t column_block = 0; column_block < columns;
             column_block += block_size.columns) {
            const auto column_end =
                std::min(column_block + block_size.columns, columns);
            for (std::size_t inner_block = 0; inner_block < inner;
                 inner_block += block_size.inner) {
                const auto inner_end =
                    std::min(inner_block + block_size.inner, inner);
                for (std::size_t row = row_block; row < row_end; ++row) {
                    const auto* lhs_row = lhs_data + row * lhs_row_stride;
                    auto* output_row = output_data + row * columns;
                    auto column = column_block;
                    const auto width = simd_width();
                    if (width > 1) {
                        for (; column + width <= column_end; column += width) {
                            accumulate_simd_dot(
                                output_row, lhs_row, lhs_inner_stride,
                                rhs_data, columns, column, inner_block,
                                inner_end);
                        }
                    }
                    for (; column < column_end; ++column) {
                        auto sum = output_row[column];
                        for (std::size_t k = inner_block; k < inner_end; ++k) {
                            sum += lhs_row[k * lhs_inner_stride] *
                                   rhs_data[k * columns + column];
                        }
                        output_row[column] = sum;
                    }
                }
            }
        }
    }
}

std::size_t matmul_simd_width() noexcept {
    return simd_width();
}

}  // namespace tinyinfer::cpu
