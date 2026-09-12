#include "tinyinfer/backend/cpu/matmul_kernel.h"

#include <algorithm>
#include <stdexcept>

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

}  // namespace

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

}  // namespace tinyinfer::cpu
