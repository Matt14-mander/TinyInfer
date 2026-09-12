#include "tinyinfer/ops/basic_ops.h"

#include "tinyinfer/core/tensor_iterator.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace tinyinfer::ops {
namespace {

void require_f32(const Tensor& tensor, const char* operation) {
    if (tensor.dtype() != DataType::Float32) {
        throw std::invalid_argument(std::string(operation) + " currently supports only float32 tensors");
    }
}

float logical_value(const Tensor& tensor, std::size_t index) {
    return tensor.data<float>()[tensor.layout().storage_offset(index)];
}

}  // namespace

Tensor add(const Tensor& lhs, const Tensor& rhs) {
    require_f32(lhs, "add");
    require_f32(rhs, "add");

    TensorIterator iterator({lhs.layout(), rhs.layout()});
    Tensor output(iterator.shape());
    const auto* lhs_data = lhs.data<float>();
    const auto* rhs_data = rhs.data<float>();
    auto* output_data = output.data<float>();
    if (iterator.has_contiguous_fast_path()) {
        for (std::size_t i = 0; i < iterator.numel(); ++i) {
            output_data[i] = lhs_data[i] + rhs_data[i];
        }
        return output;
    }
    for (std::size_t i = 0; i < iterator.numel(); ++i) {
        output_data[i] = lhs_data[iterator.operand_offset(0, i)] +
                         rhs_data[iterator.operand_offset(1, i)];
    }
    return output;
}

Tensor matmul(const Tensor& lhs, const Tensor& rhs) {
    require_f32(lhs, "matmul");
    require_f32(rhs, "matmul");
    if (lhs.rank() != 2 || rhs.rank() != 2) throw std::invalid_argument("matmul expects two rank-2 tensors");
    if (lhs.shape()[1] != rhs.shape()[0]) throw std::invalid_argument("matmul inner dimensions must match");

    const auto rows = static_cast<std::size_t>(lhs.shape()[0]);
    const auto inner = static_cast<std::size_t>(lhs.shape()[1]);
    const auto cols = static_cast<std::size_t>(rhs.shape()[1]);
    Tensor output({static_cast<std::int64_t>(rows), static_cast<std::int64_t>(cols)});

    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            float sum = 0.0F;
            for (std::size_t k = 0; k < inner; ++k) {
                sum += lhs.at({static_cast<std::int64_t>(row), static_cast<std::int64_t>(k)}) *
                       rhs.at({static_cast<std::int64_t>(k), static_cast<std::int64_t>(col)});
            }
            output.at(row * cols + col) = sum;
        }
    }
    return output;
}

Tensor relu(const Tensor& input) {
    require_f32(input, "relu");
    TensorIterator iterator({input.layout()});
    Tensor output(input.shape());
    const auto* input_data = input.data<float>();
    auto* output_data = output.data<float>();
    if (iterator.has_contiguous_fast_path()) {
        for (std::size_t i = 0; i < iterator.numel(); ++i) {
            output_data[i] = std::max(0.0F, input_data[i]);
        }
        return output;
    }
    for (std::size_t i = 0; i < iterator.numel(); ++i) {
        output_data[i] = std::max(0.0F,
                                  input_data[iterator.operand_offset(0, i)]);
    }
    return output;
}

Tensor softmax(const Tensor& input) {
    require_f32(input, "softmax");
    if (input.rank() == 0 || input.shape().back() <= 0) {
        throw std::invalid_argument("softmax expects a non-empty final dimension");
    }

    Tensor output(input.shape());
    const auto width = static_cast<std::size_t>(input.shape().back());
    const auto rows = input.numel() / width;
    for (std::size_t row = 0; row < rows; ++row) {
        const auto offset = row * width;
        float maximum = logical_value(input, offset);
        for (std::size_t col = 1; col < width; ++col) {
            maximum = std::max(maximum, logical_value(input, offset + col));
        }

        float denominator = 0.0F;
        for (std::size_t col = 0; col < width; ++col) {
            output.at(offset + col) = std::exp(logical_value(input, offset + col) - maximum);
            denominator += output.at(offset + col);
        }
        for (std::size_t col = 0; col < width; ++col) output.at(offset + col) /= denominator;
    }
    return output;
}

Tensor linear(const Tensor& input, const Tensor& weight, const Tensor& bias) {
    return add(matmul(input, weight), bias);
}

}  // namespace tinyinfer::ops
