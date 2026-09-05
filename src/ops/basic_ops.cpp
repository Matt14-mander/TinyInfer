#include "tinyinfer/ops/basic_ops.h"

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

}  // namespace

Tensor add(const Tensor& lhs, const Tensor& rhs) {
    require_f32(lhs, "add");
    require_f32(rhs, "add");

    Tensor output(lhs.shape());
    if (lhs.shape() == rhs.shape()) {
        for (std::size_t i = 0; i < lhs.numel(); ++i) output.at(i) = lhs.at(i) + rhs.at(i);
        return output;
    }

    // The Phase 0 Linear operator needs one simple broadcast form: [..., N] + [N].
    if (rhs.rank() == 1 && !lhs.shape().empty() && rhs.shape()[0] == lhs.shape().back()) {
        const auto width = static_cast<std::size_t>(rhs.shape()[0]);
        for (std::size_t i = 0; i < lhs.numel(); ++i) output.at(i) = lhs.at(i) + rhs.at(i % width);
        return output;
    }

    throw std::invalid_argument("add expects equal shapes or a one-dimensional bias matching the last dimension");
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
                sum += lhs.at(row * inner + k) * rhs.at(k * cols + col);
            }
            output.at(row * cols + col) = sum;
        }
    }
    return output;
}

Tensor relu(const Tensor& input) {
    require_f32(input, "relu");
    Tensor output(input.shape());
    for (std::size_t i = 0; i < input.numel(); ++i) output.at(i) = std::max(0.0F, input.at(i));
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
        float maximum = input.at(offset);
        for (std::size_t col = 1; col < width; ++col) maximum = std::max(maximum, input.at(offset + col));

        float denominator = 0.0F;
        for (std::size_t col = 0; col < width; ++col) {
            output.at(offset + col) = std::exp(input.at(offset + col) - maximum);
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
