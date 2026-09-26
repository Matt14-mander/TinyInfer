#include "tinyinfer/ops/basic_ops.h"

#include "tinyinfer/ops/kernel_runner.h"
#include "tinyinfer/core/reduction_iterator.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace tinyinfer::ops {
namespace {

// the data type for float32
void require_f32(const Tensor& tensor, const char* operation) {
    if (tensor.dtype() != DataType::Float32) {
        throw std::invalid_argument(std::string(operation) + " currently supports only float32 tensors");
    }
}

void layer_norm_impl(Tensor& output, const Tensor& input, const Tensor* weight,
                     const Tensor* bias, float epsilon) {
    require_f32(input, "layer_norm");
    if (input.rank() == 0 || input.shape().back() <= 0) {
        throw std::invalid_argument(
            "layer_norm expects a non-empty final dimension");
    }
    if (!std::isfinite(epsilon) || epsilon < 0.0F) {
        throw std::invalid_argument("layer_norm epsilon must be finite and non-negative");
    }

    const auto width = input.shape().back();
    for (const auto* parameter : {weight, bias}) {
        if (!parameter) continue;
        require_f32(*parameter, "layer_norm");
        if (parameter->shape() != Shape({width})) {
            throw std::invalid_argument(
                "layer_norm weight and bias must match the final dimension");
        }
    }

    if (output.shape() != input.shape() ||
        output.dtype() != DataType::Float32 || !output.is_contiguous()) {
        throw std::invalid_argument(
            "layer_norm output must be contiguous and match the input");
    }
    ReductionIterator iterator(input.layout(), {-1});
    const auto* input_data = input.data<float>();
    auto* output_data = output.data<float>();
    const auto* weight_data = weight ? weight->data<float>() : nullptr;
    const auto* bias_data = bias ? bias->data<float>() : nullptr;
    const auto contiguous = iterator.has_contiguous_reduction();

    for (std::size_t group = 0; group < iterator.output_numel(); ++group) {
        const auto base = group * iterator.reduction_numel();
        float mean = 0.0F;
        for (std::size_t index = 0; index < iterator.reduction_numel(); ++index) {
            const auto offset = contiguous ? base + index
                                           : iterator.input_offset(group, index);
            mean += input_data[offset]; // sum
        }
        mean /= static_cast<float>(iterator.reduction_numel()); // mean

        float variance = 0.0F;
        for (std::size_t index = 0; index < iterator.reduction_numel(); ++index) {
            const auto offset = contiguous ? base + index
                                           : iterator.input_offset(group, index);
            const auto centered = input_data[offset] - mean;
            variance += centered * centered;
        }
        variance /= static_cast<float>(iterator.reduction_numel());
        const auto inverse_stddev = 1.0F / std::sqrt(variance + epsilon); // inverse standard deviation

        for (std::size_t index = 0; index < iterator.reduction_numel(); ++index) {
            const auto input_offset = contiguous
                                          ? base + index
                                          : iterator.input_offset(group, index);
            const auto output_index = contiguous
                                          ? base + index
                                          : iterator.input_logical_index(group, index);
            auto value = (input_data[input_offset] - mean) * inverse_stddev;
            // AffineTransform
            if (weight_data) value *= weight_data[weight->layout().storage_offset(index)];
            if (bias_data) value += bias_data[bias->layout().storage_offset(index)];
            output_data[output_index] = value;
        }
    }
}

}  // namespace

// add operator for two tensor
Tensor add(const Tensor& lhs, const Tensor& rhs) {
    TensorIterator iterator({lhs.layout(), rhs.layout()});
    Tensor output(iterator.shape(), DataType::Float32);
    add_out(output, lhs, rhs);
    return output;
}

// write the output tensor.
void add_out(Tensor& output, const Tensor& lhs, const Tensor& rhs) {
    run_binary_kernel_into<float>(output, lhs, rhs,
                                  [](float left, float right) {
                                      return left + right;
                                  });
}

// subtract operator for two tensor
Tensor sub(const Tensor& lhs, const Tensor& rhs) {
    TensorIterator iterator({lhs.layout(), rhs.layout()});
    Tensor output(iterator.shape(), DataType::Float32);
    sub_out(output, lhs, rhs);
    return output;
}

// write the output tensor.
void sub_out(Tensor& output, const Tensor& lhs, const Tensor& rhs) {
    run_binary_kernel_into<float>(output, lhs, rhs,
                                  [](float left, float right) {
                                      return left - right;
                                  });
}

// multiply operator for two tensor
Tensor mul(const Tensor& lhs, const Tensor& rhs) {
    TensorIterator iterator({lhs.layout(), rhs.layout()});
    Tensor output(iterator.shape(), DataType::Float32);
    mul_out(output, lhs, rhs);
    return output;
}

// write the output tensor.
void mul_out(Tensor& output, const Tensor& lhs, const Tensor& rhs) {
    run_binary_kernel_into<float>(output, lhs, rhs,
                                  [](float left, float right) {
                                      return left * right;
                                  });
}

// the rectified linear unit of each element in the input tensor.
Tensor relu(const Tensor& input) {
    Tensor output(input.shape(), DataType::Float32);
    relu_out(output, input);
    return output;
}

// writes the result to the output tensor.
void relu_out(Tensor& output, const Tensor& input) {
    run_unary_kernel_into<float>(output, input,
                                 [](float value) {
                                     return std::max(0.0F, value);
                                 });
}

// the hyperbolic tangent of each element in the input tensor.
Tensor tanh(const Tensor& input) {
    Tensor output(input.shape(), DataType::Float32);
    tanh_out(output, input);
    return output;
}

// writes the result to the output tensor.
void tanh_out(Tensor& output, const Tensor& input) {
    run_unary_kernel_into<float>(output, input,
                                 [](float value) { return std::tanh(value); });
}

// the Gaussian error linear unit of each element in the input tensor.
Tensor gelu(const Tensor& input) {
    Tensor output(input.shape(), DataType::Float32);
    gelu_out(output, input);
    return output;
}

// write the output tensor.
void gelu_out(Tensor& output, const Tensor& input,
              std::string_view approximate) {
    if (approximate == "none") {
        constexpr float kInverseSqrtTwo = 0.7071067811865475F;
        run_unary_kernel_into<float>(output, input, [](float value) {
            return 0.5F * value * (1.0F + std::erf(value * kInverseSqrtTwo));
        });
        return;
    }
    if (approximate != "tanh") {
        throw std::invalid_argument("gelu approximate must be 'none' or 'tanh'");
    }
    // tanh approximation used by many inference runtimes.
    constexpr float kSqrtTwoOverPi = 0.7978845608028654F;
    constexpr float kCubicCoefficient = 0.044715F;
    run_unary_kernel_into<float>(
        output, input,
        [](float value) {
            const auto cubic = value * value * value;
            return 0.5F * value *
                   (1.0F + std::tanh(kSqrtTwoOverPi *
                                     (value + kCubicCoefficient * cubic)));
        });
}

// reduce_sum computes the sum of the input tensor along the specified axes.
Tensor reduce_sum(const Tensor& input,
                  const std::vector<std::int64_t>& axes, bool keepdim) {
    require_f32(input, "reduce_sum");
    ReductionIterator iterator(input.layout(), axes, keepdim);
    Tensor output(iterator.output_shape());
    const auto* input_data = input.data<float>();
    auto* output_data = output.data<float>();

    for (std::size_t group = 0; group < iterator.output_numel(); ++group) {
        float result = 0.0F;
        if (iterator.has_contiguous_reduction()) {
            const auto base = group * iterator.reduction_numel();
            for (std::size_t index = 0; index < iterator.reduction_numel(); ++index) {
                result += input_data[base + index];
            }
        } else {
            for (std::size_t index = 0; index < iterator.reduction_numel(); ++index) {
                result += input_data[iterator.input_offset(group, index)];
            }
        }
        output_data[group] = result;
    }
    return output;
}

// reduce_max computes the maximum of the input tensor along the specified axes.
Tensor reduce_max(const Tensor& input,
                  const std::vector<std::int64_t>& axes, bool keepdim) {
    require_f32(input, "reduce_max");
    ReductionIterator iterator(input.layout(), axes, keepdim);
    if (iterator.reduction_numel() == 0) {
        throw std::invalid_argument("reduce_max cannot reduce an empty dimension");
    }
    Tensor output(iterator.output_shape());
    const auto* input_data = input.data<float>();
    auto* output_data = output.data<float>();

    for (std::size_t group = 0; group < iterator.output_numel(); ++group) {
        const auto first_offset = iterator.has_contiguous_reduction()
                                      ? group * iterator.reduction_numel()
                                      : iterator.input_offset(group, 0);
        float result = input_data[first_offset];
        for (std::size_t index = 1; index < iterator.reduction_numel(); ++index) {
            const auto offset = iterator.has_contiguous_reduction()
                                    ? group * iterator.reduction_numel() + index
                                    : iterator.input_offset(group, index);
            result = std::max(result, input_data[offset]);
        }
        output_data[group] = result;
    }
    return output;
}

// softmax of the input tensor along the specified axis.
Tensor softmax(const Tensor& input, std::int64_t axis) {
    Tensor output(input.shape(), DataType::Float32);
    softmax_out(output, input, axis);
    return output;
}

// write the output tensor.
void softmax_out(Tensor& output, const Tensor& input, std::int64_t axis) {
    require_f32(input, "softmax");
    ReductionIterator iterator(input.layout(), {axis});
    if (iterator.reduction_numel() == 0) {
        throw std::invalid_argument("softmax cannot normalize an empty dimension");
    }
    if (output.shape() != input.shape() ||
        output.dtype() != DataType::Float32 || !output.is_contiguous()) {
        throw std::invalid_argument(
            "softmax output must be contiguous and match the input");
    }
    const auto* input_data = input.data<float>();
    auto* output_data = output.data<float>();
    const auto contiguous = iterator.has_contiguous_reduction();
    for (std::size_t group = 0; group < iterator.output_numel(); ++group) {
        const auto base = group * iterator.reduction_numel();
        const auto first_offset = contiguous ? base
                                             : iterator.input_offset(group, 0);
        float maximum = input_data[first_offset];
        for (std::size_t index = 1; index < iterator.reduction_numel(); ++index) {
            const auto offset = contiguous ? base + index
                                           : iterator.input_offset(group, index);
            maximum = std::max(
                maximum, input_data[offset]);
        }

        float denominator = 0.0F;
        for (std::size_t index = 0; index < iterator.reduction_numel(); ++index) {
            const auto input_offset = contiguous
                                          ? base + index
                                          : iterator.input_offset(group, index);
            const auto output_index = contiguous
                                          ? base + index
                                          : iterator.input_logical_index(group, index);
            output_data[output_index] =
                std::exp(input_data[input_offset] - maximum);
            denominator += output_data[output_index];
        }
        for (std::size_t index = 0; index < iterator.reduction_numel(); ++index) {
            const auto output_index = contiguous
                                          ? base + index
                                          : iterator.input_logical_index(group, index);
            output_data[output_index] /= denominator;
        }
    }
}

//  Layer normalization
// 仿射变换： weight * x_norm + bias
Tensor layer_norm(const Tensor& input, float epsilon) {
    Tensor output(input.shape(), DataType::Float32);
    layer_norm_impl(output, input, nullptr, nullptr, epsilon); // epsilon (minimum value)
    return output;
}

// write the output tensor.
void layer_norm_out(Tensor& output, const Tensor& input, float epsilon) {
    layer_norm_impl(output, input, nullptr, nullptr, epsilon);
}

// add weight value 缩放参数
Tensor layer_norm(const Tensor& input, const Tensor& weight, float epsilon) {
    Tensor output(input.shape(), DataType::Float32);
    layer_norm_impl(output, input, &weight, nullptr, epsilon);
    return output;
}

// write the output tensor.
void layer_norm_out(Tensor& output, const Tensor& input,
                    const Tensor& weight, float epsilon) {
    layer_norm_impl(output, input, &weight, nullptr, epsilon);
}

// add bias value 偏移参数
Tensor layer_norm(const Tensor& input, const Tensor& weight,
                  const Tensor& bias, float epsilon) {
    Tensor output(input.shape(), DataType::Float32);
    layer_norm_impl(output, input, &weight, &bias, epsilon);
    return output;
}

// write the output tensor.
void layer_norm_out(Tensor& output, const Tensor& input,
                    const Tensor& weight, const Tensor& bias, float epsilon) {
    layer_norm_impl(output, input, &weight, &bias, epsilon);
}

// linear computes the linear transformation of the input tensor using the weight and bias tensors.
Tensor linear(const Tensor& input, const Tensor& weight, const Tensor& bias) {
    return add(matmul(input, weight), bias);
}

}  // namespace tinyinfer::ops
