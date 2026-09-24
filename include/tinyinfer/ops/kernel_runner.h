#pragma once

#include <cstddef>
#include <functional>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "tinyinfer/core/tensor.h"
#include "tinyinfer/core/tensor_iterator.h"

namespace tinyinfer::ops {

namespace detail {

template <typename T>
void require_kernel_dtype(const Tensor& tensor) {
    using StorageType = std::remove_cv_t<T>;
    static_assert(is_supported_storage_type_v<StorageType>,
                  "unsupported TinyInfer kernel storage type");
    if (tensor.dtype() != data_type_of<StorageType>()) {
        throw std::invalid_argument(
            "kernel storage type does not match tensor dtype");
    }
}

}  // namespace detail

// Runs an elementwise unary operation and materializes a contiguous output.
template <typename T, typename Operation>
void run_unary_kernel_into(Tensor& output, const Tensor& input,
                           Operation&& operation) {
    using StorageType = std::remove_cv_t<T>;
    detail::require_kernel_dtype<StorageType>(input);
    detail::require_kernel_dtype<StorageType>(output);

    TensorIterator iterator({input.layout()});
    if (output.shape() != iterator.shape() || !output.is_contiguous()) {
        throw std::invalid_argument(
            "unary kernel output must be contiguous and match its input shape");
    }
    const auto* input_data = input.data<StorageType>();
    auto* output_data = output.data<StorageType>();

    if (iterator.has_contiguous_fast_path()) {
        for (std::size_t index = 0; index < iterator.numel(); ++index) {
            output_data[index] = static_cast<StorageType>(
                std::invoke(operation, input_data[index]));
        }
        return;
    }

    for (std::size_t index = 0; index < iterator.numel(); ++index) {
        output_data[index] = static_cast<StorageType>(std::invoke(
            operation, input_data[iterator.operand_offset(0, index)]));
    }
}

template <typename T, typename Operation>
Tensor run_unary_kernel(const Tensor& input, Operation&& operation) {
    TensorIterator iterator({input.layout()});
    Tensor output(iterator.shape(), data_type_of<std::remove_cv_t<T>>());
    run_unary_kernel_into<T>(output, input,
                             std::forward<Operation>(operation));
    return output;
}

// Runs an elementwise binary operation with NumPy-style broadcasting and
// materializes a contiguous output.
template <typename T, typename Operation>
void run_binary_kernel_into(Tensor& output, const Tensor& lhs,
                            const Tensor& rhs, Operation&& operation) {
    using StorageType = std::remove_cv_t<T>;
    detail::require_kernel_dtype<StorageType>(lhs);
    detail::require_kernel_dtype<StorageType>(rhs);
    detail::require_kernel_dtype<StorageType>(output);

    TensorIterator iterator({lhs.layout(), rhs.layout()});
    if (output.shape() != iterator.shape() || !output.is_contiguous()) {
        throw std::invalid_argument(
            "binary kernel output must be contiguous and match broadcast shape");
    }
    const auto* lhs_data = lhs.data<StorageType>();
    const auto* rhs_data = rhs.data<StorageType>();
    auto* output_data = output.data<StorageType>();

    if (iterator.has_contiguous_fast_path()) {
        for (std::size_t index = 0; index < iterator.numel(); ++index) {
            output_data[index] = static_cast<StorageType>(
                std::invoke(operation, lhs_data[index], rhs_data[index]));
        }
        return;
    }

    for (std::size_t index = 0; index < iterator.numel(); ++index) {
        output_data[index] = static_cast<StorageType>(std::invoke(
            operation,
            lhs_data[iterator.operand_offset(0, index)],
            rhs_data[iterator.operand_offset(1, index)]));
    }
}

template <typename T, typename Operation>
Tensor run_binary_kernel(const Tensor& lhs, const Tensor& rhs,
                         Operation&& operation) {
    TensorIterator iterator({lhs.layout(), rhs.layout()});
    Tensor output(iterator.shape(), data_type_of<std::remove_cv_t<T>>());
    run_binary_kernel_into<T>(output, lhs, rhs,
                              std::forward<Operation>(operation));
    return output;
}

}  // namespace tinyinfer::ops
