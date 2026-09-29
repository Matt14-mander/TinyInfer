#include "tinyinfer/ops/gemm.h"

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <string>

#include "tinyinfer/core/tensor_iterator.h"
#include "tinyinfer/ops/matmul.h"
#include "ops/gemm_internal.h"

namespace tinyinfer::ops {

namespace detail {

void apply_gemm_epilogue(Tensor& output, const Tensor* bias,
                         float alpha, float beta, bool apply_relu) {
    auto* output_data = output.data<float>();
    if (bias != nullptr) {
        TensorIterator iterator({output.layout(), bias->layout()});
        if (iterator.shape() != output.shape()) {
            throw std::invalid_argument(
                "Gemm bias must broadcast to the output shape");
        }
        const auto* bias_data = bias->data<float>();
        for (std::size_t index = 0; index < iterator.numel(); ++index) {
            auto value = alpha * output_data[index] +
                         beta * bias_data[iterator.operand_offset(1, index)];
            if (apply_relu) {
                value = std::max(0.0F, value);
            }
            output_data[index] = value;
        }
        return;
    }

    if (alpha != 1.0F || apply_relu) {
        for (std::size_t index = 0; index < output.numel(); ++index) {
            auto value = alpha * output_data[index];
            if (apply_relu) {
                value = std::max(0.0F, value);
            }
            output_data[index] = value;
        }
    }
}

}  // namespace detail

namespace {

enum class Epilogue { None, ReLU };

void validate_transpose(std::int64_t value, const char* name) {
    if (value != 0 && value != 1) {
        throw std::invalid_argument(std::string("Gemm ") + name +
                                    " must be 0 or 1");
    }
}

void run_gemm(Tensor& output, const Tensor& lhs, const Tensor& rhs,
              const Tensor* bias, float alpha, float beta,
              std::int64_t trans_a, std::int64_t trans_b,
              Epilogue epilogue) {
    validate_transpose(trans_a, "transA");
    validate_transpose(trans_b, "transB");

    const Tensor* lhs_operand = &lhs;
    const Tensor* rhs_operand = &rhs;
    std::optional<Tensor> transposed_lhs;
    std::optional<Tensor> transposed_rhs;
    if (trans_a != 0) {
        transposed_lhs.emplace(lhs);
        transposed_lhs->transpose(0, 1);
        lhs_operand = &*transposed_lhs;
    }
    if (trans_b != 0) {
        transposed_rhs.emplace(rhs);
        transposed_rhs->transpose(0, 1);
        rhs_operand = &*transposed_rhs;
    }

    matmul_out(output, *lhs_operand, *rhs_operand);
    detail::apply_gemm_epilogue(output, bias, alpha, beta,
                                epilogue == Epilogue::ReLU);
}

}  // namespace

void gemm_out(Tensor& output, const Tensor& lhs, const Tensor& rhs,
              const Tensor* bias, float alpha, float beta,
              std::int64_t trans_a, std::int64_t trans_b) {
    run_gemm(output, lhs, rhs, bias, alpha, beta, trans_a, trans_b,
             Epilogue::None);
}

void gemm_relu_out(Tensor& output, const Tensor& lhs, const Tensor& rhs,
                   const Tensor* bias, float alpha, float beta,
                   std::int64_t trans_a, std::int64_t trans_b) {
    run_gemm(output, lhs, rhs, bias, alpha, beta, trans_a, trans_b,
             Epilogue::ReLU);
}

}  // namespace tinyinfer::ops
