#include "tinyinfer/backend/cpu/operator_registry.h"

#include <cstdint>
#include <stdexcept>
#include <utility>

#include "tinyinfer/ops/basic_ops.h"
#include "tinyinfer/ops/gemm.h"
#include "tinyinfer/runtime/execution_context.h"

namespace tinyinfer::cpu {
namespace {

const Tensor& input(const Node& node, ExecutionContext& context,
                    std::size_t index) {
    return context.value(node.inputs.at(index));
}

Tensor& output(const Node& node, ExecutionContext& context) {
    if (node.outputs.size() != 1) {
        throw std::logic_error("CPU kernel expects exactly one output");
    }
    return context.prepare_output(node.outputs.front());
}

std::int64_t integer_attribute(const Node& node, const char* name,
                               std::int64_t default_value) {
    const auto iterator = node.attributes.find(name);
    return iterator == node.attributes.end()
               ? default_value
               : std::get<std::int64_t>(iterator->second);
}

float float_attribute(const Node& node, const char* name, float default_value) {
    const auto iterator = node.attributes.find(name);
    return iterator == node.attributes.end()
               ? default_value
               : std::get<float>(iterator->second);
}

void execute_gemm(const Node& node, ExecutionContext& context,
                  bool apply_relu) {
    const auto trans_a = integer_attribute(node, "transA", 0);
    const auto trans_b = integer_attribute(node, "transB", 0);
    const auto alpha = float_attribute(node, "alpha", 1.0F);
    const auto beta = float_attribute(node, "beta", 1.0F);
    const Tensor* bias =
        node.inputs.size() == 3 ? &input(node, context, 2) : nullptr;
    if (apply_relu) {
        ops::gemm_relu_out(output(node, context), input(node, context, 0),
                           input(node, context, 1), bias, alpha, beta,
                           trans_a, trans_b);
        return;
    }
    ops::gemm_out(output(node, context), input(node, context, 0),
                  input(node, context, 1), bias, alpha, beta,
                  trans_a, trans_b);
}

}  // namespace

OperatorRegistry::OperatorRegistry() {
    register_kernel(OpType::Add, [](const Node& node, ExecutionContext& context) {
        ops::add_out(output(node, context), input(node, context, 0),
                     input(node, context, 1));
    });
    register_kernel(OpType::Subtract,
                    [](const Node& node, ExecutionContext& context) {
        ops::sub_out(output(node, context), input(node, context, 0),
                     input(node, context, 1));
    });
    register_kernel(OpType::Multiply,
                    [](const Node& node, ExecutionContext& context) {
        ops::mul_out(output(node, context), input(node, context, 0),
                     input(node, context, 1));
    });
    register_kernel(OpType::MatMul,
                    [](const Node& node, ExecutionContext& context) {
        ops::matmul_out(output(node, context), input(node, context, 0),
                        input(node, context, 1));
    });
    register_kernel(OpType::Gemm, [](const Node& node,
                                     ExecutionContext& context) {
        execute_gemm(node, context, false);
    });
    register_kernel(OpType::FusedGemmActivation,
                    [](const Node& node, ExecutionContext& context) {
        const auto activation = node.attribute<std::string>("activation");
        if (activation != "relu") {
            throw std::invalid_argument(
                "FusedGemmActivation currently supports only relu");
        }
        execute_gemm(node, context, true);
    });
    register_kernel(OpType::ReLU, [](const Node& node, ExecutionContext& context) {
        ops::relu_out(output(node, context), input(node, context, 0));
    });
    register_kernel(OpType::Tanh, [](const Node& node, ExecutionContext& context) {
        ops::tanh_out(output(node, context), input(node, context, 0));
    });
    register_kernel(OpType::GELU, [](const Node& node, ExecutionContext& context) {
        const auto found = node.attributes.find("approximate");
        const auto approximate = found == node.attributes.end()
                                     ? std::string_view{"tanh"}
                                     : std::string_view{std::get<std::string>(found->second)};
        ops::gelu_out(output(node, context), input(node, context, 0),
                      approximate);
    });
    register_kernel(OpType::Softmax,
                    [](const Node& node, ExecutionContext& context) {
        const auto axis = integer_attribute(node, "axis", -1);
        ops::softmax_out(output(node, context), input(node, context, 0), axis);
    });
    register_kernel(OpType::LayerNorm,
                    [](const Node& node, ExecutionContext& context) {
        const auto epsilon = float_attribute(node, "epsilon", 1e-5F);
        if (node.inputs.size() == 1) {
            ops::layer_norm_out(output(node, context),
                                input(node, context, 0), epsilon);
            return;
        }
        if (node.inputs.size() == 2) {
            ops::layer_norm_out(output(node, context), input(node, context, 0),
                                input(node, context, 1), epsilon);
            return;
        }
        ops::layer_norm_out(output(node, context), input(node, context, 0),
                            input(node, context, 1),
                            input(node, context, 2), epsilon);
    });
}

void OperatorRegistry::register_kernel(OpType op, OperatorKernel kernel) {
    if (!kernel) throw std::invalid_argument("CPU operator kernel must be callable");
    if (!kernels_.emplace(op, std::move(kernel)).second) {
        throw std::invalid_argument("CPU operator kernel is already registered");
    }
}

bool OperatorRegistry::supports(OpType op) const noexcept {
    return kernels_.find(op) != kernels_.end();
}

void OperatorRegistry::execute(const Node& node,
                               ExecutionContext& context) const {
    const auto iterator = kernels_.find(node.op);
    if (iterator == kernels_.end()) {
        throw std::runtime_error("operator is not registered for the CPU backend");
    }
    iterator->second(node, context);
}

}  // namespace tinyinfer::cpu
