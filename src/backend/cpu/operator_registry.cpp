#include "tinyinfer/backend/cpu/operator_registry.h"

#include <cstdint>
#include <stdexcept>
#include <utility>

#include "tinyinfer/ops/basic_ops.h"
#include "tinyinfer/runtime/execution_context.h"

namespace tinyinfer::cpu {
namespace {

const Tensor& input(const Node& node, ExecutionContext& context,
                    std::size_t index) {
    return context.value(node.inputs.at(index));
}

void store_output(const Node& node, ExecutionContext& context, Tensor tensor) {
    if (node.outputs.size() != 1) {
        throw std::logic_error("CPU kernel expects exactly one output");
    }
    context.set_value(node.outputs.front(), std::move(tensor));
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

}  // namespace

OperatorRegistry::OperatorRegistry() {
    register_kernel(OpType::Add, [](const Node& node, ExecutionContext& context) {
        store_output(node, context,
                     ops::add(input(node, context, 0), input(node, context, 1)));
    });
    register_kernel(OpType::Subtract,
                    [](const Node& node, ExecutionContext& context) {
        store_output(node, context,
                     ops::sub(input(node, context, 0), input(node, context, 1)));
    });
    register_kernel(OpType::Multiply,
                    [](const Node& node, ExecutionContext& context) {
        store_output(node, context,
                     ops::mul(input(node, context, 0), input(node, context, 1)));
    });
    register_kernel(OpType::MatMul,
                    [](const Node& node, ExecutionContext& context) {
        store_output(node, context,
                     ops::matmul(input(node, context, 0),
                                 input(node, context, 1)));
    });
    register_kernel(OpType::Gemm, [](const Node& node, ExecutionContext& context) {
        const auto trans_a = integer_attribute(node, "transA", 0);
        const auto trans_b = integer_attribute(node, "transB", 0);
        const auto alpha = float_attribute(node, "alpha", 1.0F);
        const auto beta = float_attribute(node, "beta", 1.0F);

        auto lhs = input(node, context, 0);
        auto rhs = input(node, context, 1);
        if (trans_a != 0) lhs.transpose(0, 1);
        if (trans_b != 0) rhs.transpose(0, 1);
        auto output = ops::matmul(lhs, rhs);
        if (alpha != 1.0F) {
            output = ops::mul(
                output, Tensor::from_vector(Shape{}, std::vector<float>{alpha}));
        }
        if (node.inputs.size() == 3) {
            auto bias = input(node, context, 2);
            if (beta != 1.0F) {
                bias = ops::mul(
                    bias, Tensor::from_vector(Shape{}, std::vector<float>{beta}));
            }
            output = ops::add(output, bias);
        }
        store_output(node, context, std::move(output));
    });
    register_kernel(OpType::ReLU, [](const Node& node, ExecutionContext& context) {
        store_output(node, context, ops::relu(input(node, context, 0)));
    });
    register_kernel(OpType::GELU, [](const Node& node, ExecutionContext& context) {
        store_output(node, context, ops::gelu(input(node, context, 0)));
    });
    register_kernel(OpType::Softmax,
                    [](const Node& node, ExecutionContext& context) {
        const auto axis = integer_attribute(node, "axis", -1);
        store_output(node, context,
                     ops::softmax(input(node, context, 0), axis));
    });
    register_kernel(OpType::LayerNorm,
                    [](const Node& node, ExecutionContext& context) {
        const auto epsilon = float_attribute(node, "epsilon", 1e-5F);
        if (node.inputs.size() == 1) {
            store_output(node, context,
                         ops::layer_norm(input(node, context, 0), epsilon));
            return;
        }
        store_output(node, context,
                     ops::layer_norm(input(node, context, 0),
                                     input(node, context, 1),
                                     input(node, context, 2), epsilon));
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
