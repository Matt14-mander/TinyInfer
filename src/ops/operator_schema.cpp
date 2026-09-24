#include "tinyinfer/ops/operator_schema.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "tinyinfer/core/tensor_iterator.h"

namespace tinyinfer {
namespace {

const OperatorSchema kAdd{OpType::Add, "Add", 2, 2, 1, {}};
const OperatorSchema kMultiply{OpType::Multiply, "Multiply", 2, 2, 1, {}};
const OperatorSchema kSubtract{OpType::Subtract, "Subtract", 2, 2, 1, {}};
const OperatorSchema kMatMul{OpType::MatMul, "MatMul", 2, 2, 1, {}};
const OperatorSchema kGemm{
    OpType::Gemm, "Gemm", 2, 3, 1,
    {{"alpha", AttributeType::Float, false},
     {"beta", AttributeType::Float, false},
     {"transA", AttributeType::Integer, false},
     {"transB", AttributeType::Integer, false}}};
const OperatorSchema kReLU{OpType::ReLU, "ReLU", 1, 1, 1, {}};
const OperatorSchema kGELU{
    OpType::GELU, "GELU", 1, 1, 1,
    {{"approximate", AttributeType::String, false}}};
const OperatorSchema kSoftmax{
    OpType::Softmax, "Softmax", 1, 1, 1,
    {{"axis", AttributeType::Integer, false}}};
const OperatorSchema kLayerNorm{
    OpType::LayerNorm, "LayerNorm", 1, 3, 1,
    {{"epsilon", AttributeType::Float, false}}};

AttributeType attribute_type(const AttributeValue& value) {
    switch (value.index()) {
        case 0: return AttributeType::Integer;
        case 1: return AttributeType::Float;
        case 2: return AttributeType::Boolean;
        case 3: return AttributeType::String;
        case 4: return AttributeType::Integers;
        default: throw std::logic_error("unknown graph attribute type");
    }
}

void validate_attributes(const OperatorSchema& schema,
                         const NodeAttributes& attributes) {
    for (const auto& entry : attributes) {
        const auto& name = entry.first;
        const auto& value = entry.second;
        const auto definition = std::find_if(
            schema.attributes.begin(), schema.attributes.end(),
            [&](const AttributeSchema& attribute) {
                return attribute.name == name;
            });
        if (definition == schema.attributes.end()) {
            throw std::invalid_argument(std::string(schema.name) +
                                        " has an unknown attribute: " + name);
        }
        if (definition->type != attribute_type(value)) {
            throw std::invalid_argument(std::string(schema.name) +
                                        " attribute has the wrong type: " + name);
        }
    }
    for (const auto& definition : schema.attributes) {
        if (definition.required &&
            attributes.find(std::string(definition.name)) == attributes.end()) {
            throw std::invalid_argument(std::string(schema.name) +
                                        " is missing a required attribute");
        }
    }
}

void require_float32(const OperatorSchema& schema,
                     const std::vector<TensorSpec>& inputs) {
    for (const auto& input : inputs) {
        if (input.dtype != DataType::Float32) {
            throw std::invalid_argument(std::string(schema.name) +
                                        " currently requires float32 inputs");
        }
    }
}

TensorSpec infer_elementwise(const std::vector<TensorSpec>& inputs) {
    TensorIterator iterator(
        {TensorLayout(inputs[0].shape), TensorLayout(inputs[1].shape)});
    return TensorSpec{iterator.shape(), inputs[0].dtype};
}

std::int64_t normalized_axis(const TensorSpec& input,
                             const NodeAttributes& attributes) {
    auto axis = std::int64_t{-1};
    const auto iterator = attributes.find("axis");
    if (iterator != attributes.end()) axis = std::get<std::int64_t>(iterator->second);
    if (axis < 0) axis += static_cast<std::int64_t>(input.shape.size());
    if (axis < 0 || axis >= static_cast<std::int64_t>(input.shape.size())) {
        throw std::out_of_range("Softmax axis is out of range");
    }
    if (input.shape[static_cast<std::size_t>(axis)] == 0) {
        throw std::invalid_argument("Softmax axis must not be empty");
    }
    return axis;
}

}  // namespace

const OperatorSchema& operator_schema(OpType op) {
    switch (op) {
        case OpType::Add: return kAdd;
        case OpType::Multiply: return kMultiply;
        case OpType::Subtract: return kSubtract;
        case OpType::MatMul: return kMatMul;
        case OpType::Gemm: return kGemm;
        case OpType::ReLU: return kReLU;
        case OpType::GELU: return kGELU;
        case OpType::Softmax: return kSoftmax;
        case OpType::LayerNorm: return kLayerNorm;
    }
    throw std::invalid_argument("unknown operator type");
}

std::vector<TensorSpec> infer_output_specs(
    OpType op, const std::vector<TensorSpec>& inputs,
    const NodeAttributes& attributes) {
    const auto& schema = operator_schema(op);
    if (inputs.size() < schema.minimum_inputs ||
        inputs.size() > schema.maximum_inputs) {
        throw std::invalid_argument(std::string(schema.name) +
                                    " received the wrong number of inputs");
    }
    for (const auto& input : inputs) {
        const TensorLayout layout(input.shape);
        layout.size_bytes(input.dtype);
    }
    validate_attributes(schema, attributes);
    require_float32(schema, inputs);

    switch (op) {
        case OpType::Add:
        case OpType::Multiply:
        case OpType::Subtract:
            if (inputs[0].dtype != inputs[1].dtype) {
                throw std::invalid_argument("Elementwise input dtypes must match");
            }
            return {infer_elementwise(inputs)};

        case OpType::MatMul:
            if (inputs[0].dtype != inputs[1].dtype) {
                throw std::invalid_argument("MatMul input dtypes must match");
            }
            if (inputs[0].shape.size() != 2 || inputs[1].shape.size() != 2) {
                throw std::invalid_argument("MatMul expects rank-2 inputs");
            }
            if (inputs[0].shape[1] != inputs[1].shape[0]) {
                throw std::invalid_argument("MatMul inner dimensions must match");
            }
            return {TensorSpec{{inputs[0].shape[0], inputs[1].shape[1]},
                               inputs[0].dtype}};

        case OpType::Gemm: {
            if (inputs[0].dtype != inputs[1].dtype ||
                (inputs.size() == 3 && inputs[0].dtype != inputs[2].dtype)) {
                throw std::invalid_argument("Gemm input dtypes must match");
            }
            if (inputs[0].shape.size() != 2 || inputs[1].shape.size() != 2) {
                throw std::invalid_argument("Gemm expects rank-2 A and B inputs");
            }
            const auto integer_attribute = [&](const char* name) {
                const auto iterator = attributes.find(name);
                return iterator == attributes.end()
                           ? std::int64_t{0}
                           : std::get<std::int64_t>(iterator->second);
            };
            const auto trans_a = integer_attribute("transA");
            const auto trans_b = integer_attribute("transB");
            if ((trans_a != 0 && trans_a != 1) ||
                (trans_b != 0 && trans_b != 1)) {
                throw std::invalid_argument("Gemm transpose attributes must be 0 or 1");
            }
            const auto m = inputs[0].shape[trans_a == 0 ? 0 : 1];
            const auto k_a = inputs[0].shape[trans_a == 0 ? 1 : 0];
            const auto k_b = inputs[1].shape[trans_b == 0 ? 0 : 1];
            const auto n = inputs[1].shape[trans_b == 0 ? 1 : 0];
            if (k_a != k_b) {
                throw std::invalid_argument("Gemm inner dimensions must match");
            }
            const TensorSpec output{{m, n}, inputs[0].dtype};
            if (inputs.size() == 3) {
                TensorIterator iterator(
                    {TensorLayout(output.shape), TensorLayout(inputs[2].shape)});
                if (iterator.shape() != output.shape) {
                    throw std::invalid_argument(
                        "Gemm bias must broadcast to the output shape");
                }
            }
            return {output};
        }

        case OpType::ReLU:
            return {inputs[0]};

        case OpType::GELU: {
            const auto found = attributes.find("approximate");
            if (found != attributes.end()) {
                const auto& value = std::get<std::string>(found->second);
                if (value != "none" && value != "tanh") {
                    throw std::invalid_argument(
                        "GELU approximate must be 'none' or 'tanh'");
                }
            }
            return {inputs[0]};
        }

        case OpType::Softmax:
            if (inputs[0].shape.empty()) {
                throw std::invalid_argument("Softmax expects rank >= 1");
            }
            normalized_axis(inputs[0], attributes);
            return {inputs[0]};

        case OpType::LayerNorm: {
            if (inputs[0].shape.empty() || inputs[0].shape.back() == 0) {
                throw std::invalid_argument(
                    "LayerNorm expects a non-empty final dimension");
            }
            if (inputs.size() >= 2) {
                const Shape parameter_shape{inputs[0].shape.back()};
                if (inputs[1].shape != parameter_shape ||
                    inputs[1].dtype != inputs[0].dtype) {
                    throw std::invalid_argument(
                        "LayerNorm weight must match the final dimension");
                }
                if (inputs.size() == 3 &&
                    (inputs[2].shape != parameter_shape ||
                     inputs[2].dtype != inputs[0].dtype)) {
                    throw std::invalid_argument(
                        "LayerNorm bias must match the final dimension");
                }
            }
            const auto epsilon = attributes.find("epsilon");
            if (epsilon != attributes.end() &&
                (!std::isfinite(std::get<float>(epsilon->second)) ||
                 std::get<float>(epsilon->second) < 0.0F)) {
                throw std::invalid_argument(
                    "LayerNorm epsilon must be finite and non-negative");
            }
            return {inputs[0]};
        }
    }
    throw std::logic_error("operator shape inference is not implemented");
}

}  // namespace tinyinfer
