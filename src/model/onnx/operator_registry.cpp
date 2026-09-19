#include "tinyinfer/model/onnx/operator_registry.h"

#include <stdexcept>
#include <utility>

namespace tinyinfer::onnx {
namespace {

OperatorTranslator direct(OpType op) {
    return [op](const NodeProto& node) {
        return TranslatedOperator{op, node.attributes};
    };
}

}  // namespace

OperatorRegistry::OperatorRegistry() {
    register_translator("Add", direct(OpType::Add));
    register_translator("Sub", direct(OpType::Subtract));
    register_translator("Mul", direct(OpType::Multiply));
    register_translator("MatMul", direct(OpType::MatMul));
    register_translator("Gemm", direct(OpType::Gemm));
    register_translator("Relu", direct(OpType::ReLU));
    register_translator("Gelu", direct(OpType::GELU));
    register_translator("Softmax", direct(OpType::Softmax));
    register_translator("LayerNormalization", direct(OpType::LayerNorm));
}

void OperatorRegistry::register_translator(
    std::string op_type, OperatorTranslator translator) {
    if (op_type.empty() || !translator) {
        throw std::invalid_argument(
            "ONNX operator translator must have a name and callable");
    }
    if (!translators_.emplace(std::move(op_type), std::move(translator)).second) {
        throw std::invalid_argument("ONNX operator translator is already registered");
    }
}

bool OperatorRegistry::supports(const std::string& op_type) const noexcept {
    return translators_.find(op_type) != translators_.end();
}

TranslatedOperator OperatorRegistry::translate(const NodeProto& node) const {
    const auto iterator = translators_.find(node.op_type);
    if (iterator == translators_.end()) {
        throw std::invalid_argument("unsupported ONNX operator: " + node.op_type);
    }
    return iterator->second(node);
}

}  // namespace tinyinfer::onnx
