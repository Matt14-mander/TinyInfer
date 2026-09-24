#include "tinyinfer/model/onnx/operator_registry.h"

#include <cassert>
#include <stdexcept>
#include <string>

int main() {
    using tinyinfer::OpType;
    using tinyinfer::onnx::NodeProto;
    using tinyinfer::onnx::OperatorRegistry;
    using tinyinfer::onnx::TranslatedOperator;

    OperatorRegistry registry;
    assert(registry.supports("", "Gemm", 13));
    assert(registry.supports("ai.onnx", "Gemm", 17));
    assert(!registry.supports("custom", "Gemm", 17));
    assert(!registry.supports("", "Gelu", 19));
    assert(registry.supports("", "Gelu", 20));
    assert(!registry.supports("", "LayerNormalization", 16));
    assert(registry.supports("", "LayerNormalization", 17));

    NodeProto default_domain_node;
    default_domain_node.domain = "ai.onnx";
    default_domain_node.op_type = "Gemm";
    assert(registry.translate(default_domain_node, 17).op == OpType::Gemm);

    registry.register_translator(
        "custom", "Demo", 2, 4,
        [](const NodeProto&) { return TranslatedOperator{OpType::Add, {}}; });
    registry.register_translator(
        "custom", "Demo", 5, 6,
        [](const NodeProto&) { return TranslatedOperator{OpType::Multiply, {}}; });
    assert(!registry.supports("custom", "Demo", 1));
    assert(registry.supports("custom", "Demo", 4));
    assert(registry.supports("custom", "Demo", 5));
    assert(!registry.supports("custom", "Demo", 7));
    NodeProto node;
    node.domain = "custom";
    node.op_type = "Demo";
    assert(registry.translate(node, 4).op == OpType::Add);
    assert(registry.translate(node, 5).op == OpType::Multiply);

    bool overlap_rejected = false;
    try {
        registry.register_translator(
            "custom", "Demo", 4, 8,
            [](const NodeProto&) { return TranslatedOperator{OpType::ReLU, {}}; });
    } catch (const std::invalid_argument&) {
        overlap_rejected = true;
    }
    assert(overlap_rejected);

    bool invalid_range_rejected = false;
    try {
        registry.register_translator(
            "custom", "Demo", 0, 1,
            [](const NodeProto&) { return TranslatedOperator{OpType::ReLU, {}}; });
    } catch (const std::invalid_argument&) {
        invalid_range_rejected = true;
    }
    assert(invalid_range_rejected);

    bool unsupported_version_rejected = false;
    try {
        static_cast<void>(registry.translate(node, 7));
    } catch (const std::invalid_argument& error) {
        unsupported_version_rejected =
            std::string(error.what()).find("opset=7") != std::string::npos;
    }
    assert(unsupported_version_rejected);
}
