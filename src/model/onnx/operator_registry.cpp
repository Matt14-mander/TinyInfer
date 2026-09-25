#include "tinyinfer/model/onnx/operator_registry.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace tinyinfer::onnx {
namespace {

OperatorTranslator direct(OpType op) {
    return [op](const NodeProto& node) {
        return TranslatedOperator{op, node.attributes};
    };
}

TranslatedOperator translate_gelu(const NodeProto& node) {
    auto attributes = node.attributes;
    const auto found = attributes.find("approximate");
    if (found == attributes.end()) {
        attributes.emplace("approximate", std::string{"none"});
    } else {
        const auto* value = std::get_if<std::string>(&found->second);
        if (!value || (*value != "none" && *value != "tanh")) {
            throw std::invalid_argument(
                "ONNX Gelu approximate must be 'none' or 'tanh'");
        }
    }
    return {OpType::GELU, std::move(attributes)};
}

TranslatedOperator translate_layer_norm(const NodeProto& node) {
    if (node.inputs.size() < 2 || node.inputs.size() > 3) {
        throw std::invalid_argument(
            "ONNX LayerNormalization requires X and Scale, with optional Bias");
    }
    auto attributes = node.attributes;
    const auto axis = attributes.find("axis");
    if (axis != attributes.end()) {
        const auto* value = std::get_if<std::int64_t>(&axis->second);
        if (!value || *value != -1) {
            throw std::invalid_argument(
                "ONNX LayerNormalization currently supports axis=-1 only");
        }
        attributes.erase(axis);
    }
    const auto stash_type = attributes.find("stash_type");
    if (stash_type != attributes.end()) {
        const auto* value = std::get_if<std::int64_t>(&stash_type->second);
        if (!value || *value != 1) {
            throw std::invalid_argument(
                "ONNX LayerNormalization currently supports stash_type=1 only");
        }
        attributes.erase(stash_type);
    }
    return {OpType::LayerNorm, std::move(attributes)};
}

std::string canonical_domain(const std::string& domain) {
    return domain == "ai.onnx" ? "" : domain;
}

}  // namespace

OperatorRegistry::OperatorRegistry() {
    constexpr auto latest = std::numeric_limits<std::int64_t>::max();
    register_translator("", "Add", 13, latest, direct(OpType::Add));
    register_translator("", "Sub", 13, latest, direct(OpType::Subtract));
    register_translator("", "Mul", 13, latest, direct(OpType::Multiply));
    register_translator("", "MatMul", 13, latest, direct(OpType::MatMul));
    register_translator("", "Gemm", 13, latest, direct(OpType::Gemm));
    register_translator("", "Relu", 13, latest, direct(OpType::ReLU));
    register_translator("", "Softmax", 13, latest, direct(OpType::Softmax));
    register_translator("", "LayerNormalization", 17, latest,
                        translate_layer_norm);
    register_translator("", "Gelu", 20, latest, translate_gelu);
}

void OperatorRegistry::register_translator(
    std::string domain, std::string op_type, std::int64_t first_opset,
    std::int64_t last_opset, OperatorTranslator translator) {
    if (op_type.empty() || !translator || first_opset <= 0 ||
        last_opset < first_opset) {
        throw std::invalid_argument(
            "ONNX translator requires an operator, callable, and valid opset range");
    }
    auto& versions = translators_[{canonical_domain(domain), std::move(op_type)}];
    for (const auto& version : versions) {
        if (first_opset <= version.last_opset &&
            version.first_opset <= last_opset) {
            throw std::invalid_argument(
                "ONNX operator translator opset ranges overlap");
        }
    }
    versions.push_back({first_opset, last_opset, std::move(translator)});
    std::sort(versions.begin(), versions.end(),
              [](const VersionedTranslator& lhs,
                 const VersionedTranslator& rhs) {
                  return lhs.first_opset < rhs.first_opset;
              });
}

bool OperatorRegistry::supports(const std::string& domain,
                                const std::string& op_type,
                                std::int64_t opset_version) const {
    if (opset_version <= 0) return false;
    const auto found = translators_.find({canonical_domain(domain), op_type});
    if (found == translators_.end()) return false;
    return std::any_of(found->second.begin(), found->second.end(),
                       [opset_version](const VersionedTranslator& version) {
                           return version.first_opset <= opset_version &&
                                  opset_version <= version.last_opset;
                       });
}

TranslatedOperator OperatorRegistry::translate(
    const NodeProto& node, std::int64_t opset_version) const {
    const auto found = translators_.find(
        {canonical_domain(node.domain), node.op_type});
    if (found != translators_.end()) {
        for (const auto& version : found->second) {
            if (version.first_opset <= opset_version &&
                opset_version <= version.last_opset) {
                return version.translate(node);
            }
        }
    }
    const auto domain = canonical_domain(node.domain);
    throw std::invalid_argument(
        "unsupported ONNX operator or opset: domain='" + domain +
        "', op='" + node.op_type + "', opset=" +
        std::to_string(opset_version));
}

std::vector<OperatorRegistry::Registration>
OperatorRegistry::registrations() const {
    std::vector<Registration> result;
    for (const auto& entry : translators_) {
        for (const auto& version : entry.second) {
            result.push_back({entry.first.first, entry.first.second,
                              version.first_opset, version.last_opset});
        }
    }
    return result;
}

}  // namespace tinyinfer::onnx
