#pragma once

#include <functional>
#include <string>
#include <unordered_map>

#include "tinyinfer/model/onnx/model_proto.h"

namespace tinyinfer::onnx {

struct TranslatedOperator {
    OpType op;
    NodeAttributes attributes;
};

using OperatorTranslator =
    std::function<TranslatedOperator(const NodeProto&)>;

class OperatorRegistry {
public:
    OperatorRegistry();

    void register_translator(std::string op_type,
                             OperatorTranslator translator);
    bool supports(const std::string& op_type) const noexcept;
    TranslatedOperator translate(const NodeProto& node) const;

private:
    std::unordered_map<std::string, OperatorTranslator> translators_;
};

}  // namespace tinyinfer::onnx
