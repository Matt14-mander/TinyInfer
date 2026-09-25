#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>

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
    struct Registration {
        std::string domain;
        std::string op_type;
        std::int64_t first_opset;
        std::int64_t last_opset;
    };

    OperatorRegistry();

    void register_translator(std::string domain, std::string op_type,
                             std::int64_t first_opset,
                             std::int64_t last_opset,
                             OperatorTranslator translator);
    bool supports(const std::string& domain, const std::string& op_type,
                  std::int64_t opset_version) const;
    TranslatedOperator translate(const NodeProto& node,
                                 std::int64_t opset_version) const;
    std::vector<Registration> registrations() const;

private:
    struct VersionedTranslator {
        std::int64_t first_opset;
        std::int64_t last_opset;
        OperatorTranslator translate;
    };
    using OperatorKey = std::pair<std::string, std::string>;
    std::map<OperatorKey, std::vector<VersionedTranslator>> translators_;
};

}  // namespace tinyinfer::onnx
