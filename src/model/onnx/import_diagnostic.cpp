#include "tinyinfer/model/onnx/import_diagnostic.h"

#include <sstream>
#include <utility>

namespace tinyinfer::onnx {

const char* to_string(OnnxImportStage stage) noexcept {
    switch (stage) {
        case OnnxImportStage::FileRead: return "FileRead";
        case OnnxImportStage::ProtobufParse: return "ProtobufParse";
        case OnnxImportStage::TensorDecode: return "TensorDecode";
        case OnnxImportStage::ModelValidation: return "ModelValidation";
        case OnnxImportStage::GraphImport: return "GraphImport";
        case OnnxImportStage::OperatorTranslation:
            return "OperatorTranslation";
        case OnnxImportStage::ShapeInference: return "ShapeInference";
        case OnnxImportStage::GraphValidation: return "GraphValidation";
    }
    return "Unknown";
}

OnnxImportError::OnnxImportError(OnnxImportDiagnostic diagnostic)
    : std::runtime_error(format(diagnostic)),
      diagnostic_(std::move(diagnostic)) {}

std::string OnnxImportError::format(
    const OnnxImportDiagnostic& diagnostic) {
    std::ostringstream message;
    message << "ONNX import failed [stage=" << to_string(diagnostic.stage);
    if (!diagnostic.model_path.empty()) {
        message << ", path=" << diagnostic.model_path.string();
    }
    if (!diagnostic.graph_name.empty()) {
        message << ", graph=" << diagnostic.graph_name;
    }
    if (diagnostic.node_index) {
        message << ", node_index=" << *diagnostic.node_index;
    }
    if (!diagnostic.node_name.empty()) {
        message << ", node=" << diagnostic.node_name;
    }
    if (!diagnostic.op_type.empty()) {
        message << ", op=";
        if (!diagnostic.domain.empty()) message << diagnostic.domain << "::";
        message << diagnostic.op_type;
    } else if (!diagnostic.domain.empty()) {
        message << ", domain=" << diagnostic.domain;
    }
    if (!diagnostic.value_name.empty()) {
        message << ", value=" << diagnostic.value_name;
    }
    message << "]: " << diagnostic.message;
    return message.str();
}

}  // namespace tinyinfer::onnx
