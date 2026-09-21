#pragma once

#include <cstddef>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>

namespace tinyinfer::onnx {

enum class OnnxImportStage {
    FileRead,
    ProtobufParse,
    TensorDecode,
    ModelValidation,
    GraphImport,
    OperatorTranslation,
    ShapeInference,
    GraphValidation,
};

const char* to_string(OnnxImportStage stage) noexcept;

struct OnnxImportDiagnostic {
    OnnxImportStage stage{OnnxImportStage::GraphImport};
    std::filesystem::path model_path;
    std::string graph_name;
    std::optional<std::size_t> node_index;
    std::string node_name;
    std::string op_type;
    std::string domain;
    std::string value_name;
    std::string message;
};

class OnnxImportError final : public std::runtime_error {
public:
    explicit OnnxImportError(OnnxImportDiagnostic diagnostic);

    const OnnxImportDiagnostic& diagnostic() const noexcept {
        return diagnostic_;
    }

private:
    static std::string format(const OnnxImportDiagnostic& diagnostic);

    OnnxImportDiagnostic diagnostic_;
};

}  // namespace tinyinfer::onnx
