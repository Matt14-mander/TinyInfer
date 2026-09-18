#pragma once

#include <filesystem>
#include <memory>

#include "tinyinfer/model/model_loader.h"
#include "tinyinfer/model/onnx/model_proto.h"
#include "tinyinfer/model/onnx/operator_registry.h"

namespace tinyinfer::onnx {

class ModelParser {
public:
    virtual ~ModelParser() = default;
    virtual ModelProto parse(const std::filesystem::path& path) const = 0;
};

class OnnxImporter final : public ModelLoader {
public:
    explicit OnnxImporter(
        std::shared_ptr<const ModelParser> parser = nullptr);

    Model load(const std::filesystem::path& path) const override;
    Model import_model(const ModelProto& model) const;

    const OperatorRegistry& operators() const noexcept { return operators_; }

private:
    std::shared_ptr<const ModelParser> parser_;
    OperatorRegistry operators_;
};

}  // namespace tinyinfer::onnx
