#pragma once

#include <filesystem>
#include <string_view>

#include "tinyinfer/model/onnx/onnx_importer.h"

namespace tinyinfer::onnx {

class ProtobufModelParser final : public ModelParser {
public:
    ModelProto parse(const std::filesystem::path& path) const override;
    Tensor decode_tensor_proto(std::string_view serialized) const;
};

}  // namespace tinyinfer::onnx
