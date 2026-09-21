#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "tinyinfer/core/tensor.h"
#include "tinyinfer/graph/node.h"

namespace tinyinfer::onnx {

struct ValueInfo {
    std::string name;
    TensorSpec spec;
};

struct Initializer {
    std::string name;
    Tensor value;
};

struct NodeProto {
    std::string name;
    std::string op_type;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    NodeAttributes attributes;
    std::string domain;
};

struct GraphProto {
    std::string name;
    std::vector<ValueInfo> inputs;
    std::vector<Initializer> initializers;
    std::vector<NodeProto> nodes;
    std::vector<ValueInfo> outputs;
};

struct ModelProto {
    std::int64_t opset_version{0};
    GraphProto graph;
};

}  // namespace tinyinfer::onnx
