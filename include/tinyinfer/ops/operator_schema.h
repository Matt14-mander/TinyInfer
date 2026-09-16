#pragma once

#include <cstddef>
#include <string_view>
#include <vector>

#include "tinyinfer/graph/node.h"

namespace tinyinfer {

enum class AttributeType {
    Integer,
    Float,
    Boolean,
    String,
    Integers,
};

struct AttributeSchema {
    std::string_view name;
    AttributeType type;
    bool required{false};
};

struct OperatorSchema {
    OpType op;
    std::string_view name;
    std::size_t minimum_inputs;
    std::size_t maximum_inputs;
    std::size_t output_count;
    std::vector<AttributeSchema> attributes;
};

const OperatorSchema& operator_schema(OpType op);

std::vector<TensorSpec> infer_output_specs(
    OpType op, const std::vector<TensorSpec>& inputs,
    const NodeAttributes& attributes = {});

}  // namespace tinyinfer
