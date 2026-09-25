#pragma once
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "tinyinfer/graph/value.h"

namespace tinyinfer {
enum class OpType {
    Add,
    Multiply,
    Subtract,
    MatMul,
    Gemm,
    ReLU,
    GELU,
    Softmax,
    LayerNorm,
    Count  // Sentinel: keep last so coverage tests enumerate every real OpType.
};
using AttributeValue = std::variant<std::int64_t, float, bool, std::string,
                                    std::vector<std::int64_t>>;
using NodeAttributes = std::map<std::string, AttributeValue>;

struct Node {
    NodeId id{kInvalidNodeId};
    std::string name;
    OpType op{OpType::Add};
    std::vector<ValueId> inputs;
    std::vector<ValueId> outputs;
    NodeAttributes attributes;

    bool has_attribute(const std::string& key) const {
        return attributes.find(key) != attributes.end();
    }

    template <typename T>
    const T& attribute(const std::string& key) const {
        const auto iterator = attributes.find(key);
        if (iterator == attributes.end()) {
            throw std::out_of_range("node attribute does not exist");
        }
        return std::get<T>(iterator->second);
    }
};
}  // namespace tinyinfer
