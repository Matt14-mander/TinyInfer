#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>

#include "tinyinfer/core/dtype.h"
#include "tinyinfer/core/tensor_layout.h"

namespace tinyinfer {

using ValueId = std::size_t;
using NodeId = std::size_t;

inline constexpr ValueId kInvalidValueId = std::numeric_limits<ValueId>::max();
inline constexpr NodeId kInvalidNodeId = std::numeric_limits<NodeId>::max();

struct TensorSpec {
    Shape shape;
    DataType dtype{DataType::Float32};
};

inline bool operator==(const TensorSpec& lhs, const TensorSpec& rhs) {
    return lhs.shape == rhs.shape && lhs.dtype == rhs.dtype;
}

inline bool operator!=(const TensorSpec& lhs, const TensorSpec& rhs) {
    return !(lhs == rhs);
}

enum class ValueKind : std::uint8_t { Input, Constant, Intermediate };

struct GraphValue {
    ValueId id{kInvalidValueId};
    std::string name;
    TensorSpec spec;
    ValueKind kind{ValueKind::Intermediate};
    std::optional<NodeId> producer;
};

}  // namespace tinyinfer
