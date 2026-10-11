#pragma once
#include <optional>
#include <string>
#include <vector>
#include "tinyinfer/graph/node.h"
#include "tinyinfer/runtime/device/device.h"

namespace tinyinfer {
struct TensorDescriptor {
    TensorSpec spec;
    // Absent at preparation for runtime inputs; present when current layout is known.
    std::optional<Strides> strides;
};
struct NodeCapability {
    bool supported{false};
    // Backends that need runtime layout validation must say so explicitly.
    bool requires_runtime_layout_check{false};
    std::string reason;
};
struct CapabilityQuery {
    DeviceId device;
    Node node;
    std::vector<TensorDescriptor> inputs, outputs;
};
enum class FallbackPolicy { Strict, WholeModelCpu };
struct NodeCapabilityDiagnostic {
    NodeId node;
    std::string name;
    NodeCapability capability;
};
struct ModelPlacement {
    DeviceId requested, effective;
    bool used_fallback{false};
    std::string reason;
    std::vector<NodeCapabilityDiagnostic> nodes;
};
class Model;
class Backend;
// Preparation metadata only, not a GPU execution plan or a per-node partitioner.
// Any runtime layout condition must also be enforced by the future device plan.
ModelPlacement select_model_backend(const Model&, const Backend&, FallbackPolicy);
} // namespace tinyinfer
