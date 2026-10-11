#include "tinyinfer/backend/backend.h"
#include "tinyinfer/backend/cpu_backend.h"
#include "tinyinfer/model/model.h"
#include "tinyinfer/ops/operator_schema.h"
#include <stdexcept>
#include <utility>

namespace tinyinfer {
DeviceInfo Backend::device_info() const { return {{}, std::string(name()), true, {}}; }
NodeCapability Backend::query_capability(const CapabilityQuery&) const {
    return {false, false, "backend has no node-level capability implementation"};
}
DeviceInfo CpuBackend::device_info() const { return {{}, "CPU", true, {}}; }
NodeCapability CpuBackend::query_capability(const CapabilityQuery& query) const {
    if (query.device != DeviceId{}) return {false, false, "CPU backend requires cpu:0"};
    if (!supports(query.node.op)) return {false, false, "operator has no CPU kernel"};
    if (query.node.inputs.size() != query.inputs.size() || query.node.outputs.size() != query.outputs.size())
        return {false, false, "node descriptor arity differs from node operands"};
    try {
        std::vector<TensorSpec> inputs;
        for (const auto& descriptor : query.inputs) {
            if (descriptor.strides) {
                const TensorLayout layout(descriptor.spec.shape, *descriptor.strides);
                layout.size_bytes(descriptor.spec.dtype);
            }
            inputs.push_back(descriptor.spec);
        }
        const auto outputs = infer_output_specs(query.node.op, inputs, query.node.attributes);
        if (outputs.size() != query.outputs.size()) return {false, false, "output count differs from schema"};
        for (std::size_t i = 0; i < outputs.size(); ++i) {
            if (outputs[i] != query.outputs[i].spec) return {false, false, "output spec differs from inference"};
            if (query.outputs[i].strides && !TensorLayout(outputs[i].shape, *query.outputs[i].strides).is_contiguous())
                return {false, false, "planned CPU output must be contiguous"};
        }
    } catch (const std::invalid_argument& error) {
        return {false, false, error.what()};
    } catch (const std::overflow_error& error) {
        return {false, false, error.what()};
    }
    return {true, false, "CPU schema and layout supported"};
}
namespace {
CapabilityQuery make_query(const Graph& graph, const Node& node, DeviceId device) {
    CapabilityQuery query{device, node, {}, {}};
    for (auto id : node.inputs) {
        TensorDescriptor descriptor{graph.value(id).spec, std::nullopt};
        if (graph.is_constant(id)) descriptor.strides = graph.constant(id).strides();
        query.inputs.push_back(std::move(descriptor));
    }
    for (auto id : node.outputs) {
        auto spec = graph.value(id).spec;
        query.outputs.push_back({spec, TensorLayout(spec.shape).strides()});
    }
    return query;
}
} // namespace
ModelPlacement select_model_backend(const Model& model, const Backend& backend, FallbackPolicy policy) {
    if (policy != FallbackPolicy::Strict && policy != FallbackPolicy::WholeModelCpu)
        throw std::invalid_argument("unknown fallback policy");
    const auto& graph = model.graph();
    graph.validate();
    const auto info = backend.device_info();
    validate_device_id(info.id);
    ModelPlacement placement{info.id, info.id, false, {}, {}};
    if (!info.available) placement.reason = "device unavailable: " + info.unavailable_reason;
    else for (auto id : graph.topological_order()) {
        const auto& node = graph.node(id);
        const auto capability = backend.query_capability(make_query(graph, node, info.id));
        placement.nodes.push_back({id, node.name, capability});
        if (!capability.supported && placement.reason.empty())
            placement.reason = "unsupported node '" + node.name + "': " + capability.reason;
    }
    if (placement.reason.empty()) return placement;
    if (policy == FallbackPolicy::Strict || info.id.kind == DeviceKind::Cpu)
        throw std::runtime_error(placement.reason);
    // Verify the entire CPU model before selecting it. Never mix individual nodes.
    CpuBackend cpu;
    for (const auto& node : graph.nodes()) {
        const auto capability = cpu.query_capability(make_query(graph, node, {}));
        if (!capability.supported) throw std::runtime_error("CPU fallback unavailable: " + capability.reason);
    }
    placement.effective = {};
    placement.used_fallback = true;
    return placement;
}
} // namespace tinyinfer
