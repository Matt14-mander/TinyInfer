#pragma once
#include <memory>
#include "tinyinfer/backend/capability.h"
#include "tinyinfer/runtime/device/device_runtime.h"
#include "tinyinfer/model/model.h"

namespace tinyinfer {
namespace detail { struct MetalPlanState; struct MetalContextState; }
struct MetalExecutionPlanOptions {
    std::uint32_t ordinal{0};
    FallbackPolicy fallback{FallbackPolicy::Strict};
};
struct MetalPlanStatistics {
    std::size_t constant_uploads{}, constant_bytes{}, activation_bytes{}, activation_slots{};
    std::size_t device_allocations{}, pipeline_creations{};
};
struct MetalContextStatistics {
    std::size_t runs{}, gpu_runs{}, cpu_runs{}, input_uploads{}, input_bytes{};
    std::size_t output_downloads{}, output_bytes{}, reuse_waits{};
};
class MetalExecutionContext;
// Immutable model snapshot and shared device constants. Contexts sharing a plan
// must be externally serialized; each context owns its input and activation buffers.
class MetalExecutionPlan {
public:
    explicit MetalExecutionPlan(const Model&, MetalExecutionPlanOptions = {});
    MetalExecutionContext create_context() const;
    ValueId input_id(const std::string&) const;
    ValueId output_id(const std::string&) const;
    const std::vector<Model::NamedValue>& inputs() const;
    const std::vector<Model::NamedValue>& outputs() const;
    TensorSpec input_spec(ValueId) const;
    const ModelPlacement& placement() const;
    MetalPlanStatistics statistics() const;
    DeviceRuntimeStatistics device_statistics() const;
private:
    std::shared_ptr<const detail::MetalPlanState> state_;
};
class MetalExecutionContext {
public:
    ~MetalExecutionContext();
    MetalExecutionContext(MetalExecutionContext&&) noexcept;
    MetalExecutionContext& operator=(MetalExecutionContext&&) noexcept;
    MetalExecutionContext(const MetalExecutionContext&) = delete;
    MetalExecutionContext& operator=(const MetalExecutionContext&) = delete;
    void bind_input(ValueId, Tensor);
    void bind_input(const std::string&, Tensor);
    void run();
    Tensor output(ValueId) const;
    Tensor output(const std::string&) const;
    ModelPlacement placement() const;
    MetalContextStatistics statistics() const;
private:
    friend class MetalExecutionPlan;
    explicit MetalExecutionContext(std::shared_ptr<const detail::MetalPlanState>);
    detail::MetalContextState& live() const;
    std::unique_ptr<detail::MetalContextState> state_;
};
} // namespace tinyinfer
