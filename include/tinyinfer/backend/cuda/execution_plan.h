#pragma once
#include <memory>
#include "tinyinfer/backend/cuda/cuda_runtime.h"
#include "tinyinfer/backend/capability.h"
#include "tinyinfer/runtime/device/device_runtime.h"
#include "tinyinfer/model/model.h"

namespace tinyinfer {
namespace detail { struct CudaPlanState; struct CudaContextState; }
struct CudaExecutionPlanOptions {
    std::uint32_t ordinal{0};
    FallbackPolicy fallback{FallbackPolicy::Strict};
};
struct CudaPlanStatistics {
    // Preparation snapshot. Activation bytes/slots describe each context's arena;
    // device_allocations excludes later context creation and cuBLAS internals.
    std::size_t constant_uploads{}, constant_bytes{}, activation_bytes{}, activation_slots{};
    std::size_t device_allocations{};
};
struct CudaContextStatistics {
    std::size_t runs{}, gpu_runs{}, cpu_runs{}, input_uploads{}, input_bytes{};
    std::size_t output_downloads{}, output_bytes{}, reuse_waits{};
};
class CudaExecutionContext;
// Immutable model snapshot and shared device constants. Contexts sharing a plan
// must be externally serialized; each context owns its input and activation buffers.
class CudaExecutionPlan {
public:
    explicit CudaExecutionPlan(const Model&, CudaExecutionPlanOptions = {});
    CudaExecutionContext create_context() const;
    ValueId input_id(const std::string&) const;
    ValueId output_id(const std::string&) const;
    const std::vector<Model::NamedValue>& inputs() const;
    const std::vector<Model::NamedValue>& outputs() const;
    TensorSpec input_spec(ValueId) const;
    const ModelPlacement& placement() const;
    CudaPlanStatistics statistics() const;
    DeviceRuntimeStatistics device_statistics() const;
    CudaTransferStatistics transfer_statistics() const;
    CudaKernelStatistics kernel_statistics() const;
private:
    std::shared_ptr<const detail::CudaPlanState> state_;
};
class CudaExecutionContext {
public:
    ~CudaExecutionContext();
    CudaExecutionContext(CudaExecutionContext&&) noexcept;
    CudaExecutionContext& operator=(CudaExecutionContext&&) noexcept;
    CudaExecutionContext(const CudaExecutionContext&) = delete;
    CudaExecutionContext& operator=(const CudaExecutionContext&) = delete;
    // Synchronous snapshot/upload; an lvalue Tensor is copied at this boundary.
    // Strict rejects a moved noncontiguous view; WholeModelCpu preserves it.
    void bind_input(ValueId, Tensor);
    void bind_input(const std::string&, Tensor);
    void run(); // Synchronous; warm GPU runs do not allocate device buffers or transfer data.
    // Independent host copy. Any successful binding invalidates prior results.
    Tensor output(ValueId) const;
    Tensor output(const std::string&) const;
    ModelPlacement placement() const;
    CudaContextStatistics statistics() const;
private:
    friend class CudaExecutionPlan;
    explicit CudaExecutionContext(std::shared_ptr<const detail::CudaPlanState>);
    detail::CudaContextState& live() const;
    std::unique_ptr<detail::CudaContextState> state_;
};
} // namespace tinyinfer
