#include "tinyinfer/backend/cuda/execution_plan.h"
#include "tinyinfer/backend/cuda/cuda_runtime.h"
#include "tinyinfer/backend/cpu/execution_plan.h"
#include "tinyinfer/runtime/memory_plan.h"
#include <algorithm>
#include <optional>
#include <stdexcept>

namespace tinyinfer {
namespace {
class UnavailableCuda final : public Backend {
public:
    explicit UnavailableCuda(std::uint32_t ordinal, std::string reason = "requested Cuda device is unavailable") : ordinal_(ordinal), reason_(std::move(reason)) {}
    std::string_view name() const noexcept override { return "cuda"; }
    bool supports(OpType) const noexcept override { return false; }
    DeviceInfo device_info() const override {
        return {{DeviceKind::Cuda, ordinal_}, "Cuda", false, reason_};
    }
    void execute(const Node&, ExecutionContext&) override { throw std::logic_error("unavailable Cuda"); }
private: std::uint32_t ordinal_; std::string reason_;
};

}
namespace detail {
struct CudaPlanState {
    Model model;
    MemoryPlan memory;
    CudaExecutionPlanOptions options;
    ModelPlacement placement;
    CudaPlanStatistics statistics;
    std::vector<NodeId> order;
    std::shared_ptr<CudaDeviceRuntime> runtime;
    std::unique_ptr<CpuExecutionPlan> cpu;
    std::vector<std::optional<CudaTensorView>> constants;
    explicit CudaPlanState(const Model& source, CudaExecutionPlanOptions opts)
        : model(source), memory(model.graph()), options(opts), order(model.graph().topological_order()),
          constants(model.graph().value_count()) {
        if (opts.fallback != FallbackPolicy::Strict && opts.fallback != FallbackPolicy::WholeModelCpu)
            throw std::invalid_argument("invalid Cuda fallback policy");
        // Only discovery/runtime construction is eligible for unavailable-device
        // fallback. Allocation, upload and execution failures must propagate.
        std::string unavailable;
        try {
            const auto devices = CudaDeviceRuntime::enumerate_devices();
            if (opts.ordinal >= devices.size()) unavailable = "requested CUDA device is unavailable";
            else runtime = std::make_shared<CudaDeviceRuntime>(opts.ordinal);
        } catch (const std::runtime_error& error) {
            if (opts.fallback == FallbackPolicy::Strict) throw;
            unavailable = error.what();
        }
        if (!runtime) {
            UnavailableCuda backend(opts.ordinal, unavailable);
            placement = select_model_backend(model, backend, opts.fallback);
        } else {
            CudaBackend backend(runtime);
            placement = select_model_backend(model, backend, opts.fallback);
        }
        if (placement.effective.kind == DeviceKind::Cuda) {
            for (const auto& v : model.graph().values()) {
                // Also validate graph endpoints and disconnected constants:
                // node-level placement alone cannot check zero-node models.
                Node check; check.op = OpType::ReLU; check.inputs = {v.id}; check.outputs = {v.id};
                const auto strides = TensorLayout(v.spec.shape).strides();
                const auto capability = runtime->query_capability({runtime->device(), check,
                    {{v.spec, strides}}, {{v.spec, strides}}});
                if (!capability.supported) {
                    UnavailableCuda unsupported(opts.ordinal, "value " + v.name + ": " + capability.reason);
                    placement = select_model_backend(model, unsupported, opts.fallback);
                    break;
                }
            }
        }
        if (opts.fallback == FallbackPolicy::WholeModelCpu)
            cpu = std::make_unique<CpuExecutionPlan>(model);
        if (placement.effective.kind == DeviceKind::Cpu) { runtime.reset(); return; }
        std::vector<bool> needed(model.graph().value_count(), false);
        for (const auto& node : model.graph().nodes()) for (auto id : node.inputs) needed[id] = true;
        for (auto id : model.graph().outputs()) needed[id] = true;
        for (const auto& v : model.graph().values()) if (v.kind == ValueKind::Constant && needed[v.id]) {
            const auto& t = model.graph().constant(v.id);
            auto buffer = runtime->allocate(t.size_bytes());
            constants[v.id].emplace(v.spec, DeviceSpan{buffer, 0, t.size_bytes()});
            runtime->upload(constants[v.id]->storage, t.storage().buffer(), t.storage().byte_offset()).wait();
            ++statistics.constant_uploads; statistics.constant_bytes += t.size_bytes();
        }
        runtime->synchronize();
        statistics.activation_bytes = memory.buffer_size_bytes();
        statistics.activation_slots = memory.slot_count();
        statistics.device_allocations = runtime->statistics().allocations;
    }
};
struct CudaContextState {
    std::shared_ptr<const CudaPlanState> plan;
    std::vector<bool> bound, contiguous;
    std::vector<std::optional<CudaTensorView>> views;
    std::optional<CpuExecutionContext> cpu;
    CudaContextStatistics statistics;
    ModelPlacement placement;
    bool has_output{false};
    explicit CudaContextState(std::shared_ptr<const CudaPlanState> p)
        : plan(std::move(p)), bound(plan->model.graph().value_count(), false), contiguous(plan->model.graph().value_count(), false), views(plan->constants), placement(plan->placement) {
        if (plan->cpu) cpu.emplace(plan->cpu->create_context());
        if (!plan->runtime) return;
        auto arena = plan->runtime->allocate(plan->memory.buffer_size_bytes());
        for (const auto& v : plan->model.graph().values()) {
            if (v.kind == ValueKind::Input) {
                auto bytes = TensorLayout(v.spec.shape).size_bytes(v.spec.dtype);
                views[v.id].emplace(v.spec, DeviceSpan{plan->runtime->allocate(bytes), 0, bytes});
            } else if (plan->memory.has_allocation(v.id)) {
                const auto& a = plan->memory.allocation(v.id);
                views[v.id].emplace(v.spec, DeviceSpan{arena, a.byte_offset, a.size_bytes});
            }
        }
    }
};
}
CudaExecutionPlan::CudaExecutionPlan(const Model& m, CudaExecutionPlanOptions o)
    : state_(std::make_shared<detail::CudaPlanState>(m, o)) {}
CudaExecutionContext CudaExecutionPlan::create_context() const { return CudaExecutionContext(state_); }
ValueId CudaExecutionPlan::input_id(const std::string& n) const { return state_->model.input_id(n); }
ValueId CudaExecutionPlan::output_id(const std::string& n) const { return state_->model.output_id(n); }
const std::vector<Model::NamedValue>& CudaExecutionPlan::inputs() const { return state_->model.inputs(); }
const std::vector<Model::NamedValue>& CudaExecutionPlan::outputs() const { return state_->model.outputs(); }
TensorSpec CudaExecutionPlan::input_spec(ValueId id) const {
    const auto& v = state_->model.graph().value(id);
    if (v.kind != ValueKind::Input) throw std::invalid_argument("not a model input");
    return v.spec;
}
const ModelPlacement& CudaExecutionPlan::placement() const { return state_->placement; }
CudaPlanStatistics CudaExecutionPlan::statistics() const { return state_->statistics; }
DeviceRuntimeStatistics CudaExecutionPlan::device_statistics() const {
    return state_->runtime ? state_->runtime->statistics() : DeviceRuntimeStatistics{};
}
CudaTransferStatistics CudaExecutionPlan::transfer_statistics() const {
    return state_->runtime ? state_->runtime->transfer_statistics() : CudaTransferStatistics{};
}
CudaKernelStatistics CudaExecutionPlan::kernel_statistics() const {
    return state_->runtime ? state_->runtime->kernel_statistics() : CudaKernelStatistics{};
}
CudaExecutionContext::CudaExecutionContext(std::shared_ptr<const detail::CudaPlanState> p)
    : state_(std::make_unique<detail::CudaContextState>(std::move(p))) {}
CudaExecutionContext::~CudaExecutionContext() = default;
CudaExecutionContext::CudaExecutionContext(CudaExecutionContext&&) noexcept = default;
CudaExecutionContext& CudaExecutionContext::operator=(CudaExecutionContext&&) noexcept = default;
detail::CudaContextState& CudaExecutionContext::live() const {
    if (!state_) throw std::logic_error("moved-from Cuda context");
    return *state_;
}
void CudaExecutionContext::bind_input(const std::string& n, Tensor t) { bind_input(live().plan->model.input_id(n), std::move(t)); }
void CudaExecutionContext::bind_input(ValueId id, Tensor t) {
    auto& s = live(); const auto& v = s.plan->model.graph().value(id);
    if (v.kind != ValueKind::Input || TensorSpec{t.shape(), t.dtype()} != v.spec)
        throw std::invalid_argument("Cuda input ID, shape or dtype mismatch");
    if (s.plan->runtime && !t.is_contiguous() && s.plan->options.fallback == FallbackPolicy::Strict)
        throw std::invalid_argument("Cuda input must be contiguous; whole-model CPU fallback is disabled");
    s.has_output = false; s.bound[id] = false;
    if (s.plan->runtime && t.is_contiguous()) {
        s.plan->runtime->upload(s.views[id]->storage, t.storage().buffer(), t.storage().byte_offset()).wait();
        ++s.statistics.input_uploads; s.statistics.input_bytes += t.size_bytes();
    }
    const bool contiguous = t.is_contiguous();
    // The CPU context takes ownership of this already-owned parameter. Copying
    // it again doubles the public input snapshot cost on whole-model fallback.
    if (s.cpu) s.cpu->bind_input(id, std::move(t));
    s.contiguous[id] = contiguous;
    s.bound[id] = true;
}
void CudaExecutionContext::run() {
    auto& s = live(); s.has_output = false;
    s.placement = s.plan->placement;
    for (auto id : s.plan->model.graph().inputs()) {
        if (!s.bound[id]) throw std::logic_error("model input is not bound");
        if (s.plan->runtime && !s.contiguous[id]) {
            s.placement.effective = {DeviceKind::Cpu, 0}; s.placement.used_fallback = true;
            s.placement.reason = "runtime input " + s.plan->model.graph().value(id).name + " is noncontiguous; whole model uses Prepared CPU Auto";
        }
    }
    if (s.placement.effective.kind == DeviceKind::Cpu) {
        s.cpu->run(); ++s.statistics.cpu_runs;
    } else {
        std::vector<Completion> pending; pending.reserve(s.plan->order.size());
        std::vector<ValueId> occupants(s.plan->memory.slot_count(), kInvalidValueId);
        std::size_t waited = 0;
        for (auto node_id : s.plan->order) {
            const auto& node = s.plan->model.graph().node(node_id);
            const auto out = node.outputs.at(0);
            const auto& a = s.plan->memory.allocation(out);
            const auto previous = occupants[a.slot];
            if (previous != kInvalidValueId) {
                const auto end = s.plan->memory.lifetime(previous).last_step + 1;
                if (waited < end) {
                    while (waited < end) pending.at(waited++).wait();
                    ++s.statistics.reuse_waits;
                }
            }
            std::vector<CudaTensorView> operands;
            for (auto id : node.inputs) operands.push_back(s.views[id].value());
            pending.push_back(s.plan->runtime->execute(node.op, operands, s.views[out].value(), node.attributes));
            occupants[a.slot] = out;
        }
        while (waited < pending.size()) pending[waited++].wait();
        s.plan->runtime->synchronize(); ++s.statistics.gpu_runs;
    }
    ++s.statistics.runs; s.has_output = true;
}
Tensor CudaExecutionContext::output(const std::string& n) const { return output(live().plan->model.output_id(n)); }
Tensor CudaExecutionContext::output(ValueId id) const {
    auto& s = live();
    const auto& outputs = s.plan->model.graph().outputs();
    if (std::find(outputs.begin(), outputs.end(), id) == outputs.end()) throw std::invalid_argument("not a model output");
    if (!s.has_output) throw std::logic_error("no successful run since last input binding");
    if (s.placement.effective.kind == DeviceKind::Cpu) return s.cpu->output(id);
    const auto& view = s.views.at(id).value();
    Tensor t(view.spec.shape, view.spec.dtype);
    s.plan->runtime->download(t.storage().buffer(), view.storage, t.storage().byte_offset()).wait();
    ++s.statistics.output_downloads; s.statistics.output_bytes += t.size_bytes();
    return t;
}
ModelPlacement CudaExecutionContext::placement() const { return live().placement; }
CudaContextStatistics CudaExecutionContext::statistics() const { return live().statistics; }
} // namespace tinyinfer
