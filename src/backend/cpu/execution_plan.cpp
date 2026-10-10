#include "tinyinfer/backend/cpu/execution_plan.h"

#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>
#include <utility>

#include "tinyinfer/backend/cpu/matmul_kernel.h"
#include "tinyinfer/backend/cpu/operator_registry.h"
#include "tinyinfer/runtime/execution_context.h"
#include "ops/gemm_internal.h"

namespace tinyinfer {
namespace {

bool matrix_op(OpType op) {
    return op == OpType::MatMul || op == OpType::Gemm ||
           op == OpType::FusedGemmActivation;
}

template <typename T>
T attribute(const Node& node, const char* name, T default_value) {
    const auto found = node.attributes.find(name);
    return found == node.attributes.end() ? default_value : std::get<T>(found->second);
}

}  // namespace

namespace detail {

struct CpuExecutionPlanState {
    explicit CpuExecutionPlanState(const Model& source, CpuExecutionPlanOptions options)
        : model(source), memory_plan(model.graph()) {
        if (options.gemm_epilogue != CpuGemmEpilogueMode::Legacy &&
            options.gemm_epilogue != CpuGemmEpilogueMode::Specialized &&
            options.gemm_epilogue != CpuGemmEpilogueMode::Fused)
            throw std::invalid_argument("invalid prepared Gemm epilogue mode");
        const auto& graph = model.graph();
        std::map<std::pair<ValueId, bool>, std::size_t> packing;
        for (const auto id : graph.topological_order()) {
            const auto& node = graph.node(id);
            if (!registry.supports(node.op)) {
                throw std::invalid_argument("operator is not supported by prepared CPU execution");
            }
            CpuExecutionStep step;
            step.node = id;
            step.op = node.op;
            step.fallback_reason = "ordinary CPU operator";
            if (matrix_op(node.op)) {
                const auto rhs = node.inputs.at(1);
                const bool gemm = node.op != OpType::MatMul;
                const bool trans_b = gemm && attribute(node, "transB", std::int64_t{0}) != 0;
                step.transpose_a = gemm && attribute(node, "transA", std::int64_t{0}) != 0;
                step.alpha = gemm ? attribute(node, "alpha", 1.0F) : 1.0F;
                step.beta = gemm ? attribute(node, "beta", 1.0F) : 1.0F;
                step.relu = node.op == OpType::FusedGemmActivation;
                if (gemm) {
                    const Shape* bias_shape = node.inputs.size() == 3
                        ? &graph.value(node.inputs[2]).spec.shape : nullptr;
                    step.epilogue = cpu::GemmEpilogue(graph.value(node.outputs[0]).spec.shape,
                        bias_shape, step.alpha, step.beta, step.relu);
                    step.epilogue_mode = options.gemm_epilogue;
                }
                step.fallback_reason = "runtime RHS; use existing CPU kernel";
                if (graph.is_constant(rhs)) {
                    const auto key = std::make_pair(rhs, trans_b);
                    const auto found = packing.find(key);
                    if (found == packing.end()) {
                        const auto index = weights.size();
                        weights.emplace_back(graph.constant(rhs), trans_b);
                        const auto bytes = weights.back().size_bytes();
                        if (bytes > std::numeric_limits<std::size_t>::max() - packed_bytes) {
                            throw std::overflow_error("packed weight bytes overflow");
                        }
                        packed_bytes += bytes;
                        packing.emplace(key, index);
                        step.packed_weight_index = index;
                    } else {
                        step.packed_weight_index = found->second;
                    }
                    step.path = CpuExecutionPath::PackedConstantRhs;
                    step.fallback_reason.clear();
                }
            }
            steps.push_back(std::move(step));
        }
    }

    const Model model;
    const MemoryPlan memory_plan;
    cpu::OperatorRegistry registry;
    std::vector<cpu::PackedMatMulRhs> weights;
    std::vector<CpuExecutionStep> steps;
    std::size_t packed_bytes{0};
};

}  // namespace detail

CpuExecutionPlan::CpuExecutionPlan(const Model& model, CpuExecutionPlanOptions options)
    : state_(std::make_shared<detail::CpuExecutionPlanState>(model, options)) {}

CpuExecutionContext CpuExecutionPlan::create_context() const {
    return CpuExecutionContext(state_);
}

ValueId CpuExecutionPlan::input_id(const std::string& name) const {
    return state_->model.input_id(name);
}
ValueId CpuExecutionPlan::output_id(const std::string& name) const {
    return state_->model.output_id(name);
}
const std::vector<Model::NamedValue>& CpuExecutionPlan::inputs() const {
    return state_->model.inputs();
}
const std::vector<Model::NamedValue>& CpuExecutionPlan::outputs() const {
    return state_->model.outputs();
}
TensorSpec CpuExecutionPlan::input_spec(ValueId id) const {
    const auto& value = state_->model.graph().value(id);
    if (value.kind != ValueKind::Input) {
        throw std::invalid_argument("value is not a prepared model input");
    }
    return value.spec;
}
const std::vector<CpuExecutionStep>& CpuExecutionPlan::steps() const { return state_->steps; }
std::size_t CpuExecutionPlan::pack_count() const { return state_->weights.size(); }
std::size_t CpuExecutionPlan::packed_weight_bytes() const { return state_->packed_bytes; }
std::size_t CpuExecutionPlan::activation_bytes() const {
    return state_->memory_plan.buffer_size_bytes();
}

CpuExecutionContext::CpuExecutionContext(
    std::shared_ptr<const detail::CpuExecutionPlanState> state)
    : state_(std::move(state)),
      context_(std::make_unique<ExecutionContext>(state_->model.graph(), state_->memory_plan)) {}

CpuExecutionContext::~CpuExecutionContext() = default;
CpuExecutionContext::CpuExecutionContext(CpuExecutionContext&&) noexcept = default;
CpuExecutionContext& CpuExecutionContext::operator=(CpuExecutionContext&& other) noexcept {
    if (this != &other) {
        context_.reset();
        state_ = std::move(other.state_);
        context_ = std::move(other.context_);
        runs_ = other.runs_;
        runtime_packs_ = other.runtime_packs_;
        fused_gemms_ = other.fused_gemms_;
        specialized_gemms_ = other.specialized_gemms_;
    }
    return *this;
}

void CpuExecutionContext::require_live() const {
    if (!context_) throw std::logic_error("prepared context was moved from");
}

void CpuExecutionContext::bind_input(ValueId id, Tensor value) {
    require_live();
    context_->bind_input(id, std::move(value));
}
void CpuExecutionContext::bind_input(const std::string& name, Tensor value) {
    require_live();
    bind_input(state_->model.input_id(name), std::move(value));
}

void CpuExecutionContext::run() {
    require_live();
    context_->require_all_inputs_bound();
    context_->clear_intermediates();
    const auto& graph = state_->model.graph();
    const auto& input_context = std::as_const(*context_);
    for (std::size_t i = 0; i < state_->steps.size(); ++i) {
        const auto& step = state_->steps[i];
        const auto& node = graph.node(step.node);
        if (step.path == CpuExecutionPath::PackedConstantRhs) {
            const auto& lhs = input_context.value(node.inputs.at(0));
            // Full-range narrow explicitly shares layout/storage without copying data.
            // The alias is private and only read by the kernel.
            std::optional<Tensor> lhs_view;
            if (step.transpose_a) {
                lhs_view.emplace(lhs.narrow(0, 0, lhs.shape()[0]));
                lhs_view->transpose(0, 1);
            }
            auto& output = context_->prepare_output(node.outputs.at(0));
            const auto& operand = lhs_view ? *lhs_view : lhs;
            if (step.op == OpType::MatMul) {
                cpu::matmul_packed_simd(operand, state_->weights.at(step.packed_weight_index), output);
            } else {
                const Tensor* bias = node.inputs.size() == 3
                    ? &input_context.value(node.inputs.at(2)) : nullptr;
                // No bias has no runtime layout to bind. Specialization can use
                // the same linear helper as Legacy; an identity Fused request
                // also has no register epilogue work. Keep these decisions per
                // prepared step, outside both output traversal and K blocks.
                if (step.epilogue_mode == CpuGemmEpilogueMode::Legacy ||
                    (!bias && (step.epilogue_mode == CpuGemmEpilogueMode::Specialized ||
                               step.epilogue.is_identity()))) {
                    cpu::matmul_packed_simd(operand, state_->weights.at(step.packed_weight_index), output);
                    ops::detail::apply_gemm_epilogue(output, bias, step.alpha, step.beta, step.relu);
                    if (step.epilogue_mode != CpuGemmEpilogueMode::Legacy) ++specialized_gemms_;
                } else {
                    const auto epilogue = step.epilogue.bind(bias);
                    if (step.epilogue_mode == CpuGemmEpilogueMode::Fused) {
                        cpu::matmul_packed_gemm(operand, state_->weights.at(step.packed_weight_index), output, epilogue);
                        if (epilogue.supports_fusion()) ++fused_gemms_;
                        else if (epilogue.supports_specialization()) ++specialized_gemms_;
                    } else {
                        cpu::matmul_packed_simd(operand, state_->weights.at(step.packed_weight_index), output);
                        cpu::apply_gemm_epilogue(output, epilogue);
                        if (epilogue.supports_specialization()) ++specialized_gemms_;
                    }
                }
            }
        } else {
            state_->registry.execute(node, *context_);
            if (matrix_op(step.op)) ++runtime_packs_;
        }
        context_->release_after_step(i);
    }
    ++runs_;
}

Tensor CpuExecutionContext::output(ValueId id) const {
    require_live();
    return context_->output(id);
}
Tensor CpuExecutionContext::output(const std::string& name) const {
    require_live();
    return output(state_->model.output_id(name));
}

}  // namespace tinyinfer
