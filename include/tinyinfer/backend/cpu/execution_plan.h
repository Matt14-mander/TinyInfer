#pragma once

#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "tinyinfer/model/model.h"

namespace tinyinfer {

class ExecutionContext;
namespace detail { struct CpuExecutionPlanState; }

enum class CpuExecutionPath { PackedConstantRhs, ExistingKernel };

struct CpuExecutionStep {
    NodeId node{kInvalidNodeId};
    OpType op{OpType::Add};
    CpuExecutionPath path{CpuExecutionPath::ExistingKernel};
    std::size_t packed_weight_index{std::numeric_limits<std::size_t>::max()};
    std::string fallback_reason;
    bool transpose_a{false};
    bool relu{false};
    float alpha{1.0F};
    float beta{1.0F};
};

// Owns a model snapshot, order, memory plan, and immutable packed CPU weights.
// No Graph/Tensor accessor is exposed: even a const Tensor's Buffer can be mutable.
class CpuExecutionContext;
class CpuExecutionPlan {
public:
    explicit CpuExecutionPlan(const Model& model);
    CpuExecutionPlan(const CpuExecutionPlan&) = default;
    CpuExecutionPlan& operator=(const CpuExecutionPlan&) = default;

    CpuExecutionContext create_context() const;
    ValueId input_id(const std::string& name) const;
    ValueId output_id(const std::string& name) const;
    const std::vector<Model::NamedValue>& inputs() const;
    const std::vector<Model::NamedValue>& outputs() const;
    TensorSpec input_spec(ValueId id) const;
    const std::vector<CpuExecutionStep>& steps() const;
    std::size_t pack_count() const;
    std::size_t packed_weight_bytes() const;
    std::size_t activation_bytes() const;

private:
    std::shared_ptr<const detail::CpuExecutionPlanState> state_;
};

// A movable, non-copyable session with private mutable activation storage.
// It keeps its plan alive; output() returns an independent Tensor copy.
class CpuExecutionContext {
public:
    ~CpuExecutionContext();
    CpuExecutionContext(CpuExecutionContext&&) noexcept;
    CpuExecutionContext& operator=(CpuExecutionContext&&) noexcept;
    CpuExecutionContext(const CpuExecutionContext&) = delete;
    CpuExecutionContext& operator=(const CpuExecutionContext&) = delete;

    void bind_input(ValueId id, Tensor value);
    void bind_input(const std::string& name, Tensor value);
    void run();
    Tensor output(ValueId id) const;
    Tensor output(const std::string& name) const;
    std::size_t run_count() const noexcept { return runs_; }
    std::size_t runtime_pack_count() const noexcept { return runtime_packs_; }

private:
    friend class CpuExecutionPlan;
    explicit CpuExecutionContext(
        std::shared_ptr<const detail::CpuExecutionPlanState> state);
    void require_live() const;

    // Keep state before context so destruction releases graph references first.
    std::shared_ptr<const detail::CpuExecutionPlanState> state_;
    std::unique_ptr<ExecutionContext> context_;
    std::size_t runs_{0};
    std::size_t runtime_packs_{0};
};

}  // namespace tinyinfer
