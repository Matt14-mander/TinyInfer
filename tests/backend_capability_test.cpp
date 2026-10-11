#include "tinyinfer/tinyinfer.h"
#include <cmath>
#include <filesystem>
#include <stdexcept>
using namespace tinyinfer;
namespace {
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
template<class F> void rejects(F fn) {
    try { fn(); } catch (const std::exception&) { return; }
    throw std::runtime_error("invalid placement accepted");
}
CapabilityQuery matrix_query() {
    Node node; node.op = OpType::MatMul; node.inputs = {0, 1}; node.outputs = {2};
    return {{}, node, {{{{2, 3}, DataType::Float32}, std::nullopt},
                      {{{3, 4}, DataType::Float32}, std::nullopt}},
                     {{{{2, 4}, DataType::Float32}, std::nullopt}}};
}
// Mock for placement/error contracts only: never executes GPU operations.
class TestGpu final : public Backend {
public:
    bool available{true};
    std::string_view name() const noexcept override { return "test-gpu"; }
    bool supports(OpType) const noexcept override { return true; }
    DeviceInfo device_info() const override { return {{DeviceKind::Cuda, 0}, "test", available, "not built"}; }
    NodeCapability query_capability(const CapabilityQuery& query) const override {
        return {false, false, "unsupported layout on '" + query.node.name + "'"};
    }
    void execute(const Node&, ExecutionContext&) override { throw std::runtime_error("must not execute GPU mock"); }
};
void capability() {
    CpuBackend cpu;
    auto query = matrix_query();
    require(cpu.query_capability(query).supported, "valid matrix rejected");
    query.inputs[0].strides = Strides{6, 2};
    require(cpu.query_capability(query).supported, "valid strided lhs rejected");
    query.inputs[0].strides = Strides{3, -1};
    require(!cpu.query_capability(query).supported, "negative stride accepted");
    query = matrix_query(); query.inputs[1].spec.shape = {7, 4};
    require(!cpu.query_capability(query).supported, "invalid inner dimension accepted");
    query = matrix_query(); query.inputs[0].spec.dtype = DataType::Float16;
    require(!cpu.query_capability(query).supported, "unsupported dtype accepted");
    query = matrix_query(); query.outputs[0].spec.shape = {2, 5};
    require(!cpu.query_capability(query).supported, "invalid output accepted");
    query = matrix_query(); query.outputs[0].strides = Strides{8, 2};
    require(!cpu.query_capability(query).supported, "strided planned output accepted");
    query = matrix_query(); query.node.attributes["transA"] = std::int64_t{1};
    require(!cpu.query_capability(query).supported, "unknown matrix attribute accepted");
    query = matrix_query(); query.device = {DeviceKind::Cuda, 0};
    require(!cpu.query_capability(query).supported, "foreign device accepted");
    query = matrix_query(); query.outputs.clear();
    require(!cpu.query_capability(query).supported, "descriptor arity mismatch accepted");
    require(!cpu.query_capability(query).reason.empty(), "missing capability reason");
    query = matrix_query(); query.node.op = OpType::Gemm;
    query.node.attributes = {{"alpha", -0.5F}, {"beta", 1.5F}, {"transA", std::int64_t{0}}};
    require(cpu.query_capability(query).supported, "valid Gemm attributes rejected");
    query.node.attributes["transB"] = std::int64_t{2};
    require(!cpu.query_capability(query).supported, "invalid transpose attribute accepted");
    // Legacy OpType support still works, and does not promise shape/dtype support.
    require(cpu.supports(OpType::MatMul), "legacy backend API broken");
}
void placement_and_cpu_fixture() {
    onnx::OnnxImporter importer;
    const auto path = std::filesystem::path(TINYINFER_TEST_SOURCE_DIR) / "fixtures/phase3_mlp_gemm.onnx";
    auto model = importer.load(path);
    CpuBackend cpu;
    auto cpu_placement = select_model_backend(model, cpu, FallbackPolicy::Strict);
    require(!cpu_placement.used_fallback && cpu_placement.effective == DeviceId{}, "CPU placement differs");
    require(cpu_placement.nodes.size() == model.graph().size(), "node diagnostics missing");
    TestGpu gpu;
    rejects([&] { select_model_backend(model, gpu, FallbackPolicy::Strict); });
    auto fallback = select_model_backend(model, gpu, FallbackPolicy::WholeModelCpu);
    require(fallback.used_fallback && fallback.effective == DeviceId{} &&
            fallback.requested.kind == DeviceKind::Cuda, "whole-model fallback differs");
    require(!fallback.reason.empty() && fallback.nodes.size() == model.graph().size(), "fallback diagnostics missing");
    // Execute the selected CPU plan, verifying original ownership/output contract.
    CpuExecutionPlan plan(model);
    auto session = plan.create_context();
    session.bind_input("x", Tensor::from_vector({1, 2}, {1, -2})); session.run();
    auto output = session.output("probabilities");
    require(std::abs(output.at(0) - 0.26894143F) < 1e-5F, "fallback fixture output differs");
    output.at(0) = 99;
    require(session.output("probabilities").at(0) != 99, "output ownership changed");
    gpu.available = false;
    rejects([&] { select_model_backend(model, gpu, FallbackPolicy::Strict); });
    require(select_model_backend(model, gpu, FallbackPolicy::WholeModelCpu).used_fallback, "missing-device fallback failed");
    rejects([&] { select_model_backend(model, cpu, static_cast<FallbackPolicy>(99)); });
}
} // namespace
int main() { capability(); placement_and_cpu_fixture(); }
