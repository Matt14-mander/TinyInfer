#include "tinyinfer/backend/metal/metal_runtime.h"
#include "tinyinfer/tinyinfer.h"
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace tinyinfer;
namespace {
std::size_t checked_cases = 0;
void require(bool value, const char* text) { if (!value) throw std::runtime_error(text); }
template<class F> void rejects(F fn) {
    try { fn(); } catch (const std::exception&) { return; }
    throw std::runtime_error("invalid Metal dispatch accepted");
}
void close(const Tensor& a, const Tensor& b) {
    require(a.shape() == b.shape(), "shape differs");
    for (std::size_t i = 0; i < a.numel(); ++i) {
        const float x = a.at(i), y = b.at(i);
        if (std::isnan(y)) require(std::isnan(x), "NaN semantics differ");
        else if (std::isinf(y)) require(x == y, "infinity semantics differ");
        else if (!std::isfinite(x) || std::abs(x - y) > 1e-5F * (1 + std::abs(y)))
            throw std::runtime_error("finite result differs at " + std::to_string(i) + ": " + std::to_string(x) + " vs " + std::to_string(y));
        if (x == 0 && y == 0) require(std::signbit(x) == std::signbit(y), "signed zero differs");
    }
}
Tensor values(Shape shape, int seed) {
    Tensor t(shape); for (std::size_t i = 0; i < t.numel(); ++i) t.at(i) = float(int((i * 17 + seed) % 31) - 15) / 19;
    return t;
}
MetalTensorView upload(MetalDeviceRuntime& runtime, const Tensor& tensor, std::size_t padding = 0) {
    auto storage = runtime.allocate(tensor.size_bytes() + padding);
    auto host = std::make_shared<Buffer>(tensor.size_bytes());
    if (tensor.numel()) std::memcpy(host->data(), tensor.data<float>(), tensor.size_bytes());
    runtime.upload({storage, padding, tensor.size_bytes()}, host).wait();
    return {{tensor.shape(), tensor.dtype()}, {storage, padding, tensor.size_bytes()}};
}
Tensor download(MetalDeviceRuntime& runtime, const MetalTensorView& view) {
    Tensor result(view.spec.shape);
    auto host = std::make_shared<Buffer>(view.storage.bytes);
    runtime.download(host, view.storage).wait();
    if (result.numel()) std::memcpy(result.data<float>(), host->data(), result.size_bytes());
    return result;
}
void check(MetalDeviceRuntime& runtime, OpType op, const std::vector<Tensor>& inputs,
           NodeAttributes attributes = {}) {
    Graph graph; std::vector<ValueId> ids;
    for (std::size_t i = 0; i < inputs.size(); ++i) ids.push_back(graph.add_constant("in" + std::to_string(i), inputs[i]));
    const auto id = graph.add_node("test", op, ids, attributes);
    const auto& node = graph.node(id); graph.mark_output(node.outputs[0]);
    ExecutionContext context(graph); CpuBackend cpu; Executor executor(cpu); executor.run(graph, context);
    auto expected = context.output(node.outputs[0]);
    std::vector<MetalTensorView> views;
    for (const auto& input : inputs) views.push_back(upload(runtime, input, 4));
    MetalTensorView output({expected.shape(), DataType::Float32},
                          {runtime.allocate(expected.size_bytes() + 4), 4, expected.size_bytes()});
    runtime.execute(op, views, output, attributes).wait();
    close(download(runtime, output), expected); ++checked_cases;
}
void matrices(MetalDeviceRuntime& runtime) {
    for (bool ta : {false, true}) for (bool tb : {false, true}) {
        for (int k : {0, 1, 33, 65}) {
            const auto a = values(ta ? Shape{k, 3} : Shape{3, k}, 1);
            const auto b = values(tb ? Shape{7, k} : Shape{k, 7}, 2);
            for (auto op : {OpType::Gemm, OpType::FusedGemmActivation}) {
                NodeAttributes attrs{{"transA", std::int64_t{ta}}, {"transB", std::int64_t{tb}}, {"alpha", -0.5F}, {"beta", 1.5F}};
                if (op == OpType::FusedGemmActivation) attrs["activation"] = std::string("relu");
                check(runtime, op, {a, b}, attrs);
                for (const auto& bias : std::vector<Shape>{{}, {7}, {1, 7}, {3, 1}, {3, 7}})
                    check(runtime, op, {a, b, values(bias, 3)}, attrs);
            }
        }
    }
    for (int n : {1, 4, 8, 17, 131}) check(runtime, OpType::MatMul, {values({3, 127}, 1), values({127, n}, 2)});
    for (auto op : {OpType::Gemm, OpType::FusedGemmActivation}) {
        NodeAttributes attrs;
        if (op == OpType::FusedGemmActivation) attrs["activation"] = std::string("relu");
        for (const auto& bias : std::vector<Shape>{{}, {7}, {1, 7}, {3, 1}, {3, 7}})
            check(runtime, op, {values({3, 33}, 1), values({33, 7}, 2), values(bias, 3)}, attrs);
    }
    check(runtime, OpType::MatMul, {values({0, 3}, 1), values({3, 7}, 2)});
    check(runtime, OpType::Gemm, {values({3, 4}, 1), values({4, 0}, 2)});
}
void elementwise(MetalDeviceRuntime& runtime) {
    for (auto shape : std::vector<Shape>{{}, {1}, {67}, {2, 3, 7}, {2, 1, 3, 7}, {0, 7}}) {
        check(runtime, OpType::ReLU, {values(shape, 1)});
        check(runtime, OpType::Tanh, {values(shape, 1)});
        check(runtime, OpType::Add, {values(shape, 1), values({}, 2)});
    }
    check(runtime, OpType::Add, {values({2, 1, 3, 1}, 1), values({1, 4, 1, 7}, 2)});
    for (auto axis : {-3, -2, -1, 0, 1, 2})
        check(runtime, OpType::Softmax, {values({2, 3, 7}, 1)}, {{"axis", std::int64_t{axis}}});
    check(runtime, OpType::Softmax, {values({0, 3}, 1)});
    const float inf = std::numeric_limits<float>::infinity(), nan = std::numeric_limits<float>::quiet_NaN();
    const auto special = Tensor::from_vector({7}, {0.0F, -0.0F, 1, -1, inf, -inf, nan});
    check(runtime, OpType::ReLU, {special}); check(runtime, OpType::Tanh, {special});
    check(runtime, OpType::Add, {special, special});
    check(runtime, OpType::Softmax, {special});
    for (float alpha : {1.0F, 0.0F, -1.0F}) {
        for (auto op : {OpType::Gemm, OpType::FusedGemmActivation}) {
            NodeAttributes attrs{{"alpha", alpha}, {"beta", 0.0F}};
            if (op == OpType::FusedGemmActivation) attrs["activation"] = std::string("relu");
            check(runtime, op, {Tensor::from_vector({1, 1}, {1}), Tensor::from_vector({1, 7}, {0, -0.0F, 1, -1, inf, -inf, nan})}, attrs);
        }
    }
}
void diagnostics(MetalDeviceRuntime& runtime) {
    auto a = upload(runtime, values({2, 3}, 1)), b = upload(runtime, values({3, 7}, 2));
    auto out = upload(runtime, values({2, 7}, 3));
    Node node; node.op = OpType::MatMul; node.inputs = {0, 1}; node.outputs = {2};
    CapabilityQuery query{runtime.device(), node, {{a.spec, std::nullopt}, {b.spec, std::nullopt}}, {{out.spec, out.strides}}};
    auto capability = runtime.query_capability(query);
    require(capability.supported && capability.requires_runtime_layout_check, "unknown layouts not marked conditional");
    query.inputs[0].strides = Strides{6, 2};
    require(!runtime.query_capability(query).supported, "strided query was accepted");
    const auto before = runtime.kernel_statistics().dispatches;
    a.strides = {6, 2}; rejects([&] { runtime.execute(OpType::MatMul, {a, b}, out); });
    a.strides = {3, 1};
    rejects([&] { runtime.execute(OpType::GELU, {a}, a); });
    rejects([&] { runtime.execute(OpType::ReLU, {a}, a); });
    auto bad = a; bad.storage.offset = 1; bad.storage.bytes -= 1;
    rejects([&] { runtime.execute(OpType::ReLU, {bad}, a); });
    a.spec.dtype = DataType::Float16;
    rejects([&] { runtime.execute(OpType::MatMul, {a, b}, out); });
    require(runtime.kernel_statistics().dispatches == before, "rejected dispatch changed kernel counters");
    MetalBackend backend(std::make_shared<MetalDeviceRuntime>());
    Graph graph; auto x = graph.add_input("x", {{2, 3}, DataType::Float32});
    graph.mark_output(x); ExecutionContext context(graph);
    rejects([&] { backend.execute(node, context); });
}
void fixture_chain(MetalDeviceRuntime& runtime, const char* name, const std::vector<float>& data,
                   const std::vector<float>& golden) {
    onnx::OnnxImporter importer;
    auto model = importer.load(std::filesystem::path(TINYINFER_TEST_SOURCE_DIR) / "fixtures" / name);
    ExecutionContext reference(model.graph());
    const auto input_id = model.inputs()[0].second;
    auto input = Tensor::from_vector(model.graph().value(input_id).spec.shape, data);
    reference.bind_input(input_id, input);
    CpuBackend cpu; Executor executor(cpu); executor.run(model.graph(), reference);
    std::vector<std::optional<MetalTensorView>> values_by_id(model.graph().value_count());
    values_by_id[input_id] = upload(runtime, input);
    for (const auto& value : model.graph().values()) if (model.graph().is_constant(value.id))
        values_by_id[value.id] = upload(runtime, model.graph().constant(value.id));
    std::vector<Completion> work;
    for (auto node_id : model.graph().topological_order()) {
        const auto& node = model.graph().node(node_id);
        std::vector<MetalTensorView> inputs;
        for (auto id : node.inputs) inputs.push_back(*values_by_id[id]);
        const auto out_id = node.outputs[0]; const auto spec = model.graph().value(out_id).spec;
        const auto bytes = TensorLayout(spec.shape).size_bytes(spec.dtype);
        values_by_id[out_id].emplace(spec, DeviceSpan{runtime.allocate(bytes), 0, bytes});
        work.push_back(runtime.execute(node.op, inputs, *values_by_id[out_id], node.attributes));
    }
    runtime.synchronize();
    const auto out_id = model.outputs()[0].second;
    const auto actual = download(runtime, *values_by_id[out_id]);
    close(actual, reference.output(out_id));
    close(actual, Tensor::from_vector(actual.shape(), golden));
    for (auto& completion : work) completion.wait();
    ++checked_cases;
}
}
int main(int argc, char**) {
    try {
        if (MetalDeviceRuntime::enumerate_devices().empty()) {
            std::cerr << "No visible Metal device\n"; return argc > 1 ? 1 : 77;
        }
        MetalDeviceRuntime runtime;
        std::cout << "Device: " << runtime.device_info().name << '\n';
        matrices(runtime); elementwise(runtime); diagnostics(runtime);
        fixture_chain(runtime, "phase3_mlp_gemm.onnx", {1, -2}, {0.26894143F, 0.73105860F});
        fixture_chain(runtime, "rl_actor_mlp_tanh.onnx", {0.25F, -0.5F, 1, 0.75F}, {0.58088166F, -0.47540686F});
        fixture_chain(runtime, "rl_actor_mlp_tanh.onnx", {-1, 0, 0.5F, -0.25F}, {-0.17322102F, -0.59409916F});
        require(runtime.kernel_statistics().pipeline_creations == 5, "pipelines recreated during dispatch");
        std::cout << "PASS: " << checked_cases << " checked cases; dispatches=" << runtime.kernel_statistics().dispatches << '\n';
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
