#include "tinyinfer/tinyinfer.h"

#include <cmath>
#include <filesystem>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace tinyinfer;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename Function>
void rejects(Function&& function) {
    bool rejected = false;
    try { function(); } catch (const std::exception&) { rejected = true; }
    require(rejected, "invalid prepared operation was accepted");
}

Tensor values(Shape shape, int seed) {
    Tensor result(std::move(shape));
    for (std::size_t i = 0; i < result.numel(); ++i) {
        result.at(i) = static_cast<float>(static_cast<int>((i * 13 + seed) % 29) - 14) / 17.0F;
    }
    return result;
}

void close(const Tensor& actual, const Tensor& expected, float tolerance = 1e-5F) {
    require(actual.shape() == expected.shape() && actual.dtype() == expected.dtype(),
            "prepared output metadata differs");
    for (std::size_t i = 0; i < actual.numel(); ++i) {
        const float a = actual.at(i), b = expected.at(i);
        if (std::isnan(b)) { require(std::isnan(a), "NaN semantics changed"); continue; }
        if (std::isinf(b)) { require(a == b, "infinity semantics changed"); continue; }
        require(std::isfinite(a) && std::fabs(a - b) <= tolerance * (1.0F + std::fabs(b)),
                "prepared output differs numerically");
        if (a == 0.0F && b == 0.0F) {
            require(std::signbit(a) == std::signbit(b), "signed-zero semantics changed");
        }
    }
}

struct MatrixCase {
    Model model;
    Tensor x;
    Tensor b;
    Tensor c;
    bool dynamic_b;
    bool dynamic_c;
};

MatrixCase matrix_case(OpType op, bool ta, bool tb, bool dynamic_b,
                       const Shape* bias_shape, bool dynamic_c,
                       std::int64_t m = 2, std::int64_t k = 3, std::int64_t n = 5,
                       float alpha = -0.5F, float beta = 1.5F) {
    auto x = values(ta ? Shape{k, m} : Shape{m, k}, 1);
    auto b = values(tb ? Shape{n, k} : Shape{k, n}, 7);
    auto c = bias_shape ? values(*bias_shape, 11) : Tensor({0});
    Graph graph;
    const auto xi = graph.add_input("x", {x.shape(), DataType::Float32});
    const auto bi = dynamic_b
        ? graph.add_input("b", {b.shape(), DataType::Float32})
        : graph.add_constant("b", b);
    std::vector<ValueId> operands{xi, bi};
    std::vector<Model::NamedValue> inputs{{"x", xi}};
    if (dynamic_b) inputs.emplace_back("b", bi);
    if (bias_shape) {
        const auto ci = dynamic_c
            ? graph.add_input("c", {c.shape(), DataType::Float32})
            : graph.add_constant("c", c);
        operands.push_back(ci);
        if (dynamic_c) inputs.emplace_back("c", ci);
    }
    NodeAttributes attributes;
    if (op != OpType::MatMul) {
        attributes = {{"transA", std::int64_t{ta}}, {"transB", std::int64_t{tb}},
                      {"alpha", alpha}, {"beta", beta}};
        if (op == OpType::FusedGemmActivation) attributes["activation"] = std::string("relu");
    }
    const auto node = graph.add_node("matrix", op, operands, attributes);
    const auto out = graph.node(node).outputs.front();
    graph.mark_output(out);
    return {Model(std::move(graph), std::move(inputs), {{"y", out}}),
            std::move(x), std::move(b), std::move(c), dynamic_b,
            bias_shape != nullptr && dynamic_c};
}

Tensor ordinary(const MatrixCase& fixture) {
    ExecutionContext context(fixture.model.graph(), MemoryPlan(fixture.model.graph()));
    context.bind_input(fixture.model.input_id("x"), fixture.x);
    if (fixture.dynamic_b) context.bind_input(fixture.model.input_id("b"), fixture.b);
    if (fixture.dynamic_c) context.bind_input(fixture.model.input_id("c"), fixture.c);
    CpuBackend backend;
    Executor executor(backend);
    executor.run(fixture.model.graph(), context);
    return context.output(fixture.model.output_id("y"));
}

void check_matrix(MatrixCase fixture) {
    CpuExecutionPlan plan(fixture.model);
    require(plan.pack_count() == (fixture.dynamic_b ? 0U : 1U), "incorrect pack count");
    require(plan.packed_weight_bytes() == (fixture.dynamic_b ? 0U : fixture.b.size_bytes()),
            "incorrect packed payload size");
    require(plan.activation_bytes() == MemoryPlan(fixture.model.graph()).buffer_size_bytes(),
            "activation memory policy changed");
    require(plan.steps().front().path == (fixture.dynamic_b
                ? CpuExecutionPath::ExistingKernel : CpuExecutionPath::PackedConstantRhs),
            "incorrect prepared dispatch");
    auto context = plan.create_context();
    rejects([&] { context.run(); });
    rejects([&] { context.output("y"); });
    rejects([&] { context.bind_input("missing", fixture.x); });
    rejects([&] { plan.input_spec(plan.output_id("y")); });
    for (int run = 0; run < 3; ++run) {
        fixture.x = values(fixture.x.shape(), 1 + run * 5);
        if (fixture.dynamic_b) fixture.b = values(fixture.b.shape(), 7 + run * 7);
        if (fixture.dynamic_c) fixture.c = values(fixture.c.shape(), 11 + run * 3);
        context.bind_input("x", fixture.x);
        if (fixture.dynamic_b) context.bind_input("b", fixture.b);
        if (fixture.dynamic_c) context.bind_input("c", fixture.c);
        context.run();
        close(context.output("y"), ordinary(fixture));
    }
    require(context.run_count() == 3, "run count differs");
    require(context.runtime_pack_count() == (fixture.dynamic_b ? 3U : 0U),
            "constant RHS was packed during inference or dynamic RHS was stale");
    auto original = context.output("y");
    auto escaped = context.output("y");
    if (escaped.numel() > 0) escaped.at(0) = 999.0F;
    close(context.output("y"), original);
    auto moved = std::move(context);
    rejects([&] { context.run(); });
    close(moved.output("y"), original);
    auto assigned = plan.create_context();
    assigned = std::move(moved);
    assigned.run();
    close(assigned.output("y"), original);
}

void check_sharing_and_snapshot() {
    Graph graph;
    const auto x = graph.add_input("x", {{2, 2}, DataType::Float32});
    const auto b = graph.add_constant("b", Tensor::from_vector({2, 2}, {1, 2, 3, 4}));
    std::vector<Model::NamedValue> outputs;
    for (int i = 0; i < 3; ++i) {
        const auto node = graph.add_node("g" + std::to_string(i), OpType::Gemm, {x, b},
                                        {{"transB", std::int64_t{i == 2 ? 1 : 0}}});
        const auto out = graph.node(node).outputs.front();
        graph.mark_output(out);
        outputs.emplace_back("y" + std::to_string(i), out);
    }
    Model model(std::move(graph), {{"x", x}}, outputs);
    CpuExecutionPlan plan(model);
    require(plan.pack_count() == 2 && plan.packed_weight_bytes() == 32,
            "same weight/orientation did not deduplicate");
    require(plan.steps()[0].packed_weight_index == plan.steps()[1].packed_weight_index &&
            plan.steps()[0].packed_weight_index != plan.steps()[2].packed_weight_index,
            "transpose packing cache key is incorrect");
    auto first = plan.create_context();
    auto second = plan.create_context();
    first.bind_input("x", Tensor::from_vector({2, 2}, {1, 0, 0, 1}));
    second.bind_input("x", Tensor::from_vector({2, 2}, {2, 0, 0, 2}));
    // A const Tensor can expose a mutable Buffer. The plan must own a deep snapshot.
    static_cast<float*>(model.graph().constant(b).storage().buffer()->data())[0] = 999;
    first.run();
    second.run();
    close(first.output("y0"), Tensor::from_vector({2, 2}, {1, 2, 3, 4}));
    close(second.output("y0"), Tensor::from_vector({2, 2}, {2, 4, 6, 8}));
    close(first.output("y2"), Tensor::from_vector({2, 2}, {1, 3, 2, 4}));
    second.run();
    close(first.output("y0"), Tensor::from_vector({2, 2}, {1, 2, 3, 4}));
}

void check_lifetime_and_constant_output() {
    auto context = [] {
        Graph graph;
        const auto c = graph.add_constant("c", Tensor::from_vector({1}, {42}));
        graph.mark_output(c);
        Model model(std::move(graph), {}, {{"answer", c}});
        CpuExecutionPlan plan(model);
        return plan.create_context();
    }();
    context.run();
    auto escaped = context.output("answer");
    escaped.at(0) = -1;
    context.run();
    require(context.output("answer").at(0) == 42, "constant output escaped plan ownership");
}

void check_strides() {
    auto fixture = matrix_case(OpType::Gemm, true, true, false, nullptr, false);
    CpuExecutionPlan plan(fixture.model);
    auto context = plan.create_context();
    auto backing = values({3, 4}, 9);
    auto view = backing.slice(1, 0, 4, 2);
    fixture.x = view;
    context.bind_input("x", std::move(view));
    context.run();
    close(context.output("y"), ordinary(fixture));

    Tensor rhs_backing = values({5, 6}, 4);
    auto rhs_view = rhs_backing.slice(1, 0, 6, 2);
    cpu::PackedMatMulRhs packed(rhs_view, true);
    auto rhs_copy = rhs_view;
    rhs_copy.transpose(0, 1);
    cpu::PackedMatMulRhs reference(rhs_copy);
    require(packed.size_bytes() == reference.size_bytes(), "strided packing shape differs");
    for (std::size_t i = 0; i < packed.size_bytes() / sizeof(float); ++i) {
        require(packed.data()[i] == reference.data()[i], "direct transpose packing differs");
    }
}

void check_special_values() {
    for (auto op : {OpType::Gemm, OpType::FusedGemmActivation}) {
        for (float value : {0.0F, -0.0F, std::numeric_limits<float>::infinity(),
                            -std::numeric_limits<float>::infinity(),
                            std::numeric_limits<float>::quiet_NaN()}) {
            auto fixture = matrix_case(op, false, false, false, nullptr, false, 1, 1, 1, 1, 1);
            fixture.x.at(0) = value;
            CpuExecutionPlan plan(fixture.model);
            auto context = plan.create_context();
            context.bind_input("x", fixture.x);
            context.run();
            close(context.output("y"), ordinary(fixture));
        }
    }
}

void check_onnx(const char* name, const char* input_name, const char* output_name,
                Shape shape, const std::vector<float>& input,
                const std::vector<float>& expected) {
    onnx::OnnxImporter importer;
    const auto source = importer.load(std::filesystem::path(TINYINFER_TEST_SOURCE_DIR) / "fixtures" / name);
    optimizer::PassManager optimizer;
    optimizer.add_pass(std::make_unique<optimizer::ConstantFoldingPass>());
    optimizer.add_pass(std::make_unique<optimizer::DeadCodeEliminationPass>());
    optimizer.add_pass(std::make_unique<optimizer::MatMulAddCanonicalizationPass>());
    optimizer.add_pass(std::make_unique<optimizer::GemmActivationFusionPass>());
    optimizer.add_pass(std::make_unique<optimizer::DeadCodeEliminationPass>());
    const auto optimized = optimizer.run(source);
    CpuExecutionPlan plan(optimized.model);
    require(plan.pack_count() == 2, "ONNX model weights were not prepared");
    auto context = plan.create_context();
    for (int i = 0; i < 2; ++i) {
        context.bind_input(input_name, Tensor::from_vector(shape, input));
        context.run();
        close(context.output(output_name), Tensor::from_vector({1, 2}, expected), 1e-5F);
    }
    require(context.runtime_pack_count() == 0, "ONNX constant weights repacked");
}

}  // namespace

int main() {
    for (auto op : {OpType::Gemm, OpType::FusedGemmActivation}) {
        for (bool ta : {false, true}) for (bool tb : {false, true}) {
            for (bool dynamic_b : {false, true}) {
                check_matrix(matrix_case(op, ta, tb, dynamic_b, nullptr, false));
                for (const auto& shape : std::vector<Shape>{{}, {5}, {1, 5}, {2, 1}, {2, 5}}) {
                    for (bool dynamic_c : {false, true}) {
                        check_matrix(matrix_case(op, ta, tb, dynamic_b, &shape, dynamic_c));
                    }
                }
            }
        }
    }
    for (bool dynamic : {false, true}) {
        check_matrix(matrix_case(OpType::MatMul, false, false, dynamic, nullptr, false));
    }
    for (const Shape& dimensions : std::vector<Shape>{{0, 3, 5}, {2, 0, 5}, {2, 3, 0}, {3, 127, 131}}) {
        check_matrix(matrix_case(OpType::FusedGemmActivation, true, true, false, nullptr, false,
                                 dimensions[0], dimensions[1], dimensions[2]));
    }
    const Shape bias{5};
    check_matrix(matrix_case(OpType::Gemm, false, true, false, &bias, true, 2, 3, 5, 0, 0));
    check_sharing_and_snapshot();
    check_lifetime_and_constant_output();
    check_strides();
    check_special_values();
    check_onnx("phase3_mlp_gemm.onnx", "x", "probabilities", {1, 2},
               {1, -2}, {0.26894143F, 0.73105860F});
    check_onnx("rl_actor_mlp_tanh.onnx", "observations", "actions", {1, 4},
               {0.25F, -0.5F, 1.0F, 0.75F}, {0.58088166F, -0.47540686F});
}
