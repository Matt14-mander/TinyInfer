#include "tinyinfer/tinyinfer.h"
#include "ops/gemm_internal.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <filesystem>
#include <functional>
#include <optional>
#include <map>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
    std::size_t m{16};
    std::size_t k{128};
    std::size_t n{128};
    std::size_t warmup{10};
    std::size_t samples{50};
    std::size_t repeats{10};
    std::string format{"json"};
    bool transpose_b{true};
    bool dynamic_b{false};
    std::string model_path;
    std::string bias{"vector"};
    bool relu{true};
    bool fusion_comparison{false};
    std::size_t allocation_order{0};
};

struct Statistics {
    double p50_us{};
    double p95_us{};
    double mean_us{};
};

std::size_t positive_size(std::string_view input, const char* name) {
    if (input.empty() || input.front() < '0' || input.front() > '9') {
        throw std::invalid_argument(std::string(name) + " requires a positive integer");
    }
    std::size_t consumed = 0;
    std::size_t result = 0;
    try {
        result = std::stoull(std::string(input), &consumed);
    } catch (const std::exception&) {
        throw std::invalid_argument(std::string(name) +
                                    " requires a positive integer");
    }
    if (consumed != input.size() || result == 0) {
        throw std::invalid_argument(std::string(name) +
                                    " requires a positive integer");
    }
    return result;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto value = [&](const char* name) -> std::string_view {
            if (index + 1 >= argc) {
                throw std::invalid_argument(std::string(name) +
                                            " requires a value");
            }
            return argv[++index];
        };
        if (argument == "--help") {
            std::cout << "Usage: " << argv[0]
                      << " [--m N] [--k N] [--n N] [--warmup N]"
                         " [--samples N] [--repeats N] [--format json] [--trans-b 0|1] [--dynamic-b] [--model PATH] [--bias none|scalar|vector|row|column|full] [--activation none|relu] [--fusion-comparison] [--allocation-order 0|1|2|3]\n";
            std::exit(0);
        } else if (argument == "--m") {
            options.m = positive_size(value("--m"), "--m");
        } else if (argument == "--k") {
            options.k = positive_size(value("--k"), "--k");
        } else if (argument == "--n") {
            options.n = positive_size(value("--n"), "--n");
        } else if (argument == "--warmup") {
            options.warmup = positive_size(value("--warmup"), "--warmup");
        } else if (argument == "--samples") {
            options.samples = positive_size(value("--samples"), "--samples");
        } else if (argument == "--repeats") {
            options.repeats = positive_size(value("--repeats"), "--repeats");
        } else if (argument == "--trans-b") {
            const auto transpose = value("--trans-b");
            if (transpose != "0" && transpose != "1") throw std::invalid_argument("--trans-b must be 0 or 1");
            options.transpose_b = transpose == "1";
        } else if (argument == "--dynamic-b") {
            options.dynamic_b = true;
        } else if (argument == "--fusion-comparison") {
            options.fusion_comparison = true;
        } else if (argument == "--allocation-order") {
            const auto order = value("--allocation-order");
            if (order != "0" && order != "1" && order != "2" && order != "3")
                throw std::invalid_argument("--allocation-order must be 0, 1, 2 or 3");
            options.allocation_order = static_cast<std::size_t>(order.front() - '0');
        } else if (argument == "--bias") {
            options.bias = value("--bias");
            const std::vector<std::string> supported{"none", "scalar", "vector", "row", "column", "full"};
            if (std::find(supported.begin(), supported.end(), options.bias) == supported.end())
                throw std::invalid_argument("unsupported bias form");
        } else if (argument == "--activation") {
            const auto activation = value("--activation");
            if (activation != "none" && activation != "relu")
                throw std::invalid_argument("activation must be none or relu");
            options.relu = activation == "relu";
        } else if (argument == "--model") {
            options.model_path = value("--model");
        } else if (argument == "--format") {
            options.format = value("--format");
            if (options.format != "json") {
                throw std::invalid_argument("--format must be json");
            }
        } else {
            throw std::invalid_argument("unknown option: " +
                                        std::string(argument));
        }
    }
    return options;
}

std::vector<float> generated_values(std::size_t count, int seed) {
    std::vector<float> values(count);
    for (std::size_t index = 0; index < count; ++index) {
        const auto raw = static_cast<int>((index * 37 + seed * 17) % 101) - 50;
        values[index] = static_cast<float>(raw) / 31.0F;
    }
    return values;
}

using namespace tinyinfer;
using Bindings = std::vector<std::pair<ValueId, Tensor>>;

Model make_model(const Options& options) {
    if (!options.model_path.empty()) {
        onnx::OnnxImporter importer;
        auto model = importer.load(options.model_path);
        optimizer::PassManager passes;
        passes.add_pass(std::make_unique<optimizer::ConstantFoldingPass>());
        passes.add_pass(std::make_unique<optimizer::DeadCodeEliminationPass>());
        passes.add_pass(std::make_unique<optimizer::MatMulAddCanonicalizationPass>());
        passes.add_pass(std::make_unique<optimizer::GemmActivationFusionPass>());
        passes.add_pass(std::make_unique<optimizer::DeadCodeEliminationPass>());
        auto result = passes.run(model);
        return std::move(result.model);
    }
    const auto m = static_cast<std::int64_t>(options.m);
    const auto k = static_cast<std::int64_t>(options.k);
    const auto n = static_cast<std::int64_t>(options.n);
    const Shape rhs_shape = options.transpose_b ? Shape{n, k} : Shape{k, n};
    Graph graph;
    const auto x = graph.add_input("x", {{m, k}, DataType::Float32});
    const auto b = options.dynamic_b
        ? graph.add_input("b", {rhs_shape, DataType::Float32})
        : graph.add_constant("b", Tensor::from_vector(
              rhs_shape, generated_values(options.k * options.n, 3)));
    std::vector<ValueId> operands{x, b};
    if (options.bias != "none") {
        Shape shape{n};
        if (options.bias == "scalar") shape = {};
        else if (options.bias == "row") shape = {1, n};
        else if (options.bias == "column") shape = {m, 1};
        else if (options.bias == "full") shape = {m, n};
        Tensor c(shape);
        const auto data = generated_values(c.numel(), 11);
        std::copy(data.begin(), data.end(), c.data<float>());
        operands.push_back(graph.add_constant("c", std::move(c)));
    }
    NodeAttributes attributes{{"transB", std::int64_t{options.transpose_b}}};
    if (options.relu) attributes["activation"] = std::string("relu");
    const auto node = graph.add_node("affine", options.relu ? OpType::FusedGemmActivation : OpType::Gemm,
                                   operands, std::move(attributes));
    const auto out = graph.node(node).outputs.front();
    graph.mark_output(out);
    std::vector<Model::NamedValue> inputs{{"x", x}};
    if (options.dynamic_b) inputs.emplace_back("b", b);
    return Model(std::move(graph), std::move(inputs), {{"y", out}});
}

Bindings make_inputs(const Model& model) {
    Bindings inputs;
    for (const auto& input : model.inputs()) {
        const auto& spec = model.graph().value(input.second).spec;
        if (spec.dtype != DataType::Float32) throw std::invalid_argument("benchmark inputs must be FP32");
        Tensor tensor(spec.shape);
        const auto data = generated_values(tensor.numel(), input.first == "b" ? 3 : 23);
        std::copy(data.begin(), data.end(), tensor.data<float>());
        inputs.emplace_back(input.second, std::move(tensor));
    }
    return inputs;
}

template <typename Context>
void bind(Context& context, const Bindings& inputs) {
    for (const auto& input : inputs) context.bind_input(input.first, input.second);
}

void expect_close(const Tensor& actual, const Tensor& expected) {
    if (actual.shape() != expected.shape() || actual.dtype() != expected.dtype()) {
        throw std::runtime_error("prepared output metadata changed");
    }
    for (std::size_t i = 0; i < actual.numel(); ++i) {
        const auto a = actual.at(i), b = expected.at(i);
        if (!std::isfinite(a) || !std::isfinite(b) ||
            std::fabs(a - b) > 1e-5F * (1.0F + std::fabs(b))) {
            throw std::runtime_error("output value changed at " + std::to_string(i) + ": actual=" + std::to_string(a) + ", expected=" + std::to_string(b));
        }
    }
}

double percentile(const std::vector<double>& sorted, double p) {
    const auto position = p * static_cast<double>(sorted.size() - 1);
    const auto lower = static_cast<std::size_t>(position);
    const auto upper = std::min(lower + 1, sorted.size() - 1);
    return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - static_cast<double>(lower));
}

Statistics summarize(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    const auto mean = std::accumulate(samples.begin(), samples.end(), 0.0) /
                      static_cast<double>(samples.size());
    return {percentile(samples, 0.5), percentile(samples, 0.95), mean};
}

template <typename Function>
double time_repeats(Function&& function, std::size_t repeats) {
    const auto start = Clock::now();
    for (std::size_t i = 0; i < repeats; ++i) function();
    return std::chrono::duration<double, std::micro>(Clock::now() - start).count() /
           static_cast<double>(repeats);
}

// Frozen operands are captured from a checked ordinary execution outside timers.
// They permit direct per-node diagnostics without altering the production runtime.
struct MatrixWork {
    NodeId node;
    Tensor lhs;
    Tensor rhs;
    std::optional<Tensor> bias;
    cpu::PackedMatMulRhs packed;
    Tensor product;
    Tensor output;
    Tensor expected;
    std::vector<Tensor> epilogue_pool;
    float alpha;
    float beta;
    bool relu;
    bool dynamic;

    MatrixWork(const Node& op, const ExecutionContext& context, const Graph& graph,
               std::size_t repeats)
        : node(op.id), lhs(context.value(op.inputs.at(0))),
          rhs(context.value(op.inputs.at(1))), packed(effective_rhs(op, rhs)),
          product(graph.value(op.outputs.at(0)).spec.shape), output(product.shape()),
          expected(context.value(op.outputs.at(0))),
          alpha(attr<float>(op, "alpha", 1.0F)), beta(attr<float>(op, "beta", 1.0F)),
          relu(op.op == OpType::FusedGemmActivation), dynamic(!graph.is_constant(op.inputs.at(1))) {
        if (attr<std::int64_t>(op, "transA", 0)) lhs.transpose(0, 1);
        if (attr<std::int64_t>(op, "transB", 0)) rhs.transpose(0, 1);
        if (op.inputs.size() == 3) bias.emplace(context.value(op.inputs[2]));
        cpu::matmul_packed_simd(lhs, packed, product);
        for (std::size_t i = 0; i < repeats; ++i) epilogue_pool.emplace_back(product);
        complete();
        expect_close(output, expected);
    }
    template <typename T> static T attr(const Node& node, const char* name, T fallback) {
        const auto found = node.attributes.find(name);
        return found == node.attributes.end() ? fallback : std::get<T>(found->second);
    }
    static Tensor effective_rhs(const Node& node, const Tensor& rhs) {
        Tensor result(rhs);
        if (attr<std::int64_t>(node, "transB", 0)) result.transpose(0, 1);
        return result;
    }
    void epilogue(Tensor& target) const {
        ops::detail::apply_gemm_epilogue(target, bias ? &*bias : nullptr, alpha, beta, relu);
    }
    void complete() {
        cpu::matmul_packed_simd(lhs, packed, output);
        epilogue(output);
    }
    void reset_epilogue_pool() {
        for (auto& target : epilogue_pool)
            std::copy(product.data<float>(), product.data<float>() + product.numel(), target.data<float>());
    }
};

bool matrix_op(OpType op) {
    return op == OpType::MatMul || op == OpType::Gemm || op == OpType::FusedGemmActivation;
}

// Control executes input checks, slot/layout preparation and release, but no math.
// It is a diagnostic workload, not instrumented wall-clock attribution.
void bookkeeping(const Graph& graph, const std::vector<NodeId>& order, ExecutionContext& context) {
    context.require_all_inputs_bound();
    context.clear_intermediates();
    for (std::size_t i = 0; i < order.size(); ++i) {
        const auto& node = graph.node(order[i]);
        for (auto input : node.inputs) (void)std::as_const(context).value(input);
        if (matrix_op(node.op) && MatrixWork::attr<std::int64_t>(node, "transA", 0)) {
            const auto& lhs = std::as_const(context).value(node.inputs[0]);
            auto view = lhs.narrow(0, 0, lhs.shape()[0]);
            view.transpose(0, 1);
        }
        for (auto output : node.outputs) (void)context.prepare_output(output);
        context.release_after_step(i);
    }
}

struct Metric {
    std::string name;
    Statistics stats;
};
using Metrics = std::vector<Metric>;

struct Task {
    std::string name;
    std::function<void()> reset;
    std::function<void()> execute;
    bool pool{false};
    std::vector<double> samples;
};

Metrics measure(const Options& options, std::vector<Task>& tasks) {
    for (auto& task : tasks) for (std::size_t i = 0; i < options.warmup; ++i) {
        task.reset(); task.execute();
    }
    for (std::size_t sample = 0; sample < options.samples; ++sample) {
        for (std::size_t j = 0; j < tasks.size(); ++j) {
            auto& task = tasks[(j + sample) % tasks.size()];
            task.reset();
            const auto repeats = task.pool ? 1 : options.repeats;
            auto duration = time_repeats(task.execute, repeats);
            if (task.pool) duration /= static_cast<double>(options.repeats);
            task.samples.push_back(duration);
        }
    }
    Metrics result;
    for (auto& task : tasks) result.push_back({task.name, summarize(task.samples)});
    return result;
}

void print_metric(const Metric& metric, bool comma = true) {
    if (comma) std::cout << ",\n";
    std::cout << '"' << metric.name << "\": {\"p50_us\": " << metric.stats.p50_us
              << ", \"p95_us\": " << metric.stats.p95_us
              << ", \"mean_us\": " << metric.stats.mean_us << '}';
}

const char* compiler_name() {
#if defined(__clang__)
    return "Clang " __clang_version__;
#elif defined(_MSC_VER)
    return "MSVC " TINYINFER_COMPILER_VERSION;
#else
    return "GCC " __VERSION__;
#endif
}
std::string escape(const std::string& source) {
    std::string out;
    for (char c : source) {
        if (c == '\\' || c == '"') out += '\\';
        if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else out += c;
    }
    return out;
}

int run_fusion_comparison(const Options& options) {
    const auto model = make_model(options);
    const auto inputs = make_inputs(model);
    std::vector<CpuGemmEpilogueMode> modes{
        CpuGemmEpilogueMode::Legacy, CpuGemmEpilogueMode::Specialized,
        CpuGemmEpilogueMode::Fused, CpuGemmEpilogueMode::Auto};
    // Timing order already rotates per sample. Also balance which mode gets
    // each separately allocated model/packed-weight/context position across
    // processes: identical kernels can otherwise inherit fixed buffer bias.
    std::rotate(modes.begin(), modes.begin() + options.allocation_order, modes.end());
    std::vector<CpuExecutionPlan> plans;
    std::vector<CpuExecutionContext> sessions;
    plans.reserve(4); sessions.reserve(4);
    for (auto mode : modes) {
        plans.emplace_back(model, CpuExecutionPlanOptions{mode});
        sessions.push_back(plans.back().create_context());
        bind(sessions.back(), inputs); sessions.back().run();
    }
    // Restore canonical metric/counter indices without copying model payloads.
    std::rotate(plans.begin(), plans.end() - options.allocation_order, plans.end());
    std::rotate(sessions.begin(), sessions.end() - options.allocation_order, sessions.end());
    for (const auto& out : model.outputs()) for (std::size_t i=1;i<4;++i)
        expect_close(sessions[i].output(out.second), sessions[0].output(out.second));
    std::vector<Task> tasks;
    const std::vector<std::string> names{"legacy", "specialized", "fused", "auto"};
    for (std::size_t i=0;i<4;++i) tasks.push_back({names[i], []{}, [&,i]{sessions[i].run();}});
    const auto metrics = measure(options, tasks);
    std::vector<double> specialized, fused, incremental, automatic;
    for (std::size_t i=0;i<options.samples;++i) {
        specialized.push_back(tasks[0].samples[i]/tasks[1].samples[i]);
        fused.push_back(tasks[0].samples[i]/tasks[2].samples[i]);
        incremental.push_back(tasks[1].samples[i]/tasks[2].samples[i]);
        automatic.push_back(tasks[1].samples[i]/tasks[3].samples[i]);
    }
    for (const auto& out : model.outputs()) for (std::size_t i=1;i<4;++i)
        expect_close(sessions[i].output(out.second), sessions[0].output(out.second));
    for (std::size_t i=0;i<4;++i) {
        if (plans[i].activation_bytes()!=plans[0].activation_bytes() || plans[i].pack_count()!=plans[0].pack_count())
            throw std::runtime_error("comparison memory/packing policy changed");
        const auto dynamic = std::count_if(plans[i].steps().begin(),plans[i].steps().end(),[](const auto& step){
            return matrix_op(step.op) && step.path==CpuExecutionPath::ExistingKernel;
        });
        if (sessions[i].runtime_pack_count()!=sessions[i].run_count()*dynamic)
            throw std::runtime_error("comparison runtime packing differs");
        // Benchmark inputs/constants are contiguous; strided runtime rebinding is
        // covered by the kernel integration tests rather than this timing suite.
        std::size_t eligible = 0, expected_fused = 0;
        if (i != 0) for (const auto& step : plans[i].steps()) {
            if (step.path == CpuExecutionPath::PackedConstantRhs && step.op != OpType::MatMul &&
                ((i == 2 || (i == 3 && step.epilogue_mode == CpuGemmEpilogueMode::Fused))
                    ? step.epilogue.supports_fusion() : step.epilogue.supports_specialization()))
                ++eligible;
            if (step.path == CpuExecutionPath::PackedConstantRhs && step.op != OpType::MatMul &&
                step.epilogue_mode == CpuGemmEpilogueMode::Fused && step.epilogue.supports_fusion())
                ++expected_fused;
        }
        const auto selected = i == 2 ? sessions[i].fused_gemm_count()
            : sessions[i].specialized_gemm_count() + sessions[i].fused_gemm_count();
        if (selected != eligible * sessions[i].run_count())
            throw std::runtime_error("comparison epilogue dispatch differs");
        if (sessions[i].fused_gemm_count() != expected_fused * sessions[i].run_count())
            throw std::runtime_error("comparison selected fusion policy differs");
    }
    std::cout << std::fixed << std::setprecision(6)
              << "{\n\"build_type\": \"" << TINYINFER_BUILD_TYPE
              << "\",\n\"compiler\": \"" << escape(compiler_name())
              << "\",\n\"native_arch\": " << TINYINFER_NATIVE_ARCH
              << ",\n\"simd_width\": " << cpu::matmul_simd_width()
              << ",\n\"workload\": \"" << escape(options.model_path.empty()?"synthetic_gemm":options.model_path)
              << "\",\n\"samples\": " << options.samples << ",\n\"repeats\": " << options.repeats
              << ",\n\"warmup\": " << options.warmup << ",\n\"pack_count\": " << plans[0].pack_count()
              << ",\n\"allocation_order\": " << options.allocation_order
              << ",\n\"activation_bytes\": " << plans[0].activation_bytes()
              << ",\n\"packed_weight_bytes\": " << plans[0].packed_weight_bytes();
    if (options.model_path.empty()) {
        std::cout << ",\n\"m\": " << options.m << ",\n\"k\": " << options.k << ",\n\"n\": " << options.n
                  << ",\n\"trans_b\": " << options.transpose_b << ",\n\"dynamic_b\": " << options.dynamic_b
                  << ",\n\"bias\": \"" << options.bias << "\",\n\"relu\": " << options.relu;
    }
    for (const auto& metric : metrics) print_metric(metric);
    std::cout << ",\n\"specialized_vs_legacy_paired_p50\": " << summarize(specialized).p50_us
              << ",\n\"fused_vs_legacy_paired_p50\": " << summarize(fused).p50_us
              << ",\n\"fused_vs_specialized_paired_p50\": " << summarize(incremental).p50_us
              << ",\n\"auto_vs_specialized_paired_p50\": " << summarize(automatic).p50_us
              << ",\n\"auto_fused_gemm_count\": " << sessions[3].fused_gemm_count()
              << ",\n\"auto_specialized_gemm_count\": " << sessions[3].specialized_gemm_count()
              << ",\n\"fused_gemm_count\": " << sessions[2].fused_gemm_count()
              << ",\n\"specialized_gemm_count\": " << sessions[1].specialized_gemm_count()
              << ",\n\"runtime_pack_count\": " << sessions[2].runtime_pack_count()
              << ",\n\"run_count\": " << sessions[2].run_count() << "\n}\n";
    return 0;
}

int run(const Options& options) {
    const auto model = make_model(options);
    const auto& graph = model.graph();
    const auto inputs = make_inputs(model);
    const auto order = graph.topological_order();
    cpu::OperatorRegistry registry;
    ExecutionContext captured(graph); // No plan: retain all outputs for capture.
    bind(captured, inputs);
    for (auto id : order) registry.execute(graph.node(id), captured);
    const CpuExecutionPlan plan(model, {CpuGemmEpilogueMode::Legacy});
    auto session = plan.create_context();
    bind(session, inputs);
    session.run();
    for (const auto& output : model.outputs())
        expect_close(session.output(output.second), captured.output(output.second));

    std::vector<std::unique_ptr<MatrixWork>> matrices;
    for (auto id : order) if (matrix_op(graph.node(id).op))
        matrices.push_back(std::make_unique<MatrixWork>(graph.node(id), captured, graph, options.repeats));

    const MemoryPlan memory(graph);
    ExecutionContext control(graph, memory);
    bind(control, inputs);
    std::size_t checksum = 0;
    std::vector<Task> whole{
        {"prepared_run", [] {}, [&] { session.run(); }},
        {"context_bookkeeping_control", [] {}, [&] { bookkeeping(graph, order, control); }},
        {"captured_matrix_chain", [] {}, [&] { for (auto& matrix : matrices) matrix->complete(); }},
        {"input_binding_copy", [] {}, [&] { bind(session, inputs); }},
        {"output_extraction_copy", [] {}, [&] {
            for (const auto& out : model.outputs()) { auto copy = session.output(out.second); checksum += copy.numel(); }
        }},
        {"clock_loop_floor", [] {}, [] {}}
    };
    auto metrics = measure(options, whole);
    session.run();
    if (session.runtime_pack_count() != session.run_count() * std::count_if(
            matrices.begin(), matrices.end(), [](const auto& work) { return work->dynamic; }))
        throw std::runtime_error("runtime packing count differs");
    for (const auto& out : model.outputs())
        expect_close(session.output(out.second), captured.output(out.second));

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "{\n\"build_type\": \"" << TINYINFER_BUILD_TYPE
              << "\",\n\"compiler\": \"" << escape(compiler_name())
              << "\",\n\"native_arch\": " << TINYINFER_NATIVE_ARCH
              << ",\n\"simd_width\": " << cpu::matmul_simd_width()
              << ",\n\"workload\": \"" << escape(options.model_path.empty() ? "synthetic_gemm" : options.model_path)
              << "\",\n\"warmup\": " << options.warmup << ",\n\"samples\": " << options.samples
              << ",\n\"repeats\": " << options.repeats << ",\n\"graph_nodes\": " << graph.size()
              << ",\n\"pack_count\": " << plan.pack_count()
              << ",\n\"runtime_pack_count\": " << session.runtime_pack_count()
              << ",\n\"run_count\": " << session.run_count()
              << ",\n\"activation_bytes\": " << plan.activation_bytes()
              << ",\n\"packed_weight_bytes\": " << plan.packed_weight_bytes();
    if (options.model_path.empty()) {
        std::cout << ",\n\"m\": " << options.m << ",\n\"k\": " << options.k << ",\n\"n\": " << options.n
                  << ",\n\"trans_b\": " << options.transpose_b << ",\n\"dynamic_b\": " << options.dynamic_b
                  << ",\n\"bias\": \"" << options.bias << "\",\n\"relu\": " << options.relu;
    }
    for (const auto& metric : metrics) print_metric(metric);
    std::cout << ",\n\"matrix_nodes\": [";
    for (std::size_t i = 0; i < matrices.size(); ++i) {
        auto& work = *matrices[i];
        std::vector<Task> tasks{
            {"matmul", [] {}, [&] { cpu::matmul_packed_simd(work.lhs, work.packed, work.output); }},
            {"epilogue", [&] { work.reset_epilogue_pool(); }, [&] {
                for (auto& output : work.epilogue_pool) work.epilogue(output);
            }, true},
            {"complete", [] {}, [&] { work.complete(); }}
        };
        const auto tail = static_cast<std::size_t>(work.output.shape()[1]) % cpu::matmul_simd_width();
        std::optional<Tensor> tail_rhs, tail_output;
        if (tail) {
            const auto columns = work.rhs.shape()[1];
            tail_rhs.emplace(work.rhs.narrow(1, columns - static_cast<std::int64_t>(tail), tail));
            tail_output.emplace(Shape{work.lhs.shape()[0], static_cast<std::int64_t>(tail)});
            tasks.push_back({"scalar_tail_control", [] {}, [&] {
                cpu::matmul_strided(work.lhs, *tail_rhs, *tail_output);
            }});
        }
        auto node_metrics = measure(options, tasks);
        work.complete(); expect_close(work.output, work.expected);
        for (const auto& output : work.epilogue_pool) {
            try { expect_close(output, work.expected); }
            catch (const std::exception& error) { throw std::runtime_error(std::string("epilogue pool: ") + error.what()); }
        }
        if (tail) {
            const auto expected_view = work.product.narrow(1, work.product.shape()[1] - static_cast<std::int64_t>(tail), tail);
            const Tensor expected(expected_view); // Deep copy for linear-index comparison.
            try { expect_close(*tail_output, expected); }
            catch (const std::exception& error) { throw std::runtime_error(std::string("scalar tail control: ") + error.what()); }
        }
        std::cout << (i ? ",\n" : "\n") << "{\"node\": " << work.node
                  << ", \"name\": \"" << escape(graph.node(work.node).name) << '\"'
                  << ", \"m\": " << work.lhs.shape()[0] << ", \"k\": " << work.lhs.shape()[1]
                  << ", \"n\": " << work.output.shape()[1] << ", \"tail_columns\": " << tail
                  << ", \"dynamic_b\": " << work.dynamic << ", \"alpha\": " << work.alpha
                  << ", \"beta\": " << work.beta << ", \"relu\": " << work.relu
                  << ", \"bias_elements\": " << (work.bias ? work.bias->numel() : 0)
                  << ", \"epilogue_pool_bytes\": " << work.product.size_bytes() * options.repeats;
        for (const auto& metric : node_metrics) print_metric(metric);
        std::cout << '}';
    }
    // Ordinary nodes are measured in their own one-node, planned contexts.
    // This includes registry lookup/output slot setup, not pure arithmetic only.
    std::cout << "\n],\n\"ordinary_nodes\": [";
    std::size_t ordinary_count = 0;
    for (auto id : order) {
        const auto& original = graph.node(id);
        if (matrix_op(original.op)) continue;
        Graph isolated;
        std::vector<ValueId> operands;
        for (auto input : original.inputs)
            operands.push_back(isolated.add_constant("operand" + std::to_string(operands.size()), captured.value(input)));
        const auto node_id = isolated.add_node(original.name, original.op, operands, original.attributes);
        const auto output_id = isolated.node(node_id).outputs.at(0);
        isolated.mark_output(output_id);
        ExecutionContext context(isolated, MemoryPlan(isolated));
        std::vector<Task> task{{"registry_dispatch", [] {}, [&] { registry.execute(isolated.node(node_id), context); }}};
        const auto measured = measure(options, task);
        expect_close(context.output(output_id), captured.value(original.outputs.at(0)));
        std::cout << (ordinary_count++ ? ",\n" : "\n") << "{\"node\": " << id
                  << ", \"name\": \"" << escape(original.name) << '\"'
                  << ", \"op\": " << static_cast<int>(original.op);
        for (const auto& metric : measured) print_metric(metric);
        std::cout << '}';
    }
    std::cout << "\n],\n\"diagnostic_checksum\": " << checksum << "\n}\n";
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    try { const auto options=parse_options(argc, argv);
        return options.fusion_comparison ? run_fusion_comparison(options) : run(options); }
    catch (const std::exception& error) {
        std::cerr << "cpu performance analysis: " << error.what() << '\n';
        return 1;
    }
}
