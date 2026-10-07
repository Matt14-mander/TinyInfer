#include "tinyinfer/tinyinfer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <filesystem>
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
    std::string format{"text"};
    bool transpose_b{true};
    bool dynamic_b{false};
    std::string model_path;
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
                         " [--samples N] [--repeats N] [--format text|json] [--trans-b 0|1] [--dynamic-b] [--model PATH]\n";
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
        } else if (argument == "--model") {
            options.model_path = value("--model");
        } else if (argument == "--format") {
            options.format = value("--format");
            if (options.format != "text" && options.format != "json") {
                throw std::invalid_argument("--format must be text or json");
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
    const auto bias = graph.add_constant("c", Tensor::from_vector(
        {n}, generated_values(options.n, 11)));
    const auto node = graph.add_node("affine_relu", OpType::FusedGemmActivation,
        {x, b, bias}, {{"activation", std::string("relu")},
                      {"transB", std::int64_t{options.transpose_b}}});
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
            throw std::runtime_error("prepared output value changed");
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

template <typename Function>
Statistics measure_stage(const Options& options, Function&& function) {
    function();
    std::vector<double> samples;
    for (std::size_t i = 0; i < std::min<std::size_t>(20, options.samples); ++i) {
        samples.push_back(time_repeats(function, 1));
    }
    return summarize(std::move(samples));
}

struct Report {
    std::map<std::string, double> metrics;
    void add(const std::string& name, Statistics value) {
        metrics[name + "_p50_us"] = value.p50_us;
        metrics[name + "_p95_us"] = value.p95_us;
        metrics[name + "_mean_us"] = value.mean_us;
    }
};

Report benchmark(const Options& options, const Model& model, const Bindings& inputs) {
    Report report;
    const auto& graph = model.graph();
    const MemoryPlan memory(graph);
    const CpuExecutionPlan prepared(model);
    CpuBackend backend;
    Executor executor(backend);
    ExecutionContext baseline(graph, memory);
    auto session = prepared.create_context();
    bind(baseline, inputs);
    bind(session, inputs);
    executor.run(graph, baseline);
    session.run();
    for (const auto& out : model.outputs()) {
        expect_close(session.output(out.second), baseline.output(out.second));
    }

    report.metrics["pack_count"] = static_cast<double>(prepared.pack_count());
    report.metrics["packed_weight_bytes"] = static_cast<double>(prepared.packed_weight_bytes());
    report.metrics["activation_bytes"] = static_cast<double>(prepared.activation_bytes());
    report.metrics["graph_nodes"] = static_cast<double>(graph.size());
    report.metrics["stage_samples"] = static_cast<double>(std::min<std::size_t>(20, options.samples));
    std::size_t constant_bytes = 0;
    for (const auto& value : graph.values()) {
        if (graph.is_constant(value.id)) {
            constant_bytes += TensorLayout(value.spec.shape).size_bytes(value.spec.dtype);
        }
    }
    std::size_t input_bytes = 0;
    for (const auto& input : inputs) {
        input_bytes += TensorLayout(input.second.shape()).size_bytes(input.second.dtype());
    }
    report.metrics["snapshot_constant_bytes"] = static_cast<double>(constant_bytes);
    report.metrics["context_constant_bytes"] = static_cast<double>(constant_bytes);
    report.metrics["context_input_bytes"] = static_cast<double>(input_bytes);
    report.metrics["plan_accounted_payload_bytes"] =
        static_cast<double>(constant_bytes + prepared.packed_weight_bytes());
    report.metrics["context_accounted_payload_bytes"] =
        static_cast<double>(constant_bytes + input_bytes + prepared.activation_bytes());
    std::size_t scratch_result = 0;
    report.add("model_snapshot", measure_stage(options, [&] {
        const Model candidate(model);
        scratch_result += candidate.graph().value_count();
    }));
    report.add("topological_order", measure_stage(options, [&] {
        const auto order = graph.topological_order();
        scratch_result += order.size();
    }));
    report.add("baseline_memory_plan", measure_stage(options, [&] {
        const MemoryPlan candidate(graph);
        scratch_result += candidate.buffer_size_bytes();
    }));
    report.add("preparation", measure_stage(options, [&] {
        const CpuExecutionPlan candidate(model);
        scratch_result += candidate.packed_weight_bytes();
    }));
    report.add("baseline_context_init", measure_stage(options, [&] {
        const ExecutionContext candidate(graph, memory);
        scratch_result += candidate.graph().value_count();
    }));
    report.add("prepared_context_init", measure_stage(options, [&] {
        const auto candidate = prepared.create_context();
        scratch_result += candidate.run_count();
    }));

    // Component measurements: model-wide RHS packing and RHS transpose copies.
    // They are diagnostics; their medians are not added to predict warm latency.
    std::vector<Tensor> effective_rhs;
    std::vector<const Tensor*> transposed_sources;
    for (const auto& node : graph.nodes()) {
        if (node.op != OpType::MatMul && node.op != OpType::Gemm &&
            node.op != OpType::FusedGemmActivation) continue;
        const auto id = node.inputs.at(1);
        const Tensor* rhs = nullptr;
        if (graph.is_constant(id)) rhs = &graph.constant(id);
        else for (const auto& input : inputs) if (input.first == id) rhs = &input.second;
        if (!rhs) throw std::invalid_argument("benchmark requires constant or input matrix RHS");
        const auto found = node.attributes.find("transB");
        const bool transposed = found != node.attributes.end() && std::get<std::int64_t>(found->second) == 1;
        effective_rhs.push_back(*rhs);
        if (transposed) {
            transposed_sources.push_back(rhs);
            effective_rhs.back().transpose(0, 1);
        }
    }
    report.add("baseline_rhs_pack", measure_stage(options, [&] {
        for (const auto& rhs : effective_rhs) {
            const cpu::PackedMatMulRhs candidate(rhs);
            scratch_result += candidate.size_bytes();
        }
    }));
    report.add("baseline_rhs_transpose_copy", measure_stage(options, [&] {
        for (const auto* rhs : transposed_sources) {
            auto candidate = *rhs;
            candidate.transpose(0, 1);
            scratch_result += candidate.numel();
        }
    }));

    std::vector<double> first_before, first_after;
    for (std::size_t i = 0; i < std::min<std::size_t>(20, options.samples); ++i) {
        ExecutionContext fresh_before(graph, memory);
        auto fresh_after = prepared.create_context();
        bind(fresh_before, inputs);
        bind(fresh_after, inputs);
        const auto run_before = [&] { executor.run(graph, fresh_before); };
        const auto run_after = [&] { fresh_after.run(); };
        if (i % 2 == 0) {
            first_before.push_back(time_repeats(run_before, 1));
            first_after.push_back(time_repeats(run_after, 1));
        } else {
            first_after.push_back(time_repeats(run_after, 1));
            first_before.push_back(time_repeats(run_before, 1));
        }
    }
    report.add("baseline_first_inference", summarize(std::move(first_before)));
    report.add("prepared_first_inference", summarize(std::move(first_after)));

    const auto run_before = [&] { executor.run(graph, baseline); };
    const auto run_after = [&] { session.run(); };
    for (std::size_t i = 0; i < options.warmup; ++i) { run_before(); run_after(); }
    std::vector<double> before_samples, after_samples;
    for (std::size_t i = 0; i < options.samples; ++i) {
        if (i % 2 == 0) {
            before_samples.push_back(time_repeats(run_before, options.repeats));
            after_samples.push_back(time_repeats(run_after, options.repeats));
        } else {
            after_samples.push_back(time_repeats(run_after, options.repeats));
            before_samples.push_back(time_repeats(run_before, options.repeats));
        }
    }
    std::vector<double> paired_ratios;
    for (std::size_t i = 0; i < before_samples.size(); ++i) {
        paired_ratios.push_back(before_samples[i] / after_samples[i]);
    }
    const auto paired = summarize(std::move(paired_ratios));
    report.metrics["paired_speedup_p50"] = paired.p50_us;
    report.metrics["paired_speedup_p95"] = paired.p95_us;
    report.metrics["paired_speedup_mean"] = paired.mean_us;
    const auto before = summarize(std::move(before_samples));
    const auto after = summarize(std::move(after_samples));
    report.add("baseline_warm", before);
    report.add("prepared_warm", after);
    report.metrics["p50_speedup"] = before.p50_us / after.p50_us;
    report.metrics["runtime_pack_count"] = static_cast<double>(session.runtime_pack_count());
    report.metrics["session_run_count"] = static_cast<double>(session.run_count());
    report.metrics["diagnostic_checksum"] = static_cast<double>(scratch_result);
    const auto savings = before.mean_us - after.mean_us;
    if (savings > 0) {
        // Conservative estimate charging the entire preparation cost, not just
        // its difference from a baseline MemoryPlan. Not an end-to-end timer.
        report.metrics["full_preparation_amortization_calls"] =
            std::ceil(report.metrics.at("preparation_mean_us") / savings);
    }
    for (const auto& out : model.outputs()) {
        expect_close(session.output(out.second), baseline.output(out.second));
    }
    std::size_t dynamic_matrix_steps = 0;
    for (const auto& step : prepared.steps()) {
        if (step.path == CpuExecutionPath::ExistingKernel &&
            (step.op == OpType::MatMul || step.op == OpType::Gemm ||
             step.op == OpType::FusedGemmActivation)) ++dynamic_matrix_steps;
    }
    if (session.runtime_pack_count() != dynamic_matrix_steps * session.run_count()) {
        throw std::runtime_error("unexpected runtime weight packing count");
    }
    return report;
}

const char* compiler_name() {
#if defined(__clang__)
    return "Clang " __clang_version__;
#elif defined(_MSC_VER)
    return "MSVC";
#elif defined(__GNUC__)
    return "GCC " __VERSION__;
#else
    return "unknown";
#endif
}

std::string escape(const std::string& input) {
    std::string result;
    for (char c : input) {
        if (c == '\\' || c == '"') result += '\\';
        if (c == '\n') result += "\\n";
        else if (c == '\r') result += "\\r";
        else if (c == '\t') result += "\\t";
        else result += c;
    }
    return result;
}

void print(const Options& options, const Report& report) {
    std::cout << std::fixed << std::setprecision(6);
    const auto workload = options.model_path.empty() ? "synthetic_fused_gemm" : options.model_path;
    if (options.format == "json") {
        std::cout << "{\n  \"build_type\": \"" << TINYINFER_BUILD_TYPE
                  << "\",\n  \"compiler\": \"" << escape(compiler_name())
                  << "\",\n  \"native_arch\": " << TINYINFER_NATIVE_ARCH
                  << ",\n  \"workload\": \"" << escape(workload)
                  << "\"";
        if (options.model_path.empty()) {
            std::cout << ",\n  \"m\": " << options.m
                  << ",\n  \"k\": " << options.k
                  << ",\n  \"n\": " << options.n
                  << ",\n  \"trans_b\": " << options.transpose_b
                  << ",\n  \"dynamic_b\": " << options.dynamic_b;
        }
        std::cout << ",\n  \"warmup\": " << options.warmup
                  << ",\n  \"samples\": " << options.samples
                  << ",\n  \"repeats\": " << options.repeats;
        for (const auto& item : report.metrics) {
            std::cout << ",\n  \"" << item.first << "\": " << item.second;
        }
        std::cout << "\n}\n";
    } else {
        std::cout << "Phase 4.4 prepared CPU inference\nworkload: " << workload
                  << "\nbuild: " << TINYINFER_BUILD_TYPE
                  << "\ncompiler: " << compiler_name() << '\n';
        for (const auto& item : report.metrics) std::cout << item.first << ": " << item.second << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse_options(argc, argv);
        const auto model = make_model(options);
        const auto inputs = make_inputs(model);
        print(options, benchmark(options, model, inputs));
    } catch (const std::exception& error) {
        std::cerr << "tinyinfer_cpu_execution_plan_benchmark: " << error.what() << '\n';
        return 1;
    }
}
