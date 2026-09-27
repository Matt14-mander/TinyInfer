#include "tinyinfer/tinyinfer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
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
    std::size_t warmup{20};
    std::size_t samples{200};
    std::size_t repeats{100};
    std::string format{"text"};
};

struct Statistics {
    double p50_us{};
    double p95_us{};
    double mean_us{};
};

struct PairResult {
    Statistics before;
    Statistics after;
};

std::size_t positive_size(std::string_view input, const char* name) {
    std::size_t consumed = 0;
    std::size_t result = 0;
    try {
        result = std::stoull(std::string(input), &consumed);
    } catch (const std::exception&) {
        throw std::invalid_argument(std::string(name) + " requires a positive integer");
    }
    if (consumed != input.size() || result == 0) {
        throw std::invalid_argument(std::string(name) + " requires a positive integer");
    }
    return result;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto value = [&](const char* name) -> std::string_view {
            if (index + 1 >= argc) {
                throw std::invalid_argument(std::string(name) + " requires a value");
            }
            return argv[++index];
        };
        if (argument == "--help") {
            std::cout << "Usage: " << argv[0]
                      << " [--warmup N] [--samples N] [--repeats N]"
                         " [--format text|json]\n";
            std::exit(0);
        } else if (argument == "--warmup") {
            options.warmup = positive_size(value("--warmup"), "--warmup");
        } else if (argument == "--samples") {
            options.samples = positive_size(value("--samples"), "--samples");
        } else if (argument == "--repeats") {
            options.repeats = positive_size(value("--repeats"), "--repeats");
        } else if (argument == "--format") {
            options.format = value("--format");
            if (options.format != "text" && options.format != "json") {
                throw std::invalid_argument("--format must be text or json");
            }
        } else {
            throw std::invalid_argument("unknown option: " + std::string(argument));
        }
    }
    return options;
}

const tinyinfer::Tensor& constant_named(const tinyinfer::Model& model,
                                       std::string_view name) {
    for (const auto& value : model.graph().values()) {
        if (value.name == name && model.graph().is_constant(value.id)) {
            return model.graph().constant(value.id);
        }
    }
    throw std::runtime_error("missing ONNX fixture constant: " + std::string(name));
}

tinyinfer::Model make_optimization_fixture(const tinyinfer::Model& imported) {
    using namespace tinyinfer;
    Graph graph;
    const auto x = graph.add_input(
        "x", imported.graph().value(imported.input_id("x")).spec);
    const auto w1 = graph.add_constant("w1", constant_named(imported, "w1"));
    const auto b1 = graph.add_constant("b1", constant_named(imported, "b1"));
    const auto w2 = graph.add_constant("w2", constant_named(imported, "w2"));
    const auto b2 = graph.add_constant("b2", constant_named(imported, "b2"));
    const auto zero3 = graph.add_constant(
        "zero3", Tensor::from_vector({3}, {0.0F, 0.0F, 0.0F}));
    const auto zero2 = graph.add_constant(
        "zero2", Tensor::from_vector({2}, {0.0F, 0.0F}));
    const auto output_of = [&](NodeId id) { return graph.node(id).outputs.front(); };

    const auto b1_sum = output_of(graph.add_node("b1_plus_zero", OpType::Add,
                                               {b1, zero3}));
    const auto b1_ready = output_of(graph.add_node(
        "b1_relu", OpType::ReLU, {b1_sum}));
    const auto b2_ready = output_of(graph.add_node(
        "b2_plus_zero", OpType::Add, {b2, zero2}));
    const auto hidden_pre = output_of(graph.add_node(
        "linear1", OpType::Gemm, {x, w1, b1_ready},
        {{"transB", std::int64_t{1}}}));
    const auto hidden = output_of(graph.add_node(
        "relu", OpType::ReLU, {hidden_pre}));
    const auto logits = output_of(graph.add_node(
        "linear2", OpType::Gemm, {hidden, w2, b2_ready},
        {{"transB", std::int64_t{1}}}));
    const auto probabilities = output_of(graph.add_node(
        "softmax", OpType::Softmax, {logits},
        {{"axis", std::int64_t{-1}}}));

    // A disconnected model head makes output-driven DCE observable.
    const auto dead_pre = output_of(graph.add_node(
        "unused_linear1", OpType::Gemm, {x, w1, b1},
        {{"transB", std::int64_t{1}}}));
    const auto dead_hidden = output_of(graph.add_node(
        "unused_relu", OpType::ReLU, {dead_pre}));
    const auto dead_logits = output_of(graph.add_node(
        "unused_linear2", OpType::Gemm, {dead_hidden, w2, b2},
        {{"transB", std::int64_t{1}}}));
    graph.add_node("unused_softmax", OpType::Softmax, {dead_logits},
                   {{"axis", std::int64_t{-1}}});

    graph.mark_output(probabilities);
    return Model(std::move(graph), {{"x", x}},
                 {{"probabilities", probabilities}});
}

tinyinfer::optimizer::PassManager make_pipeline() {
    using namespace tinyinfer::optimizer;
    PassManager pipeline;
    pipeline.add_pass(std::make_unique<ConstantFoldingPass>());
    pipeline.add_pass(std::make_unique<DeadCodeEliminationPass>());
    return pipeline;
}

tinyinfer::Tensor run_once(const tinyinfer::Model& model,
                           const std::vector<float>& input) {
    tinyinfer::ExecutionContext context(model.graph());
    context.bind_input(model.input_id("x"),
                       tinyinfer::Tensor::from_vector({1, 2}, input));
    tinyinfer::CpuBackend backend;
    tinyinfer::Executor executor(backend);
    executor.run(model.graph(), context);
    return context.output(model.output_id("probabilities"));
}

void expect_close(const tinyinfer::Tensor& actual,
                  const tinyinfer::Tensor& expected) {
    if (actual.shape() != expected.shape() || actual.dtype() != expected.dtype()) {
        throw std::runtime_error("Phase 4.2 changed the model output spec");
    }
    for (std::size_t i = 0; i < actual.numel(); ++i) {
        if (std::fabs(actual.at(i) - expected.at(i)) > 1e-5F) {
            throw std::runtime_error("Phase 4.2 changed the model output value");
        }
    }
}

void verify(const tinyinfer::Model& imported,
            const tinyinfer::optimizer::OptimizationResult& control,
            const tinyinfer::Model& before,
            const tinyinfer::optimizer::OptimizationResult& optimized) {
    if (control.changed() || before.graph().size() != 11 ||
        optimized.model.graph().size() != 4 ||
        optimized.pass_statistics.size() != 2 ||
        optimized.pass_statistics[0].nodes_rewritten != 3 ||
        optimized.pass_statistics[1].nodes_rewritten != 4 ||
        optimized.model.inputs() != before.inputs() ||
        optimized.model.outputs().front().first != before.outputs().front().first ||
        optimized.mapped_value(before.output_id("probabilities")) !=
            optimized.model.output_id("probabilities")) {
        throw std::runtime_error("Phase 4.2 graph acceptance check failed");
    }
    for (const auto& input : std::vector<std::vector<float>>{
             {1.0F, -2.0F}, {-0.5F, 0.25F}, {2.0F, 1.0F}}) {
        const auto reference = run_once(imported, input);
        expect_close(run_once(control.model, input), reference);
        expect_close(run_once(before, input), reference);
        expect_close(run_once(optimized.model, input), reference);
    }
    auto pipeline = make_pipeline();
    if (pipeline.run(optimized.model).changed()) {
        throw std::runtime_error("Phase 4.2 pipeline is not stable");
    }
}

double percentile(const std::vector<double>& sorted, double p) {
    const double position = p * static_cast<double>(sorted.size() - 1);
    const auto lower = static_cast<std::size_t>(position);
    const auto upper = std::min(lower + 1, sorted.size() - 1);
    return sorted[lower] + (sorted[upper] - sorted[lower]) *
                               (position - static_cast<double>(lower));
}

Statistics summarize(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    const double mean = std::accumulate(samples.begin(), samples.end(), 0.0) /
                        static_cast<double>(samples.size());
    return {percentile(samples, 0.5), percentile(samples, 0.95), mean};
}

template <typename Function>
double time_repeats(Function&& function, std::size_t repeats) {
    const auto start = Clock::now();
    for (std::size_t i = 0; i < repeats; ++i) function();
    return std::chrono::duration<double, std::micro>(Clock::now() - start)
               .count() / static_cast<double>(repeats);
}

PairResult measure_pair(const Options& options,
                        const tinyinfer::Model& before,
                        const tinyinfer::Model& after) {
    tinyinfer::CpuBackend backend;
    tinyinfer::Executor executor(backend);
    tinyinfer::ExecutionContext before_context(before.graph());
    tinyinfer::ExecutionContext after_context(after.graph());
    const auto input = tinyinfer::Tensor::from_vector(
        {1, 2}, {1.0F, -2.0F});
    before_context.bind_input(before.input_id("x"), input);
    after_context.bind_input(after.input_id("x"), input);
    const auto run_before = [&] {
        executor.run(before.graph(), before_context);
        static_cast<void>(before_context.output(before.output_id("probabilities")));
    };
    const auto run_after = [&] {
        executor.run(after.graph(), after_context);
        static_cast<void>(after_context.output(after.output_id("probabilities")));
    };
    for (std::size_t i = 0; i < options.warmup; ++i) {
        run_before();
        run_after();
    }
    std::vector<double> before_samples;
    std::vector<double> after_samples;
    before_samples.reserve(options.samples);
    after_samples.reserve(options.samples);
    for (std::size_t i = 0; i < options.samples; ++i) {
        if (i % 2 == 0) {
            before_samples.push_back(time_repeats(run_before, options.repeats));
            after_samples.push_back(time_repeats(run_after, options.repeats));
        } else {
            after_samples.push_back(time_repeats(run_after, options.repeats));
            before_samples.push_back(time_repeats(run_before, options.repeats));
        }
    }
    return {summarize(std::move(before_samples)),
            summarize(std::move(after_samples))};
}

const char* platform_name() {
#if defined(__APPLE__)
    return "macOS";
#elif defined(__linux__)
    return "Linux";
#else
    return "other";
#endif
}

const char* architecture_name() {
#if defined(__aarch64__)
    return "arm64";
#elif defined(__x86_64__)
    return "x86_64";
#else
    return "other";
#endif
}

void print_results(const Options& options, const PairResult& control,
                   const PairResult& result,
                   const tinyinfer::Model& before,
                   const tinyinfer::Model& after) {
    const double speedup = result.before.p50_us / result.after.p50_us;
    if (options.format == "json") {
        std::cout << std::fixed << std::setprecision(6)
                  << "{\n  \"build_type\": \"" << TINYINFER_BUILD_TYPE
                  << "\",\n  \"platform\": \"" << platform_name()
                  << "\",\n  \"architecture\": \"" << architecture_name()
                  << "\",\n  \"warmup\": " << options.warmup
                  << ",\n  \"samples\": " << options.samples
                  << ",\n  \"repeats\": " << options.repeats
                  << ",\n  \"nodes_before\": " << before.graph().size()
                  << ",\n  \"nodes_after\": " << after.graph().size()
                  << ",\n  \"values_before\": " << before.graph().value_count()
                  << ",\n  \"values_after\": " << after.graph().value_count()
                  << ",\n  \"before_p50_us\": " << result.before.p50_us
                  << ",\n  \"before_p95_us\": " << result.before.p95_us
                  << ",\n  \"before_mean_us\": " << result.before.mean_us
                  << ",\n  \"after_p50_us\": " << result.after.p50_us
                  << ",\n  \"after_p95_us\": " << result.after.p95_us
                  << ",\n  \"after_mean_us\": " << result.after.mean_us
                  << ",\n  \"control_before_p50_us\": " << control.before.p50_us
                  << ",\n  \"control_after_p50_us\": " << control.after.p50_us
                  << ",\n  \"p50_speedup\": " << speedup << "\n}\n";
        return;
    }
    std::cout << "Phase 4.2 graph optimization benchmark\n"
              << "build: " << TINYINFER_BUILD_TYPE << '\n'
              << "platform: " << platform_name() << '\n'
              << "architecture: " << architecture_name() << '\n'
              << "warmup: " << options.warmup << '\n'
              << "samples: " << options.samples << '\n'
              << "repeats: " << options.repeats << '\n'
              << "nodes: " << before.graph().size() << " -> "
              << after.graph().size() << '\n'
              << "values: " << before.graph().value_count() << " -> "
              << after.graph().value_count() << "\n\n"
              << std::fixed << std::setprecision(3)
              << "warm inference (us)       p50       p95      mean\n"
              << "before                  " << std::setw(8)
              << result.before.p50_us << std::setw(10)
              << result.before.p95_us << std::setw(10)
              << result.before.mean_us << '\n'
              << "after                   " << std::setw(8)
              << result.after.p50_us << std::setw(10)
              << result.after.p95_us << std::setw(10)
              << result.after.mean_us << '\n'
              << "p50 speedup: " << speedup << "x\n"
              << "unchanged ONNX control p50 (us): "
              << control.before.p50_us << " -> "
              << control.after.p50_us << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse_options(argc, argv);
        const auto fixture = std::filesystem::path(TINYINFER_PROJECT_SOURCE_DIR) /
                             "tests/fixtures/phase3_mlp_gemm.onnx";
        tinyinfer::onnx::OnnxImporter importer;
        const auto imported = importer.load(fixture);
        const auto before = make_optimization_fixture(imported);
        auto pipeline = make_pipeline();
        const auto control = pipeline.run(imported);
        const auto optimized = pipeline.run(before);
        verify(imported, control, before, optimized);
        const auto control_result = measure_pair(
            options, imported, control.model);
        const auto result = measure_pair(options, before, optimized.model);
        print_results(options, control_result, result, before, optimized.model);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "tinyinfer_graph_optimization_benchmark: "
                  << error.what() << '\n';
        return 1;
    }
}
