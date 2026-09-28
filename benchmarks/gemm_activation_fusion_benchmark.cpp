#include "tinyinfer/tinyinfer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
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
    std::size_t m{16};
    std::size_t k{128};
    std::size_t n{128};
    std::size_t warmup{10};
    std::size_t samples{50};
    std::size_t repeats{10};
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
                         " [--samples N] [--repeats N] [--format text|json]\n";
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

tinyinfer::Model make_model(const Options& options,
                            tinyinfer::OpType activation) {
    using namespace tinyinfer;
    const auto m = static_cast<std::int64_t>(options.m);
    const auto k = static_cast<std::int64_t>(options.k);
    const auto n = static_cast<std::int64_t>(options.n);
    Graph graph;
    const auto x = graph.add_input(
        "x", TensorSpec{{m, k}, DataType::Float32});
    const auto weight = graph.add_constant(
        "weight", Tensor::from_vector(
                      {k, n},
                      generated_values(options.k * options.n, 3)));
    const auto bias = graph.add_constant(
        "bias", Tensor::from_vector(
                    {n}, generated_values(options.n, 11)));
    const auto gemm = graph.add_node("affine", OpType::Gemm,
                                     {x, weight, bias});
    const auto affine = graph.node(gemm).outputs.front();
    const auto activated = graph.add_node("activation", activation, {affine});
    const auto output = graph.node(activated).outputs.front();
    graph.mark_output(output);
    return Model(std::move(graph), {{"x", x}}, {{"y", output}});
}

tinyinfer::Tensor make_input(const Options& options) {
    return tinyinfer::Tensor::from_vector(
        {static_cast<std::int64_t>(options.m),
         static_cast<std::int64_t>(options.k)},
        generated_values(options.m * options.k, 23));
}

tinyinfer::Tensor run_checked(const tinyinfer::Model& model,
                              const tinyinfer::Tensor& input) {
    tinyinfer::MemoryPlan plan(model.graph());
    tinyinfer::ExecutionContext context(model.graph(), plan);
    context.bind_input(model.input_id("x"), input);
    tinyinfer::CpuBackend backend;
    tinyinfer::Executor executor(backend);
    executor.run(model.graph(), context);
    return context.output(model.output_id("y"));
}

void expect_close(const tinyinfer::Tensor& actual,
                  const tinyinfer::Tensor& expected) {
    if (actual.shape() != expected.shape() || actual.dtype() != expected.dtype()) {
        throw std::runtime_error("fusion changed the output TensorSpec");
    }
    for (std::size_t index = 0; index < actual.numel(); ++index) {
        if (std::fabs(actual.at(index) - expected.at(index)) > 1e-5F) {
            throw std::runtime_error("fusion changed an output value");
        }
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
    for (std::size_t index = 0; index < repeats; ++index) function();
    return std::chrono::duration<double, std::micro>(Clock::now() - start)
               .count() /
           static_cast<double>(repeats);
}

PairResult measure_pair(const Options& options, const tinyinfer::Model& before,
                        const tinyinfer::Model& after,
                        const tinyinfer::Tensor& input) {
    tinyinfer::CpuBackend backend;
    tinyinfer::Executor executor(backend);
    tinyinfer::ExecutionContext before_context(
        before.graph(), tinyinfer::MemoryPlan(before.graph()));
    tinyinfer::ExecutionContext after_context(
        after.graph(), tinyinfer::MemoryPlan(after.graph()));
    before_context.bind_input(before.input_id("x"), input);
    after_context.bind_input(after.input_id("x"), input);
    const auto run_before = [&] { executor.run(before.graph(), before_context); };
    const auto run_after = [&] { executor.run(after.graph(), after_context); };
    for (std::size_t index = 0; index < options.warmup; ++index) {
        run_before();
        run_after();
    }
    std::vector<double> before_samples;
    std::vector<double> after_samples;
    before_samples.reserve(options.samples);
    after_samples.reserve(options.samples);
    for (std::size_t index = 0; index < options.samples; ++index) {
        if (index % 2 == 0) {
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
#if defined(_WIN32)
    return "Windows";
#elif defined(__APPLE__)
    return "macOS";
#elif defined(__linux__)
    return "Linux";
#else
    return "other";
#endif
}

const char* architecture_name() {
#if defined(_M_X64) || defined(__x86_64__)
    return "x86_64";
#elif defined(_M_ARM64) || defined(__aarch64__)
    return "arm64";
#else
    return "other";
#endif
}

void print_results(const Options& options, const PairResult& result,
                   const tinyinfer::Model& before,
                   const tinyinfer::Model& after) {
    const tinyinfer::MemoryPlan before_plan(before.graph());
    const tinyinfer::MemoryPlan after_plan(after.graph());
    const double speedup = result.before.p50_us / result.after.p50_us;
    if (options.format == "json") {
        std::cout << std::fixed << std::setprecision(6)
                  << "{\n  \"build_type\": \"" << TINYINFER_BUILD_TYPE
                  << "\",\n  \"platform\": \"" << platform_name()
                  << "\",\n  \"architecture\": \"" << architecture_name()
                  << "\",\n  \"m\": " << options.m
                  << ",\n  \"k\": " << options.k
                  << ",\n  \"n\": " << options.n
                  << ",\n  \"warmup\": " << options.warmup
                  << ",\n  \"samples\": " << options.samples
                  << ",\n  \"repeats\": " << options.repeats
                  << ",\n  \"nodes_before\": " << before.graph().size()
                  << ",\n  \"nodes_after\": " << after.graph().size()
                  << ",\n  \"naive_bytes_before\": "
                  << before_plan.naive_intermediate_bytes()
                  << ",\n  \"naive_bytes_after\": "
                  << after_plan.naive_intermediate_bytes()
                  << ",\n  \"planned_bytes_before\": "
                  << before_plan.buffer_size_bytes()
                  << ",\n  \"planned_bytes_after\": "
                  << after_plan.buffer_size_bytes()
                  << ",\n  \"before_p50_us\": " << result.before.p50_us
                  << ",\n  \"before_p95_us\": " << result.before.p95_us
                  << ",\n  \"before_mean_us\": " << result.before.mean_us
                  << ",\n  \"after_p50_us\": " << result.after.p50_us
                  << ",\n  \"after_p95_us\": " << result.after.p95_us
                  << ",\n  \"after_mean_us\": " << result.after.mean_us
                  << ",\n  \"p50_speedup\": " << speedup << "\n}\n";
        return;
    }
    std::cout << "Phase 4.3 Gemm + ReLU fusion benchmark\n"
              << "build: " << TINYINFER_BUILD_TYPE << '\n'
              << "platform: " << platform_name() << '\n'
              << "architecture: " << architecture_name() << '\n'
              << "shape: [" << options.m << ',' << options.k << "] x ["
              << options.k << ',' << options.n << "]\n"
              << "warmup/samples/repeats: " << options.warmup << '/'
              << options.samples << '/' << options.repeats << '\n'
              << "nodes: " << before.graph().size() << " -> "
              << after.graph().size() << '\n'
              << "naive intermediate bytes: "
              << before_plan.naive_intermediate_bytes() << " -> "
              << after_plan.naive_intermediate_bytes() << '\n'
              << "planned buffer bytes: " << before_plan.buffer_size_bytes()
              << " -> " << after_plan.buffer_size_bytes() << "\n\n"
              << std::fixed << std::setprecision(3)
              << "planned warm inference (us)    p50       p95      mean\n"
              << "before                      " << std::setw(8)
              << result.before.p50_us << std::setw(10)
              << result.before.p95_us << std::setw(10)
              << result.before.mean_us << '\n'
              << "after                       " << std::setw(8)
              << result.after.p50_us << std::setw(10)
              << result.after.p95_us << std::setw(10)
              << result.after.mean_us << '\n'
              << "p50 speedup: " << speedup << "x\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse_options(argc, argv);
        const auto before = make_model(options, tinyinfer::OpType::ReLU);
        tinyinfer::optimizer::GemmActivationFusionPass pass;
        const auto optimized = pass.run(before);
        if (!optimized.changed() || before.graph().size() != 2 ||
            optimized.model.graph().size() != 1 ||
            optimized.model.graph().node(0).op !=
                tinyinfer::OpType::FusedGemmActivation) {
            throw std::runtime_error("eligible Gemm + ReLU was not fused");
        }
        if (pass.run(optimized.model).changed()) {
            throw std::runtime_error("fusion pass is not idempotent");
        }
        const auto control = pass.run(make_model(options, tinyinfer::OpType::Tanh));
        if (control.changed()) {
            throw std::runtime_error("non-ReLU control was fused");
        }
        const auto input = make_input(options);
        expect_close(run_checked(optimized.model, input),
                     run_checked(before, input));
        const tinyinfer::MemoryPlan before_plan(before.graph());
        const tinyinfer::MemoryPlan after_plan(optimized.model.graph());
        if (after_plan.naive_intermediate_bytes() >=
            before_plan.naive_intermediate_bytes()) {
            throw std::runtime_error("fusion did not reduce intermediate bytes");
        }
        const auto result = measure_pair(options, before, optimized.model, input);
        print_results(options, result, before, optimized.model);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "tinyinfer_gemm_activation_fusion_benchmark: "
                  << error.what() << '\n';
        return 1;
    }
}
