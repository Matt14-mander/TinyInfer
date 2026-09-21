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
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
    std::filesystem::path model_path =
        std::filesystem::path(TINYINFER_PROJECT_SOURCE_DIR) /
        "tests/fixtures/phase3_mlp_gemm.onnx";
    std::size_t warmup{10};
    std::size_t samples{100};
    std::size_t repeats{100};
    std::string format{"text"};
};

struct Statistics {
    double minimum_us{};
    double median_us{};
    double p90_us{};
    double p95_us{};
    double p99_us{};
    double mean_us{};
    double standard_deviation_us{};
};

struct BenchmarkResult {
    std::string name;
    Statistics statistics;
};

std::size_t positive_size(std::string_view value, const char* option) {
    std::size_t consumed = 0;
    std::size_t result = 0;
    try {
        result = std::stoull(std::string(value), &consumed);
    } catch (const std::exception&) {
        throw std::invalid_argument(std::string(option) +
                                    " requires a positive integer");
    }
    if (consumed != value.size() || result == 0) {
        throw std::invalid_argument(std::string(option) +
                                    " requires a positive integer");
    }
    return result;
}

void print_usage(const char* program) {
    std::cout
        << "Usage: " << program << " [model.onnx] [options]\n\n"
        << "Options:\n"
        << "  --model PATH      ONNX model path (defaults to the Phase 3 Gemm MLP)\n"
        << "  --warmup N        Warm-up invocations per benchmark (default: 10)\n"
        << "  --samples N       Timed samples per benchmark (default: 100)\n"
        << "  --repeats N       Invocations inside each timed sample (default: 100)\n"
        << "  --format FORMAT   text or json (default: text)\n"
        << "  --help             Show this message\n";
}

Options parse_options(int argc, char** argv) {
    Options options;
    bool positional_model_seen = false;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto require_value = [&](const char* option) -> std::string_view {
            if (index + 1 >= argc) {
                throw std::invalid_argument(std::string(option) +
                                            " requires a value");
            }
            return argv[++index];
        };

        if (argument == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (argument == "--model") {
            options.model_path = require_value("--model");
            positional_model_seen = true;
        } else if (argument == "--warmup") {
            options.warmup = positive_size(require_value("--warmup"),
                                           "--warmup");
        } else if (argument == "--samples") {
            options.samples = positive_size(require_value("--samples"),
                                            "--samples");
        } else if (argument == "--repeats") {
            options.repeats = positive_size(require_value("--repeats"),
                                            "--repeats");
        } else if (argument == "--format") {
            options.format = require_value("--format");
            if (options.format != "text" && options.format != "json") {
                throw std::invalid_argument("--format must be text or json");
            }
        } else if (!argument.empty() && argument.front() == '-') {
            throw std::invalid_argument("unknown option: " +
                                        std::string(argument));
        } else if (!positional_model_seen) {
            options.model_path = argument;
            positional_model_seen = true;
        } else {
            throw std::invalid_argument("only one positional model path is allowed");
        }
    }
    return options;
}

double percentile(const std::vector<double>& sorted, double probability) {
    const auto position = probability * static_cast<double>(sorted.size() - 1);
    const auto lower = static_cast<std::size_t>(position);
    const auto upper = std::min(lower + 1, sorted.size() - 1);
    const auto fraction = position - static_cast<double>(lower);
    return sorted[lower] + (sorted[upper] - sorted[lower]) * fraction;
}

Statistics summarize(std::vector<double> samples_us) {
    if (samples_us.empty()) {
        throw std::invalid_argument("benchmark requires at least one sample");
    }
    std::sort(samples_us.begin(), samples_us.end());
    const auto mean = std::accumulate(samples_us.begin(), samples_us.end(), 0.0) /
                      static_cast<double>(samples_us.size());
    double squared_difference = 0.0;
    for (const auto sample : samples_us) {
        const auto difference = sample - mean;
        squared_difference += difference * difference;
    }
    return Statistics{
        samples_us.front(),
        percentile(samples_us, 0.50),
        percentile(samples_us, 0.90),
        percentile(samples_us, 0.95),
        percentile(samples_us, 0.99),
        mean,
        std::sqrt(squared_difference / static_cast<double>(samples_us.size()))};
}

template <typename Function>
BenchmarkResult measure(std::string name, const Options& options,
                        Function&& function) {
    for (std::size_t index = 0; index < options.warmup; ++index) function();

    std::vector<double> samples_us;
    samples_us.reserve(options.samples);
    for (std::size_t sample = 0; sample < options.samples; ++sample) {
        const auto start = Clock::now();
        for (std::size_t repeat = 0; repeat < options.repeats; ++repeat) {
            function();
        }
        const auto elapsed =
            std::chrono::duration<double, std::micro>(Clock::now() - start)
                .count();
        samples_us.push_back(elapsed / static_cast<double>(options.repeats));
    }
    return BenchmarkResult{std::move(name), summarize(std::move(samples_us))};
}

const char* compiler_name() {
#if defined(__clang__)
    return "Clang " __clang_version__;
#elif defined(__GNUC__)
    return "GCC " __VERSION__;
#elif defined(_MSC_VER)
    return "MSVC";
#else
    return "unknown";
#endif
}

const char* platform_name() {
#if defined(__APPLE__)
    return "macOS";
#elif defined(__linux__)
    return "Linux";
#elif defined(_WIN32)
    return "Windows";
#else
    return "unknown";
#endif
}

const char* architecture_name() {
#if defined(__aarch64__) || defined(_M_ARM64)
    return "arm64";
#elif defined(__x86_64__) || defined(_M_X64)
    return "x86_64";
#else
    return "unknown";
#endif
}

std::string json_escape(std::string_view input) {
    std::string output;
    output.reserve(input.size());
    constexpr char hex[] = "0123456789abcdef";
    for (const auto character : input) {
        const auto byte = static_cast<unsigned char>(character);
        switch (character) {
            case '\\': output += "\\\\"; break;
            case '"': output += "\\\""; break;
            case '\n': output += "\\n"; break;
            case '\r': output += "\\r"; break;
            case '\t': output += "\\t"; break;
            default:
                if (byte < 0x20U) {
                    output += "\\u00";
                    output += hex[byte >> 4U];
                    output += hex[byte & 0x0FU];
                } else {
                    output.push_back(character);
                }
        }
    }
    return output;
}

void print_text(const Options& options,
                const std::vector<BenchmarkResult>& results) {
    std::cout << "TinyInfer Model Benchmark\n"
              << "model: " << options.model_path.string() << '\n'
              << "build: " << TINYINFER_BUILD_TYPE << '\n'
              << "platform: " << platform_name() << '\n'
              << "architecture: " << architecture_name() << '\n'
              << "compiler: " << compiler_name() << '\n'
              << "warmup: " << options.warmup << '\n'
              << "samples: " << options.samples << '\n'
              << "repeats_per_sample: " << options.repeats << "\n\n";

    std::cout << std::left << std::setw(22) << "benchmark" << std::right
              << std::setw(12) << "min_us" << std::setw(12) << "p50_us"
              << std::setw(12) << "p90_us" << std::setw(12) << "p95_us"
              << std::setw(12) << "p99_us" << std::setw(12) << "mean_us"
              << std::setw(12) << "stddev_us" << '\n';
    for (const auto& result : results) {
        const auto& value = result.statistics;
        std::cout << std::left << std::setw(22) << result.name << std::right
                  << std::fixed << std::setprecision(3)
                  << std::setw(12) << value.minimum_us
                  << std::setw(12) << value.median_us
                  << std::setw(12) << value.p90_us
                  << std::setw(12) << value.p95_us
                  << std::setw(12) << value.p99_us
                  << std::setw(12) << value.mean_us
                  << std::setw(12) << value.standard_deviation_us << '\n';
    }
}

void print_json(const Options& options,
                const std::vector<BenchmarkResult>& results) {
    std::cout << std::fixed << std::setprecision(6)
              << "{\n"
              << "  \"model\": \""
              << json_escape(options.model_path.string()) << "\",\n"
              << "  \"build_type\": \"" << json_escape(TINYINFER_BUILD_TYPE)
              << "\",\n"
              << "  \"platform\": \"" << platform_name() << "\",\n"
              << "  \"architecture\": \"" << architecture_name() << "\",\n"
              << "  \"compiler\": \"" << json_escape(compiler_name())
              << "\",\n"
              << "  \"warmup\": " << options.warmup << ",\n"
              << "  \"samples\": " << options.samples << ",\n"
              << "  \"repeats_per_sample\": " << options.repeats << ",\n"
              << "  \"unit\": \"microseconds_per_invocation\",\n"
              << "  \"results\": [\n";
    for (std::size_t index = 0; index < results.size(); ++index) {
        const auto& result = results[index];
        const auto& value = result.statistics;
        std::cout << "    {\"name\": \"" << json_escape(result.name)
                  << "\", \"min\": " << value.minimum_us
                  << ", \"p50\": " << value.median_us
                  << ", \"p90\": " << value.p90_us
                  << ", \"p95\": " << value.p95_us
                  << ", \"p99\": " << value.p99_us
                  << ", \"mean\": " << value.mean_us
                  << ", \"stddev\": " << value.standard_deviation_us
                  << '}' << (index + 1 == results.size() ? "\n" : ",\n");
    }
    std::cout << "  ]\n}\n";
}

void verify_fixture(const tinyinfer::Model& model) {
    if (model.inputs().size() != 1 || model.outputs().size() != 1) {
        throw std::runtime_error(
            "model benchmark currently expects one input and one output");
    }
    const auto input_id = model.inputs().front().second;
    const auto output_id = model.outputs().front().second;
    const auto& input_spec = model.graph().value(input_id).spec;
    if (input_spec.shape != tinyinfer::Shape({1, 2}) ||
        input_spec.dtype != tinyinfer::DataType::Float32) {
        throw std::runtime_error(
            "model benchmark currently expects one Float32 [1, 2] input");
    }

    tinyinfer::ExecutionContext context(model.graph());
    context.bind_input(input_id,
                       tinyinfer::Tensor::from_vector(
                           {1, 2}, {1.0F, -2.0F}));
    tinyinfer::CpuBackend backend;
    tinyinfer::Executor executor(backend);
    executor.run(model.graph(), context);
    const auto& output = context.output(output_id);
    if (output.shape() != tinyinfer::Shape({1, 2}) ||
        std::fabs(output.at(0) - 0.26894143F) >= 1e-5F ||
        std::fabs(output.at(1) - 0.73105860F) >= 1e-5F) {
        throw std::runtime_error(
            "model benchmark correctness check failed for the Phase 3 MLP");
    }
}

std::vector<BenchmarkResult> run_benchmarks(const Options& options) {
    tinyinfer::onnx::ProtobufModelParser parser;
    tinyinfer::onnx::OnnxImporter importer;
    const auto parsed = parser.parse(options.model_path);
    const auto model = importer.import_model(parsed);
    verify_fixture(model);

    std::vector<BenchmarkResult> results;
    results.reserve(5);
    results.push_back(measure("file_and_parse", options, [&] {
        const auto candidate = parser.parse(options.model_path);
        if (candidate.graph.nodes.empty()) {
            throw std::runtime_error("parsed benchmark model has no nodes");
        }
    }));
    results.push_back(measure("graph_import", options, [&] {
        const auto candidate = importer.import_model(parsed);
        if (candidate.graph().size() == 0) {
            throw std::runtime_error("imported benchmark model has no nodes");
        }
    }));
    results.push_back(measure("context_init", options, [&] {
        const tinyinfer::ExecutionContext context(model.graph());
        if (context.graph().size() == 0) {
            throw std::runtime_error("benchmark context has an empty graph");
        }
    }));

    tinyinfer::CpuBackend backend;
    tinyinfer::Executor executor(backend);
    const auto input_id = model.inputs().front().second;
    const auto output_id = model.outputs().front().second;
    results.push_back(measure("cold_inference", options, [&] {
        tinyinfer::ExecutionContext context(model.graph());
        context.bind_input(input_id,
                           tinyinfer::Tensor::from_vector(
                               {1, 2}, {1.0F, -2.0F}));
        executor.run(model.graph(), context);
        static_cast<void>(context.output(output_id));
    }));

    tinyinfer::ExecutionContext warm_context(model.graph());
    warm_context.bind_input(input_id,
                            tinyinfer::Tensor::from_vector(
                                {1, 2}, {1.0F, -2.0F}));
    results.push_back(measure("warm_inference", options, [&] {
        executor.run(model.graph(), warm_context);
        static_cast<void>(warm_context.output(output_id));
    }));
    return results;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse_options(argc, argv);
        const auto results = run_benchmarks(options);
        if (options.format == "json") {
            print_json(options, results);
        } else {
            print_text(options, results);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "tinyinfer_model_benchmark: " << error.what() << '\n';
        return 1;
    }
}
