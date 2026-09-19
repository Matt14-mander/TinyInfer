#include "tinyinfer/tinyinfer.h"

#include <cassert>
#include <cmath>
#include <cstdint>

int main() {
    using tinyinfer::CpuBackend;
    using tinyinfer::DataType;
    using tinyinfer::ExecutionContext;
    using tinyinfer::Executor;
    using tinyinfer::Graph;
    using tinyinfer::OpType;
    using tinyinfer::Tensor;
    using tinyinfer::TensorSpec;

    Graph graph;
    const auto a = graph.add_input(
        "a", TensorSpec{{1, 2}, DataType::Float32});
    const auto b = graph.add_constant(
        "b", Tensor::from_vector({3, 2},
                                  {1.0F, 1.0F, 0.0F, -1.0F, -1.0F, 1.0F}));
    const auto c = graph.add_constant(
        "c", Tensor::from_vector({3}, {1.0F, 2.0F, 3.0F}));
    const auto gemm = graph.add_node(
        "gemm", OpType::Gemm, {a, b, c},
        {{"alpha", 2.0F}, {"beta", 0.5F},
         {"transB", std::int64_t{1}}});
    const auto output_id = graph.node(gemm).outputs.front();
    graph.mark_output(output_id);

    ExecutionContext context(graph);
    context.bind_input(a, Tensor::from_vector({1, 2}, {1.0F, 2.0F}));
    CpuBackend backend;
    Executor executor(backend);
    executor.run(graph, context);

    const auto& output = context.output(output_id);
    assert(output.shape() == tinyinfer::Shape({1, 3}));
    assert(std::fabs(output.at(0) - 6.5F) < 1e-6F);
    assert(std::fabs(output.at(1) - -3.0F) < 1e-6F);
    assert(std::fabs(output.at(2) - 3.5F) < 1e-6F);
}
