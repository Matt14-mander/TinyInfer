# TinyInfer

TinyInfer is a lightweight AI inference runtime built for learning and experimentation. Its long-term scope spans tensors, computation graphs, execution, kernel optimization, quantization, and hardware backends.

The repository has completed **Phase 0** and is now developing **Phase 2 — Graph Execution Engine**. TinyInfer can execute a two-layer MLP through its eager Tensor API and includes explicit Tensor layout and storage abstractions, views and strided iteration, Elementwise and Reduction kernels, normalization operators, and an optimized CPU MatMul path. The graph layer now has a complete data model, operator schema inference, validation, stable topological ordering, an execution context, and CPU operator dispatch.

## Current development stage

```text
Phase 0  Learning Prototype       Complete
Phase 1  Tensor Runtime           Substantially complete
Phase 2  Graph Execution Engine   In progress (CPU graph execution connected)
Phase 3  Model Runtime            Not started
Phase 4  Optimization Engine      Partially explored through CPU MatMul
Phase 5  Hardware Backend         Not started
Phase 6  ML Compiler Direction    Not started
```

The current graph execution path is:

```text
Graph -> validation -> topological order -> Executor
      -> ExecutionContext -> CpuBackend -> CPU OperatorRegistry
      -> eager CPU kernel -> output Tensor
```

Graph values and operator nodes can be assembled with Tensor metadata, constants,
typed attributes, producers, and registered outputs. Graph validation checks their
integrity before a stable topological order is produced. `Executor` consumes this
order and dispatches nodes through the CPU registry using `ExecutionContext`.

Operator outputs are inferred automatically during graph construction. The
current schemas cover Add, Subtract, Multiply, MatMul, ReLU, GELU, Softmax, and
LayerNorm, including broadcasting, dtype, rank, axis, and attribute validation.

## Roadmap

| Milestone | Scope |
| --- | --- |
| v0.1 Tensor Engine | Substantially complete: Tensor, eager operators, CPU execution, MLP inference |
| v0.2 Graph Runtime | In progress: graph model, validation, execution context, and CPU dispatch complete |
| v0.3 Model Runtime | Planned: ONNX import, graph optimization, model benchmarks |
| v0.4 Accelerated | Partially explored: CPU SIMD MatMul; Metal, CUDA, and quantization remain planned |

See [docs/architecture.md](docs/architecture.md) for module responsibilities, [docs/tensor.md](docs/tensor.md) for the current Tensor model, and [docs/roadmap.md](docs/roadmap.md) for the phased development plan.

## Repository layout

```text
include/tinyinfer/   Public C++ API
src/core/            Tensor, dtype, and memory foundations
src/graph/           Nodes, graphs, and future graph passes
src/ops/             Operator definitions and dispatch
src/runtime/         Execution and scheduling
src/backend/         CPU and future accelerator backends
src/compiler/        Future IR and fusion passes
examples/            Small end-to-end programs
tests/               Unit and smoke tests
benchmarks/          Performance harnesses
cmake/               CMake helpers
docs/                Architecture and roadmap notes
```

## Build

Requirements: a C++17 compiler and CMake 3.20 or newer.

```bash
cmake -S . -B build -DTINYINFER_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

Run the scaffold example:

```bash
./build/examples/tinyinfer_quickstart
```

Run the Phase 0 MLP:

```bash
./build/examples/tinyinfer_phase0_mlp
```

Run the same MLP through the Graph Runtime:

```bash
./build/examples/tinyinfer_phase0_graph_mlp
```

Build and run the optional MatMul benchmark:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DTINYINFER_BUILD_BENCHMARKS=ON \
  -DTINYINFER_ENABLE_NATIVE_ARCH=ON
cmake --build build --target tinyinfer_matmul_benchmark
./build/benchmarks/tinyinfer_matmul_benchmark
```

## Implemented modules

- **Tensor core:** multidimensional indexing, checked dtypes, deep copy and move semantics, reshape, transpose, contiguous conversion, views, narrow, and stepped slice.
- **Layout and memory:** `TensorLayout`, `Allocator`, `ArenaAllocator`, reference-counted `Buffer`, and byte-range `Storage`.
- **Elementwise execution:** broadcasting, dimension coalescing, contiguous fast paths, `TensorIterator`, and reusable unary/binary Kernel Runners.
- **Elementwise operators:** Add, Sub, Mul, ReLU, and GELU.
- **Reduction and normalization:** `ReductionIterator`, ReduceSum, ReduceMax, arbitrary-axis Softmax, and LayerNorm with optional affine weight and bias.
- **CPU MatMul:** reference, direct stride-aware, cache-blocked, reusable RHS packing, and AVX2/SSE2/ARM NEON micro-kernel paths with scalar fallback.
- **Graph runtime:** Value-based graphs, schema inference, validation, stable topological sorting, `ExecutionContext`, CPU Operator Registry, and end-to-end numerical execution.
- **Verification:** focused unit tests, eager and Graph MLP integration tests, sanitizer validation, and a correctness-checking MatMul benchmark.

## Current boundary

TinyInfer is now an executable CPU graph runtime, but it is not yet a model runtime. The current Executor materializes every intermediate Tensor for the entire invocation and does not yet perform lifetime analysis or buffer reuse. There is no ONNX importer, graph optimizer, quantization pipeline, Metal backend, or CUDA backend yet.

## Design principles

- Keep the public API small and explicit.
- Separate graph semantics from backend execution.
- Make ownership and memory lifetime visible.
- Add optimizations only after correctness tests and benchmarks exist.
- Keep optional dependencies out of the core runtime.

## Next milestone

The active milestone is **v0.2 Graph Runtime**. Graph Values, Operator
Schema/Shape Inference, whole-graph validation, stable topological sorting, and
the per-invocation `ExecutionContext`, CPU operator dispatch, and the end-to-end
Graph MLP are now implemented. The next slice is observable Tensor lifetimes,
followed by memory planning and buffer reuse.
