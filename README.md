# TinyInfer

TinyInfer is a lightweight AI inference runtime built for learning and experimentation. Its long-term scope spans tensors, computation graphs, execution, kernel optimization, quantization, and hardware backends.

TinyInfer has completed **Phase 3 — Model Runtime** and has entered **Phase 4 — Optimization Engine**. It can execute the Phase 0 two-layer MLP through both the eager Tensor API and the Graph Runtime, and it can load and execute a documented restricted ONNX model through its format-independent `ModelLoader` and named `Model` interface.

## Current development stage

```text
Phase 0  Learning Prototype       Complete
Phase 1  Tensor Runtime           Complete (v0.1 scope)
Phase 2  Graph Execution Engine   Complete (v0.2 MVP scope)
Phase 3  Model Runtime            Complete (v0.3 restricted ONNX scope)
Phase 4  Optimization Engine      In progress (graph + memory optimization)
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
current schemas cover Add, Subtract, Multiply, MatMul, Gemm, ReLU, GELU,
Softmax, and LayerNorm, including broadcasting, dtype, rank, axis, and attribute
validation.

## Roadmap

| Milestone | Scope |
| --- | --- |
| v0.1 Tensor Engine | Complete: Tensor, eager operators, CPU execution, MLP inference |
| v0.2 Graph Runtime | Complete (MVP): graph model, validation, execution context, CPU dispatch, Graph MLP |
| v0.3 Model Runtime | Complete: protobuf parser, real Gemm MLP fixture, restricted ONNX import, diagnostics, compatibility profile, and Mac/ROG CPU baselines |
| v0.4 Optimization Engine | In progress: graph simplification, lifetime planning, buffer reuse, and CPU SIMD MatMul; fusion, threading, and quantization remain planned |

See the [documentation index](docs/README.md) for an overview. The current
system design lives under `docs/architecture`, learning-oriented walkthroughs
under `docs/tutorials`, and design rationale under `docs/decisions`.

## Repository layout

```text
include/tinyinfer/   Public C++ API
src/core/            Tensor, dtype, and memory foundations
src/graph/           Nodes, graphs, and future graph passes
src/model/           Model abstraction and format importers
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
- **Graph optimization:** composable passes, checked Model interface preservation, Graph reconstruction, ValueId-map composition, pass statistics, cascading Constant Folding, and output-driven Dead Code Elimination.
- **Runtime memory planning:** static intermediate lifetimes, deterministic slot assignment, one shared aligned Buffer, direct output kernels, dead-value release, and cross-invocation buffer reuse.
- **Verification:** focused unit tests, eager and Graph MLP integration tests, sanitizer validation, and a correctness-checking MatMul benchmark.

## Current boundary

TinyInfer is now an executable CPU graph runtime with restricted direct `.onnx` loading. Its built-in protobuf reader supports static tensor shapes, the default ONNX domain, one output per node, the current operator subset, and Float32/Float16/Int8/Int32 initializers. It does not support arbitrary ONNX models, dynamic shapes, external tensor files, sparse tensors, or general operator coverage. Planned execution can reuse intermediate storage, including storage retained for intermediate graph outputs; inputs, constants, packed weights, and backend scratch allocations remain outside that accounting.

ONNX failures are reported with structured stages and available model, graph,
node, operator, domain, and value context. See the
[v0.3 ONNX compatibility profile](docs/architecture/onnx_compatibility.md) for
the exact parser, importer, dtype, operator, and rejection matrix.

## Design principles

- Keep the public API small and explicit.
- Separate graph semantics from backend execution.
- Make ownership and memory lifetime visible.
- Add optimizations only after correctness tests and benchmarks exist.
- Keep optional dependencies out of the core runtime.

## Next milestone

The active milestone is **v0.4 Optimization Engine**. The initial CPU MatMul
foundation already includes reference, stride-aware, cache-blocked, reusable
RHS packing, and SIMD paths. The GraphPass/PassManager pipeline now includes a
rebuilding GraphRewriter, cascading Constant Folding, and output-driven Dead
Code Elimination with composed ValueId mappings. Lifetime analysis and buffer
reuse are now connected to `ExecutionContext` through a static `MemoryPlan`.
The next work is fused inference operators, shape-aware
kernel selection, optional threading, and a measured Int8 path. Every
optimization must preserve the v0.3 correctness contract and be justified by
reproducible before-and-after benchmarks.
