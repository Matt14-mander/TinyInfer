# TinyInfer

TinyInfer is a lightweight AI inference runtime built for learning and experimentation. Its long-term scope spans tensors, computation graphs, execution, kernel optimization, quantization, and hardware backends.

TinyInfer has completed the static FP32 CPU optimization slice and is developing **Phase 5 — Hardware Backends**. It can execute the Phase 0 two-layer MLP through both the eager Tensor API and the Graph Runtime, and it can load and execute a documented restricted ONNX model through its format-independent `ModelLoader` and named `Model` interface.

## Current development stage

```text
Phase 0  Learning Prototype       Complete
Phase 1  Tensor Runtime           Complete (v0.1 scope)
Phase 2  Graph Execution Engine   Complete (v0.2 MVP scope)
Phase 3  Model Runtime            Complete (v0.3 restricted ONNX scope)
Phase 4  Optimization Engine      Complete (measured static FP32 CPU slice)
Phase 5  Hardware Backend         In progress (Prepared Metal models; performance gates pending)
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
current schemas cover Add, Subtract, Multiply, MatMul, Gemm, the internal
FusedGemmActivation operator, ReLU, GELU,
Softmax, and LayerNorm, including broadcasting, dtype, rank, axis, and attribute
validation.

## Roadmap

| Milestone | Scope |
| --- | --- |
| v0.1 Tensor Engine | Complete: Tensor, eager operators, CPU execution, MLP inference |
| v0.2 Graph Runtime | Complete (MVP): graph model, validation, execution context, CPU dispatch, Graph MLP |
| v0.3 Model Runtime | Complete: protobuf parser, real Gemm MLP fixture, restricted ONNX import, diagnostics, compatibility profile, and Mac/ROG CPU baselines |
| v0.4 Optimization Engine | In progress: graph simplification, lifetime planning, buffer reuse, CPU SIMD MatMul, Gemm + ReLU fusion, and prepared CPU execution; threading and quantization remain planned |

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
- **Graph optimization:** composable passes, checked Model interface preservation, Graph reconstruction, ValueId-map composition, pass statistics, cascading Constant Folding, output-driven Dead Code Elimination, MatMul + Add canonicalization, and Gemm + ReLU fusion.
- **Runtime memory planning:** static intermediate lifetimes, deterministic slot assignment, one shared aligned Buffer, direct output kernels, dead-value release, and cross-invocation buffer reuse.
- **Prepared CPU execution:** stable model snapshot, cached steps and MemoryPlan, deduplicated constant matrix packing, private per-context activations, dynamic RHS fallback, and latency/initialization diagnostics. See the [prepared API](docs/architecture/cpu_execution_plan.md).
- **Verification:** focused unit tests, eager and Graph MLP integration tests, sanitizer validation, and a correctness-checking MatMul benchmark.

## Current boundary

TinyInfer is now an executable CPU graph runtime with restricted direct `.onnx` loading. Its built-in protobuf reader supports static tensor shapes, the default ONNX domain, one output per node, the current operator subset, and Float32/Float16/Int8/Int32 initializers. It does not support arbitrary ONNX models, dynamic shapes, external tensor files, sparse tensors, or general operator coverage. Planned execution can reuse intermediate storage, including storage retained for intermediate graph outputs; inputs, constants, packed weights, and backend scratch allocations remain outside that accounting.

ONNX failures are reported with structured stages and available model, graph,
node, operator, domain, and value context. ONNX operator translation now checks
the model opset against per-operator version ranges. See the
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
The first fused inference operator now combines Gemm scaling, bias, and ReLU
in one CPU epilogue. Static FP32 models can now use an explicit prepared CPU
plan to reuse execution steps and constant matrix packing.
Phase 4.4 is accepted on Mac and ROG; the
[ROG report](docs/benchmarks/rog-cpu-phase4.4.md) includes portable/native tests,
24 scenarios per configuration, preparation costs and persistent memory.
Phase 4.5 now defaults to conservative `Auto` Gemm epilogue selection: measured
MSVC AVX2 biased K=N=128 shapes can fuse, while other shapes/ISAs keep Specialized
or the existing dynamic-RHS path. The [ROG selection acceptance](docs/benchmarks/rog-cpu-phase4.5-selection.md)
records 41/41 tests per build, 208 full comparison processes, 112 independent
target/control rechecks and 12 extra p95 rechecks. This closes the measured FP32
CPU slice, not unrestricted register fusion. Explicit Fused remains experimental;
SSE2/NEON fusion performance is not claimed. MSVC portable selects scalar width 1.
The [earlier fixes](docs/benchmarks/rog-cpu-phase4.5-fix.md) and failed loop-split
experiment remain documented. Next work is broader shape/ISA tuning, multi-row
tiling, optional threading, and a measured Int8 path. Every
optimization must preserve the v0.3 correctness contract and be justified by
reproducible before-and-after benchmarks.


## Optional Prepared Metal models (Phase 5.1/5.2)

macOS can build `TinyInfer::metal` with `-DTINYINFER_ENABLE_METAL=ON` (default OFF).
The device runtime supports initial FP32 Add/ReLU/Tanh/MatMul/Gemm/ReLU epilogues
and Softmax, with explicit device views, owned transfers and completion. The
[Metal example](examples/metal_ops.cpp) runs a checked GPU Gemm → ReLU chain.
[Actual-device acceptance](docs/acceptance/phase5.1.md) records the support scope
and all verification. `MetalExecutionPlan` uploads constants once and creates
independent contexts with reusable device activations and explicit whole-model
CPU fallback. The [model example](examples/metal_model.cpp) loads an ONNX fixture;
[Phase 5.2 acceptance](docs/acceptance/phase5.2.md) defines the supported scope.
[Phase 5.3 measurements](docs/benchmarks/mac-metal-phase5.3.md) report completed
end-to-end analysis and a resolved fallback-copy regression. The frozen target
performance gates remain unmet; Phase 5.3 is not yet accepted.
[Transfer follow-up](docs/benchmarks/mac-metal-phase5.3-transfer.md) reduces
Buffer copies and reuses staging, with retained p95 diagnostics and failed gates.

## CUDA preparation contracts (Phase 5.4, Mac slice)

`TinyInfer::cuda_contract` is an SDK-free C++ preparation library for the prospective
CUDA operator subset and cuBLAS matrix mapping. It does not provide an executable
CUDA backend or prove GPU availability. The CPU interpretation tests cover 773
matrix/bias/epilogue cases; both ONNX fixtures have static contract coverage.

```bash
./build-bench/examples/tinyinfer_cuda_contract_inspect tests/fixtures/phase3_mlp_gemm.onnx
```

[Mac preparation delivery](docs/acceptance/phase5.4-mac-preparation.md) and the
[Phase 5.4 plan](docs/plans/phase5.4.md) define the boundary. Native CUDA runtime,
operators, Prepared models and GPU performance will be developed on ROG after
collecting its hardware/toolchain inventory. Metal Phase 5.3 remains unaccepted.

## Serving scheduler learning example

[Serving scheduler](examples/serving_scheduler/README.md) implements Static and
Continuous Batching using a deterministic Python backend and virtual clock.
It includes FIFO admission, iteration-boundary refill, request traces, output
consistency tests and explicitly simulated TTFT/TPOT/throughput/P95 metrics.
Block-based KV budget reservation, FIFO capacity waiting, explicit rejection
policies and completion/error cleanup are implemented in stage two. Chunked
prefill is implemented in stage three with per-step token budgets, decode priority,
round-robin prefill progress and simulated latency/throughput tradeoffs. Stage four adds a local random-weight single-layer decoder using actual TinyInfer
CPU attention/logits/KV operations, a wall-clock driver and measured request
metrics; see the [engineering acceptance](examples/serving_scheduler/STAGE4.md).
It is neither a pretrained LLM nor hardware-batched or production network serving.
