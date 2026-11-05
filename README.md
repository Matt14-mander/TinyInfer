# TinyInfer

TinyInfer is a lightweight AI inference runtime built for learning and experimentation. Its long-term scope spans tensors, computation graphs, execution, kernel optimization, quantization, and hardware backends.

TinyInfer has completed its measured static FP32 CPU optimization and initial Metal/CUDA backend milestones. The current development focus is **MiniPrefix**, a native CPU/CUDA LLM generation engine running SmolLM2-135M-Instruct in FP32, with paged KV, exact prefix reuse, and transactional copy-on-write.

## Current development stage

Status checked against the main branch on **2026-11-05**. The core runtime and
MiniPrefix have separate roadmaps:

| Track | Current status |
| --- | --- |
| Core Phases 0–3 | Complete within the learning prototype, v0.1 Tensor, v0.2 Graph MVP, and v0.3 restricted ONNX scopes |
| Core Phase 4 | Complete for the measured static FP32 CPU optimization slice |
| Core Phase 5 | Complete for initial static FP32 Metal/CUDA support and frozen-target performance gates |
| Core Phase 6 | ML compiler direction not started |
| MiniPrefix M0–M2 | Architecture, native engine/reference path, and real-model correctness established; preemption/recompute remains deferred |
| MiniPrefix M3–M4 | GPU forward, resident paged KV, partial-page COW, local generation, and scoped FP32 baselines implemented |
| MiniPrefix M5 | In progress: exact partial-page prefix reuse, multi-turn history snapshots, cost measurement, and COW/publication optimizations implemented |

The next planned MiniPrefix execution step is cross-request batched computation.
Current scheduling supports multiple active requests, but runners execute their
chunks sequentially. See the [MiniPrefix overview](miniprefix/README.md) and
[detailed roadmap](miniprefix/docs/roadmap.md) for capabilities and acceptance boundaries.

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
| v0.4 Optimization Engine | Complete for the measured static FP32 CPU slice: graph simplification, lifetime planning, buffer reuse, CPU SIMD MatMul, Gemm + ReLU fusion, and prepared CPU execution; threading and quantization remain deferred |
| v0.5 Hardware Backends | Complete for the initial static FP32 scope: Mac Metal and ROG CUDA correctness and frozen-target performance gates passed |
| MiniPrefix M0–M5 | Active LLM track: real-model CPU/CUDA generation, paged KV, partial-page prefix reuse/COW, and measured cache costs; cross-request batched computation is next |

See the [documentation index](docs/README.md) for an overview. The current
system design lives under `docs/architecture`, learning-oriented walkthroughs
under `docs/tutorials`, and design rationale under `docs/decisions`.

The [Phase 5.0 Mac acceptance](docs/acceptance/phase5.0.md) records 43/43 tests
in both Release configurations and 2/2 targeted sanitizer tests. Device discovery,
queue creation and runtime shader compilation are verified, not GPU arithmetic.
Subsequent [Phase 5.1](docs/acceptance/phase5.1.md) and
[Phase 5.2](docs/acceptance/phase5.2.md) add actual Metal operators and prepared
model execution with Mac device validation. [Phase 5.3 closeout](docs/acceptance/phase5.3-closeout.md)
passes the original Metal performance gates on 2026-10-20; [Phase 5.4 Mac preparation](docs/acceptance/phase5.4-mac-preparation.md)
is delivered; ROG native runtime, operators and Prepared CUDA models are now
verified. Both backends now pass their frozen-target performance gates within
the documented initial static FP32 scope; tiny models can still favor CPU.

## Repository layout

```text
include/tinyinfer/   Public C++ API
src/core/            Tensor, dtype, and memory foundations
src/graph/           Nodes, graphs, validation, and optimization passes
src/model/           Model abstraction and format importers
src/ops/             Operator definitions and dispatch
src/runtime/         Execution and scheduling
src/backend/         CPU, Metal, and CUDA backends
src/compiler/        Future IR and fusion passes
examples/            Small end-to-end programs
miniprefix/          CPU/CUDA LLM engine, paged KV, prefix cache, and baselines
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

### Build MiniPrefix

MiniPrefix is optional and defaults to OFF. A CPU build needs the same C++17/CMake
toolchain; downloading model weights and installing Python tokenizer/reference
packages are explicit, separate steps.

```bash
cmake -S . -B build-miniprefix -DCMAKE_BUILD_TYPE=Release \
  -DTINYINFER_BUILD_MINIPREFIX=ON -DTINYINFER_BUILD_TESTS=ON
cmake --build build-miniprefix
ctest --test-dir build-miniprefix --output-on-failure
```

Enable `-DTINYINFER_ENABLE_CUDA=ON` in a CUDA-capable build environment to also
build `miniprefix_cuda_worker`. Model preparation and text generation commands
are in the [real-model guide](miniprefix/docs/real_model.md).

## MiniPrefix capabilities

- **Real-model inference:** native CPU and full CUDA forward for SmolLM2-135M-Instruct FP32, checked against an independent Transformers reference; local worker, Python IPC client, and text CLI.
- **Request scheduling:** an independent Scheduler, decode priority, chunked prefill, token budgets, Greedy generation, EOS/length termination, cancellation, and failure handling.
- **KV ownership and admission:** a shared physical page pool, per-sequence block tables, transactional append/rollback, and conservative worst-case page reservations with request-level capacity errors. See [capacity admission](miniprefix/docs/capacity_admission.md).
- **Exact prefix reuse:** full and partial-page history snapshots, page-trie lookup and LRU, token-exact longest-prefix matching, and CPU/CUDA copy-on-write for shared tails. See [current cache semantics](miniprefix/docs/partial_prefix_cache.md). Cache is disabled by default; at least one prompt token is recomputed for logits, and the latest sampled token is not cached until it has been processed by the model.
- **Cost-driven optimization:** independent lookup/publication/COW counters and optional timing, [CUDA COW copies merged across layers](miniprefix/docs/merged_cuda_cow.md), and [publication-path and metadata reuse](miniprefix/docs/prefix_publication.md).
- **Validation and measurement:** lifecycle and differential tests, [native stress measurement](miniprefix/docs/long_running_stress.md), [off/cold/hot cache experiments](miniprefix/docs/partial_prefix_costs.md), and independent real-model numerical checks. Performance reports identify their workload, hardware, and timing boundaries; metadata or copy-stage improvements are not universal end-to-end speedups.

The current runner processes each sequence separately. Cross-request matrix
batching, sharing-aware admission, and preemption/recompute remain planned.
Multi-turn reuse requires the caller to submit the complete token history with a
new request ID; automatic conversation storage, HTTP serving, quantization,
CUDA Graphs, and multi-GPU execution are not implemented in this path.

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

The active work is MiniPrefix M5 and its execution path. The latest
[roadmap](miniprefix/docs/roadmap.md) names **cross-request batched computation**
as the next step, building on the current scheduler, paged KV ownership, partial
prefix reuse, and COW transactions. Sharing-aware capacity estimates and
preemption/recompute remain later work.

The completed core Phase 4/5 foundations remain available for this work; their
scope and results are recorded in the [CPU optimization acceptance](docs/benchmarks/rog-cpu-phase4.5-selection.md)
and [hardware backend plan](docs/plans/phase5.md). Broader shape/ISA tuning,
threading, quantization, and compiler IR are future directions rather than
completed milestones. New optimizations should preserve the numerical and
resource-lifecycle contracts and use reproducible before/after measurements.

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
performance gates failed in the initial measurements.
[Transfer follow-up](docs/benchmarks/mac-metal-phase5.3-transfer.md) reduces
Buffer copies and reuses staging, with retained p95 diagnostics and failed gates.
The [2026-10-20 closeout](docs/acceptance/phase5.3-closeout.md) remeasures the same
implementation: both rounds pass p50, p95 and CPU fallback gates. No kernel,
workload, tolerance or threshold was changed; this is a device/workload-specific result.

## CUDA preparation contracts (Phase 5.4, Mac slice)

`TinyInfer::cuda_contract` is an SDK-free C++ preparation library for the prospective
CUDA operator subset and cuBLAS matrix mapping. It does not provide an executable
CUDA backend or prove GPU availability. The CPU interpretation tests cover 773
matrix/bias/epilogue cases; both ONNX fixtures have static contract coverage.

```bash
./build-bench/examples/tinyinfer_cuda_contract_inspect tests/fixtures/phase3_mlp_gemm.onnx
```

[Mac preparation delivery](docs/acceptance/phase5.4-mac-preparation.md) and the
[Phase 5.4 plan](docs/plans/phase5.4.md) define the boundary. The SDK-free library
remains preparation-only; executable device operators live in the optional native
target below. Metal Phase 5.3 is accepted within its frozen-target scope; see the
[latest closeout](docs/acceptance/phase5.3-closeout.md).

ROG's optional `TinyInfer::cuda` (`TINYINFER_ENABLE_CUDA=ON`) now provides device
resources/transfers plus FP32 MatMul/Gemm/FusedGemmActivation (ReLU), Add/ReLU/Tanh
and Softmax through `CudaTensorView`/`CudaDeviceRuntime::execute`. The
[operator acceptance](docs/acceptance/phase5.4-rog-operators.md) records 1032
GPU numerical/model-chain cases, 48/48 full tests and native memory checks.
Use `tools/build_cuda_rog.ps1` in a separate PowerShell process for the verified
private CUDA 12.5.1/MSVC 14.29 setup. The host Executor deliberately rejects this
backend; model execution uses the separate Prepared API below.

### Prepared CUDA models

`CudaExecutionPlan` snapshots the model and uploads shared constants once.
Each movable `CudaExecutionContext` has private input buffers and a completion-safe
activation arena; `run()` is synchronous and `output()` returns an independent
host Tensor. Contexts sharing a plan require external serialization, not concurrent
calls. Strict is the default; explicit `WholeModelCpu` permits whole-model Prepared
CPU Auto fallback with diagnostics, never CPU replay after CUDA execution errors.

```cpp
#include "tinyinfer/backend/cuda/execution_plan.h"
tinyinfer::CudaExecutionPlan plan(model);
auto context = plan.create_context();
context.bind_input(plan.inputs()[0].first, input_tensor);
context.run();
auto result = context.output(plan.outputs()[0].first);
```

Link `TinyInfer::cuda`. The optional `tinyinfer_cuda_model model.onnx` example runs
three resident inferences. [Prepared CUDA acceptance](docs/acceptance/phase5.4-rog-prepared.md)
records 49/49 ROG tests, 45/45 CPU-only and clean memcheck/full leak checks.
The [ROG performance/memory report](docs/benchmarks/rog-cuda-phase5.4.md) records
two rounds of three independent processes: target M=128/K=N=1024 E2E p50 speedups
10.62x / 10.26x, with p95 and CPU-fallback gates passed. Tiny fixtures remain
CPU-favored. Memory includes logical payloads, process working set and device-wide
snapshots, not exact per-process VRAM peak. These are scoped ROG results, not a
universal GPU speedup. Mac Metal has separate [closeout evidence](docs/acceptance/phase5.3-closeout.md).

The [CPU fallback cold-start fix](docs/acceptance/phase5.4-rog-cold-fallback.md)
rejects known unsupported models before CUDA initialization. Six isolated prepares
take 28.1–36.3 us; real-device regression now passes 51/51 and the original warm
performance gates still pass. Supported GPU models retain native validation.

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
The worker now packs token rows across requests for CPU Q/K/V projections,
output projection, FFN and vocabulary projection; ragged causal attention remains
per query. See [tensor batching validation](examples/serving_scheduler/BATCHING.md).
The model uses random weights and supports local CPU or hybrid CPU/CUDA inference.
An optional [CPU PagedAttention path](examples/serving_scheduler/PAGED_ATTENTION.md)
uses a physical KV block pool, per-request block tables and direct block-wise
online softmax, with contiguous and gather reference paths for validation.
Optional [advanced serving](examples/serving_scheduler/ADVANCED_SERVING.md) adds
dynamic admission with budgeted preempt/recompute, exact prefix sharing/COW,
and a real CUDA paged-attention kernel while projections remain CPU-batched.
