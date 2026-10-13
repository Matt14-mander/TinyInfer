# Development roadmap

## Phase 0 — Learning prototype

Build a hand-written `Linear -> ReLU -> Linear -> Softmax` inference path and verify every operation with known inputs.

Current status: complete as both an eager FP32 reference path and an equivalent
Graph Runtime path. Both produce the same checked logits and probabilities.

## Phase 1 — v0.1 Tensor Engine

- Tensor allocation, copying, views, and typed access.
- Add, subtract, multiply, ReLU, GELU.
- MatMul, transpose, reduce, Softmax, LayerNorm, and Linear.
- CPU reference kernels and correctness tests.

Exit criterion: run the Phase 0 MLP through the public Tensor API.

## Phase 2 — v0.2 Graph Runtime

Current status: complete for the v0.2 MVP. The Value-based graph data model and
Operator Schema/Shape Inference layer are implemented. Graph construction validates
operator arity, attributes, dtype, broadcasting, ranks, axes, and affine
normalization parameters before automatically creating output Values.
Whole-graph structural validation and stable topological sorting are also
implemented. `ExecutionContext` now manages per-invocation input, constant,
intermediate, and output Tensors with TensorSpec validation. The CPU Operator
Registry connects all current graph operators to their eager kernels, and the
Executor now performs numerical graph execution through the selected Backend.
The Phase 0 MLP runs end to end through two Linear stages, ReLU, and Softmax,
and is checked against the eager implementation.

- Tensor metadata on graph values.
- Graph validation and topological ordering.
- Executor, operator registry, and error reporting.
- End-to-end Graph execution of the Phase 0 MLP.

Exit criterion: express and execute the MLP as a graph.

Tensor lifetime analysis, memory planning, and buffer reuse are intentionally
deferred to Phase 4 rather than blocking the model-runtime milestone.

## Phase 3 — v0.3 Model Runtime

Current status: complete for v0.3. The format-independent `ModelLoader`, named `Model`
bindings, lightweight ONNX model description, operator translation registry,
and dependency-resolving Graph importer are implemented. An in-memory ONNX-style
MLP imports and executes with the expected output. A built-in minimal protobuf
reader now loads restricted `.onnx` files directly and decodes static Float32,
Float16, Int8, and Int32 TensorProto data. A protobuf-generated MLP fixture now
verifies the common Gemm `transB=1` export pattern end to end. Structured ONNX
diagnostics now identify the failed import stage and preserve path, graph,
node, operator, domain, and value context. A documented compatibility matrix
covers supported execution plus explicit rejection cases. Model-level loading
and inference benchmarking now separates file/protobuf parsing, graph import,
ExecutionContext initialization, cold inference, and warm inference with
correctness checks and text/JSON statistics. Stable measurements are recorded
in the [Mac CPU v0.3 baseline](../benchmarks/mac-cpu-v0.3.md) and the
[ROG CPU v0.3 baseline](../benchmarks/rog-cpu-v0.3.md), completing the target
hardware validation for the milestone.

- A deliberately small ONNX importer with a documented compatibility profile.
- A built-in protobuf parser and checked real-file Gemm MLP fixture.
- Model-level correctness and latency benchmarks on the target Mac and ROG CPUs.

Exit criterion: import and run one documented ONNX model end to end.

## Phase 4 — Optimization Engine

Current status: in progress. Reference, stride-aware, cache-blocked, reusable
RHS packing, and SIMD MatMul paths are implemented and benchmarked. The first
Graph Optimizer slice now provides GraphPass, PassManager, OptimizationResult,
composed ValueId mapping, pass statistics, interface validation, and an
end-to-end NoOpPass. GraphRewriter, cascading Constant Folding, and
output-driven Dead Code Elimination are also implemented. Static memory
planning and runtime buffer reuse are implemented. Gemm + ReLU graph fusion
and its combined CPU epilogue are implemented. Prepared CPU execution now
reuses cached steps, MemoryPlan and constant matrix packing. Conservative
register-epilogue selection is accepted for measured MSVC AVX2 shapes; broader
fusion tuning, threading, and quantization remain future work.

- Graph reconstruction, Constant Folding, and Dead Code Elimination. (Complete)
- Lifetime analysis and buffer reuse. (Complete)
- Gemm + ReLU graph/operator fusion with a combined CPU epilogue. (Complete)
- MatMul tiling, cache blocking, SIMD, and optional threading.
- Int8 quantization format, calibration, and kernels.

Every optimization requires a reference implementation, tolerance tests, and before/after benchmarks.
The Phase 4.2 graph-simplification slice has an opt-in model benchmark and a
[Mac acceptance report](../benchmarks/mac-cpu-phase4.2.md) covering numerical
equivalence, an unchanged ONNX control, graph-size reduction, and paired
before/after inference measurements. Its results do not establish a speedup
for models with no foldable constants or dead work.

The Phase 4.3 fusion slice conservatively rewrites eligible `Gemm -> ReLU`
pairs, preserves observable pre-activation values, and composes after
MatMul + Add canonicalization. Its dedicated benchmark checks correctness,
logical/planned intermediate memory, and paired planned warm latency before
reporting a speedup. ROG measurements are recorded in the
[Phase 4.3 acceptance report](../benchmarks/rog-cpu-phase4.3.md).

Phase 4.4 implements **Prepared CPU inference**: a stable execution plan,
one-time packing of constant matrix weights, and transpose handling without
operand data copies on the prepared path. Its baseline is Phase 4.3 planned
execution, with preparation cost and persistent packed memory reported
separately. See the [Phase 4.4 plan](../plans/phase4.4.md) for implementation
order, ownership rules, fallback behavior, and acceptance gates. Portable and
native Release correctness plus targeted sanitizer checks pass. The
[Mac report](../benchmarks/mac-cpu-phase4.4.md) records local measurements;
the [ROG report](../benchmarks/rog-cpu-phase4.4.md) completes acceptance on
2026-10-07 with 40/40 portable and native tests plus 24 scenarios × 3 processes
per configuration. Phase 4.4 is complete for static FP32 prepared inference.
Phase 4.5 starts with performance analysis of that prepared baseline: complete
execution, matrix arithmetic, existing affine/ReLU epilogues, metadata controls,
scalar-tail and input/output-copy diagnostics. The dedicated benchmark changes
no production execution behavior. The
[Mac analysis](../benchmarks/mac-cpu-phase4.5.md) covers 150 checked native/portable
processes and supports common-bias specialization before register
epilogue fusion. The first kernel slice now implements both paths. The
[local fusion comparison](../benchmarks/mac-cpu-phase4.5-fusion.md) checks 156
processes and selects specialization as the default. Register fusion passed
correctness but regressed on SSE2 and remains explicit experimental.
ROG reproduction is now complete: 41/41 native and portable tests, 150 analysis
processes, 156 comparison processes and 18 independent control rechecks.
The [ROG report](../benchmarks/rog-cpu-phase4.5-fusion.md) finds a repeatable
native specialization no-bias/ReLU regression and fused no-op regression.
The [targeted fixes](../benchmarks/rog-cpu-phase4.5-fix.md) resolve those two
regressions on 2026-10-10: 41/41 tests per build, 156 balanced-order comparisons
and 18 independent control rechecks. Specialization passes the ROG median/control
gates; broader register fusion remains experimental. MSVC portable selects scalar width 1,
not the Mac SSE2 path; see the
[Phase 4.5 plan](../plans/phase4.5.md).

The [2026-10-10 selection closure](../benchmarks/rog-cpu-phase4.5-selection.md)
adds default Auto with a narrow measured MSVC AVX2 shape/layout policy and
runtime stride fallbacks. Both Release suites pass 41/41; 208 full comparison,
112 independent target/control and 12 additional p95 recheck processes pass
numerical/counter checks and the documented case-median gates. A final-K loop
split that regressed was withdrawn, with evidence retained. Phase 4.5 is complete
for this measured FP32 CPU slice; unrestricted Fused, SSE2/NEON tuning and broader
shape coverage are deferred extensions, not accepted performance claims.

## Phase 5 — Hardware Backends

Stabilize capability and buffer interfaces, then add Metal and CUDA with explicit fallback behavior and cross-backend tests.

The [Phase 5 plan](../plans/phase5.md), defined on 2026-10-11, targets the same
static FP32 Model on CPU and GPU with explicit device memory, completion and
fallback contracts. [Phase 5.0 acceptance](../acceptance/phase5.0.md) implements
these contracts and CPU adapters while retaining existing host APIs. Current
Mac inventory verifies Intel Iris Plus Graphics 645, a Metal queue and runtime
shader library compilation, so Metal is the first backend; CUDA follows after
ROG GPU/toolchain verification. [Phase 5.1 acceptance](../acceptance/phase5.1.md)
on 2026-10-12 implements initial Metal FP32 device operators/copies/completion.
The actual-device Release suite passes 45/45, CPU-only 43/43; targeted native
CPU reference and host sanitizer checks pass. Prepared GPU model APIs are accepted in [Phase 5.2](../acceptance/phase5.2.md):
shared constants, independent contexts, completion-safe arena reuse and explicit
whole-model CPU fallback. [Phase 5.3 measurements](../benchmarks/mac-metal-phase5.3.md) on 2026-10-13
deliver end-to-end/stage/resident timing and fix duplicate CPU fallback input
copying. The frozen target p50/p95 performance gates remain unmet; Phase 5.3
is open for transfer overhead and tail-latency work.
First deliver one prepared GPU backend and measured host-to-host inference;
then validate the second backend against the same contract. CPU threading,
quantization and broader CPU fusion tuning remain separate extensions. GPU
submission latency alone is not a performance acceptance result.

## Phase 6 — Transformer and compiler direction

Add Conv and the primitives needed for attention, KV cache management, a small
Transformer-family model, and a lower-level IR only when fusion or code
generation requires it.

## Suggested weekly rhythm

With four to six hours per week, keep one vertical slice active at a time: one Tensor feature, one operator with tests, then one integration or benchmark session.
