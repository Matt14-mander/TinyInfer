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
reuses cached steps, MemoryPlan and constant matrix packing. In-register MatMul epilogues,
threading, and quantization remain open.

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
correctness but regressed on SSE2 and remains explicit experimental. Its
performance tuning and ROG acceptance remain pending; see the
[Phase 4.5 plan](../plans/phase4.5.md).

## Phase 5 — Hardware Backends

Stabilize capability and buffer interfaces, then add Metal and CUDA with explicit fallback behavior and cross-backend tests.

## Phase 6 — Transformer and compiler direction

Add Conv and the primitives needed for attention, KV cache management, a small
Transformer-family model, and a lower-level IR only when fusion or code
generation requires it.

## Suggested weekly rhythm

With four to six hours per week, keep one vertical slice active at a time: one Tensor feature, one operator with tests, then one integration or benchmark session.
