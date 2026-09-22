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
end-to-end NoOpPass. Graph reconstruction, transforming passes, static memory
planning, graph-level fusion, threading, and quantization remain open.

- Graph reconstruction, Constant Folding, and Dead Code Elimination.
- Lifetime analysis and buffer reuse.
- Linear + bias + activation fusion.
- MatMul tiling, cache blocking, SIMD, and optional threading.
- Int8 quantization format, calibration, and kernels.

Every optimization requires a reference implementation, tolerance tests, and before/after benchmarks.

## Phase 5 — Hardware Backends

Stabilize capability and buffer interfaces, then add Metal and CUDA with explicit fallback behavior and cross-backend tests.

## Phase 6 — Transformer and compiler direction

Add Conv and the primitives needed for attention, KV cache management, a small
Transformer-family model, and a lower-level IR only when fusion or code
generation requires it.

## Suggested weekly rhythm

With four to six hours per week, keep one vertical slice active at a time: one Tensor feature, one operator with tests, then one integration or benchmark session.
