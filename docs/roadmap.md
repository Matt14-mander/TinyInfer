# Development roadmap

## Phase 0 — Learning prototype

Build a hand-written `Linear -> ReLU -> Linear -> Softmax` inference path and verify every operation with known inputs.

Current status: complete as an eager FP32 reference path. The implementation deliberately uses readable loops and does not go through Graph or Backend dispatch.

## Phase 1 — v0.1 Tensor Engine

- Tensor allocation, copying, views, and typed access.
- Add, subtract, multiply, ReLU, GELU.
- MatMul, transpose, reduce, Softmax, LayerNorm, and Linear.
- CPU reference kernels and correctness tests.

Exit criterion: run the Phase 0 MLP through the public Tensor API.

## Phase 2 — v0.2 Graph Runtime

- Tensor metadata on graph values.
- Graph validation and topological ordering.
- Executor, operator registry, and error reporting.
- Initial arena allocator and observable tensor lifetimes.

Exit criterion: express and execute the MLP as a graph.

## Phase 3 — v0.3 Model Runtime

- A deliberately small ONNX importer, limited to one selected model.
- Conv and primitives needed for attention.
- Model-level correctness and latency benchmarks.

Exit criterion: import and run one documented ONNX model end to end.

## Phase 4 — Optimization Engine

- Lifetime analysis and buffer reuse.
- Linear + bias + activation fusion.
- MatMul tiling, cache blocking, SIMD, and optional threading.
- Int8 quantization format, calibration, and kernels.

Every optimization requires a reference implementation, tolerance tests, and before/after benchmarks.

## Phase 5 — Hardware Backends

Stabilize capability and buffer interfaces, then add Metal and CUDA with explicit fallback behavior and cross-backend tests.

## Phase 6 — Transformer and compiler direction

Add attention, KV cache management, a small Transformer-family model, and a lower-level IR only when fusion or code generation requires it.

## Suggested weekly rhythm

With four to six hours per week, keep one vertical slice active at a time: one Tensor feature, one operator with tests, then one integration or benchmark session.
