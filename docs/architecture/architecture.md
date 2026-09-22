# Architecture

TinyInfer uses a layered architecture so that correctness, execution policy, and hardware-specific code can evolve independently.

```text
Model import
        |
        v
Graph + optimization passes
        |
        v
Runtime executor + scheduler
        |
        v
Backend interface
   /       |       \
 CPU    Metal     CUDA
        |
        v
Tensor storage + memory management
```

## Modules

- **Core** owns `Tensor`, `DataType`, layout, iteration, and memory abstractions. `TensorLayout` encapsulates shape, strides, indexing, reshape/transpose metadata, and checked size calculations. `TensorIterator` builds a shared broadcasted iteration space for stride-aware elementwise kernels. `Allocator` supplies raw memory, `Buffer` owns an allocation, and `Storage` identifies a shared byte range inside a Buffer. `CpuAllocator` handles independent buffers while `ArenaAllocator` supports bulk lifetime reuse. Core does not depend on graph or runtime.
- **Ops** defines operator contracts, input/output arity, accepted attributes, dtype rules, and shape inference. Graph construction invokes these schemas before creating output Values, so invalid operators fail during model construction rather than execution.
- **Graph** represents Tensor values as data-flow edges and operator nodes as producers/consumers. Values carry shape, dtype, kind, and producer metadata; nodes carry Value inputs/outputs and typed attributes. Whole-graph validation checks table integrity, constants, producers, schemas, and registered outputs. A stable Kahn sort produces execution order and rejects cycles. The Phase 4 optimizer now provides `GraphPass`, `PassManager`, original-to-final ValueId mapping, per-pass statistics, and a checked NoOp pipeline; graph reconstruction and transforming passes follow next.
- **Model** owns an executable Graph plus stable external input/output name bindings. `ModelLoader` is the format-independent loading boundary. The ONNX importer keeps protobuf parsing behind an injected parser and translates a lightweight model description through an ONNX-specific operator registry.
- **Runtime** coordinates execution, kernel selection, value binding, memory planning, and scheduling. `ExecutionContext` owns the materialized Tensor table for one graph invocation: constants are loaded at construction, inputs are bound by the caller, and intermediate/output values are written during execution. It rejects shape/dtype mismatches and access to values that are not ready.
- **Backend** is the hardware abstraction boundary. `CpuBackend` delegates node execution to a CPU `OperatorRegistry`, which maps each `OpType` to a callable kernel without coupling `Executor` to operator implementations. Metal and CUDA remain optional modules.
- **Compiler** is reserved for a future low-level IR, fusion, code generation, and kernel selection.

## Current scaffold decisions

- C++17 keeps the toolchain accessible while providing safe ownership primitives.
- Public headers live under `include/tinyinfer`; implementations remain under `src`.
- `Tensor` composes `TensorLayout`, `DataType`, and shared `Storage`; views reuse the same Buffer with independent layout metadata.
- CPU MatMul is separated into reference, stride-aware, cache-blocked, and packed-SIMD kernels. `PackedMatMulRhs` converts constant weights to reusable contiguous K×N storage; the micro-kernel uses AVX2, SSE2, or ARM NEON with a scalar tail and fallback.
- Graph inputs reference earlier nodes, keeping the initial graph valid by construction.
- Unsupported execution fails explicitly instead of producing placeholder values.
