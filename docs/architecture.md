# Architecture

TinyInfer uses a layered architecture so that correctness, execution policy, and hardware-specific code can evolve independently.

```text
Model import (future)
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

- **Core** owns `Tensor`, `DataType`, layout, and memory abstractions. `TensorLayout` encapsulates shape, strides, indexing, reshape/transpose metadata, and checked size calculations. `Allocator` supplies raw memory, `Buffer` owns an allocation, and `Storage` identifies a shared byte range inside a Buffer. `CpuAllocator` handles independent buffers while `ArenaAllocator` supports bulk lifetime reuse. Core does not depend on graph or runtime.
- **Ops** defines operator contracts, shape inference, validation, and later reference kernels.
- **Graph** represents nodes and dependencies; validation, topological sorting, constant folding, fusion, and lifetime analysis follow later.
- **Runtime** coordinates execution, kernel selection, value binding, memory planning, and scheduling.
- **Backend** is the hardware abstraction boundary. CPU comes first; Metal and CUDA remain optional modules.
- **Compiler** is reserved for a future low-level IR, fusion, code generation, and kernel selection.

## Current scaffold decisions

- C++17 keeps the toolchain accessible while providing safe ownership primitives.
- Public headers live under `include/tinyinfer`; implementations remain under `src`.
- `Tensor` composes `TensorLayout`, `DataType`, and shared `Storage`; views reuse the same Buffer with independent layout metadata.
- Graph inputs reference earlier nodes, keeping the initial graph valid by construction.
- Unsupported execution fails explicitly instead of producing placeholder values.
