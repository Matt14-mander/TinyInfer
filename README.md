# TinyInfer

TinyInfer is a lightweight AI inference runtime built for learning and experimentation. Its long-term scope spans tensors, computation graphs, execution, kernel optimization, quantization, and hardware backends.

The repository is currently in **Phase 0**. It includes a small FP32 Tensor, clear reference implementations of the operators needed by a two-layer MLP, and the original graph/runtime scaffold. Optimized kernels and graph execution remain later milestones.

## Roadmap

| Milestone | Scope |
| --- | --- |
| v0.1 Tensor Engine | Tensor, basic operators, CPU execution, MLP inference |
| v0.2 Graph Runtime | Computation graph, executor, memory manager |
| v0.3 Model Runtime | ONNX import, graph optimization, benchmarks |
| v0.4 Accelerated | SIMD, Metal, CUDA, quantization |

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

## Design principles

- Keep the public API small and explicit.
- Separate graph semantics from backend execution.
- Make ownership and memory lifetime visible.
- Add optimizations only after correctness tests and benchmarks exist.
- Keep optional dependencies out of the core runtime.

## Status

The eager Phase 0 path can execute `Linear -> ReLU -> Linear -> Softmax` with FP32 reference operators. Graph nodes can be assembled, but graph-based numerical execution is not connected yet.
