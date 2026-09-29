# Prepared CPU execution

Phase 4.4 adds `CpuExecutionPlan` and `CpuExecutionContext` in
`tinyinfer/backend/cpu/execution_plan.h`, also included by `tinyinfer.h`.
The caller supplies a static model after choosing graph optimization passes.
Preparation does not change the imported graph or enable passes automatically.

## Public entry point

```cpp
#include "tinyinfer/tinyinfer.h"

using namespace tinyinfer;
onnx::OnnxImporter importer;
auto model = importer.load("model.onnx");
// Optionally run the explicit PassManager pipeline here.
CpuExecutionPlan plan(model);
auto session = plan.create_context();
session.bind_input("x", Tensor::from_vector({1, 2}, {1.0F, -2.0F}));
session.run();
Tensor probabilities = session.output("probabilities");
// Bind a new input and run again using the same prepared weights and buffer.
```

Named bindings are model-specific. `inputs()`, `outputs()`, `input_id()`,
`output_id()`, and `input_spec()` expose names, IDs and input metadata.
Input binding retains ExecutionContext's shape and dtype checks.

## Ownership and lifetime

The plan deep-copies the Model into a stable heap allocation before constructing
its MemoryPlan. The MemoryPlan refers to that snapshot's graph identity. Cheap
plan copies share this prepared state; contexts keep it alive after all caller
model and plan handles have been destroyed. There is no global cache.

Each context owns its input bindings, copied graph constants, and activation
Buffer. Contexts share the snapshot and packed weights but have separate writable
activations. A context is movable and cannot be copied. Its moved-from execution,
input, and output methods reject access. A single context requires external
serialization when used from multiple threads.

The snapshot is private: the API exposes no Graph, Tensor constant, Storage, or
mutable ExecutionContext. This matters because even a const Tensor can reveal
its mutable Buffer. `output()` returns a deep copy, including when the graph
output is a constant or input. Output extraction therefore has allocation/copy
cost; the warm execution benchmark excludes extraction for both paths.

## Preparation and dispatch

Preparation validates and stores the graph's topological execution steps,
constructs its MemoryPlan, and selects a CPU path for each node:

| Operator / RHS | Path |
| --- | --- |
| Rank-2 FP32 MatMul, Gemm, FusedGemmActivation with constant B | PackedConstantRhs |
| The same operators with input/intermediate B | ExistingKernel; packs the current B each run |
| Other current CPU operators | ExistingKernel |

The current matrix schemas already restrict eligible matrix execution to rank-2
FP32. Unsupported CPU operators fail preparation. `steps()` exposes the selected
path, packed index and fallback reason for inspection.

Within a snapshot, `(ValueId, transB)` deduplicates packed weights. Dtype and
packing format are fixed to FP32 and the current CPU packed format, and layout
is fixed by the snapshot. Consumers with different effective transB need separate
entries. A future dtype/format backend must extend this contract and cache key.

Packing reads effective RHS strides directly, including transB, so it creates
no temporary operand copy. Prepared transA creates a private shared-storage
layout view. The packed SIMD MatMul writes into planned output storage; Gemm
then uses the existing combined alpha/beta/bias/ReLU traversal. Dynamic bias
is read from the current context on every run. Dynamic B uses the existing
operator implementation and cannot enter the constant cache.

`run()` iterates stored steps, clears previous intermediates, and releases dead
values using stored MemoryPlan lifetimes. It does not recalculate graph order
or pack constant matrix weights.

## Diagnostics and memory accounting

- `pack_count()` counts unique packed objects created during preparation.
- `packed_weight_bytes()` counts their float payloads, excluding vector/object
  metadata and allocator capacity.
- `activation_bytes()` counts the MemoryPlan activation Buffer capacity.
- `run_count()` counts completed runs on a context.
- `runtime_pack_count()` counts successful fallback matrix dispatches, each of
  which invokes the current packing kernel once. It is a dispatch diagnostic,
  not global allocator or constructor instrumentation.

These byte counters do not represent total resident memory. The plan also owns
a deep model snapshot; each context copies its constants. Inputs, output copies,
metadata, and temporary fallback allocations are additional. Static RHS is
retained in both snapshot and packed form to keep ordinary operator/constant
output semantics and reuse the current ExecutionContext implementation.

## Validation and boundaries

The new tests compare current kernels and independent ONNX fixture outputs,
including transpose combinations, bias broadcasts, changing runtime operands,
strides, zero dimensions, tails, special values, deduplication and ownership.
The finite tolerance is `1e-5 + 1e-5 * abs(reference)`. Release tests retain
assertions, and targeted ASan/UBSan checks cover the new lifetime/buffer paths.

This is an explicit CPU FP32 preparation layer. It introduces no dynamic shapes,
new activation fusion, in-register epilogue, compiler IR, autotuning, threading,
quantization or GPU backend. See the [Phase 4.4 plan](../plans/phase4.4.md),
[ownership decision](../decisions/0006-own-prepared-cpu-state.md), and
[Mac measurements](../benchmarks/mac-cpu-phase4.4.md).
