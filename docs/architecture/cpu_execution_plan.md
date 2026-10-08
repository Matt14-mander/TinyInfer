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
layout view. The packed SIMD MatMul writes into planned output storage. Phase 4.5 adds
Gemm epilogue modes described below. Dynamic bias
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

The acceptance benchmark additionally reports snapshot/context constant payload
and bound inputs. Its accounted plan payload is snapshot constants plus packed
weights; accounted context payload is copied constants plus inputs plus planned
activations. For C live contexts, the measured payload model is
`plan_accounted_payload_bytes + C * context_accounted_payload_bytes`.
Caller source models, metadata, allocator capacity, extracted output copies and
fallback scratch are outside this model. Independent snapshot, ordering,
MemoryPlan, RHS packing and transpose-copy timers describe preparation work;
their medians must not be summed into a predicted end-to-end latency.

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
[Mac measurements](../benchmarks/mac-cpu-phase4.4.md) and
[ROG acceptance](../benchmarks/rog-cpu-phase4.4.md).

## Gemm epilogue modes (Phase 4.5)

`CpuExecutionPlanOptions{CpuGemmEpilogueMode::Legacy}` selects the existing
TensorIterator epilogue. `Specialized` uses decoded broadcast strides and a
separate direct traversal and is the default after local paired measurements.
The explicit experimental `Fused` mode applies eligible affine/bias/ReLU
operations to the accumulator on the final K block before its existing store.
This is arithmetic kernel fusion; the earlier graph Gemm/ReLU rewrite alone
still used a separate epilogue traversal.

The initial scope is rank-2 FP32 constant packed B, None/ReLU, no bias with any
alpha, or scalar/vector/row/column/full bias with alpha=beta=1. Broadcast column
stride must be zero or one for SIMD fusion. A strided column bias falls back to
the specialized traversal; biased nonunit coefficients retain the original
helper, including its compiler contraction behavior. Dynamic B uses the existing
operator path. Zero K zero-fills and then applies the epilogue once. Tail columns
apply it once after their final scalar accumulation.

The SIMD ReLU uses an ordered positive mask to match `std::max(0.0F, x)`:
NaN and either signed zero produce positive zero. MatMul reduction order, tiling,
packed format, activation buffer and weight ownership are unchanged. NEON has an
implementation but requires target execution before claiming ARM acceptance.

Steps expose pointer-free epilogue prototypes and the requested mode. Every
run binds current bias storage and strides; runtime eligibility can differ from
prototype shape eligibility. The low-level bound descriptor borrows its Tensor
and storage until the kernel returns and must not alias output. Context counters
`fused_gemm_count()` and `specialized_gemm_count()` count successful selected
Gemm dispatches, cumulatively. Fallback to the original helper is neither count.
Zero-K fused dispatches use a separate epilogue internally. Counters describe
dispatch selection, not hardware instructions or eliminated memory traffic.

The historical Phase 4.4 and Phase 4.5 analysis benchmarks explicitly select
Legacy. Use `--fusion-comparison` for paired warm prepared execution with all
three modes, identical graphs and buffer policy, and rotated measurement order.

The initial register fusion candidate passed local numerical gates but regressed
on portable SSE2 and often trailed specialization on AVX2. It is not accepted as
the default performance optimization. See the [fusion measurement report](../benchmarks/mac-cpu-phase4.5-fusion.md).
