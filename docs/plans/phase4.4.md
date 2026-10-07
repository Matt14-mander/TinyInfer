# Phase 4.4 — Prepared CPU inference

- Status: Complete; Mac and ROG correctness/performance gates passed
- Acceptance completed: 2026-10-07
- Planning date: 2026-09-29
- Source baseline: `51cfb7f` (Phase 4.3 Gemm + ReLU fusion)

## Objective

Prepare a static CPU model once and reuse its execution order and constant
matrix weights across repeated inference. Preserve the Phase 4.3 graph and
operator semantics, and measure preparation cost separately from warm execution.

Phase 4.3 already removes the Gemm pre-activation intermediate and combines
scaling, bias, and ReLU in one traversal. Its ROG p50 speedups are 1.111x,
1.053x, and 1.005x for batches 1, 16, and 64 respectively. These measurements
do not identify the dominant remaining cost. Code inspection identifies three
preparation operations that should be measured before more kernel work:

- `ops::matmul_out` constructs `PackedMatMulRhs` on every call.
- Gemm transpose handling copies Tensor data before changing layout metadata.
- `Executor::run` obtains a new topological order on every invocation.

The first task is to quantify these operations, not to assume that any one is
the bottleneck. See the [Phase 4.3 report](../benchmarks/rog-cpu-phase4.3.md).

## Deliverables and boundary

| Deliverable | Required behavior |
| --- | --- |
| CPU execution plan | Own a stable model snapshot and a validated ordered list of execution steps |
| Constant RHS preparation | Pack eligible rank-2 FP32 MatMul/Gemm/FusedGemmActivation weights once |
| Gemm layout handling | Interpret transA/transB without copying operand contents on the prepared path |
| Prepared CPU dispatch | Use packed weights for eligible nodes and the existing implementation for other nodes |
| Execution state | Keep inputs, intermediate/output storage, and MemoryPlan state separate for each invocation context |
| Diagnostics and measurements | Report selected paths, pack count, packed bytes, preparation cost, first inference, and warm inference |

The core slice targets constant B, static shapes, CPU FP32, and the existing
None/ReLU Gemm epilogues. Dynamic B remains a supported fallback and is never
cached between invocations. Bias may remain dynamic and must be read on each
run. Existing alpha, beta, transpose, and broadcasting semantics remain valid.

In-register epilogues, multi-row SIMD micro-kernels, automatic shape tuning,
new activation fusion, threading, Int8, GPU backends, and a compiler IR are
later slices. The Phase 4.4 path continues using the Phase 4.3 separate combined
epilogue so that preparation reuse has an independently measurable result.

## Architecture

```text
ONNX / Model
    -> explicit graph optimization passes
    -> CpuExecutionPlan (stable model snapshot)
         - validated execution steps
         - MemoryPlan
         - immutable packed constant weights
         - per-node CPU execution descriptor
    -> separate mutable invocation contexts
    -> prepared CPU execution
```

`CpuExecutionPlan` is now the implemented public API. The
preparation boundary receives an already optimized Model. It does not silently
change ONNX import or choose graph passes for the caller. Optimize first,
prepare second. A new model requires a new plan.

The plan should own the model snapshot rather than retain a borrowed mutable
Graph pointer. Prepared contexts must not expose mutable plan constants through
the existing `ExecutionContext::value()` API: packed bytes and runtime constant
reads must come from the same immutable snapshot. Resolve this ownership/API
contract before adding the cache. The existing unprepared API remains available
for execution that requires mutable constant tensors.

Each prepared context owns its input bindings and activation Buffer. It keeps
the plan alive. Two contexts may share read-only packed weights but must not
share writable intermediate storage. Keep prepared state in the CPU layer;
ordinary ExecutionContext should not acquire ISA-specific weight objects.

Deduplicate packed weights inside one plan using source ValueId, effective
transpose/layout, dtype, and packing format. Two consumers with the same
effective B can share one packed object; a different transB orientation needs
a different entry. Do not introduce a global pointer-keyed cache. Report packed
weight bytes separately from MemoryPlan activation bytes.

Gemm transpose handling should use internal read-only layout descriptors or
explicit shared-storage views. Preserve Tensor's deep-copy value semantics.
Constant transB can be interpreted directly during packing. Dynamic operands
must respect their current strides and values on every invocation.

## Implementation order

### 4.4.0 — Baseline and cost breakdown

Extend a dedicated benchmark to separate RHS packing, transpose preparation,
topological ordering, and complete Phase 4.3 planned warm inference. Record
constant-B and dynamic-B cases. Keep component timing distinct from measured
end-to-end latency; summing component medians is not an end-to-end measurement.

### 4.4.1 — Stable execution plan and ownership

Define model/constant ownership, context lifetime, and the prepared CPU entry
point. Cache validated execution steps and attach MemoryPlan. Initially dispatch
existing kernels so this step can verify plan semantics without packing changes.

### 4.4.2 — Constant weight preparation

Build and deduplicate packed weights for eligible nodes. Add prepared MatMul
and Gemm dispatch, preserve transpose semantics without data copies, and reuse
the current scaling/bias/ReLU epilogue. Keep unsupported preparation cases on
the existing path with an inspectable fallback reason.

### 4.4.3 — Integration and correctness

Run the full optional optimization pipeline before plan construction. Cover
MatMul, Gemm, and FusedGemmActivation, planned output storage, repeated input
binding, and real ONNX fixtures. Verify that repeated warm inference does not
pack constant B or recalculate graph ordering.

### 4.4.4 — Performance acceptance and documentation

Compare the same optimized graph using the Phase 4.3 planned execution path and
the prepared path. Report absolute latency, paired ratios, initialization cost,
packed bytes, and the number of calls needed to amortize preparation. Record
Mac and ROG results independently; never compare different hosts as a speedup.

## Correctness acceptance

- Verify outputs against the existing kernels and independently checked ONNX
  fixture references; preserve shape, dtype, and named model bindings.
- Cover transA/transB combinations, optional bias, scalar/vector/row/column/full
  bias broadcasts, non-default alpha/beta, and None/ReLU epilogues.
- Cover sliced/strided inputs, SIMD tails, non-square matrices, and supported
  zero-size cases. Preserve established NaN, infinity, and signed-zero behavior.
- Keep existing fixture tolerances. For the larger finite random matrix suite,
  document an absolute-plus-relative tolerance before measurement; use
  `1e-5 + 1e-5 * abs(reference)` as the initial proposal, and justify any change.
- Check repeated runs with changed A and dynamic bias. For dynamic B fallback,
  change B between runs and verify the new result, preventing stale-cache bugs.
- Assert that constant packing happens during preparation only; check same-B
  sharing, different-transpose separation, and packed-byte accounting.
- Check source-model independence, plan/context lifetime, and isolation of two
  contexts. Validate that constants cannot change under a prepared plan.
- Run the full suite in portable and native Release builds. Sanitizer checks
  should target the new shared-lifetime and buffer-boundary risks.

## Performance acceptance

Use identical graphs, inputs, MemoryPlan policy, ISA flags, and compiler builds
for each comparison. Alternate measurement order per sample and repeat each
case in three processes. The baseline is Phase 4.3 planned fused execution,
not the earlier unfused Gemm + ReLU graph.

| Case | Purpose |
| --- | --- |
| M=1/16/64, K=N=128 | Continue the Phase 4.3 baseline |
| M=1/16/64, K=N=512 | Increase weight preparation work |
| M=3, K=127, N=131 | Rectangular shape and SIMD tails |
| Constant B with transB=0 and 1 | Preparation reuse and transpose handling |
| Dynamic B control | Fallback semantics and overhead |
| Gemm MLP and RL actor ONNX fixtures | End-to-end integration; actor Tanh remains an ordinary operator |

Required metrics: preparation and context creation time, first inference,
warm p50/p95/mean, pack count, packed bytes, activation bytes, and paired speedup.
Preparation creates a persistent memory/time tradeoff and must be reported.
An amortization estimate may use incremental preparation cost divided by the
measured mean per-call savings, only when those savings are positive.

Exit criteria:

1. All correctness and ownership checks pass.
2. Constant-B warm execution has zero repeated packs and uses stored execution steps.
3. Representative constant-weight cases show a repeatable gain beyond paired
   measurement noise; otherwise revisit the design before claiming acceptance.
4. Any repeatable regression greater than 5% in the controls is explained and
   resolved through the path/fallback policy. This is a proposed review threshold.
5. Reports include per-run data, environment and source revision, preparation
   cost, persistent memory, and limitations. There is no universal speedup promise.

## Implementation status (2026-09-29)

- 4.4.0–4.4.3 implemented, including direct transB packing and private transA views.
- 40/40 tests pass in portable and native Release, with assertions enabled.
- 3/3 targeted ASan + UBSan tests pass.
- Mac: 24 scenarios × 3 processes measured; see the [report](../benchmarks/mac-cpu-phase4.4.md).
- ROG: awaiting target-host execution; the [runner and commands](../../benchmarks/README.md#reproducible-mac--rog-suite) are ready.
- Cross-host Phase 4.4 performance acceptance remains open until ROG results
  satisfy the control-regression and repeatability gates.

## ROG acceptance and closure (2026-10-07)

The pending ROG gate above is now complete. Fresh portable SSE2 and native
AVX2 Release builds each passed 40/40 tests with assertions enabled. Both
configurations ran all 24 scenarios in three sequential processes per case,
for 144 total processes. Static cases gained in all measured processes; all
dynamic controls passed the 5% repeatable-regression gate. Static sessions
completed 1,021 runs without runtime packing; dynamic B remained uncached.

The benchmark now reports independent model-snapshot cost and explicit plan
and per-context payload memory. The runner preserves full component/context
and per-process latency tables. See the
[ROG acceptance report](../benchmarks/rog-cpu-phase4.4.md) and
[detailed statistics](../benchmarks/rog-cpu-phase4.4-details.md). Phase 4.4 is
accepted for the static FP32 prepared CPU slice; subsequent kernel work uses
this prepared path as its baseline.

## Completion statement

Phase 4.4 is complete when a static optimized CPU model can be prepared once,
execute repeatedly with immutable prepacked constant weights and per-context
activation storage, and demonstrate measured benefits over Phase 4.3 without
changing operator semantics. A later micro-kernel epilogue phase can then use
this prepared dispatch point and compare against this stronger baseline.
