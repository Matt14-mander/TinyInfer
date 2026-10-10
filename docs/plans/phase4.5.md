# Phase 4.5 — Prepared CPU performance analysis and Gemm epilogue

- Status: ROG no-bias/ReLU specialization and fused identity regressions fixed; balanced-order specialization gates passed; general register fusion remains experimental
- Start date: 2026-10-07 (Asia/Shanghai)
- Source baseline: `f472304` (Phase 4.4 ROG acceptance)

## Objective

Measure remaining costs after Phase 4.4 preparation reuse, then select a
small CPU optimization whose benefit is verified against the prepared path.
Phase 4.4 passed Mac and ROG correctness/performance gates. Its before/after
ratios include sorting, packing and transpose copies removed from warm runs;
those gains are not evidence of a faster arithmetic kernel.

## 4.5.0 — Analysis boundary

The dedicated `tinyinfer_cpu_performance_analysis_benchmark` measures:

- Complete unmodified `CpuExecutionContext::run()` on a reused bound context.
- A context bookkeeping control: input checks, intermediate clear, node/value
  lookup, private transA view, planned output slot setup, dead-value release.
- A captured matrix chain using operands from checked ordinary execution,
  packed once outside measurement and independent preallocated outputs.
- Per-matrix packed SIMD MatMul, the existing combined Gemm epilogue, and their
  complete direct execution.
- Scalar-tail control for the last `N % SIMD_width` columns, using a strided
  scalar kernel. Aligned shape controls expose whole-kernel behavior separately.
- Ordinary operators in isolated planned one-node registry dispatches.
- Input binding and copied output extraction as separate application costs.

These are separate experiments, not a partition from an instrumented execution.
Do not sum component medians or subtract them from the full timer to claim an
exact dispatch fraction. The bookkeeping control queries all node inputs and
checks transA attributes, whereas production packed dispatch skips RHS lookup
and reads decoded descriptors. The bookkeeping control omits registry dispatch,
packing and arithmetic; captured matrix operands do not propagate outputs
between layers. Ordinary isolated dispatch includes slot setup/registry overhead.
Scalar-tail control changes loop/cache context and is not the actual in-kernel
tail timer. No zero-work operator is inserted into the production graph.

For epilogue timing, a pool of `repeats` independent output buffers is reset
from the original MatMul product outside each timed sample, then transformed
once per buffer. This prevents repeated affine application and preserves bias
and activation semantics. It also gives different cache behavior from repeatedly
updating one buffer, so the isolated time is diagnostic. Record the pool payload.
The analysis keeps additional captured operands, packed copies and output pools
alive, which can affect caches even for the full prepared timer. These timings
are not interchangeable with a minimal standalone prepared benchmark.
Warm whole/direct samples average `repeats` invocations. Measurement order rotates
between workloads each sample. Report p50/p95/mean and loop/timer floor, with
three sequential processes per case. Packing, graph optimization, setup and
correctness checks are outside warm timers.

Use M=1/16/64 with K=N=128/512, both transB forms, rectangular/tail cases,
no/scalar/vector/row/column/full bias, None/ReLU, a dynamic-B control and both
ONNX fixtures. Changing runtime B correctness remains covered by Phase 4.4;
the analysis control retains existing packing and never caches runtime B in
production. The direct diagnostic packs its captured B outside timing and must
not be interpreted as production fallback latency.

## 4.5.1 — Select and specify optimization

Review complete latency and isolated costs by workload. If common bias handling
is significant, distinguish general TensorIterator offset computation from
output memory traffic; both may be reduced by a specialized epilogue. If
matrix arithmetic dominates, multi-row reuse may warrant a separate kernel slice.
For tiny models, inspect bookkeeping and ordinary operator costs as well.
Input/output deep copies are measured separately; contiguous bulk copying is a
possible independent runtime improvement while preserving Tensor ownership.

The proposed first Gemm kernel slice targets static FP32 prepared matrix nodes,
common bias layout and None/ReLU. Select its support boundary after measurements.
Keep the existing combined epilogue as the correctness oracle/fallback. Preserve
alpha/beta, transpositions, dynamic bias, tail and zero-K behavior. Apply affine
and activation only after all K blocks have completed. SIMD ReLU must preserve
current NaN and signed-zero behavior. Any packing-format change requires an
explicit descriptor/cache key update.

## 4.5.2 — Implement, verify, measure

Implement the selected kernel slice through prepared dispatch with inspectable
eligibility. Measure specialization and register fusion independently where
practical, so their contributions can be evaluated. Keep multi-row tiling,
threading and new dtypes as separate follow-up slices.

Run finite tolerance checks (`1e-5 + 1e-5 * abs(reference)` initially), special
values, all supported attributes/bias/stride/tail cases, changed runtime operands,
independent fixture references, portable/native Release and targeted sanitizers.
Compare new execution against Phase 4.4 prepared execution on identical graphs,
inputs, buffer policy and ISA/compiler. Repeat each case in three processes.
Require repeatable target-case improvement, inspect p95 as well as p50, and
resolve repeatable control regressions over 5%. Publish Mac and ROG results
independently. A component-cost estimate does not establish an achieved speedup.

## Deliverables

- Reproducible benchmark/runner with numerical checks and measurement boundaries.
- Local analysis report and target-host reproduction commands.
- Evidence-based kernel support specification before implementation.
- Correctness and paired prepared-versus-new performance acceptance for the
  eventual kernel change. Completing analysis alone does not complete Phase 4.5.

## Analysis result and implementation order (2026-10-07)

The [Mac report](../benchmarks/mac-cpu-phase4.5.md) records 25 cases × 3 processes
in native AVX2 and portable SSE2 Release (150 numerically checked processes).
The [appendix](../benchmarks/mac-cpu-phase4.5-details.md) retains per-process
p50/p95/mean. Production execution, graph semantics and public API are unchanged.

Code inspection and bias controls support common-bias specialization first:

1. Introduce an explicit epilogue layout descriptor and direct scalar/vector/row/
   full loops, retaining generic broadcasting for other strides/layouts. Classify
   static constants at preparation; validate current dynamic bias layout on run.
2. Measure this specialization against the current epilogue before integrating
   eligible affine/ReLU operations into final K-block stores. Preserve arithmetic
   and special-value behavior; keep the same packed RHS format and MatMul tiling.
3. Evaluate multi-row reuse separately for compute-heavy shapes. Tensor contiguous
   bulk copies are a separate application/runtime candidate; preserve deep-copy
   ownership and measure real bind-run-extract costs before accepting changes.

The tiny fixtures show very short matrix diagnostics and visible epilogue/
ordinary-operator overhead. Larger K512 cases remain mainly matrix computation.
These controls cannot establish exact stage percentages or a future speedup.
ROG Phase 4.5 reproduction was pending at this stage; the 2026-10-09 results
below complete reproduction and record the remaining host-specific failures.


## First kernel slice result (2026-10-08)

The decoded epilogue, direct broadcast traversal and final-K register fusion
candidate are implemented. Local native/portable Release suites passed 41/41;
ASan/UBSan passed four targeted tests. A new runner compares three explicit
prepared modes in 26 cases × three processes × two builds (156 checked processes).
See the [fusion report](../benchmarks/mac-cpu-phase4.5-fusion.md) and
[per-process evidence](../benchmarks/mac-cpu-phase4.5-fusion-details.md).

Specialized is the default: common bias workloads and fixtures improve, while
no-bias/no-op/dynamic-B controls have no repeatable >5% paired regression.
Fused is an explicit experimental mode. It frequently trails specialization
and regresses on SSE2 controls, so register fusion performance is not accepted.
Nonunit biased coefficients retain the old helper; strided bias columns use
specialization. Dynamic B, packing format, reduction and buffer policies are unchanged.

Next acceptance work:

1. Inspect generated SIMD loops/final-K dispatch and register pressure; remove
   the observed SSE2 overhead, then repeat paired controls before enabling fusion.
2. ROG reproduction completed on 2026-10-09, native and portable, with all
   process statistics and environment metadata retained. Resolve the native
   specialization no-bias/ReLU control regression before accepting that host.
3. Validate NEON on ARM before claiming that backend. Evaluate multi-row tiling
   separately once the fused candidate has a justified support/selection policy.

Phase 4.5 remains open for general register fusion performance, not for
unexecuted ROG reproduction. The native specialization gate failure below is
historical; the 2026-10-10 targeted fix and balanced-order rerun resolves it.

## ROG test closure (2026-10-09)

Fresh MSVC Release builds passed 41/41 tests each, with assertions retained.
The analysis runner completed 25 cases × 3 processes × 2 builds (150); the
three-mode runner completed 26 cases × 3 processes × 2 builds (156). Three
controls were independently rechecked in three processes per build (18).
All 324 benchmark processes passed numerical and packing/dispatch checks.

See the [ROG analysis](../benchmarks/rog-cpu-phase4.5.md) and its
[per-process appendix](../benchmarks/rog-cpu-phase4.5-details.md), plus the
[three-mode acceptance report](../benchmarks/rog-cpu-phase4.5-fusion.md) and
[initial/recheck evidence](../benchmarks/rog-cpu-phase4.5-fusion-details.md).

Native AVX2 specialization improves common biased workloads and both fixtures,
but the no-bias/ReLU control fails the 5% latency regression gate in all three
initial and all three follow-up processes (about 9–11% and 10–14% more latency).
Native register fusion also fails the no-op control in every initial/recheck
process. Therefore testing is complete, but native specialization performance
acceptance and general fusion acceptance are not complete. Existing production
defaults were not changed by the validation.

MSVC portable reports explicit kernel width 1 (scalar fallback); native reports
8. Portable specialization meets its control gate, but this is not validation
of the Mac SSE2 width-4 path. SIMD feature detection on MSVC and NEON execution
remain explicit boundaries. ROG sanitizer execution is not claimed; historical
Mac ASan/UBSan evidence is preserved.

Next: remediate the native no-bias/ReLU specialization path and fused no-op
overhead, then rerun the target/control gates without discarding these results.
Inspect code generation before attributing the regressions to a particular
cause. Keep portable SIMD detection and ARM execution as separately verified
work; do not enable experimental fusion from biased-target gains alone.

## Targeted ROG fixes (2026-10-10)

The no-bias epilogue now reuses the existing linear helper instead of evaluating
generic layout/attribute branches per output element. Prepared no-bias
specialization skips descriptor binding; identity Fused requests select plain
packed MatMul and are no longer counted as register fusion. Both full Release
suites pass 41/41 with additional special-value, zero-K, tail and counter tests.

Fixed-allocation-order intermediate results are retained, including a remaining
no-bias median failure after the shared-call fast path. The benchmark now cycles
plan/context allocation positions across processes as well as rotating timing
order within each process. This controls the fixed mode/buffer-position
confound without asserting a particular hardware stall cause or attributing the
protocol change to kernel speedup.

The final 26 × 3 × 2 comparison and 18 independent balanced control processes
pass numerical/counter checks. The two requested native target controls pass
the 5% gate in all six initial/recheck processes each. Every full-suite
Specialized case median and each recheck control median passes in native and
portable; isolated non-target process outliers remain documented. MSVC portable
still selects scalar width 1, not SSE2. No default-mode change was made.

See the [fix and code-generation analysis](../benchmarks/rog-cpu-phase4.5-fix.md)
and [all intermediate/final statistics](../benchmarks/rog-cpu-phase4.5-fix-details.md).
General fusion remains experimental: the native fused no-bias/ReLU path can
still trail Specialized, and Mac SSE2/ARM acceptance is not supplied by this run.
