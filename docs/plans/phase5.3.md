# Phase 5.3 — Metal performance acceptance

Date: 2026-10-13 (Asia/Shanghai). Baseline: Phase 5.2 `8c919c9`.
This stage measures the existing implementation; it does not change its kernels
or completion/arena reuse semantics.

## Frozen gate

Retain the Phase 5.0 target: static FP32 Gemm + vector bias + ReLU,
M=128, K=N=1024, alpha=beta=1, transA=transB=0. Measure two rounds with three
independent processes per round. Each round's median of process p50 speedups
must be >=1.2; the median of process GPU/CPU p95 ratios must be <=1.05.
Whole-model CPU fallback p50 ratio must be <=1.05, or require an investigation.
No target substitution or relaxed numerical tolerances after seeing results.

Controls: the two ONNX fixtures; M=64,K=N=512 Gemm/bias/ReLU; CPU fallback
GELU on 64x512 values. The matrix/bias/ReLU expression is represented by the
supported FusedGemmActivation node for both Prepared CPU Auto and Metal.
Inputs are deterministic FP32, with two values alternated for measured E2E calls.
CPU and Metal use the same host Tensor lvalue binding and independent output
ownership. Correctness gate: 1e-5*(1+abs(reference)); NaN/Inf are not admitted for
these finite workload inputs. Every measured E2E output is checked outside its timer.

## Timers

- Model import (fixtures) or construction (synthetic): once per process/case.
- Preparation: owned snapshot, CPU packing or Metal library/pipeline compilation,
  constant upload and completion. One sample per process/case.
- Context creation: writable buffers, once per process/case.
- First inference: binding + completed run + independent host output.
- Warm E2E: host input -> binding -> completed inference -> owned host output;
  includes destruction of the temporary output before the timer ends.
- Independent diagnostic input binding, completed resident run, output copy.
  Binding includes CPU Tensor snapshot and Metal host staging/upload/wait.
  Output includes host allocation and staging/download/wait. These are API-boundary
  wall times, not isolated bus transfer throughput; do not sum stage percentiles.
- Resident-only run: no rebind or host output copy. Runtime counts must show
  no additional logical device allocation/upload/download. This is diagnostic,
  not a basis for an application E2E speedup claim.

Completed run is host-observed elapsed time including dispatch/driver/wait and
necessary reuse waits; it is not a hardware timestamp for pure GPU kernel time.
First inference and preparation are process-first measurements. No attempt is
made to flush driver/compiler/disk caches, so they are not hardware cold-start
claims. Within one process, model cases also share driver caches.

## Reproduction

Release, native CPU architecture enabled, Metal required and benchmarks enabled:

```bash
cmake -S . -B build-bench -DCMAKE_BUILD_TYPE=Release \
  -DTINYINFER_ENABLE_NATIVE_ARCH=ON -DTINYINFER_ENABLE_METAL=ON \
  -DTINYINFER_METAL_REQUIRE_DEVICE=ON -DTINYINFER_BUILD_BENCHMARKS=ON
cmake --build build-bench --target tinyinfer_metal_model_benchmark -j 4
python3 tools/run_phase53.py \
  --binary build-bench/benchmarks/tinyinfer_metal_model_benchmark \
  --output phase53-results --samples 31 --warmup 5
```

For this Mac's SDK include workaround, see Phase 5.1 acceptance. Use host GPU
access. No GPU skip or CPU substitute is permitted. Performance runs disable
Metal debug/shader validation; correctness tests use validation separately.

Each process uses five warmups and 31 samples per steady-state mode. Backend
order alternates per sample; process starting order and model order alternate.
Six processes run sequentially with no other benchmarks run concurrently by this
workflow. Nearest-rank p50/p95 uses ceil(n*p), with no interpolation. Full raw
sample arrays, process stderr, build cache, source/binary hashes, commit/dirty
state, device/OS/SDK/compiler, and gate calculations are retained by the runner.

Memory: device allocated bytes are cumulative logical allocation requests;
there are no releases/reallocations during the measured workload. This includes
constants, input buffers and the context arena, but excludes Metal staging,
pipelines and driver internals. CPU weight bytes count packed weights and
activation bytes count the arena, excluding original Model snapshots and inputs.
Process peak RSS is a whole-process high-water mark on macOS (bytes), shared by
both backend reports; it does not isolate CPU versus GPU ownership.

Crossovers are observations across these selected model sizes, not a fitted
universal threshold. Integrated Intel GPU/CPU share machine resources; thermal
and OS scheduling variation remain visible in the raw data. Results apply to
this Mac, not untested ROG/CUDA or Apple Silicon.

## First delivery outcome

Two versions, twelve actual-device processes, all numerical/resource checks passed.
The redundant fallback input copy was fixed and the fallback gate now passes.
Target p50/p95 gates remain unmet. See [the full report](../benchmarks/mac-metal-phase5.3.md).
Phase 5.3 remains open; do not label the first Metal performance milestone accepted.

## Transfer-cost follow-up

The follow-up retains the same gate and sampling protocol. Prepared input uploads
use the owned Tensor Buffer and byte offset directly; host outputs receive downloads
directly in independent Tensor storage. A bounded two-entry native staging cache
uses terminal-completion leases and reports allocations/reuse/cached capacity.
Normal gate measurements leave command profiling disabled. Separate target-only
CPU-first and GPU-first diagnostic processes record completed GPU command durations,
CPU process time, host waits and involuntary context-switch counts per E2E sample.
These diagnostic processes are excluded from all acceptance ratios. Command GPU
intervals include blits and compute, and cannot isolate scheduling, thermal or other
GPU-client causes. CPU time/context switches describe the benchmark process, not
whole-system load. The normal runner records load averages and power source before
and between processes; these snapshots do not prove within-sample causes.
