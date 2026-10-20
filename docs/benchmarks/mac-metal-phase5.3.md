# Mac Metal Phase 5.3 — end-to-end performance

> 历史报告：2026-10-20 同门槛复测已通过，最新状态见[收尾报告](mac-metal-phase5.3-closeout.md)。本页保留当时失败结果。

Date: 2026-10-13 (Asia/Shanghai). Hardware: Intel Iris Plus Graphics 645, x86_64 Mac, macOS 15.8.1. Release native CPU Prepared Auto baseline.

**Status: measurements and fallback-copy fix delivered; performance acceptance NOT passed.** The frozen target was retained. These measurements do not justify advancing the first Metal performance milestone as accepted.

## Method

See [the protocol](../plans/phase5.3.md). Each version has two rounds of three sequential independent processes, five warmups, and 31 samples per warm mode. Backend/sample and process/model starting orders alternate. p50/p95 use nearest rank. No sample or process is discarded. End-to-end includes host input snapshot, completed inference and independent host output; stage and resident timings are independent diagnostics. Hardware completion is waited; no submission-only timing is used. All finite results pass the existing FP32 1e-5*(1+abs(reference)) check.

The two versions used the same target, inputs, sample counts and measurement harness. First results precede the fix and final results follow it; this is not a simultaneous controlled thermal experiment. Device/driver and OS state can vary. Exact source/binary hashes, build cache, source revision/dirty state and all raw samples are retained. Process-first initialization does not flush driver/disk caches.

## Frozen gates

| Version | Round | Target p50 speedup (>=1.2) | Target p95 GPU/CPU ratio (<=1.05) | Fallback p50 GPU-wrapper/CPU ratio (<=1.05) |
|---|---:|---:|---:|---:|
| Original | 1 | 1.278 | 0.417 | 1.424 |
| Original | 2 | 0.983 | 0.990 | 1.337 |
| After input-copy fix | 1 | 1.027 | 0.946 | 0.938 |
| After input-copy fix | 2 | 1.029 | 1.070 | 1.028 |

Gate values are medians of process ratios, not ratios derived from independently aggregated table latencies. Final target p50 fails both rounds, target p95 fails round 2, and fallback passes both rounds.

## Final workload results

| Workload | Round | CPU E2E p50 (ms) | Metal/wrapper E2E p50 (ms) | p50 speedup | p95 ratio |
|---|---:|---:|---:|---:|---:|
| phase3_mlp_gemm | 1 | 0.0474 | 17.2493 | 0.003 | 334.278 |
| rl_actor_mlp_tanh | 1 | 0.0432 | 18.0582 | 0.002 | 711.555 |
| control | 1 | 9.5475 | 22.7591 | 0.428 | 3.404 |
| target | 1 | 82.0817 | 75.9196 | 1.027 | 0.946 |
| fallback | 1 | 4.2265 | 3.9245 | 1.066 | 0.931 |
| phase3_mlp_gemm | 2 | 0.0452 | 17.5669 | 0.003 | 463.860 |
| rl_actor_mlp_tanh | 2 | 0.0411 | 16.4307 | 0.002 | 408.253 |
| control | 2 | 12.1142 | 24.2319 | 0.500 | 1.614 |
| target | 2 | 72.8465 | 75.8434 | 1.029 | 1.070 |
| fallback | 2 | 3.7406 | 3.8472 | 0.972 | 1.024 |

The two tiny fixtures are strongly CPU-favored; driver/command and transfer overhead dominate their very small compute work. M=64,K=N=512 remains CPU-favored in both rounds. The fixed M=128,K=N=1024 target is close to E2E parity, with process-level variation; it is not a stable >=1.2 crossover. No universal size threshold or automatic backend-selection policy is established by this sample grid.

## Final target stage diagnostics

Numbers below are medians of the three process p50s, in milliseconds. First/prepare/context values have one observation per process, so these are not within-process cold-start percentile estimates.

| Round | Backend | Prepare | Context | First inference | Input bind/upload | Completed run after bind | Host output | Resident-only | Measured E2E |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | cpu_auto | 74.059 | 60.887 | 91.942 | 7.436 | 81.475 | 6.799 | 75.557 | 82.082 |
| 1 | metal | 127.277 | 0.119 | 93.792 | 17.446 | 52.031 | 6.546 | 54.939 | 75.920 |
| 2 | cpu_auto | 59.695 | 58.997 | 96.035 | 6.531 | 68.114 | 5.929 | 72.841 | 72.846 |
| 2 | metal | 115.125 | 0.147 | 87.020 | 13.031 | 54.094 | 4.818 | 52.966 | 75.843 |

Resident Metal execution is faster in these target diagnostics, but host binding/upload and output download reduce the E2E benefit. Stage medians must not be added to replace measured E2E; scheduling, cache and thermal conditions differ between modes. The completed-run timer includes driver/host wait and reuse synchronization; no pure GPU timestamp is claimed.

## Memory and resource checks

For the target, CPU packed weights use 4,194,304 bytes and its activation arena uses 524,288 bytes. Metal constants use 4,198,400 bytes, its arena uses 524,288 bytes, and cumulative logical device allocation requests total 5,246,976 bytes (constants + input + arena). This is not a VRAM measurement: staging, pipeline and driver memory are excluded. CPU original snapshots/inputs are excluded from packed/arena counts. Each raw report also records whole-process macOS peak RSS; it cannot be attributed separately to the two backends.

Steady-state checks confirm no repeated device allocations and no uploads/downloads during resident-only runs. Correctness-output downloads outside timing remain visible in cumulative counters. Five pipelines prepare once per plan. All source and binary versions are recorded.

## Resolved fallback regression

Original wrapper fallback p50 regressions were 42.4% and 33.7%. Diagnostics showed input binding cost about twice the direct CPU path. Code inspection found that the by-value public parameter already copied the lvalue, then CPU-backup binding copied it again and the wrapper retained the first host Tensor.

The wrapper now records bound/layout flags and moves the already-owned parameter into the CPU context after any upload completes. Strict GPU contexts release the host input after upload. A counting allocator checks one lvalue snapshot, zero additional moved-input copies, preserved layout, and lvalue isolation. Final fallback latency ratios 0.938 and 1.028 meet the original 5% gate; occasional process variation is retained. No new CPU kernel or GPU shader was introduced.

## Verification and remaining work

- Actual-device Release suite, Metal API validation enabled: 46/46 passed.
- Actual-device execution-plan ASan/UBSan host instrumentation: 1/1 passed. Shader addresses are not covered.
- Twelve performance processes (six per version), all numerical/resource checks passed; validation disabled for timing.
- Target performance gates remain unmet. Keep Phase 5.3 open.

Next work: profile and reduce H2D/D2H staging/allocation/copy overhead, investigate target p95 variation, then repeat the frozen gate. Reducing resident compute or changing submission/reuse waits requires separate correctness proof. A tiled/library GEMM is a broader extension explicitly deferred by the current scope; no target substitution is proposed. CUDA/ROG performance remains untested.

## Evidence

- [Original raw results and summary](../acceptance/phase5.3-results/summary.json)
- [Final raw results and summary](../acceptance/phase5.3-results-fixed/summary.json)
- [Final environment/source/build metadata](../acceptance/phase5.3-results-fixed/metadata.json)
- [Release correctness log](../acceptance/phase5.3-correctness.log)
- [ASan/UBSan log](../acceptance/phase5.3-asan.log)
- [Benchmark source](../../benchmarks/metal_model_benchmark.cpp)
- [Sequential runner and gate evaluator](../../tools/run_phase53.py)

## Transfer follow-up

The [transfer optimization and p95 investigation](mac-metal-phase5.3-transfer.md)
removes two intermediate host Buffer copies and caches native staging with completion
leases. A further six-process same-gate run still fails the target performance gate;
profiling and environment data expose device/wait/scheduling variability. Reduced
allocation counts are established; stable E2E speedup is not.
