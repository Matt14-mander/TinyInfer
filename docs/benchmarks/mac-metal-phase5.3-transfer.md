# Mac Metal Phase 5.3 — transfer optimization and p95 investigation

> 历史报告：2026-10-20 同门槛复测已通过，最新状态见[收尾报告](mac-metal-phase5.3-closeout.md)。本页保留当时失败结果。

Date: 2026-10-13 (Asia/Shanghai). Baseline source: `bcf2c72`. Same Intel Iris Plus Graphics 645 Mac, Release native Prepared CPU Auto, original frozen workloads and FP32 correctness tolerance.

**Outcome: transfer resource costs reduced; performance gates remain unmet.** No workload, threshold, sample count or outlier policy was changed.

## Implementation

- Upload directly from the by-value Tensor snapshot Buffer at its storage byte offset. Remove the extra C++ Buffer allocation and memcpy.
- Download directly into the independent output Tensor Buffer. Remove the extra C++ Buffer allocation and memcpy after download. Owned output/lvalue isolation and contiguous view offsets remain covered.
- Cache at most two native shared staging buffers per runtime, with completion leases. Downloads retain the lease until host publication; busy stages are never reused, and extra concurrent requests use uncached buffers. Cache entries can grow, so the count bound is not a small fixed byte bound.
- Add native staging allocation/reuse/cached-capacity diagnostics. Target preparation plus all benchmark transfers now use **one native staging allocation and 171 reuses**; the previous implementation allocated a new native staging Buffer for every nonempty upload/download.
- The target retains **4,194,304 bytes (4 MiB)** of shared staging capacity until runtime destruction. Logical device allocation bytes still exclude this separately reported cache and driver overhead.
- Optional completed-command GPU timestamp/host-wait profiling is off by default. Target-only profiling runs are separate and excluded from acceptance.

Synchronization, private device resources, output ownership, arena reuse completion proof, shaders and numerical precision are unchanged. No command failure is retried on CPU.

## Same-gate reproduction

Two rounds, three independent sequential processes each; five warmups, 31 samples per warm timing mode. Backend/sample and model/process order alternate. All six runs pass numerical and resource checks, including no warm native staging allocation. No samples or processes are discarded.

```bash
python3 tools/run_phase53.py \
  --binary build-bench/benchmarks/tinyinfer_metal_model_benchmark \
  --output phase53-transfer-results --samples 31 --warmup 5
```

| Version | Round | Target p50 speedup, >=1.2 | Target p95 GPU/CPU, <=1.05 | Fallback p50 ratio, <=1.05 |
|---|---:|---:|---:|---:|
| Before transfer changes | 1 | 1.027 | 0.946 | 0.938 |
| Before transfer changes | 2 | 1.029 | 1.070 | 1.028 |
| After transfer changes | 1 | 0.995 | 1.127 | 0.980 |
| After transfer changes | 2 | 1.073 | 0.974 | 0.979 |

Values are medians of the three process ratios. Target p50 fails both rounds; target p95 fails round 1; fallback passes both rounds. Before/after runs are sequential measurements under varying system conditions, so their differences are not clean estimates of the optimization effect. Reduced allocation/copy counts are proven; stable E2E speedup is not.

## Final target phase measurements

Milliseconds, median of three process p50s per round. Phase diagnostics are independent and must not be summed to substitute for measured E2E.

| Round | Backend | Input bind/upload | Completed run | Host output | Resident-only | E2E p50 | E2E p95 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | cpu_auto | 5.793 | 64.491 | 5.223 | 75.489 | 77.203 | 88.097 |
| 1 | metal | 13.433 | 52.981 | 12.590 | 55.031 | 77.629 | 98.954 |
| 2 | cpu_auto | 5.231 | 59.300 | 5.032 | 60.799 | 72.394 | 107.895 |
| 2 | metal | 12.993 | 41.095 | 11.380 | 50.462 | 71.093 | 101.094 |

## p95 diagnostic observations

Two separate target-only profiled processes use the same 31 samples/five warmups. Three valid completed command timestamp intervals are present for each Metal E2E sample: upload, compute and download. GPU values sum command intervals, not pure compute kernel time. `host_wait_us` measures completion waiting; subtracting GPU time does not yield exact queue delay. CPU process time and involuntary context-switch counts are process-wide counters read around each call, outside the E2E timer.

| Profile | Backend | E2E p50 ms | E2E p95 ms | Median GPU command ms | Median host wait ms | Median process CPU ms |
|---|---|---:|---:|---:|---:|---:|
| cpu-first.json | cpu_auto | 76.053 | 117.988 | 0.000 | 0.000 | 69.795 |
| cpu-first.json | metal | 56.558 | 89.873 | 36.692 | 49.678 | 7.397 |
| gpu-first.json | cpu_auto | 76.290 | 103.967 | 0.000 | 0.000 | 70.322 |
| gpu-first.json | metal | 74.839 | 105.353 | 52.358 | 68.337 | 7.336 |

CPU rows have no GPU/wait metric; their displayed zeros mean not applicable. Profiling affects execution and must not be used for the performance gate.

Specific retained samples:

- CPU-first Metal: median E2E 56.558 ms, median command duration 36.692 ms, median wait 49.678 ms. A tail sample takes 92.978 ms, with 53.051 ms of command durations and 86.348 ms of host wait. GPU durations and host completion time both vary.
- GPU-first Metal: median E2E 74.839 ms; the worst sample takes 106.962 ms, with 69.497 ms of command durations and 100.802 ms of wait.
- CPU-first CPU: median E2E 76.053 ms and process CPU time 69.795 ms. Its worst sample takes 147.348 ms with 82.829 ms of process CPU time and 413 involuntary switches. The substantial wall/CPU-time gap is evidence of time not executing this process on CPU; it does not identify a specific competing app.

These observations do not support treating p95 as only host memcpy/allocation cost. Device-command variability and host completion/scheduling delays coexist. Thermal, frequency, queue scheduling and concurrent GPU activity were not isolated, so no single root cause is established.

## Environment limitations

The runner now retains power source, load average and start/end timestamps for each process. All six runs were on AC power. End-of-process one-minute load averages ranged from **9.72 to 19.51** on a machine with **4 physical / 8 logical CPUs**. Load average includes more than CPU utilization; these are coarse snapshots, not per-sample causal measurements. They indicate an unsuitable environment for a strong isolated-performance claim.

A read-only post-run snapshot records active GUI/WebKit and other processes, and no reported CPU thermal warning/speed limit (scheduler/speed limit 100). It is a post-run observation and cannot prove conditions throughout each measured sample or exclude GPU frequency changes. No external process was stopped or altered. The failed gate remains recorded instead of attributing it entirely to background load.

## Verification and next boundary

- Metal API-validation Release suite: **46/46** passed.
- Metal ASan/UBSan host instrumentation: **3/3** passed; shader memory is not instrumented.
- New runtime tests cover warm reuse, multiple queued transfers with distinct sentinels, reverse waits, host publication, bounded cache and optional profiling. Existing lifetime/error/zero/offset tests remain active.
- Six unprofiled acceptance processes plus two excluded diagnostic processes; all FP32 checks passed.
- Phase 5.3 remains unaccepted. Repeat the original gate on an idle Mac before making timing-based algorithm changes. Continue profiling private-buffer transfer/command-wait overhead if target still fails in an idle environment. A tiled/library GEMM remains a separately scoped extension; this follow-up did not add one.

## Evidence

- [Transfer-version raw reports, environments and gate summary](../acceptance/phase5.3-results-transfer/summary.json)
- [Environment/source/binary/build metadata](../acceptance/phase5.3-results-transfer/metadata.json)
- [Profiling sample/tail analysis](../acceptance/phase5.3-transfer-profiles/analysis.json)
- [Post-run system observation](../acceptance/phase5.3-transfer-profiles/system-after.json)
- [Release test log](../acceptance/phase5.3-transfer-correctness.log)
- [ASan/UBSan log](../acceptance/phase5.3-transfer-asan.log)
