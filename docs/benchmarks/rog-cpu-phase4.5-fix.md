# ROG CPU — Phase 4.5 targeted regression fixes

- Date: 2026-10-10 (Asia/Shanghai).
- Source: `b60c5f2a3e053192da3ddd623e26df96b49e2eab` plus the measured runtime/test/benchmark changes.
- Status: the native Specialized no-bias/ReLU and Fused identity controls are
  remediated and pass the balanced-order full-suite plus independent recheck.
  Specialized passes both ROG configuration median/control gates.
  Register fusion remains experimental; general/default fusion acceptance is
  **not** claimed.

## Specific problems and fixes

### No-bias/ReLU specialization

The original direct epilogue used nested rows/columns and `finish()` even
without bias. MSVC diagnostic assembly retained per-element bias-layout,
alpha and ReLU descriptor tests. Legacy's no-bias helper instead has a flat
linear traversal with alpha passed once. MSVC `/Qvec-report:2` reported both
loops as not vectorized: this is **not evidence of a lost SIMD vectorization**.

The low-level Specialized helper now delegates no-bias work to the existing
linear Legacy helper after validation. The prepared dispatcher also bypasses
descriptor binding/output revalidation when no bias layout exists. It uses
the same packed MatMul and linear helper as Legacy. The Legacy condition
short-circuits first and does not acquire an identity-descriptor check merely
to improve the comparator. Bias specialization and nonunit biased fallback
are unchanged. Alpha multiplication and ReLU NaN/signed-zero semantics are
preserved, including nonunit/nonfinite alpha and zero K.

### Identity epilogue still entered register fusion

With no bias, alpha=1 and no activation, there is no epilogue arithmetic.
Previously `supports_fusion()` still classified it as eligible and called
`run_packed<true>`, retaining final-K and descriptor checks for each output
tile. This also incremented the fusion counter despite no register epilogue work.

`GemmEpilogue::is_identity()` now recognizes exactly this case (beta is
irrelevant without bias). Identity is not fusion-eligible. The low-level
entry validates the descriptor, then calls plain packed MatMul. Prepared
execution selects its simple path before descriptor binding. Invalid output
metadata and block sizes are still rejected. Identity requests record zero
fused dispatches; the successful simple fallback is a Specialized dispatch.
This removes unnecessary fusion rather than claiming arithmetic fusion sped
up an operation with nothing to fuse.

### Measurement control: independent allocation order

All three modes own separate model snapshots, packed weights and contexts.
The old benchmark rotated timing order but **always allocated Legacy first**.
After both modes shared the same no-bias arithmetic calls, fixed-order
comparisons still showed process-dependent relative slowdowns. A diagnostic
binary logged packed addresses modulo 64 as 0 or 32 across modes/processes.
That does not establish cache-line crossing, SIMD misalignment, register
pressure or a unique microarchitectural cause. No alignment/packing change
was made based on this observation.

The benchmark now accepts `--allocation-order 0|1|2`. The runner cycles the
three construction positions across processes, then restores canonical
mode/metric indices without copying payloads. Timing order still rotates per
sample. This controls a fixed mode-to-buffer-position confound; **it is a
measurement improvement, not a kernel optimization**. Host scheduling/cache
effects are not eliminated. Old fixed-order results are retained below and
in the appendix; they are not silently replaced.

## Verification and environment

| Item | Result |
| --- | --- |
| CPU | ROG Intel Core i9-13980HX; 24 cores / 32 logical processors |
| OS / power | Windows 11 Home Chinese 10.0.26200 / Performance |
| Compiler | MSVC 19.43.34808.0; VS 2022 Build Tools, Ninja Release |
| Native ON | AVX2, explicit kernel width 8 |
| Portable OFF | Scalar fallback, explicit width 1; **not SSE2 validation** |
| Shared flags | `/EHsc /permissive-`; tests retain `/UNDEBUG` |
| Final portable CTest | 41/41 passed, 4.99 s |
| Final native CTest | 41/41 passed, 3.52 s |
| Final balanced full comparison | 26 cases × 3 processes × 2 builds = 156 |
| Final independent control recheck | 3 controls × 3 processes × 2 builds = 18 |
| Warm-up / samples / repeats | 20 / 100 / 10 |
| Numerical/counter checks | All 174 final comparison/recheck processes passed |

New tests cover no-bias alpha=1/0/-0/-1/Inf/NaN, irrelevant NaN beta, None/ReLU,
NaN/Inf/signed zero outputs, zero K, multiple K blocks, scalar columns/custom
blocks, invalid descriptors/blocks and prepared identity counters. Existing
tests retain transpose/stride/bias/rebinding and independent fixture checks.

The first incremental build mixed updated kernel objects with an old execution
plan object because the local Ninja/MSVC include prefix was garbled; the new
identity counter test caught it. Fresh `build-phase45-fix-{native,portable}`
directories rebuilt all header users and passed 41/41 each (8.55/7.19 s).
The subsequent prepared-dispatch change explicitly recompiled its .cpp and
passed the full suites again; the final times above refer to that version.
Benchmark-only allocation-order changes did not change runtime/test code.
All builds/tests completed before each measurement batch. No ROG sanitizer
or ARM execution is claimed, and no automatic/default fusion was enabled.

Packing format, reduction order, block sizes, activation payloads, snapshot
ownership and dynamic-B fallback are unchanged. Identity counts now report
actual eligibility rather than a requested mode. Every session records 1,021
runs; dynamic B packs 1,021 times, static cases zero times. Identity Fused
contexts have `fused_gemm_count=0` and `specialized_gemm_count=1021`.

## Target controls before and after

A speedup is Legacy time / candidate time. Above one is faster. The 5% latency
regression threshold is `1 / 1.05 = 0.95238`.
All entries below are ranges of per-process **paired-p50** ratios, not ratios
of unrelated historical absolute latencies. The new protocol changes allocation
balance, so old/new differences must not be presented as an isolated kernel
speedup measurement.

| Build / control / candidate | Original fixed-order initial | Final balanced initial | Final balanced recheck |
| --- | ---: | ---: | ---: |
| native / bias_none / Specialized | 0.903–0.917 | 0.995–1.020 | 0.984–1.009 |
| native / no_epilogue / Fused | 0.916–0.942 | 0.978–0.999 | 0.979–1.028 |
| portable / bias_none / Specialized | 0.981–0.985 | 0.995–1.024 | 0.995–1.005 |
| portable / no_epilogue / Fused | 0.976–1.034 | 0.997–1.000 | 0.953–0.997 |

Native target ratios exceed the threshold in all six final initial/recheck
processes per target. Portable target ratios do as well. In both builds every
full-suite Specialized case median and every recheck control median passes
the 5% gate. Individual non-target controls still vary: native initial
Specialized no-op has a 0.931 process, and portable recheck Specialized no-op
has a 0.918 process, while their three-process medians pass. Those outliers
are retained, not declared nonexistent.

Register fusion is not a uniformly better replacement for specialization:
native no-bias/ReLU has a full-suite median >5% latency regression versus
Specialized; one independent process has Fused/Legacy speedup 0.917. This
separate fusion/ReLU path remains experimental. Existing Mac width-4 SSE2
fusion issues are not validated by this MSVC scalar portable build.

p95/mean and p95 quotients are retained in the
[complete appendix](rog-cpu-phase4.5-fix-details.md). Tail variability is
not converted into a universal latency guarantee. Temperature, frequency,
affinity, RSS and unrelated host activity were not controlled or recorded.
These results do not identify exact hardware stall percentages.

## Intermediate evidence retained

| Stage | Runtime change | Protocol | Processes |
| --- | --- | --- | ---: |
| Low-level candidate | No-bias linear helper + identity MatMul guard | Fixed allocation order, full suite + recheck | 174 |
| Prepared fast-path candidate | Also bypass no-bias descriptor binding | Fixed allocation order, full suite | 156 |
| Final validation | Same prepared/runtime code | Cyclic allocation order, full suite + recheck | 174 |

The low-level candidate removed the repeated initial no-bias failure at the
three-process median but retained a 0.940 individual process. The subsequent
prepared fast-path candidate, still fixed-order, recorded native no-bias ratios
0.967, 0.923, 0.949 (median gate failure). These are shown in full rather than
concealed by the final balanced protocol. Rotating construction addresses the
comparison confound; it does not prove the precise cause of every outlier.
Three alignment diagnostics and three allocation-order probes are additional
investigations, not part of the 504 comparison/recheck process count.

## Final full-suite summary

Latencies are medians of three process p50 values in microseconds. Speedups
are independently aggregated process paired medians; division of displayed
latencies need not reproduce them.

### native

| Case | Legacy p50 | Specialized p50 | Fused p50 | Specialized / Legacy | Fused / Legacy | Fused / Specialized |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| m1_k128_n128_t0 | 2.870 | 1.450 | 1.390 | 2.033 | 2.050 | 1.014 |
| m1_k128_n128_t1 | 3.110 | 1.510 | 1.400 | 2.002 | 2.237 | 1.050 |
| m16_k128_n128_t0 | 28.790 | 14.050 | 13.295 | 2.060 | 2.213 | 1.058 |
| m16_k128_n128_t1 | 26.550 | 12.620 | 12.520 | 2.115 | 2.194 | 1.024 |
| m64_k128_n128_t0 | 103.980 | 49.920 | 47.590 | 2.086 | 2.175 | 1.046 |
| m64_k128_n128_t1 | 112.350 | 55.075 | 53.180 | 2.059 | 2.177 | 1.056 |
| m1_k512_n512_t0 | 38.985 | 28.700 | 30.130 | 1.134 | 1.104 | 0.959 |
| m1_k512_n512_t1 | 26.455 | 23.110 | 23.535 | 1.144 | 1.117 | 0.991 |
| m16_k512_n512_t0 | 302.620 | 245.070 | 249.095 | 1.242 | 1.217 | 0.981 |
| m16_k512_n512_t1 | 294.000 | 237.110 | 241.160 | 1.248 | 1.216 | 0.983 |
| m64_k512_n512_t0 | 1224.700 | 983.615 | 1008.305 | 1.235 | 1.209 | 0.981 |
| m64_k512_n512_t1 | 1201.020 | 958.695 | 1001.225 | 1.247 | 1.209 | 0.969 |
| m3_k127_n131_t0 | 7.265 | 3.455 | 3.340 | 2.106 | 2.141 | 1.024 |
| m3_k127_n131_t1 | 7.600 | 3.660 | 3.680 | 2.085 | 2.088 | 1.025 |
| m16_k128_n128_bias_none | 12.945 | 12.820 | 13.575 | 0.998 | 0.964 | 0.950 |
| m16_k128_n128_bias_scalar | 19.260 | 13.620 | 13.390 | 1.413 | 1.447 | 1.025 |
| m16_k128_n128_bias_row | 26.830 | 13.410 | 12.615 | 2.087 | 2.145 | 1.068 |
| m16_k128_n128_bias_column | 27.325 | 12.950 | 12.490 | 2.092 | 2.186 | 1.045 |
| m16_k128_n128_bias_full | 21.390 | 14.850 | 14.130 | 1.387 | 1.473 | 1.063 |
| m16_k128_n128_no_relu | 28.250 | 18.440 | 12.940 | 1.665 | 2.220 | 1.374 |
| m16_k128_n128_no_epilogue | 12.050 | 12.370 | 12.340 | 0.991 | 0.983 | 1.008 |
| m16_k128_n128_dynamic | 220.525 | 219.410 | 220.485 | 1.005 | 1.000 | 0.996 |
| m3_k127_n128_aligned_n | 6.365 | 2.840 | 2.760 | 2.255 | 2.231 | 0.977 |
| m3_k128_n128_aligned_kn | 6.760 | 3.060 | 2.860 | 2.253 | 2.345 | 1.038 |
| phase3_mlp_gemm | 2.720 | 1.100 | 1.110 | 2.464 | 2.464 | 1.000 |
| rl_actor_mlp_tanh | 3.330 | 1.810 | 1.820 | 1.835 | 1.834 | 1.000 |

### portable

| Case | Legacy p50 | Specialized p50 | Fused p50 | Specialized / Legacy | Fused / Legacy | Fused / Specialized |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| m1_k128_n128_t0 | 8.755 | 7.200 | 7.280 | 1.214 | 1.196 | 0.986 |
| m1_k128_n128_t1 | 6.365 | 5.310 | 5.090 | 1.252 | 1.280 | 1.023 |
| m16_k128_n128_t0 | 94.940 | 79.220 | 81.515 | 1.182 | 1.167 | 0.980 |
| m16_k128_n128_t1 | 96.370 | 80.730 | 81.855 | 1.186 | 1.174 | 0.991 |
| m64_k128_n128_t0 | 402.505 | 357.630 | 346.855 | 1.172 | 1.166 | 0.995 |
| m64_k128_n128_t1 | 422.540 | 358.210 | 360.530 | 1.173 | 1.157 | 0.992 |
| m1_k512_n512_t0 | 112.435 | 109.590 | 113.525 | 1.030 | 0.986 | 0.965 |
| m1_k512_n512_t1 | 108.125 | 104.335 | 108.680 | 1.026 | 0.998 | 0.977 |
| m16_k512_n512_t0 | 1975.730 | 1946.060 | 2019.975 | 1.046 | 1.004 | 0.966 |
| m16_k512_n512_t1 | 1889.460 | 1833.045 | 1916.650 | 1.028 | 0.997 | 0.969 |
| m64_k512_n512_t0 | 7472.140 | 7209.820 | 7374.575 | 1.007 | 0.991 | 0.988 |
| m64_k512_n512_t1 | 6652.640 | 6464.520 | 6740.805 | 1.036 | 1.010 | 0.968 |
| m3_k127_n131_t0 | 21.230 | 17.225 | 17.270 | 1.235 | 1.222 | 0.990 |
| m3_k127_n131_t1 | 25.675 | 20.615 | 20.665 | 1.251 | 1.212 | 0.988 |
| m16_k128_n128_bias_none | 97.660 | 94.850 | 99.105 | 1.003 | 0.980 | 0.978 |
| m16_k128_n128_bias_scalar | 103.115 | 92.385 | 95.215 | 1.073 | 1.063 | 0.987 |
| m16_k128_n128_bias_row | 120.720 | 104.710 | 105.175 | 1.176 | 1.166 | 0.988 |
| m16_k128_n128_bias_column | 112.000 | 93.245 | 95.940 | 1.182 | 1.167 | 0.987 |
| m16_k128_n128_bias_full | 115.535 | 107.745 | 108.750 | 1.066 | 1.060 | 0.984 |
| m16_k128_n128_no_relu | 103.690 | 87.830 | 89.490 | 1.183 | 1.170 | 0.990 |
| m16_k128_n128_no_epilogue | 86.420 | 90.985 | 86.880 | 0.994 | 0.997 | 1.002 |
| m16_k128_n128_dynamic | 378.540 | 381.895 | 380.600 | 0.998 | 0.997 | 1.001 |
| m3_k127_n128_aligned_n | 21.985 | 18.400 | 17.780 | 1.220 | 1.219 | 1.000 |
| m3_k128_n128_aligned_kn | 33.670 | 26.735 | 27.575 | 1.236 | 1.218 | 0.991 |
| phase3_mlp_gemm | 4.060 | 1.630 | 1.630 | 2.488 | 2.485 | 1.000 |
| rl_actor_mlp_tanh | 4.240 | 2.310 | 2.290 | 1.844 | 1.850 | 1.005 |

## Reproduction and artifacts

Use a working Visual Studio x64 developer environment. Prefer fresh directories
to avoid the observed local header-dependency issue; see the
[initial build commands](rog-cpu-phase4.5-fusion.md#reproduction), replacing
build directories with `build-phase45-fix-native` and
`build-phase45-fix-portable`. Build/test both before benchmarking.

```powershell
python benchmarks/run_phase45_fusion.py --executable build-phase45-fix-native/benchmarks/tinyinfer_cpu_performance_analysis_benchmark.exe --output benchmark-results/phase4.5-fix-balanced/rog-native --warmup 20 --samples 100 --repeats 10 --runs 3
python benchmarks/run_phase45_fusion.py --executable build-phase45-fix-portable/benchmarks/tinyinfer_cpu_performance_analysis_benchmark.exe --output benchmark-results/phase4.5-fix-balanced/rog-portable --warmup 20 --samples 100 --repeats 10 --runs 3
```

For independent controls, run each executable in fresh processes with
`--fusion-comparison --allocation-order 0`, then 1, then 2, keeping the same
20/100/10 settings. Controls are `--bias none`, `--bias none --activation none`,
and `--dynamic-b`. Invalid allocation order 3 is rejected. Default CLI order 0
retains fixed-order construction, whereas the runner balances its process orders.

Raw artifacts (ignored): `benchmark-results/phase4.5-fix` (low-level),
`benchmark-results/phase4.5-fix-final` (prepared/fixed), and
`benchmark-results/phase4.5-fix-balanced` (final balanced and recheck).
The original [2026-10-09 report](rog-cpu-phase4.5-fusion.md) measurement tables and raw artifacts
are retained as historical evidence. Compiler diagnostic logs/assembly
remain in the first directory; the independent alignment diagnostic source,
binary and logs remain in the second. No preparation or isolated-component
speedup is claimed from these whole-run comparisons.

