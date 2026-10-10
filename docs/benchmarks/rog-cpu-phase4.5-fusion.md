# ROG CPU — Phase 4.5 specialization and register fusion

- Measurement date: 2026-10-09 (Asia/Shanghai).
- Update (2026-10-10): the two reported regressions are remediated; see the
  [targeted fix report](rog-cpu-phase4.5-fix.md) and its full intermediate/final
  evidence. The measurements and failed gates below are historical, not the
  current specialization acceptance status. General fusion remains experimental.
- Selection closure (2026-10-10): [conservative Auto acceptance](rog-cpu-phase4.5-selection.md)
  supersedes the default-mode status below within a narrow measured MSVC AVX2
  policy; unrestricted Fused remains experimental. Historical data is unchanged.
- Status at measurement: ROG correctness and benchmark campaign complete; **native specialization
  performance gate fails** on the no-bias/ReLU control. Register fusion is not
  accepted as the default. Phase 4.5 remains open for performance remediation.
- Source: `9394f9f4aff08e7d6ce4ce3627b36e3b59a9ecbb`, clean tracked worktree during measurement.
- No production code, mode default or compiler-detection logic was changed by
  this validation. Specialized remains the existing default; Fused remains an
  explicit experimental mode.

## Verification and environment

| Item | Result |
| --- | --- |
| Host | ROG, Intel Core i9-13980HX; 24 cores / 32 logical processors |
| OS / power scheme | Windows 11 Home Chinese 10.0.26200 / Performance |
| Compiler / generator | MSVC 19.43.34808.0 / Ninja, Visual Studio 2022 Build Tools |
| Shared configuration | Release, `/EHsc /permissive-`, assertions enabled with `/UNDEBUG` |
| Native ON | `/arch:AVX2`, measured explicit kernel width 8 |
| Portable OFF | Measured explicit kernel width 1, scalar fallback; **not SSE2** |
| portable CTest | 41/41 passed, 8.01 s |
| native CTest | 41/41 passed, 7.36 s |
| Component analysis | 25 × 3 × 2 = 150 checked processes |
| Three-mode comparison | 26 × 3 × 2 = 156 checked processes |
| Independent control recheck | 3 controls × 3 processes × 2 builds = 18 checked processes |
| Total benchmark processes | 324; all numerical and counter checks passed |
| Timing parameters | Warm-up 20, samples 100, repeats 10 |

The current explicit SIMD selector uses `__AVX2__` / `__SSE2__`.
The MSVC native-OFF binary reports width 1; compiler-generated vectorization
may still occur. Consequently these portable results do **not** validate the
width-4 SSE2 implementation measured on Mac, nor the unexecuted NEON backend.
Do not infer SSE2 support solely from an x86-64 build.

The tests cover common bias layouts, transpositions, None/ReLU, nonunit
coefficients/fallback, current dynamic bias layouts and changed bindings,
tails/strides/custom blocks, zero dimensions, final-K-only application, NaN,
infinities, signed zeros, and independent ONNX fixture references. ROG
ASan/UBSan was not run; historical Mac targeted sanitizer evidence remains
in the [Mac report](mac-cpu-phase4.5-fusion.md).

Build warnings include existing source code-page warnings C4819 and the
intentional NDEBUG override D9025. Build and test logs are saved under
`benchmark-results/phase4.5/rog-validation`. CPU temperature, frequency,
affinity, RSS and unrelated system activity were not recorded/controlled.

## Modes and measurement boundary

Legacy explicitly selects the Phase 4.4 prepared MatMul plus generic epilogue.
Specialized decodes broadcast strides for direct epilogue traversal.
Fused applies eligible affine/bias/ReLU in final K-block register stores.
All three modes have identical graph, bound inputs, RHS packing and activation
policy. Dynamic B stays on the existing fallback; input/output deep copies,
setup and output checks are outside warm timers. Mode order rotates each sample.

Bias specialization/fusion requires alpha=beta=1. No-bias supports arbitrary
alpha. Nonunit biased coefficients retain Legacy, strided bias columns retain
Specialized, and runtime bias is validated on each call. Packing format,
MatMul tiling and reduction order are unchanged.

Each process compares the two candidates to Legacy before and after timing with
`1e-5 + 1e-5 * abs(reference)`, verifies packing/activation invariants and
expected dispatch counters. Static cases have no runtime packs. Dynamic B packs
1,021 times for 1,021 runs in each mode. Fixture plans prepare two matrix packs
and use two eligible epilogues per run; other static cases prepare one pack.
No-op selection counts a selected dispatch, not arithmetic performed.

Speedup is baseline sample time / candidate sample time; above one is faster.
Summary latency values and paired speedups are **separate medians across three
process statistics**. Dividing summary latency medians need not reproduce the
paired speedup. Full process p50/p95/mean, paired medians and p95 quotients are
in the [appendix](rog-cpu-phase4.5-fusion-details.md).

Both builds/tests finished before measurements. Initial suites ran sequentially:
native analysis, native comparison, portable analysis, portable comparison.
Only then were the three controls rechecked in fresh sequential processes,
with the same unchanged binaries/settings. Initial data was not replaced by
follow-up measurements. The [component report](rog-cpu-phase4.5.md) describes
independent diagnostic experiments, not additive stages or achieved speedups.

## Acceptance result

The 5% latency regression gate corresponds to paired speedup below
`1 / 1.05 = 0.95238`.

- Native Specialized: no-bias/ReLU (`m16_k128_n128_bias_none`) fails in all
  three initial processes: 0.903, 0.917, 0.904.
  Independent recheck also fails all three: 0.907, 0.889, 0.881.
  Initial latency increase is 9.057–10.760%,
  and follow-up increase is 10.196–13.511%.
  This is a repeatable control regression, not an accepted ROG optimization.
- Native no-op Specialized passes the initial three-process median gate.
  Recheck has one process below the threshold; its three-process median passes.
  This variation is retained and does not establish improved tail latency.
- Portable (scalar fallback) Specialized: every initial and recheck control
  process is above the 5% threshold. All initial 26-case median paired
  speedups are above that threshold, with repeatable gains on biased targets.
  This host configuration passes the specialization control gate; native does not.
- Native Fused: no-op control (`m16_k128_n128_no_epilogue`) fails in every
  initial and follow-up process. This prevents enabling fusion generally.
  Some targets improve over Specialized, but native gains are not uniform.
- Portable Fused has no initial case median regression over 5% against Legacy.
  It often stays close to Specialized rather than providing an incremental
  gain. This scalar result does not overturn the Mac SSE2 fusion failures.

The two fixtures improve repeatably under Specialized versus Legacy:

- native: Gemm MLP 2.452–2.656×; RL actor 1.842–1.931×.

- portable: Gemm MLP 2.457–2.462×; RL actor 1.933–1.974×.

p95 distributions and their baseline/candidate quotients are retained, but
these are not paired-sample p95 speedups. Tail variability and host scheduling
preclude a universal tail-latency claim. A single native K512 process showed
a Specialized regression while its other two processes improved; this is
distinct from the six consistently failing no-bias/ReLU processes.

Next work is to explain/remediate the native no-bias/ReLU path and native fused
no-op overhead, then rerun target and control cases in both builds. Inspecting
code generation and register pressure is an investigation direction, not a
hardware-counter-established cause. MSVC portable SIMD detection and ARM/NEON
execution are separate validation boundaries. No kernel/default change was
made merely to make this test report pass.

## Initial full-suite results

Latencies are median process p50 in microseconds. Speedup columns are median
process paired-p50 ratios.

### native

| Case | Legacy p50 | Specialized p50 | Fused p50 | Specialized / Legacy | Fused / Legacy | Fused / Specialized |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| m1_k128_n128_t0 | 2.540 | 1.210 | 1.245 | 2.110 | 2.047 | 0.961 |
| m1_k128_n128_t1 | 2.580 | 1.270 | 1.120 | 2.024 | 2.295 | 1.134 |
| m16_k128_n128_t0 | 26.870 | 12.995 | 12.510 | 2.071 | 2.180 | 1.051 |
| m16_k128_n128_t1 | 27.465 | 13.190 | 12.465 | 2.078 | 2.208 | 1.076 |
| m64_k128_n128_t0 | 109.435 | 52.285 | 50.760 | 2.064 | 2.137 | 1.027 |
| m64_k128_n128_t1 | 104.315 | 50.540 | 48.395 | 2.067 | 2.165 | 1.044 |
| m1_k512_n512_t0 | 19.890 | 18.090 | 18.970 | 1.116 | 1.089 | 0.998 |
| m1_k512_n512_t1 | 21.175 | 18.900 | 19.360 | 1.112 | 1.085 | 0.986 |
| m16_k512_n512_t0 | 274.400 | 218.080 | 228.940 | 1.266 | 1.214 | 0.960 |
| m16_k512_n512_t1 | 277.375 | 221.465 | 227.705 | 1.249 | 1.218 | 0.976 |
| m64_k512_n512_t0 | 1148.435 | 915.130 | 945.355 | 1.234 | 1.201 | 0.969 |
| m64_k512_n512_t1 | 1137.240 | 899.835 | 923.405 | 1.249 | 1.213 | 0.975 |
| m3_k127_n131_t0 | 6.230 | 3.070 | 2.960 | 2.057 | 2.125 | 1.038 |
| m3_k127_n131_t1 | 6.370 | 3.080 | 2.910 | 2.060 | 2.168 | 1.052 |
| m16_k128_n128_bias_none | 11.885 | 12.945 | 12.245 | 0.904 | 0.962 | 1.059 |
| m16_k128_n128_bias_scalar | 18.135 | 12.930 | 12.460 | 1.434 | 1.486 | 1.044 |
| m16_k128_n128_bias_row | 26.785 | 12.530 | 12.035 | 2.135 | 2.222 | 1.040 |
| m16_k128_n128_bias_column | 26.810 | 13.105 | 12.160 | 2.057 | 2.204 | 1.071 |
| m16_k128_n128_bias_full | 17.805 | 13.510 | 12.235 | 1.338 | 1.472 | 1.111 |
| m16_k128_n128_no_relu | 27.010 | 13.070 | 12.145 | 2.064 | 2.249 | 1.086 |
| m16_k128_n128_no_epilogue | 11.380 | 11.625 | 12.410 | 0.991 | 0.920 | 0.942 |
| m16_k128_n128_dynamic | 193.615 | 192.600 | 193.265 | 1.001 | 0.998 | 0.995 |
| m3_k127_n128_aligned_n | 5.625 | 2.660 | 2.490 | 2.145 | 2.255 | 1.063 |
| m3_k128_n128_aligned_kn | 6.460 | 2.880 | 2.810 | 2.273 | 2.313 | 1.028 |
| phase3_mlp_gemm | 2.260 | 0.920 | 0.920 | 2.454 | 2.465 | 1.010 |
| rl_actor_mlp_tanh | 3.330 | 1.800 | 1.760 | 1.851 | 1.886 | 1.007 |

### portable

| Case | Legacy p50 | Specialized p50 | Fused p50 | Specialized / Legacy | Fused / Legacy | Fused / Specialized |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| m1_k128_n128_t0 | 7.100 | 5.635 | 5.610 | 1.255 | 1.261 | 1.004 |
| m1_k128_n128_t1 | 6.440 | 5.130 | 5.110 | 1.200 | 1.227 | 0.985 |
| m16_k128_n128_t0 | 92.615 | 78.840 | 79.855 | 1.171 | 1.165 | 0.990 |
| m16_k128_n128_t1 | 95.580 | 81.135 | 81.925 | 1.180 | 1.164 | 0.991 |
| m64_k128_n128_t0 | 363.150 | 311.305 | 313.170 | 1.176 | 1.159 | 0.989 |
| m64_k128_n128_t1 | 372.045 | 318.300 | 319.960 | 1.173 | 1.155 | 0.986 |
| m1_k512_n512_t0 | 94.885 | 93.265 | 95.265 | 1.021 | 0.992 | 0.967 |
| m1_k512_n512_t1 | 96.075 | 93.775 | 97.115 | 1.024 | 0.987 | 0.965 |
| m16_k512_n512_t0 | 1687.260 | 1607.425 | 1700.380 | 1.038 | 1.000 | 0.974 |
| m16_k512_n512_t1 | 2205.410 | 2124.225 | 2285.020 | 1.034 | 1.012 | 0.968 |
| m64_k512_n512_t0 | 7643.965 | 7455.930 | 7534.955 | 1.032 | 1.022 | 1.004 |
| m64_k512_n512_t1 | 7418.790 | 7169.225 | 7295.590 | 1.018 | 1.012 | 0.986 |
| m3_k127_n131_t0 | 22.330 | 18.135 | 18.305 | 1.233 | 1.217 | 0.986 |
| m3_k127_n131_t1 | 19.630 | 15.920 | 16.200 | 1.232 | 1.217 | 0.985 |
| m16_k128_n128_bias_none | 78.250 | 79.295 | 79.920 | 0.985 | 0.984 | 1.000 |
| m16_k128_n128_bias_scalar | 85.455 | 81.000 | 80.545 | 1.059 | 1.055 | 0.991 |
| m16_k128_n128_bias_row | 92.870 | 78.530 | 79.210 | 1.181 | 1.167 | 0.991 |
| m16_k128_n128_bias_column | 96.840 | 82.425 | 82.820 | 1.175 | 1.165 | 0.994 |
| m16_k128_n128_bias_full | 88.050 | 82.760 | 83.205 | 1.064 | 1.059 | 0.993 |
| m16_k128_n128_no_relu | 92.965 | 80.545 | 79.525 | 1.183 | 1.178 | 0.996 |
| m16_k128_n128_no_epilogue | 81.140 | 81.055 | 83.375 | 0.999 | 0.980 | 0.976 |
| m16_k128_n128_dynamic | 275.535 | 277.680 | 270.710 | 1.005 | 1.002 | 1.000 |
| m3_k127_n128_aligned_n | 18.670 | 15.180 | 15.290 | 1.222 | 1.222 | 1.003 |
| m3_k128_n128_aligned_kn | 19.825 | 16.360 | 16.480 | 1.230 | 1.208 | 0.987 |
| phase3_mlp_gemm | 1.970 | 0.800 | 0.800 | 2.458 | 2.457 | 1.000 |
| rl_actor_mlp_tanh | 2.900 | 1.505 | 1.500 | 1.970 | 1.961 | 1.000 |

## Independent control recheck

These are additional processes, not replacement/selected initial results.

| Build / control / run | Specialized / Legacy paired p50 | Fused / Legacy paired p50 | Fused / Specialized paired p50 |
| --- | ---: | ---: | ---: |
| native-bias_none-run1 | 0.907 | 0.966 | 1.064 |
| native-bias_none-run2 | 0.889 | 0.959 | 1.081 |
| native-bias_none-run3 | 0.881 | 0.955 | 1.085 |
| native-dynamic-run1 | 0.985 | 0.999 | 1.009 |
| native-dynamic-run2 | 0.993 | 0.986 | 0.992 |
| native-dynamic-run3 | 0.997 | 1.003 | 1.002 |
| native-no_epilogue-run1 | 0.954 | 0.926 | 0.971 |
| native-no_epilogue-run2 | 0.927 | 0.929 | 1.001 |
| native-no_epilogue-run3 | 0.992 | 0.947 | 0.957 |
| portable-bias_none-run1 | 0.990 | 0.979 | 0.995 |
| portable-bias_none-run2 | 0.983 | 0.988 | 1.006 |
| portable-bias_none-run3 | 0.973 | 0.986 | 1.008 |
| portable-dynamic-run1 | 1.003 | 0.995 | 1.001 |
| portable-dynamic-run2 | 0.994 | 0.995 | 1.006 |
| portable-dynamic-run3 | 0.999 | 1.001 | 0.998 |
| portable-no_epilogue-run1 | 0.991 | 0.980 | 0.986 |
| portable-no_epilogue-run2 | 1.000 | 0.964 | 0.973 |
| portable-no_epilogue-run3 | 1.000 | 0.982 | 0.981 |

## Reproduction

From the working Visual Studio x64 developer PowerShell environment:

```powershell
cmake -S . -B build-phase45-portable -G Ninja -DCMAKE_BUILD_TYPE=Release -DTINYINFER_BUILD_TESTS=ON -DTINYINFER_BUILD_EXAMPLES=OFF -DTINYINFER_BUILD_BENCHMARKS=ON "-DCMAKE_CXX_FLAGS=/EHsc /permissive-" -DTINYINFER_ENABLE_NATIVE_ARCH=OFF
cmake --build build-phase45-portable --parallel 8
ctest --test-dir build-phase45-portable --output-on-failure
cmake -S . -B build-phase45-native -G Ninja -DCMAKE_BUILD_TYPE=Release -DTINYINFER_BUILD_TESTS=ON -DTINYINFER_BUILD_EXAMPLES=OFF -DTINYINFER_BUILD_BENCHMARKS=ON "-DCMAKE_CXX_FLAGS=/EHsc /permissive-" -DTINYINFER_ENABLE_NATIVE_ARCH=ON
cmake --build build-phase45-native --parallel 8
ctest --test-dir build-phase45-native --output-on-failure
python benchmarks/run_phase45_fusion.py --executable build-phase45-native/benchmarks/tinyinfer_cpu_performance_analysis_benchmark.exe --output benchmark-results/phase4.5-fusion/rog-native --warmup 20 --samples 100 --repeats 10 --runs 3
python benchmarks/run_phase45_fusion.py --executable build-phase45-portable/benchmarks/tinyinfer_cpu_performance_analysis_benchmark.exe --output benchmark-results/phase4.5-fusion/rog-portable --warmup 20 --samples 100 --repeats 10 --runs 3
```

For each control, run the following command three times in fresh processes for
each unchanged executable, retaining JSON separately:

```powershell
# No bias / ReLU (default activation)
./build-phase45-native/benchmarks/tinyinfer_cpu_performance_analysis_benchmark.exe --fusion-comparison --bias none --warmup 20 --samples 100 --repeats 10 --format json
# No bias / no activation
./build-phase45-native/benchmarks/tinyinfer_cpu_performance_analysis_benchmark.exe --fusion-comparison --bias none --activation none --warmup 20 --samples 100 --repeats 10 --format json
# Runtime RHS packing control
./build-phase45-native/benchmarks/tinyinfer_cpu_performance_analysis_benchmark.exe --fusion-comparison --dynamic-b --warmup 20 --samples 100 --repeats 10 --format json
```

Repeat with `build-phase45-portable`. Do not run builds/tests concurrently
with benchmarks. Raw output is retained under
`benchmark-results/phase4.5-fusion/rog-{native,portable}` and
`benchmark-results/phase4.5-fusion/rog-controls-recheck` (ignored artifacts).
The [appendix](rog-cpu-phase4.5-fusion-details.md) retains all initial/recheck
latency statistics, buffer/dispatch counters and binary/source hashes.

## Gate checklist

- [x] ROG fresh native and portable Release suites, 41/41 each.
- [x] 150 checked analysis processes with component/copy/memory diagnostics.
- [x] 156 checked three-mode comparison processes, p50/p95/mean retained.
- [x] 18 independent control-recheck processes, original data retained.
- [x] Publish ROG reports and update reproduction status.
- [ ] Native Specialized no-bias/ReLU regression resolved.
- [ ] Register fusion performance accepted before default/automatic selection.
- [ ] Width-4 SSE2 selected/validated on MSVC portable; ARM/NEON executed.

**ROG tests are finished; Phase 4.5 performance acceptance is not finished.**

