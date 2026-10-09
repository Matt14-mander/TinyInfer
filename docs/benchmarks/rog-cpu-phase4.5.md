# ROG CPU — Phase 4.5 prepared performance analysis

- Measurement date: 2026-10-09 (Asia/Shanghai).
- Status: ROG analysis and correctness reproduction complete; production
  specialization/fusion acceptance is recorded separately in the
  [three-mode report](rog-cpu-phase4.5-fusion.md).
- Evidence: 25 cases × 3 sequential processes × 2 builds = 150 checked processes.

## Environment and verification

- Host: ROG, Intel Core i9-13980HX, 24 cores / 32 logical processors.
- Windows 11 Home Chinese 10.0.26200; active power scheme: Performance.
- MSVC 19.43.34808.0, Visual Studio 2022 Build Tools, Ninja, Release.
- Source: `9394f9f4aff08e7d6ce4ce3627b36e3b59a9ecbb`; clean tracked worktree during all measurements.
- Fresh builds: `build-phase45-native` and `build-phase45-portable`.
- Shared C++ flags: `/EHsc /permissive-`; tests keep assertions through `/UNDEBUG`.
- Native ON: `/arch:AVX2`, measured explicit kernel width 8.
- Portable OFF: measured explicit kernel width 1 (scalar fallback), **not SSE2**.
  The current selector tests `__SSE2__`, which this MSVC OFF build does not select.
  Compiler-generated vectorization is not excluded by the width diagnostic.
  These results cannot validate the Mac width-4 SSE2 path or NEON.
- portable Release: 41/41 tests passed, 8.01 s.
- native Release: 41/41 tests passed, 7.36 s.
- Build warnings include existing code-page C4819 warnings and the intentional
  D9025 `/DNDEBUG` to `/UNDEBUG` override. No build/test failure occurred.
- ROG sanitizers were not run in this validation. Existing Mac targeted
  ASan/UBSan evidence is in the [Mac fusion report](mac-cpu-phase4.5-fusion.md);
  it is not a Windows sanitizer result.

CPU temperature/frequency/affinity, process RSS and unrelated system activity
were not recorded or controlled. Results are host measurements, not universal
latency guarantees. Both builds and tests completed before measurements.

## Measurement boundary

The analysis explicitly selects Legacy, preserving the Phase 4.4 prepared
baseline even though the public default is now Specialized. Warm-up 20, samples
100, repeats 10. A timed sample averages repeated invocations, and workload
order rotates. Import, graph optimization, packing, setup and correctness
checks are outside warm timers.

Whole prepared execution, bookkeeping, captured matrix chain, binding/output
deep copies, isolated MatMul/epilogue/complete and ordinary registry dispatch
are independent experiments with different cache/loop contexts. Do not add
their medians or subtract them from whole execution to claim an exact dispatch
fraction. Captured matrix operands are frozen and do not propagate layer
outputs. The bookkeeping control is not production dispatch. Ordinary-node
timing includes registry and planned slot setup. Scalar-tail control is a
separate strided kernel, not a timer inside the production MatMul.

Each epilogue pool has ten independent output buffers, reset outside timing
and transformed once per sample. Pool payload is reported below. Diagnostic
captured tensors, packed operands and pools remain alive and can affect caches.
Dynamic B still packs on every production invocation; the captured direct
diagnostic packs once outside timing and is not production fallback latency.
See the full [analysis boundary](../plans/phase4.5.md#450--analysis-boundary).

Each process checks prepared vs ordinary outputs before/after measurement,
captured direct outputs, epilogue pool outputs, scalar-tail outputs where
present, and isolated ordinary outputs using
`1e-5 + 1e-5 * abs(reference)`. Runtime packing counts were checked.
All 150 processes completed without a numerical or counter failure.
Each prepared session records 1,022 runs; the dynamic-B case records 1,022
runtime packs, while static cases record zero runtime packs.

Tables show the median of three **process p50 values**, in microseconds.
The [appendix](rog-cpu-phase4.5-details.md) retains every process p50/p95/mean,
loop floor, per-node metadata and artifact hashes. A zero loop-floor median
is below the observable clock/rounding floor, not proof that overhead is absent.

## Interpretation

Common-bias K128 and the small fixtures have visible independent epilogue
costs, supporting measurement of direct broadcast specialization. Large K512
workloads have larger matrix arithmetic costs. These are diagnostic observations,
not exact cost percentages and not achieved optimization speedups.
The three-mode report measures achieved gains and exposes a repeatable native
no-bias/ReLU control regression; analysis completion does not imply performance
acceptance.

## native — whole execution and application costs

| Case | Legacy prepared | Bookkeeping control | Captured matrix chain | Input binding copy | Output extraction copy |
| --- | ---: | ---: | ---: | ---: | ---: |
| m1_k128_n128_t0 | 2.365 | 0.150 | 2.275 | 1.140 | 1.130 |
| m1_k128_n128_t1 | 2.270 | 0.140 | 2.060 | 1.110 | 1.100 |
| m16_k128_n128_t0 | 26.445 | 0.150 | 26.290 | 14.875 | 14.860 |
| m16_k128_n128_t1 | 26.915 | 0.150 | 26.835 | 14.910 | 14.855 |
| m64_k128_n128_t0 | 100.460 | 0.160 | 101.020 | 57.215 | 56.565 |
| m64_k128_n128_t1 | 100.820 | 0.160 | 100.930 | 57.790 | 57.630 |
| m1_k512_n512_t0 | 20.395 | 0.160 | 19.395 | 3.755 | 3.710 |
| m1_k512_n512_t1 | 20.150 | 0.150 | 20.380 | 3.930 | 3.880 |
| m16_k512_n512_t0 | 279.025 | 0.220 | 278.950 | 60.605 | 60.415 |
| m16_k512_n512_t1 | 274.760 | 0.210 | 275.175 | 59.945 | 59.820 |
| m64_k512_n512_t0 | 1112.290 | 0.280 | 1142.645 | 237.365 | 235.885 |
| m64_k512_n512_t1 | 1097.860 | 0.270 | 1133.150 | 235.610 | 234.755 |
| m3_k127_n131_t0 | 7.375 | 0.160 | 6.955 | 3.230 | 3.230 |
| m3_k127_n131_t1 | 7.640 | 0.170 | 7.470 | 3.410 | 3.410 |
| m16_k128_n128_bias_none | 11.920 | 0.150 | 11.845 | 15.045 | 15.130 |
| m16_k128_n128_bias_scalar | 18.625 | 0.160 | 18.155 | 15.460 | 15.240 |
| m16_k128_n128_bias_row | 27.215 | 0.160 | 26.705 | 14.965 | 14.890 |
| m16_k128_n128_bias_column | 27.645 | 0.155 | 27.280 | 15.125 | 14.870 |
| m16_k128_n128_bias_full | 17.810 | 0.150 | 17.935 | 14.870 | 14.850 |
| m16_k128_n128_no_relu | 25.725 | 0.140 | 25.490 | 14.230 | 14.205 |
| m16_k128_n128_dynamic | 197.895 | 0.180 | 25.375 | 143.845 | 13.880 |
| m3_k127_n128_aligned_n | 6.125 | 0.150 | 6.050 | 3.080 | 3.000 |
| m3_k128_n128_aligned_kn | 5.995 | 0.140 | 5.780 | 2.960 | 2.940 |
| phase3_mlp_gemm | 2.455 | 0.360 | 1.460 | 0.210 | 0.190 |
| rl_actor_mlp_tanh | 2.870 | 0.450 | 1.350 | 0.210 | 0.180 |

### Matrix controls

| Case / node | MatMul | Epilogue | Complete | Scalar-tail control |
| --- | ---: | ---: | ---: | ---: |
| m1_k128_n128_t0 / 0 | 1.010 | 1.090 | 2.160 | n/a |
| m1_k128_n128_t1 / 0 | 0.900 | 1.070 | 2.040 | n/a |
| m16_k128_n128_t0 / 0 | 10.450 | 15.000 | 25.500 | n/a |
| m16_k128_n128_t1 / 0 | 11.235 | 15.615 | 27.025 | n/a |
| m64_k128_n128_t0 / 0 | 44.020 | 60.810 | 105.615 | n/a |
| m64_k128_n128_t1 / 0 | 45.325 | 61.700 | 108.105 | n/a |
| m1_k512_n512_t0 / 0 | 17.660 | 2.150 | 20.165 | n/a |
| m1_k512_n512_t1 / 0 | 18.005 | 2.310 | 20.400 | n/a |
| m16_k512_n512_t0 / 0 | 222.285 | 61.460 | 280.790 | n/a |
| m16_k512_n512_t1 / 0 | 212.905 | 59.890 | 274.430 | n/a |
| m64_k512_n512_t0 / 0 | 981.830 | 260.110 | 1350.080 | n/a |
| m64_k512_n512_t1 / 0 | 943.830 | 246.820 | 1197.645 | n/a |
| m3_k127_n131_t0 / 0 | 3.045 | 3.790 | 6.915 | 0.420 |
| m3_k127_n131_t1 / 0 | 3.070 | 4.020 | 7.180 | 0.440 |
| m16_k128_n128_bias_none / 0 | 11.180 | 0.700 | 11.860 | n/a |
| m16_k128_n128_bias_scalar / 0 | 12.340 | 7.330 | 19.730 | n/a |
| m16_k128_n128_bias_row / 0 | 11.430 | 15.995 | 27.540 | n/a |
| m16_k128_n128_bias_column / 0 | 12.330 | 17.480 | 30.095 | n/a |
| m16_k128_n128_bias_full / 0 | 11.270 | 6.740 | 18.030 | n/a |
| m16_k128_n128_no_relu / 0 | 10.550 | 15.025 | 25.635 | n/a |
| m16_k128_n128_dynamic / 0 | 11.070 | 15.465 | 27.015 | n/a |
| m3_k127_n128_aligned_n / 0 | 2.275 | 3.440 | 5.800 | n/a |
| m3_k128_n128_aligned_kn / 0 | 2.220 | 3.450 | 5.730 | n/a |
| phase3_mlp_gemm / 0 | 0.020 | 0.670 | 0.690 | 0.020 |
| phase3_mlp_gemm / 1 | 0.020 | 0.670 | 0.690 | 0.010 |
| rl_actor_mlp_tanh / 0 | 0.030 | 0.640 | 0.670 | 0.020 |
| rl_actor_mlp_tanh / 2 | 0.020 | 0.630 | 0.660 | 0.020 |

### Ordinary operators

| Case / node / op enum | Registry dispatch |
| --- | ---: |
| phase3_mlp_gemm / 2 / 9 | 0.600 |
| rl_actor_mlp_tanh / 1 / 7 | 0.600 |
| rl_actor_mlp_tanh / 3 / 7 | 0.580 |

## portable — whole execution and application costs

| Case | Legacy prepared | Bookkeeping control | Captured matrix chain | Input binding copy | Output extraction copy |
| --- | ---: | ---: | ---: | ---: | ---: |
| m1_k128_n128_t0 | 6.930 | 0.160 | 7.010 | 1.235 | 1.220 |
| m1_k128_n128_t1 | 6.255 | 0.140 | 6.010 | 1.045 | 1.040 |
| m16_k128_n128_t0 | 90.665 | 0.155 | 91.010 | 14.570 | 14.520 |
| m16_k128_n128_t1 | 91.560 | 0.160 | 91.175 | 14.650 | 14.555 |
| m64_k128_n128_t0 | 359.765 | 0.210 | 361.165 | 57.930 | 57.575 |
| m64_k128_n128_t1 | 371.330 | 0.200 | 368.060 | 59.095 | 58.565 |
| m1_k512_n512_t0 | 92.810 | 0.160 | 92.950 | 3.790 | 3.760 |
| m1_k512_n512_t1 | 94.895 | 0.160 | 93.310 | 3.830 | 3.780 |
| m16_k512_n512_t0 | 1552.175 | 0.300 | 1572.430 | 59.700 | 60.455 |
| m16_k512_n512_t1 | 1526.980 | 0.270 | 1525.675 | 59.440 | 60.365 |
| m64_k512_n512_t0 | 6490.990 | 0.340 | 6664.055 | 237.205 | 241.305 |
| m64_k512_n512_t1 | 6307.520 | 0.320 | 6265.120 | 238.975 | 237.260 |
| m3_k127_n131_t0 | 17.760 | 0.140 | 17.845 | 2.910 | 2.880 |
| m3_k127_n131_t1 | 18.325 | 0.150 | 18.220 | 2.990 | 2.940 |
| m16_k128_n128_bias_none | 78.735 | 0.160 | 78.880 | 15.105 | 15.015 |
| m16_k128_n128_bias_scalar | 81.855 | 0.160 | 81.690 | 14.560 | 14.520 |
| m16_k128_n128_bias_row | 90.180 | 0.150 | 90.415 | 14.540 | 14.500 |
| m16_k128_n128_bias_column | 91.485 | 0.160 | 91.645 | 14.640 | 14.590 |
| m16_k128_n128_bias_full | 82.970 | 0.160 | 85.105 | 14.565 | 14.520 |
| m16_k128_n128_no_relu | 89.725 | 0.150 | 89.745 | 14.470 | 14.500 |
| m16_k128_n128_dynamic | 267.375 | 0.200 | 90.360 | 146.150 | 14.230 |
| m3_k127_n128_aligned_n | 18.440 | 0.150 | 18.295 | 2.990 | 2.920 |
| m3_k128_n128_aligned_kn | 18.205 | 0.150 | 18.290 | 2.950 | 2.930 |
| phase3_mlp_gemm | 2.400 | 0.380 | 1.430 | 0.210 | 0.190 |
| rl_actor_mlp_tanh | 2.830 | 0.460 | 1.360 | 0.210 | 0.180 |

### Matrix controls

| Case / node | MatMul | Epilogue | Complete | Scalar-tail control |
| --- | ---: | ---: | ---: | ---: |
| m1_k128_n128_t0 / 0 | 5.570 | 1.170 | 6.805 | n/a |
| m1_k128_n128_t1 / 0 | 5.200 | 1.070 | 6.350 | n/a |
| m16_k128_n128_t0 / 0 | 75.860 | 15.225 | 91.220 | n/a |
| m16_k128_n128_t1 / 0 | 75.455 | 15.115 | 91.515 | n/a |
| m64_k128_n128_t0 / 0 | 308.510 | 59.240 | 370.565 | n/a |
| m64_k128_n128_t1 / 0 | 309.710 | 59.930 | 366.770 | n/a |
| m1_k512_n512_t0 / 0 | 90.960 | 2.160 | 93.215 | n/a |
| m1_k512_n512_t1 / 0 | 96.385 | 2.290 | 97.100 | n/a |
| m16_k512_n512_t0 / 0 | 1803.300 | 68.155 | 1782.370 | n/a |
| m16_k512_n512_t1 / 0 | 1567.830 | 61.795 | 1633.265 | n/a |
| m64_k512_n512_t0 / 0 | 6838.750 | 251.245 | 7057.360 | n/a |
| m64_k512_n512_t1 / 0 | 6296.055 | 244.480 | 6625.515 | n/a |
| m3_k127_n131_t0 / 0 | 14.460 | 3.445 | 17.990 | n/a |
| m3_k127_n131_t1 / 0 | 15.050 | 3.580 | 18.680 | n/a |
| m16_k128_n128_bias_none / 0 | 76.930 | 0.675 | 77.725 | n/a |
| m16_k128_n128_bias_scalar / 0 | 76.425 | 6.475 | 82.940 | n/a |
| m16_k128_n128_bias_row / 0 | 75.185 | 15.035 | 90.370 | n/a |
| m16_k128_n128_bias_column / 0 | 77.065 | 15.370 | 92.525 | n/a |
| m16_k128_n128_bias_full / 0 | 81.965 | 6.940 | 89.025 | n/a |
| m16_k128_n128_no_relu / 0 | 77.540 | 15.500 | 93.330 | n/a |
| m16_k128_n128_dynamic / 0 | 79.345 | 15.805 | 95.420 | n/a |
| m3_k127_n128_aligned_n / 0 | 14.560 | 3.370 | 18.000 | n/a |
| m3_k128_n128_aligned_kn / 0 | 15.350 | 3.600 | 19.095 | n/a |
| phase3_mlp_gemm / 0 | 0.020 | 0.690 | 0.710 | n/a |
| phase3_mlp_gemm / 1 | 0.020 | 0.690 | 0.710 | n/a |
| rl_actor_mlp_tanh / 0 | 0.030 | 0.640 | 0.670 | n/a |
| rl_actor_mlp_tanh / 2 | 0.020 | 0.630 | 0.660 | n/a |

### Ordinary operators

| Case / node / op enum | Registry dispatch |
| --- | ---: |
| phase3_mlp_gemm / 2 / 9 | 0.630 |
| rl_actor_mlp_tanh / 1 / 7 | 0.600 |
| rl_actor_mlp_tanh / 3 / 7 | 0.590 |

## Reported buffer payloads and counts

The following values match between ISA configurations and all three processes.
Activation and packed-weight payloads are production plan diagnostics; epilogue
pool bytes are additional analysis-only memory, summed across captured matrix
nodes. This is not process RSS or the total plan/context memory footprint.
Model snapshots, inputs/outputs, captures, Tensor metadata, capacity/allocator
overhead and fallback scratch are not included. Pool memory does not belong to
normal production inference.

| Case | Activation bytes | Packed weights bytes | Analysis epilogue pools bytes | Prepare packs | Runtime packs / runs |
| --- | ---: | ---: | ---: | ---: | --- |
| m1_k128_n128_t0 | 512 | 65536 | 5120 | 1 | 0 / 1022 |
| m1_k128_n128_t1 | 512 | 65536 | 5120 | 1 | 0 / 1022 |
| m16_k128_n128_t0 | 8192 | 65536 | 81920 | 1 | 0 / 1022 |
| m16_k128_n128_t1 | 8192 | 65536 | 81920 | 1 | 0 / 1022 |
| m64_k128_n128_t0 | 32768 | 65536 | 327680 | 1 | 0 / 1022 |
| m64_k128_n128_t1 | 32768 | 65536 | 327680 | 1 | 0 / 1022 |
| m1_k512_n512_t0 | 2048 | 1048576 | 20480 | 1 | 0 / 1022 |
| m1_k512_n512_t1 | 2048 | 1048576 | 20480 | 1 | 0 / 1022 |
| m16_k512_n512_t0 | 32768 | 1048576 | 327680 | 1 | 0 / 1022 |
| m16_k512_n512_t1 | 32768 | 1048576 | 327680 | 1 | 0 / 1022 |
| m64_k512_n512_t0 | 131072 | 1048576 | 1310720 | 1 | 0 / 1022 |
| m64_k512_n512_t1 | 131072 | 1048576 | 1310720 | 1 | 0 / 1022 |
| m3_k127_n131_t0 | 1572 | 66548 | 15720 | 1 | 0 / 1022 |
| m3_k127_n131_t1 | 1572 | 66548 | 15720 | 1 | 0 / 1022 |
| m16_k128_n128_bias_none | 8192 | 65536 | 81920 | 1 | 0 / 1022 |
| m16_k128_n128_bias_scalar | 8192 | 65536 | 81920 | 1 | 0 / 1022 |
| m16_k128_n128_bias_row | 8192 | 65536 | 81920 | 1 | 0 / 1022 |
| m16_k128_n128_bias_column | 8192 | 65536 | 81920 | 1 | 0 / 1022 |
| m16_k128_n128_bias_full | 8192 | 65536 | 81920 | 1 | 0 / 1022 |
| m16_k128_n128_no_relu | 8192 | 65536 | 81920 | 1 | 0 / 1022 |
| m16_k128_n128_dynamic | 8192 | 0 | 81920 | 0 | 1022 / 1022 |
| m3_k127_n128_aligned_n | 1536 | 65024 | 15360 | 1 | 0 / 1022 |
| m3_k128_n128_aligned_kn | 1536 | 65536 | 15360 | 1 | 0 / 1022 |
| phase3_mlp_gemm | 24 | 48 | 200 | 2 | 0 / 1022 |
| rl_actor_mlp_tanh | 44 | 120 | 280 | 2 | 0 / 1022 |

## Reproduction and artifacts

Run the same MSVC developer environment and fresh Release configurations described
in the [three-mode report](rog-cpu-phase4.5-fusion.md#reproduction).
Then run sequentially, after all builds/tests finish:

```powershell
python benchmarks/run_phase45.py --executable build-phase45-native/benchmarks/tinyinfer_cpu_performance_analysis_benchmark.exe --output benchmark-results/phase4.5/rog-native --warmup 20 --samples 100 --repeats 10 --runs 3
python benchmarks/run_phase45.py --executable build-phase45-portable/benchmarks/tinyinfer_cpu_performance_analysis_benchmark.exe --output benchmark-results/phase4.5/rog-portable --warmup 20 --samples 100 --repeats 10 --runs 3
```

Raw JSON and environment metadata remain under
`benchmark-results/phase4.5/rog-{native,portable}` (ignored artifacts).
Validation logs remain under `benchmark-results/phase4.5/rog-validation`.
The committed reports retain the process statistics even if local raw artifacts
are removed.

