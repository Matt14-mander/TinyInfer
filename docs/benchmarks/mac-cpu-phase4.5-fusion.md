# Mac CPU — Phase 4.5 Gemm specialization and register fusion

- Measurement date: 2026-10-08 (Asia/Shanghai).
- Source: `89d5418` plus the working kernel/API/test/benchmark changes.
- Status: specialization accepted locally as the default; register fusion passes
  correctness but **fails performance acceptance** and remains explicit experimental.
- Phase 4.5 target-host acceptance remains pending; Phase 4.4 ROG acceptance is unchanged.

## Implementation and boundary

Prepared constant-B FP32 Gemm/FusedGemmActivation can use three explicit modes.
Legacy retains the original TensorIterator epilogue. Specialized decodes broadcast
strides and uses a direct traversal. Fused applies bias/affine/ReLU before the
existing final K-block accumulator store, with scalar tails and a zero-K path.
Bias fusion/specialization requires alpha=beta=1; no-bias supports any alpha.
Nonunit biased coefficients retain the original helper. Noncontiguous bias
columns use specialization. Dynamic B retains the existing packing kernel.
Dynamic bias is rebound every invocation. SIMD ReLU preserves NaN/signed-zero
semantics of std::max(0, x). Packing format, tiling, reduction order and planned
activation memory are unchanged. NEON is implemented but has not been executed.

`CpuExecutionPlanOptions` now defaults to Specialized. Fused is available only
by explicit selection, with no portable performance guarantee. Local evidence
does not justify automatic fusion or a shape threshold. The next kernel slice
must examine final-K dispatch/code generation and register pressure; those are
investigation candidates, not causes established by hardware counters here.

## Verification

- Native AVX2/FMA Release: 41/41 tests passed.
- Portable SSE2 Release: 41/41 tests passed.
- Portable Debug ASan + UBSan: 4/4 targeted tests passed (Gemm, packed MatMul,
  execution plan, new kernel fusion).
- Tests cover all common bias forms, None/ReLU, nonunit fallback, transA/transB,
  dynamic bias/storage strides, changed bindings, SIMD tails, custom blocks,
  zero dimensions, final-K-only application, NaN, infinities, signed zeros and
  independent ONNX fixture references.
- All 156 benchmark processes check outputs before/after measurement and verify
  memory, packing and selected epilogue dispatch counts.
- The full suites above ran before choosing Specialized as default. Subsequent
  targeted execution-plan/kernel tests passed 2/2 in each Release build and verify
  the final selection; numeric kernel
  code and the three explicitly benchmarked modes were unchanged by that choice.

## Measurement boundary

Host: Intel x86_64 Mac; macOS 15.8.1 (24H32), Apple Clang 17.0.0
(clang-1700.0.13.5), CMake 4.4.3. Native ON uses AVX2 width 8; OFF uses SSE2
width 4. Release builds use the same toolchain and SDK. CPU frequency/power mode
and unrelated host activity were not controlled. Absolute process times vary
substantially; use paired ratios and process evidence, not historical latencies.

26 cases × 3 sequential processes × 2 builds = 156 processes. Each process uses
20 warm-up runs, 100 samples and 10 repeated invocations per timed sample.
Three pre-bound reused contexts share identical graph/input/packing/buffer
policies. Their order rotates every sample. Import, preparation, binding, output
copy and numerical checks are outside warm run timers. All builds/tests finished
before measurements; native then portable suites ran sequentially.

A sample speedup is Legacy time / candidate time; above one means faster.
Tables show the median of three process p50 latencies and, separately, the median
of three process paired-speedup medians. **Dividing the displayed latency medians
need not reproduce the paired speedup.** All p50/p95/mean and process paired ratios
are retained in the [appendix](mac-cpu-phase4.5-fusion-details.md). Component
analysis from the earlier report is not summed into these achieved speedups.

The source hashes/binary hashes in the appendix identify the measured artifacts.
The subsequent default-mode choice changes the API default and tests, while the
measured three modes were always explicitly selected. Raw JSON remains under
`benchmark-results/phase4.5-fusion/mac-{native,portable}` (ignored build artifacts).
Use `benchmarks/run_phase45_fusion.py` to reproduce.

## Result and acceptance decision

Specialized has no case with a paired regression above 5% in all three processes,
and none with a median paired regression above 5%, in either build. Common biased
K128 workloads improve repeatably. The two ONNX fixtures improve about 2.84× and
2.03–2.04×. The no-bias, no-op and dynamic-B controls stay near parity.

Register fusion does not improve uniformly over specialization. Native has
several cases with approximately 5–7% more latency than Specialized. Portable
matrix workloads often have about 20–29% more latency than Specialized; no-bias
and no-op controls have repeatable regressions above 5% against Legacy too.
The SSE2 M64/K512/N512 cases have fused/Legacy speedups about 0.883–0.884
(approximately 13% more latency). Therefore register fusion is **not accepted**
as the default optimization. Speedups against the costly Legacy bias traversal
must not be attributed entirely to register fusion.

p95 is also reported, but noisy controls have inconsistent process tail ratios
(e.g. portable no-bias Specialized/Legacy speedup based on p95: 0.613, 1.117,
1.070). This does not establish stable tail-latency improvement. Fixture p95
improvements repeat in all processes. These are host measurements, not a universal
latency guarantee. ROG correctness/performance and ARM execution remain pending.

## Full results

### native

| Case | Legacy p50 us | Specialized p50 us | Fused p50 us | Specialized / legacy speedup | Fused / legacy speedup | Fused / specialized speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| m1_k128_n128_t0 | 25.829 | 7.939 | 8.165 | 2.990 | 2.792 | 0.974 |
| m1_k128_n128_t1 | 23.882 | 7.901 | 8.197 | 2.932 | 2.834 | 0.967 |
| m16_k128_n128_t0 | 305.341 | 133.427 | 116.459 | 2.181 | 2.169 | 0.965 |
| m16_k128_n128_t1 | 207.408 | 99.547 | 107.817 | 2.049 | 1.893 | 0.943 |
| m64_k128_n128_t0 | 860.267 | 438.478 | 454.904 | 2.066 | 1.885 | 0.931 |
| m64_k128_n128_t1 | 870.936 | 434.925 | 446.466 | 1.907 | 1.848 | 0.998 |
| m1_k512_n512_t0 | 176.627 | 146.697 | 148.150 | 1.217 | 1.181 | 0.986 |
| m1_k512_n512_t1 | 180.329 | 153.161 | 158.812 | 1.174 | 1.163 | 0.979 |
| m16_k512_n512_t0 | 2297.350 | 1853.814 | 1953.984 | 1.199 | 1.195 | 0.971 |
| m16_k512_n512_t1 | 1608.209 | 1433.794 | 1420.395 | 1.136 | 1.195 | 1.024 |
| m64_k512_n512_t0 | 5290.880 | 4209.315 | 4268.345 | 1.177 | 1.170 | 1.006 |
| m64_k512_n512_t1 | 4261.874 | 3684.508 | 3746.786 | 1.170 | 1.173 | 1.003 |
| m3_k127_n131_t0 | 18.743 | 9.077 | 9.576 | 2.043 | 1.926 | 0.947 |
| m3_k127_n131_t1 | 18.591 | 8.996 | 9.699 | 2.045 | 1.918 | 0.942 |
| m16_k128_n128_bias_none | 37.125 | 37.793 | 38.404 | 0.997 | 0.966 | 0.969 |
| m16_k128_n128_bias_scalar | 63.356 | 41.402 | 41.849 | 1.535 | 1.551 | 1.013 |
| m16_k128_n128_bias_row | 70.102 | 37.227 | 38.445 | 1.847 | 1.784 | 0.969 |
| m16_k128_n128_bias_column | 70.759 | 39.302 | 39.136 | 1.768 | 1.779 | 1.015 |
| m16_k128_n128_bias_full | 62.297 | 37.220 | 38.699 | 1.630 | 1.574 | 0.964 |
| m16_k128_n128_no_relu | 71.204 | 37.593 | 39.256 | 1.820 | 1.776 | 0.976 |
| m16_k128_n128_no_epilogue | 36.946 | 36.800 | 37.910 | 1.002 | 0.971 | 0.971 |
| m16_k128_n128_dynamic | 401.309 | 399.026 | 402.051 | 1.003 | 1.008 | 1.013 |
| m3_k127_n128_aligned_n | 16.294 | 7.468 | 7.970 | 2.201 | 2.044 | 0.933 |
| m3_k128_n128_aligned_kn | 16.413 | 7.576 | 7.831 | 2.174 | 2.098 | 0.965 |
| phase3_mlp_gemm | 11.109 | 3.916 | 3.941 | 2.836 | 2.831 | 0.995 |
| rl_actor_mlp_tanh | 14.337 | 7.047 | 7.028 | 2.030 | 2.052 | 1.000 |

### portable

| Case | Legacy p50 us | Specialized p50 us | Fused p50 us | Specialized / legacy speedup | Fused / legacy speedup | Fused / specialized speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| m1_k128_n128_t0 | 18.345 | 9.567 | 10.493 | 1.840 | 1.505 | 0.817 |
| m1_k128_n128_t1 | 14.191 | 6.177 | 8.489 | 1.842 | 1.477 | 0.806 |
| m16_k128_n128_t0 | 137.598 | 109.202 | 125.776 | 1.384 | 1.066 | 0.789 |
| m16_k128_n128_t1 | 133.208 | 94.876 | 120.618 | 1.385 | 1.066 | 0.773 |
| m64_k128_n128_t0 | 609.121 | 451.872 | 564.779 | 1.387 | 1.089 | 0.823 |
| m64_k128_n128_t1 | 577.850 | 425.096 | 543.174 | 1.356 | 1.042 | 0.778 |
| m1_k512_n512_t0 | 158.112 | 146.877 | 166.809 | 1.085 | 0.974 | 0.894 |
| m1_k512_n512_t1 | 158.594 | 148.241 | 168.250 | 1.076 | 0.978 | 0.893 |
| m16_k512_n512_t0 | 2436.429 | 2288.879 | 2854.127 | 1.090 | 0.892 | 0.823 |
| m16_k512_n512_t1 | 2029.598 | 1878.160 | 2375.159 | 1.086 | 0.865 | 0.812 |
| m64_k512_n512_t0 | 7397.018 | 6819.331 | 8290.649 | 1.075 | 0.884 | 0.820 |
| m64_k512_n512_t1 | 7996.391 | 7314.163 | 9049.085 | 1.077 | 0.883 | 0.820 |
| m3_k127_n131_t0 | 27.416 | 18.460 | 22.676 | 1.509 | 1.219 | 0.806 |
| m3_k127_n131_t1 | 27.936 | 18.162 | 22.316 | 1.501 | 1.210 | 0.807 |
| m16_k128_n128_bias_none | 102.871 | 104.964 | 128.110 | 1.007 | 0.799 | 0.797 |
| m16_k128_n128_bias_scalar | 138.397 | 111.253 | 134.539 | 1.231 | 0.988 | 0.792 |
| m16_k128_n128_bias_row | 165.851 | 112.538 | 143.590 | 1.437 | 1.102 | 0.784 |
| m16_k128_n128_bias_column | 213.261 | 144.716 | 178.106 | 1.422 | 1.125 | 0.794 |
| m16_k128_n128_bias_full | 139.165 | 110.915 | 139.747 | 1.272 | 0.994 | 0.788 |
| m16_k128_n128_no_relu | 156.518 | 113.940 | 145.264 | 1.357 | 1.062 | 0.779 |
| m16_k128_n128_no_epilogue | 113.767 | 112.394 | 144.152 | 1.003 | 0.784 | 0.780 |
| m16_k128_n128_dynamic | 605.592 | 604.859 | 614.645 | 1.025 | 0.976 | 0.999 |
| m3_k127_n128_aligned_n | 35.191 | 21.620 | 27.832 | 1.562 | 1.214 | 0.778 |
| m3_k128_n128_aligned_kn | 35.510 | 22.281 | 29.363 | 1.544 | 1.212 | 0.783 |
| phase3_mlp_gemm | 14.790 | 5.189 | 5.179 | 2.846 | 2.841 | 0.998 |
| rl_actor_mlp_tanh | 18.895 | 9.269 | 9.176 | 2.042 | 2.036 | 0.999 |

