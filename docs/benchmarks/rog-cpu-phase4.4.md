# ROG CPU — Phase 4.4 prepared execution

- Status: ROG portable/native correctness and performance gates passed; Phase 4.4 complete
- Measurement date: 2026-10-07 (Asia/Shanghai)
- Base source: `b97463b3ac723a8f9d8d8c266fab2eadf2fde13d`, with benchmark diagnostic and runner additions
- Baseline: Phase 4.3 optimized graph, Executor + MemoryPlan + combined Gemm epilogue

## Environment and correctness

| Item | Value |
| --- | --- |
| CPU | Intel Core i9-13980HX; 24 cores, 32 logical processors |
| OS | Windows 11 Home Chinese, 10.0.26200 |
| Power scheme | Performance |
| Compiler | MSVC 19.43.34808, Visual Studio 2022 Build Tools |
| CMake / generator | Visual Studio bundled CMake, Ninja |
| Configuration | Release, `/EHsc /permissive-` |
| portable | Native architecture OFF; baseline x86-64/SSE2 |
| native | Native architecture ON; `/arch:AVX2` |
| Tests | Assertions retained through `/UNDEBUG` |
| Warm-up / samples / repeats | 20 / 100 / 10 |
| Independent processes | 24 cases × 3 processes × 2 configurations = 144 |
| Preparation/component/context/first-run samples | 20 single invocations per process |

Fresh build directories were used. Compilation and correctness tests finished
before measurement; all benchmark processes ran sequentially.

| Configuration | CTest result | Total time |
| --- | ---: | ---: |
| portable Release | 40/40 passed | 10.69 s |
| native Release | 40/40 passed | 10.05 s |

Each process verified numerical results before and after timing using
`1e-5 + 1e-5 * abs(reference)`, then verified fallback packing counts.
The test suite covers independent ONNX references, transpositions/bias forms,
strides/tails/zero sizes, changed runtime operands, snapshot ownership, context
lifetime/moves, output copies and weight deduplication. Existing Mac targeted
ASan/UBSan results are recorded in the [Mac report](mac-cpu-phase4.4.md).

## Method and interpretation

Both paths execute the same optimized graph with identical inputs and planned
activation policy in separate contexts. Variant order alternates per sample.
The synthetic graph has one FusedGemmActivation node, default alpha/beta and
constant bias. Dynamic controls bind B as a runtime input; the two ONNX fixtures
use the full explicit optimization pipeline and retain ordinary Tanh dispatch.

Import, optimization, input construction/binding and copied output extraction
are outside inference timers. First execution uses a newly allocated/bound
context with process caches already warm. Preparation and context measurements
include temporary destruction, so they measure lifecycle cost.

Every static case showed a paired speedup greater than one in all three
processes, in both configurations. All eight dynamic controls met the proposed
5% regression gate. A 5% latency regression corresponds to a speedup below
`1 / 1.05 = 0.95238`; every measured control process was above that threshold.
Some native ratios vary substantially between processes; the full ranges and
per-process p95/mean are retained rather than presenting a single universal gain.

Gemm MLP improves 3.187–3.213× on native and
3.159–3.199× on portable. RL actor improves
2.789–2.793× and 2.800–2.843×
respectively. The compute-heavy M64/K512/transB=0 case improves
1.336–1.368× on native and
1.047–1.054× on portable. This is preparation reuse
against the current runtime, not an arithmetic micro-kernel optimization.
Large transB=1 gains include eliminating repeated Tensor deep-copy preparation.
Mac and ROG measurements are separate host results, not a cross-host speedup.

## Warm latency and full preparation amortization

Latency and preparation entries are medians of the three process p50 values,
in microseconds. Paired ranges use each process's median same-sample ratio and
are not confidence intervals. Estimated calls charge all mean preparation
cost against positive mean warm savings, rounded up. They exclude context and
output-extraction costs; `n/a` means at least one process lacks positive mean
savings. They are estimates, not measured startup break-even points.

### native AVX2

| Case | Baseline p50 us | Prepared p50 us | Paired range | Prep p50 us | Estimated calls |
| --- | ---: | ---: | --- | ---: | --- |
| m1_k128_n128_t0 | 31.220 | 2.605 | 9.858–11.980× | 235.150 | 8–10 |
| m1_k128_n128_t1 | 272.755 | 3.560 | 70.778–77.048× | 256.900 | 1–2 |
| m1_k128_n128_dynamic | 224.385 | 220.960 | 1.022–1.027× | 12.100 | 2–4 |
| m16_k128_n128_t0 | 73.720 | 36.700 | 1.451–2.594× | 224.300 | 6–12 |
| m16_k128_n128_t1 | 237.275 | 30.060 | 7.898–8.207× | 211.950 | 1–3 |
| m16_k128_n128_dynamic | 240.420 | 235.795 | 1.016–1.019× | 14.400 | 4–8 |
| m64_k128_n128_t0 | 148.495 | 119.360 | 1.115–1.229× | 232.750 | 7–17 |
| m64_k128_n128_t1 | 342.295 | 123.120 | 2.737–2.837× | 232.900 | 1–2 |
| m64_k128_n128_dynamic | 401.810 | 394.695 | 1.016–1.020× | 16.750 | 2–3 |
| m1_k512_n512_t0 | 691.880 | 34.350 | 17.691–21.431× | 3882.900 | 6–7 |
| m1_k512_n512_t1 | 3537.000 | 31.565 | 87.497–116.299× | 2980.150 | 1–1 |
| m1_k512_n512_dynamic | 3450.810 | 3511.930 | 0.997–1.005× | 15.100 | n/a |
| m16_k512_n512_t0 | 843.160 | 353.785 | 2.346–2.577× | 3029.650 | 5–9 |
| m16_k512_n512_t1 | 3850.665 | 398.855 | 8.700–10.345× | 3190.350 | 1–2 |
| m16_k512_n512_dynamic | 4539.590 | 4675.850 | 0.991–1.001× | 13.450 | n/a |
| m64_k512_n512_t0 | 2849.810 | 2079.210 | 1.336–1.368× | 3584.750 | 5–7 |
| m64_k512_n512_t1 | 6175.675 | 2377.200 | 2.428–2.667× | 3123.850 | 1–1 |
| m64_k512_n512_dynamic | 5642.085 | 5605.735 | 0.998–1.004× | 13.850 | n/a |
| m3_k127_n131_t0 | 74.425 | 10.330 | 2.916–7.289× | 342.100 | 5–13 |
| m3_k127_n131_t1 | 274.475 | 9.200 | 29.685–30.106× | 256.250 | 1–2 |
| m3_k127_n131_dynamic | 201.565 | 197.830 | 1.013–1.017× | 13.000 | 3–7 |
| m16_k128_n128_dynamic_t0 | 50.585 | 46.700 | 1.088–1.092× | 13.800 | 3–4 |
| phase3_mlp_gemm | 10.890 | 3.400 | 3.187–3.213× | 23.700 | 4–4 |
| rl_actor_mlp_tanh | 12.110 | 4.340 | 2.789–2.793× | 32.550 | 5–5 |

### portable SSE2

| Case | Baseline p50 us | Prepared p50 us | Paired range | Prep p50 us | Estimated calls |
| --- | ---: | ---: | --- | ---: | --- |
| m1_k128_n128_t0 | 26.905 | 6.790 | 3.325–4.138× | 205.050 | 8–12 |
| m1_k128_n128_t1 | 222.650 | 7.615 | 29.159–30.252× | 235.150 | 1–2 |
| m1_k128_n128_dynamic | 253.020 | 250.455 | 1.017–1.024× | 13.200 | 2–4 |
| m16_k128_n128_t0 | 136.485 | 119.535 | 1.131–1.212× | 218.750 | 8–16 |
| m16_k128_n128_t1 | 315.320 | 101.795 | 2.999–3.067× | 210.300 | 1–2 |
| m16_k128_n128_dynamic | 346.340 | 337.895 | 1.019–1.020× | 11.900 | 2–3 |
| m64_k128_n128_t0 | 494.675 | 453.945 | 1.048–1.121× | 194.300 | 4–9 |
| m64_k128_n128_t1 | 672.820 | 446.800 | 1.497–1.517× | 203.850 | 1–2 |
| m64_k128_n128_dynamic | 642.275 | 634.125 | 1.005–1.011× | 15.900 | 5–46 |
| m1_k512_n512_t0 | 696.960 | 140.065 | 4.656–5.319× | 2796.450 | 5–6 |
| m1_k512_n512_t1 | 3426.600 | 134.970 | 26.044–27.616× | 2703.500 | 1–1 |
| m1_k512_n512_dynamic | 3436.115 | 3472.405 | 0.998–1.005× | 19.100 | n/a |
| m16_k512_n512_t0 | 2775.260 | 2260.765 | 1.230–1.268× | 2809.700 | 4–6 |
| m16_k512_n512_t1 | 5909.085 | 2736.830 | 2.284–2.432× | 2834.100 | 1–1 |
| m16_k512_n512_dynamic | 6218.890 | 6196.990 | 1.000–1.007× | 14.800 | 1–1 |
| m64_k512_n512_t0 | 12249.245 | 11930.650 | 1.047–1.054× | 2841.550 | n/a |
| m64_k512_n512_t1 | 15344.650 | 11872.650 | 1.288–1.372× | 3135.100 | 1–2 |
| m64_k512_n512_dynamic | 15118.730 | 14992.405 | 1.012–1.016× | 12.100 | n/a |
| m3_k127_n131_t0 | 67.260 | 20.025 | 3.324–3.361× | 222.450 | 5–6 |
| m3_k127_n131_t1 | 220.530 | 20.580 | 11.074–11.320× | 230.000 | 1–2 |
| m3_k127_n131_dynamic | 242.360 | 239.360 | 1.022–1.027× | 14.500 | 3–4 |
| m16_k128_n128_dynamic_t0 | 130.550 | 128.540 | 1.019–1.037× | 12.550 | 3–6 |
| phase3_mlp_gemm | 12.100 | 3.820 | 3.159–3.199× | 26.350 | 4–4 |
| rl_actor_mlp_tanh | 14.105 | 4.940 | 2.800–2.843× | 38.450 | 4–5 |

## Preparation components and first execution

Preparation owns a Model snapshot, builds MemoryPlan, stores execution steps
and packs eligible constants. The benchmark separately measures a deep Model
copy, graph ordering, baseline MemoryPlan, effective RHS packing and baseline
RHS transpose copy. These are independent lifecycle diagnostics. They include
validation, allocation and destruction at their respective boundaries; they do
not partition the full preparation timer and must not be summed into a predicted
warm or preparation latency. Component and complete-run cache conditions differ.
For transB=0 the transpose-copy diagnostic is an empty loop/timer floor.

The [detailed appendix](rog-cpu-phase4.4-details.md) records every case's
component medians, context creation and first-inference medians, plus every
process's warm p50/p95/mean, paired ratio and amortization estimate.

## Persistent payload memory

The following byte counts are deterministic and identical in portable/native.
Plan payload = snapshot constants + packed RHS payload.
Context payload = copied constants + bound inputs + MemoryPlan activation
capacity. Source models, caller input copies, metadata, vector/allocator
capacity, extracted outputs and dynamic fallback temporaries are additional.
These counts are an explicit payload model, not process RSS or peak memory.

| Case | Snapshot constants B | Packed weights B | Plan payload B | Context constants B | Context inputs B | Activation B | Context payload B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| m1_k128_n128_t0 | 66048 | 65536 | 131584 | 66048 | 512 | 512 | 67072 |
| m1_k128_n128_t1 | 66048 | 65536 | 131584 | 66048 | 512 | 512 | 67072 |
| m1_k128_n128_dynamic | 512 | 0 | 512 | 512 | 66048 | 512 | 67072 |
| m16_k128_n128_t0 | 66048 | 65536 | 131584 | 66048 | 8192 | 8192 | 82432 |
| m16_k128_n128_t1 | 66048 | 65536 | 131584 | 66048 | 8192 | 8192 | 82432 |
| m16_k128_n128_dynamic | 512 | 0 | 512 | 512 | 73728 | 8192 | 82432 |
| m64_k128_n128_t0 | 66048 | 65536 | 131584 | 66048 | 32768 | 32768 | 131584 |
| m64_k128_n128_t1 | 66048 | 65536 | 131584 | 66048 | 32768 | 32768 | 131584 |
| m64_k128_n128_dynamic | 512 | 0 | 512 | 512 | 98304 | 32768 | 131584 |
| m1_k512_n512_t0 | 1050624 | 1048576 | 2099200 | 1050624 | 2048 | 2048 | 1054720 |
| m1_k512_n512_t1 | 1050624 | 1048576 | 2099200 | 1050624 | 2048 | 2048 | 1054720 |
| m1_k512_n512_dynamic | 2048 | 0 | 2048 | 2048 | 1050624 | 2048 | 1054720 |
| m16_k512_n512_t0 | 1050624 | 1048576 | 2099200 | 1050624 | 32768 | 32768 | 1116160 |
| m16_k512_n512_t1 | 1050624 | 1048576 | 2099200 | 1050624 | 32768 | 32768 | 1116160 |
| m16_k512_n512_dynamic | 2048 | 0 | 2048 | 2048 | 1081344 | 32768 | 1116160 |
| m64_k512_n512_t0 | 1050624 | 1048576 | 2099200 | 1050624 | 131072 | 131072 | 1312768 |
| m64_k512_n512_t1 | 1050624 | 1048576 | 2099200 | 1050624 | 131072 | 131072 | 1312768 |
| m64_k512_n512_dynamic | 2048 | 0 | 2048 | 2048 | 1179648 | 131072 | 1312768 |
| m3_k127_n131_t0 | 67072 | 66548 | 133620 | 67072 | 1524 | 1572 | 70168 |
| m3_k127_n131_t1 | 67072 | 66548 | 133620 | 67072 | 1524 | 1572 | 70168 |
| m3_k127_n131_dynamic | 524 | 0 | 524 | 524 | 68072 | 1572 | 70168 |
| m16_k128_n128_dynamic_t0 | 512 | 0 | 512 | 512 | 73728 | 8192 | 82432 |
| phase3_mlp_gemm | 68 | 48 | 116 | 68 | 8 | 24 | 100 |
| rl_actor_mlp_tanh | 148 | 120 | 268 | 148 | 16 | 44 | 208 |

For C live contexts, accounted payload is `plan_payload + C * context_payload`.
For example, static M1/K512/N512 retains 2,099,200 B in its plan and
1,054,720 B in each bound context; one plan plus one context accounts for
3,153,920 B. Original weights are retained alongside packed weights, and each
context still copies its constants. Prepared execution trades persistent memory
and one-time work for lower warm latency.

All static synthetic cases create one packed object and the ONNX fixtures
create two. Each measured session completes 1,021 runs and reports zero runtime
packs for static matrices. Dynamic cases create zero preparation packs and
report 1,021 runtime packs. Source-weight sharing and distinct transB packing
are separately covered by correctness tests.

## Reproduction and provenance

Configure in the Visual Studio x64 developer environment:

```powershell
cmake -S . -B build-phase44-native -G Ninja -DCMAKE_BUILD_TYPE=Release `
  -DTINYINFER_BUILD_TESTS=ON -DTINYINFER_BUILD_EXAMPLES=OFF `
  -DTINYINFER_BUILD_BENCHMARKS=ON -DTINYINFER_ENABLE_NATIVE_ARCH=ON `
  "-DCMAKE_CXX_FLAGS=/EHsc /permissive-"
cmake --build build-phase44-native --parallel 4
ctest --test-dir build-phase44-native -C Release --output-on-failure
python benchmarks/run_phase44.py `
  --executable build-phase44-native/benchmarks/tinyinfer_cpu_execution_plan_benchmark.exe `
  --output benchmark-results/phase4.4/rog-native --samples 100 --repeats 10 --runs 3
```

Repeat with `build-phase44-portable`, native architecture OFF and output
`benchmark-results/phase4.4/rog-portable`.
Raw per-process JSON, environment metadata and generated detailed summaries
are under those ignored directories. Configuration/build/CTest logs, host
metadata and measurement-source diff are under
`benchmark-results/phase4.4/rog-validation/`.

- native executable SHA-256:
  `bcf61be85beda7bbbe491480b9e45414203d49bd797c592c12b7a39028011833`
- portable executable SHA-256:
  `34165553a98b17ff9ab0eed33476839039545a2f64f7ca97e04205f01ea3268e`

The two runner environment files have different tracked-diff hashes because
documentation was updated while the sequential suites ran. Both binaries were
built from the same runtime source and the same benchmark diagnostics additions;
only native ISA policy differs. The runner records source revision, pending
changes, arguments and executable hashes. Current clock/temperature and process
RSS were not measured; ratio variability limits precise latency predictions.

## Acceptance

ROG correctness, static packing reuse, repeated performance gain and dynamic
control gates pass. Together with the existing Mac report, this closes Phase 4.4
for static FP32 prepared CPU inference. In-register epilogues, shape tuning,
threading and quantization remain later optimization work.

