# ROG CPU acceptance — Phase 4.3 Gemm + ReLU fusion

- Status: Development acceptance
- Measurement date: 2026-09-28
- Base Git commit: `b8dbdb3`
- Branch: `main` with the Phase 4.3 working-tree changes

## Environment

| Item | Value |
| --- | --- |
| CPU | 13th Gen Intel Core i9-13980HX |
| Logical processors | 32 |
| Architecture | x86_64 |
| Operating system | Windows NT 10.0, build 26200 |
| Compiler | Microsoft C/C++ 19.43.34808 |
| CMake | Visual Studio bundled CMake |
| Build type | Release |
| Native CPU instructions | Enabled (`/arch:AVX2`) |
| Additional compiler flags | `/EHsc /permissive-` |

The hardware and OS metadata reuse the same ROG target recorded in the
[v0.3 baseline](rog-cpu-v0.3.md). WMI metadata lookup was unavailable in this
session, so this report does not claim that those host fields were freshly
queried. The benchmark itself reports Windows x86_64 and Release.

## Acceptance contract

Before timing, the benchmark verifies that:

- `Gemm -> ReLU` becomes one `FusedGemmActivation` node;
- a second pass is unchanged;
- `Gemm -> Tanh` is not fused;
- planned execution produces the same shape, dtype, and FP32 values within
  `1e-5` absolute tolerance; and
- logical intermediate memory decreases.

The dedicated CTest additionally covers bias/no-bias, constant and dynamic
bias, `transB=0/1`, alpha/beta scaling, downstream consumers, planned and
unplanned execution, special floating-point values, observable or multiply
used Gemm results, pass composition, mapping, schema rejection, and source
model immutability.

| Configuration | Result | Total test time |
| --- | ---: | ---: |
| Release, native AVX2 | 39/39 passed | 1.00 s |
| Release, portable SSE2, clean rebuild | 39/39 passed | 3.71 s |

## Method

Each pair uses separate planned execution contexts with pre-bound deterministic
inputs. The original and fused variants alternate first position per sample.
Optimization, MemoryPlan construction, input binding, and correctness checks
are outside the timer.

```powershell
tinyinfer_gemm_activation_fusion_benchmark.exe `
  --m M --k 128 --n 128 `
  --warmup 20 --samples S --repeats R --format json
```

## Results

| Shape | Samples × repeats | Nodes | Planned bytes | Before p50 | Fused p50 | Before p95 | Fused p95 | p50 speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `[1,128] x [128,128]` | 200 × 50 | 2 → 1 | 1,024 → 512 | 11.408 us | 10.264 us | 11.718 us | 10.783 us | 1.111× |
| `[16,128] x [128,128]` | 200 × 20 | 2 → 1 | 16,384 → 8,192 | 36.233 us | 34.413 us | 53.481 us | 51.995 us | 1.053× |
| `[64,128] x [128,128]` | 100 × 10 | 2 → 1 | 65,536 → 32,768 | 109.390 us | 108.880 us | 123.464 us | 122.142 us | 1.005× |

For all three shapes, naive intermediate bytes equal the reported planned
bytes and decrease by 50%. The latency benefit is largest for the small batch,
where removing a separate ReLU traversal is a larger fraction of total work.
At batch 64, MatMul dominates and the measured p50 difference is close to
noise; Phase 4.3 should not be described as a universal large speedup.

## Interpretation

This result accepts the graph/operator fusion slice: it removes one node and
one full output-sized intermediate while preserving semantics. The combined
CPU epilogue yields a modest measured improvement for these shapes. A larger
gain for MatMul-dominated workloads would require applying the epilogue inside
the packed SIMD micro-kernel or avoiding repeated weight preparation, both of
which remain separate work.
