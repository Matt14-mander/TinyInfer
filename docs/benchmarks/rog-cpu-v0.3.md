# ROG CPU baseline — TinyInfer v0.3

- Status: Final v0.3 baseline
- Measurement date: 2026-09-22
- Git commit: `7f94b4550732c280bc7260804fc5c7ca2922582b`
- Branch: `main`

## Scope

This report records the TinyInfer v0.3 Model Runtime and CPU MatMul results on
the ROG target machine. It covers:

- the complete CTest suite with native AVX2 instructions enabled;
- the same suite with native instructions disabled, exercising the portable
  x86-64/SSE2 path;
- all three example programs;
- three independent Model Runtime benchmark processes; and
- the MatMul reference, stride-aware, cache-blocked, and packed-SIMD kernels
  for square matrices from 32 to 1024.

The existing Mac report used commit
`da318707b1f6e5aa167cf5b57970e74a4e0480c0`. This ROG run used the newer
commit recorded above, so the two reports are useful engineering reference
points but are not a strict same-commit performance comparison.

## Environment

| Item | Value |
| --- | --- |
| CPU | 13th Gen Intel Core i9-13980HX |
| Logical processors | 32 |
| Architecture | x86_64 |
| Operating system | Windows NT 10.0, build 26200 |
| Power scheme | Performance |
| Compiler | Microsoft C/C++ 19.43.34808 |
| CMake | 3.30.5-msvc23 |
| Build type | Release |
| Native CPU instructions | Enabled for the benchmark build (`/arch:AVX2`) |
| Additional compiler flags | `/EHsc /permissive-` |

The build used the Visual Studio 2022 Build Tools installation and Ninja
bundled with Visual Studio.

## Build and correctness validation

The native build enabled tests, examples, benchmarks, and host-specific CPU
instructions:

```powershell
cmake -S . -B build-rog-v03 -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DTINYINFER_BUILD_TESTS=ON `
  -DTINYINFER_BUILD_EXAMPLES=ON `
  -DTINYINFER_BUILD_BENCHMARKS=ON `
  -DTINYINFER_ENABLE_NATIVE_ARCH=ON `
  "-DCMAKE_CXX_FLAGS=/EHsc /permissive-"
cmake --build build-rog-v03 --parallel
ctest --test-dir build-rog-v03 --output-on-failure
```

The portable build disabled native architecture instructions and therefore
validated the baseline x86-64/SSE2 path separately:

```powershell
cmake -S . -B build-rog-v03-portable -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DTINYINFER_BUILD_TESTS=ON `
  -DTINYINFER_BUILD_EXAMPLES=OFF `
  -DTINYINFER_BUILD_BENCHMARKS=OFF `
  -DTINYINFER_ENABLE_NATIVE_ARCH=OFF `
  "-DCMAKE_CXX_FLAGS=/EHsc /permissive-"
cmake --build build-rog-v03-portable --parallel
ctest --test-dir build-rog-v03-portable --output-on-failure
```

### Correctness summary

| Configuration | Result | Total test time |
| --- | ---: | ---: |
| Release, native AVX2 | 30/30 passed | 2.28 s |
| Release, portable SSE2 | 30/30 passed | 5.37 s |

All three example executables completed successfully. The eager and Graph
Runtime MLP examples both produced:

```text
logits:        [2, 3]
probabilities: [0.268941, 0.731059]
```

### MSVC build note

With the repository's default MSVC flags, compilation stopped in the GELU
implementation because the compiler diagnosed the no-capture lambda's use of
the two local `constexpr` coefficients as error `C3493`. Adding
`/permissive-` enabled the standard-conforming interpretation and allowed the
source to compile without modification.

`/EHsc` must be retained when supplying `CMAKE_CXX_FLAGS`. An initial
configuration that replaced the default flags with only `/permissive-`
disabled exception unwinding and caused the protobuf parser test to terminate.
After restoring `/EHsc`, all 30 tests passed in both builds.

## Model Runtime methodology

The benchmark uses `tests/fixtures/phase3_mlp_gemm.onnx`, the same opset-17
Gemm MLP fixture described by the Mac baseline. Each of three independent
processes used:

```text
warmup:             20
samples:            200
repeats per sample: 1000
unit:               microseconds per invocation
```

The benchmark verified the expected probabilities before timing. The primary
table reports the median of the three run-level P50, P95, and mean values.

### Model Runtime baseline

| Stage | P50 (us) | P95 (us) | Mean (us) |
| --- | ---: | ---: | ---: |
| File read + protobuf parse | 44.136 | 75.075 | 49.117 |
| Graph import | 27.203 | 32.102 | 26.226 |
| ExecutionContext initialization | 11.211 | 16.539 | 11.760 |
| Cold inference | 27.542 | 33.106 | 26.381 |
| Warm inference | 17.851 | 21.502 | 16.828 |

Warm inference is approximately 9.691 us, or 35.2%, faster than cold
inference at P50. This difference primarily reflects context construction,
constant copying, and input setup excluded by the warm measurement.

### Run-to-run detail

| Stage | Run 1 P50/P95 (us) | Run 2 P50/P95 (us) | Run 3 P50/P95 (us) |
| --- | ---: | ---: | ---: |
| File read + protobuf parse | 44.136 / 74.596 | 43.542 / 75.075 | 46.559 / 79.824 |
| Graph import | 27.549 / 39.477 | 27.203 / 32.102 | 26.236 / 30.777 |
| ExecutionContext initialization | 13.447 / 16.794 | 10.469 / 12.316 | 11.211 / 16.539 |
| Cold inference | 30.835 / 42.746 | 25.119 / 33.106 | 27.542 / 32.760 |
| Warm inference | 17.851 / 20.659 | 15.510 / 21.502 | 18.953 / 22.698 |

File parsing has the widest and highest-latency tail. Context initialization
and cold inference also vary between processes, while graph import and warm
inference remain within a narrower range. The file measurement normally uses
the Windows page cache and is not a cold-storage startup measurement.

## MatMul methodology

The native Release executable ran:

```powershell
.\build-rog-v03\benchmarks\tinyinfer_matmul_benchmark.exe `
  32 64 128 256 512 1024
```

Allocation is excluded. The packed-SIMD path reuses a packed RHS, and every
optimized result is checked for correctness. Reference timing is intentionally
limited to sizes through 128.

### MatMul baseline

| Size | Reference ms / GFLOP/s | Strided ms / GFLOP/s | Blocked ms / GFLOP/s | Packed SIMD ms / GFLOP/s |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 2.102 / 0.031 | 0.010 / 6.489 | 0.012 / 5.301 | 0.002 / 37.719 |
| 64 | 16.705 / 0.031 | 0.069 / 7.555 | 0.139 / 3.768 | 0.013 / 39.420 |
| 128 | 137.195 / 0.031 | 1.026 / 4.086 | 0.796 / 5.272 | 0.091 / 46.244 |
| 256 | — | 8.274 / 4.056 | 5.383 / 6.233 | 0.625 / 53.721 |
| 512 | — | 70.712 / 3.796 | 56.148 / 4.781 | 7.296 / 36.793 |
| 1024 | — | 2739.740 / 0.784 | 426.629 / 5.034 | 84.011 / 25.562 |

Packed SIMD is the fastest kernel at every measured size. It peaks at 53.721
GFLOP/s for size 256 and remains approximately 5.1 times faster than the
blocked kernel at size 1024. Throughput falls after size 256 as the working set
outgrows the most favorable cache levels, but the AVX2 packed path maintains a
large advantage over both the blocked and stride-aware implementations.

The blocked kernel is slower than the stride-aware kernel at sizes 32 and 64,
then becomes faster from size 128 onward. This reinforces the need for
shape-aware kernel selection if TinyInfer later introduces automatic MatMul
dispatch heuristics.

## Limitations

- The ROG and Mac reports use different Git commits, so their results are not
  a controlled CPU-only comparison.
- The model fixture is intentionally tiny and is dominated by runtime overhead
  rather than neural-network arithmetic.
- CPU package power, temperature, and instantaneous clock frequency were not
  recorded during measurement.
- Model parsing uses a warm operating-system page cache.
- The MatMul table contains one benchmark process rather than multiple
  process-level repetitions or confidence intervals.
- MSVC required `/permissive-`; builds that omit it do not compile this commit
  on the tested toolchain.

## Reproduction note

For a strict Mac-versus-ROG comparison, check out one common commit on both
machines and repeat the same Release/native configuration, fixture, model
benchmark volume, process count, and MatMul sizes. Record those results as a
new paired baseline rather than mixing them with the current reports.
