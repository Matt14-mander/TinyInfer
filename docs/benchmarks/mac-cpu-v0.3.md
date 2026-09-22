# Mac CPU baseline — TinyInfer v0.3

- Status: Final v0.3 baseline
- Measurement date: 2026-09-21
- Git commit: `da318707b1f6e5aa167cf5b57970e74a4e0480c0`
- Branch: `main`

## Scope

This report establishes the first Mac CPU baseline for the TinyInfer v0.3
Model Runtime and the existing CPU MatMul kernels. It is intended to:

- verify that the benchmark pipeline produces stable, reproducible results;
- separate model loading, graph import, context creation, and inference costs;
- preserve a CPU reference point for the later ROG comparison;
- expose optimization opportunities without claiming production-runtime
  performance.

The raw outputs are stored locally under `benchmark-results/mac-cpu/` and are
excluded from Git. This report contains the reviewed results that should be
versioned with the project.

## Environment

| Item | Value |
| --- | --- |
| CPU | Intel Core i5-8257U @ 1.40 GHz |
| Architecture | x86_64 |
| Operating system | macOS 15.4.1, build 24E263 |
| Compiler | Apple Clang 17.0.0 (`clang-1700.0.13.3`) |
| Compiler target | `x86_64-apple-darwin24.4.0` |
| CMake | 4.4.3 |
| Build type | Release |
| Native CPU instructions | Enabled |

The Release build used `TINYINFER_ENABLE_NATIVE_ARCH=ON`. On this machine the
Command Line Tools SDK path was supplied explicitly so Clang could locate the
complete libc++ headers.

## Model Runtime methodology

The benchmark uses `tests/fixtures/phase3_mlp_gemm.onnx`, an opset-17 model:

```text
Float32 [1, 2]
    -> Gemm(transB=1)
    -> ReLU
    -> Gemm(transB=1)
    -> Softmax
    -> Float32 [1, 2]
```

Before measurement, the benchmark verifies the expected probabilities:

```text
[0.26894143, 0.73105860]
```

Each of three independent runs used:

```text
warmup:             20
samples:            200
repeats per sample: 1000
unit:               microseconds per invocation
```

The primary table reports the median of the three run-level statistics. This
avoids selecting the fastest run while keeping the result easy to compare with
future machines. The file benchmark is warmed and normally uses the macOS page
cache; it measures steady-state file loading and parsing, not cold-disk startup.

### Model Runtime baseline

| Stage | P50 (us) | P95 (us) | Mean (us) |
| --- | ---: | ---: | ---: |
| File read + protobuf parse | 32.662 | 35.247 | 33.047 |
| Graph import | 42.432 | 44.676 | 42.596 |
| ExecutionContext initialization | 17.517 | 18.223 | 17.582 |
| Cold inference | 50.703 | 52.325 | 51.180 |
| Warm inference | 31.682 | 32.852 | 32.238 |

### Run-to-run detail

| Stage | Run 1 P50/P95 (us) | Run 2 P50/P95 (us) | Run 3 P50/P95 (us) |
| --- | ---: | ---: | ---: |
| File read + protobuf parse | 33.161 / 41.347 | 32.586 / 33.742 | 32.662 / 35.247 |
| Graph import | 43.281 / 48.989 | 41.924 / 43.148 | 42.432 / 44.676 |
| ExecutionContext initialization | 17.517 / 18.223 | 17.498 / 18.354 | 17.541 / 18.069 |
| Cold inference | 50.703 / 52.325 | 50.381 / 51.868 | 50.806 / 56.956 |
| Warm inference | 31.682 / 32.755 | 31.684 / 33.040 | 31.557 / 32.852 |

The P50 values are stable across the three processes: the largest spread is
about 3.2% for graph import, while context initialization, cold inference, and
warm inference remain within about 1%. P95 shows occasional operating-system
or scheduling noise, most visibly in file parsing and the third cold-inference
run.

Warm inference is approximately 19.0 us, or 37.5%, faster than cold inference
at P50. The 17.5 us median context-initialization cost explains most of that
difference. Context construction currently deep-copies graph constants, so
this measurement is also a useful baseline for future constant sharing and
memory planning.

File parsing and graph import have a combined median of approximately 75.1 us
when their independently measured P50 values are added. This number is only a
component-level estimate; an explicit end-to-end load timer would be needed
for a latency claim about one complete load call.

## MatMul methodology

The extended MatMul run covered square Float32 matrices from 32 to 1024. Kernel
allocation is excluded. The packed-SIMD path reuses an RHS packed before the
timed loop, matching inference with constant weights. Reference timing is
reported only through size 128 because that implementation prioritizes
readability and checked indexing.

### MatMul baseline

| Size | Reference ms / GFLOP/s | Strided ms / GFLOP/s | Blocked ms / GFLOP/s | Packed SIMD ms / GFLOP/s |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 5.780 / 0.011 | 0.023 / 2.862 | 0.003 / 20.125 | 0.003 / 21.490 |
| 64 | 37.715 / 0.014 | 0.221 / 2.373 | 0.023 / 22.775 | 0.024 / 22.033 |
| 128 | 312.068 / 0.013 | 1.957 / 2.144 | 0.186 / 22.594 | 0.217 / 19.338 |
| 256 | — | 17.732 / 1.892 | 1.587 / 21.140 | 1.535 / 21.861 |
| 512 | — | 225.608 / 1.190 | 15.575 / 17.235 | 14.399 / 18.643 |
| 1024 | — | 3107.859 / 0.691 | 154.250 / 13.922 | 154.163 / 13.930 |

Cache blocking and packed SIMD both materially outperform the direct
stride-aware kernel. Packed SIMD reaches about 19–22 GFLOP/s for sizes 32–512,
and its throughput advantage over the strided kernel grows from roughly 7.5x
at size 32 to 20.2x at size 1024. It does not consistently beat the blocked
kernel at small sizes; sizes 64 and 128 favor the blocked path in this run.
This suggests that kernel selection should eventually consider matrix shape
and packing/micro-kernel overhead rather than dispatch every size identically.

Both optimized kernels lose throughput at 512 and 1024, consistent with a
larger cache and memory-system working set. The near-identical 1024 results
also show that the current SIMD path is not removing the dominant large-matrix
bottleneck.

## Limitations

- The model fixture is intentionally tiny. Fixed runtime overhead dominates,
  so these inference numbers must not be generalized to larger neural models.
- The Executor performs topological ordering and materializes operator outputs
  on every run; no static memory plan or buffer reuse exists yet.
- Gemm applies transpose handling and composed eager operators rather than a
  fully fused prepacked inference kernel.
- `file_and_parse` is page-cache warm and does not measure cold storage.
- MatMul reports one extended run; it does not yet contain independent
  process-level repetitions or confidence intervals.
- CPU frequency, thermals, background activity, and power state were not
  captured continuously during each sample.
- Results apply only to the recorded commit, compiler, build configuration,
  and machine.

## ROG comparison protocol

The ROG CPU baseline should use the same Git commit and parameters:

```text
commit:              da318707b1f6e5aa167cf5b57970e74a4e0480c0
build:               Release
native architecture: enabled
model warmup:         20
model samples:        200
model repeats:        1000
model runs:           3 independent processes
MatMul sizes:         32, 64, 128, 256, 512, 1024
```

The first comparison should remain CPU-to-CPU. CUDA results should be recorded
as a separate backend baseline so hardware, operating system, compiler, and
backend changes are not mixed into one number.
