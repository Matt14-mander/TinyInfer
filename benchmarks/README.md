# TinyInfer benchmarks

Build benchmarks in Release mode so compiler optimization is enabled:

```bash
cmake -S . -B build-bench -DCMAKE_BUILD_TYPE=Release \
  -DTINYINFER_BUILD_TESTS=OFF \
  -DTINYINFER_BUILD_EXAMPLES=OFF \
  -DTINYINFER_BUILD_BENCHMARKS=ON \
  -DTINYINFER_ENABLE_NATIVE_ARCH=ON
cmake --build build-bench -j
```

On a macOS Command Line Tools installation whose default libc++ search path is
incomplete, `cstddef file not found` can be fixed by configuring against the
active SDK explicitly:

```bash
TINYINFER_MACOS_SDK=$(xcrun --show-sdk-path)
cmake -S . -B build-bench -DCMAKE_BUILD_TYPE=Release \
  -DTINYINFER_BUILD_TESTS=OFF \
  -DTINYINFER_BUILD_EXAMPLES=OFF \
  -DTINYINFER_BUILD_BENCHMARKS=ON \
  -DTINYINFER_ENABLE_NATIVE_ARCH=ON \
  -DCMAKE_OSX_SYSROOT="$TINYINFER_MACOS_SDK" \
  -DCMAKE_CXX_FLAGS="-isystem $TINYINFER_MACOS_SDK/usr/include/c++/v1"
```

Performance results from Debug builds are not meaningful. Keep the build type,
compiler, architecture, benchmark arguments, and Git commit with any published
result.

## Model Runtime benchmark

`tinyinfer_model_benchmark` measures the Phase 3 model path with the real
opset-17 Gemm MLP fixture. With no model argument it locates that fixture from
the source tree automatically:

```bash
./build-bench/benchmarks/tinyinfer_model_benchmark
```

The benchmark first runs a checked inference and rejects an incompatible model
or incorrect probabilities before timing anything. It then reports:

| Measurement | Included work |
| --- | --- |
| `file_and_parse` | Open/read `.onnx`, protobuf parsing, TensorProto decoding |
| `graph_import` | Operator translation, graph construction, shape inference and validation |
| `context_init` | ExecutionContext allocation and graph constant copies |
| `cold_inference` | New context, input creation/binding, and one CPU execution |
| `warm_inference` | CPU execution while reusing the model, context, input, and constants |

`Executor::run` clears intermediate values before each execution, so the warm
measurement still includes allocation of current operator outputs. It does not
yet represent a memory-planned runtime.

Because parsing is warmed up and repeated, `file_and_parse` normally measures
file access with the operating-system page cache already hot. It is a
steady-state loader measurement, not cold-disk startup latency.

Control the measurement volume with:

```bash
./build-bench/benchmarks/tinyinfer_model_benchmark \
  --warmup 20 \
  --samples 200 \
  --repeats 1000
```

Each timed sample contains `repeats` invocations and is divided by that count.
The output unit is microseconds per invocation. Results include minimum,
median/p50, p90, p95, p99, mean, and population standard deviation. Median and
p95 are the preferred summary values; minimum is useful mainly as a lower
bound.

The model path can be positional or passed with `--model`. The v0.3 benchmark
currently expects the documented MLP contract: one Float32 `[1, 2]` input and
the checked `[1, 2]` probability output.

Machine-readable output is available for later comparison tooling:

```bash
./build-bench/benchmarks/tinyinfer_model_benchmark \
  --samples 200 --repeats 1000 --format json
```

Run the executable on each target machine. The program records build type,
platform, architecture, and compiler; add the exact CPU model, operating
system version, power mode, and Git commit when committing a benchmark report.

## MatMul benchmark

Run:

```bash
./build-bench/benchmarks/tinyinfer_matmul_benchmark
```

The default square sizes are 64, 128, 256, and 512. Custom sizes can be
passed as command-line arguments. The allocation cost is excluded: each
kernel writes into a preallocated output Tensor. Reference results are timed
through size 128 because its indexed implementation intentionally prioritizes
clarity; larger results are checked against the stride-aware kernel.

Output reports average milliseconds and GFLOP/s for reference, stride-aware,
cache-blocked, and packed SIMD kernels. Packed SIMD timing reuses an RHS packed
before the timed loop, modeling inference where constant model weights are
packed once and reused. Every run verifies the optimized outputs before it
finishes.

`TINYINFER_ENABLE_NATIVE_ARCH` is optional and disabled by default so release
binaries remain portable. Enabling it lets x86 builds select AVX2/FMA when the
host supports them; baseline x86-64 uses SSE2, and ARM builds use NEON.
