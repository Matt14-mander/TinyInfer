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
| `memory_plan_build` | Static lifetime analysis and reusable slot assignment |
| `planned_context_init` | Constant copies plus one shared intermediate Buffer allocation |
| `cold_inference` | New context, input creation/binding, and one CPU execution |
| `warm_inference` | CPU execution while reusing the model, context, input, and constants |
| `planned_warm_inference` | CPU execution with direct writes into reusable planned storage |

`Executor::run` clears intermediate values before each execution, so the warm
measurement still includes independent allocation of current operator outputs.
Compare it with `planned_warm_inference`, which reuses one shared Buffer. The
planner reports intermediate capacity separately from inputs, constants,
packed weights, and backend scratch space; intermediate graph outputs are part
of the plan but remain live through the invocation.

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

## Phase 4.2 graph optimization benchmark

`tinyinfer_graph_optimization_benchmark` imports the real Phase 3 ONNX Gemm MLP.
It measures two pairs of warm CPU executions with pre-bound inputs and separate
reused `ExecutionContext`s:

- The imported ONNX model before and after Constant Folding + DCE is an
  unchanged control. The passes should report no graph change.
- An acceptance model built from the imported weights adds a constant-only
  bias chain and an unused output head. The same passes reduce it from 11 to
  4 nodes while preserving the imported model's probabilities.

Before timing, the executable checks outputs for three inputs against the
imported ONNX model, validates the expected node and pass counts, and checks
that a second optimization run makes no change. It alternates which variant
runs first in each timed sample to limit order bias. Model import, pass execution,
and context creation are outside the inference timers. This benchmark does not
use `MemoryPlan`, so it isolates graph simplification from runtime buffer reuse.

```bash
./build-bench/benchmarks/tinyinfer_graph_optimization_benchmark \
  --warmup 20 --samples 200 --repeats 100 --format json
```

The benchmark reports per-variant p50, p95, and mean microseconds, graph sizes,
and the p50 ratio. The acceptance model deliberately contains removable work;
its speedup is not a prediction for an arbitrary ONNX model. The unchanged
control shows measurement noise and the cost of running the optimized copy of
a model with no matching patterns. See the
[Phase 4.2 Mac acceptance report](../docs/benchmarks/mac-cpu-phase4.2.md).

## Phase 4.3 Gemm + ReLU fusion benchmark

`tinyinfer_gemm_activation_fusion_benchmark` constructs a deterministic
Float32 `Gemm -> ReLU` model, applies `GemmActivationFusionPass`, and measures
the original and fused models with separate planned `ExecutionContext`s. Before
timing it verifies numerical equivalence, the expected two-to-one node rewrite,
idempotence, reduced logical intermediate memory, and that a `Gemm -> Tanh`
control is not changed.

```bash
./build-bench/benchmarks/tinyinfer_gemm_activation_fusion_benchmark \
  --m 16 --k 128 --n 128 \
  --warmup 10 --samples 50 --repeats 10 --format json
```

The dimensions and measurement volume are configurable. Output includes p50,
p95, and mean microseconds, node counts, naive intermediate bytes, planned
buffer bytes, and the p50 speedup. The timer covers planned warm graph
execution only; model construction, optimization, correctness checks,
MemoryPlan construction, and input binding are excluded. This implementation
combines the affine and ReLU epilogue traversal but does not yet fuse that
epilogue into the SIMD MatMul micro-kernel.

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
