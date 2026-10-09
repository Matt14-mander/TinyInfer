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

## Phase 4.4 prepared CPU benchmark

`tinyinfer_cpu_execution_plan_benchmark` compares the **same optimized graph**
using Phase 4.3 Executor + MemoryPlan and CpuExecutionPlan. Both retain the
Phase 4.3 separate combined epilogue. Static B is packed once by the prepared
path; `--dynamic-b` retains the current kernel and packs B each run. Synthetic
cases use FusedGemmActivation; `--model` imports an ONNX model and applies the
explicit Constant Folding, DCE, MatMul + Add canonicalization, Gemm + ReLU
fusion, DCE pipeline. The actor's Tanh remains an ordinary operator.

```bash
cmake --build build-bench --target tinyinfer_cpu_execution_plan_benchmark
./build-bench/benchmarks/tinyinfer_cpu_execution_plan_benchmark \
  --m 16 --k 128 --n 128 --trans-b 1 \
  --warmup 20 --samples 100 --repeats 10 --format json
./build-bench/benchmarks/tinyinfer_cpu_execution_plan_benchmark \
  --m 16 --k 128 --n 128 --dynamic-b --format json
./build-bench/benchmarks/tinyinfer_cpu_execution_plan_benchmark \
  --model tests/fixtures/rl_actor_mlp_tanh.onnx --format json
```

Correctness is checked before and after timing at absolute-plus-relative
`1e-5 + 1e-5 * abs(reference)`. Timers report microseconds:

| Measurement | Included work |
| --- | --- |
| `topological_order` | One existing graph sort |
| `baseline_memory_plan` | One existing static lifetime/slot analysis |
| `preparation` | Model snapshot, MemoryPlan, stored steps and constant packing |
| `model_snapshot` | Independent deep Model copy and destruction; preparation component diagnostic |
| `baseline_context_init` / `prepared_context_init` | Context/activation allocation and constant copies |
| `baseline_rhs_pack` | Packing effective RHS for all matrix nodes; effective operand copies prepared outside this timer |
| `baseline_rhs_transpose_copy` | Baseline deep-copy plus RHS transpose metadata for transB nodes |
| `baseline_first_inference` / `prepared_first_inference` | One run in a newly created/bound context; plan/packing already prepared |
| `baseline_warm` / `prepared_warm` | Reused contexts, inputs, buffers; alternate variant order per sample |

Preparation/component/context timers measure at most 20 single-invocation
samples, with one warm-up. Local temporary destruction is inside preparation
and context timers, so these are lifecycle cost measurements. First inference
runs in a fresh context, with existing model and code caches already warm; it
is not process-cold startup. Warm samples average `repeats` runs each.

Import, optimization, input creation/binding, and copied output extraction are
outside inference timers. Do not add component medians to predict end-to-end
latency. `p50_speedup` divides variant medians; `paired_speedup_p50` is the
median of the same-sample baseline/prepared ratios. Packed bytes count payload
only, and activation bytes count MemoryPlan capacity. See
[prepared API memory boundaries](../docs/architecture/cpu_execution_plan.md).
`runtime_pack_count` is a count of successful fallback matrix dispatches.
`full_preparation_amortization_calls` charges all mean preparation cost against
positive mean per-call savings; absent values mean no positive savings.

`snapshot_constant_bytes` and `context_constant_bytes` count logical constant
payload retained in the plan and copied into each context. `context_input_bytes`
counts bound input payload; `plan_accounted_payload_bytes` adds snapshot constants
and packed weights, while `context_accounted_payload_bytes` adds context
constants, inputs and activation capacity. These totals exclude metadata,
allocator/vector capacity, caller-owned source models, output copies and runtime
fallback scratch; they are not RSS measurements. `detailed-summary.md` preserves
component, context, memory and per-process p50/p95/mean tables.

### Reproducible Mac / ROG suite

The standard-library Python runner executes all shape/transpose cases, dynamic
controls and both fixtures in **three sequential processes per case**. It stores
raw JSON, source/environment metadata and summary tables under an ignored path:

```bash
python3 benchmarks/run_phase44.py \
  --executable build-bench/benchmarks/tinyinfer_cpu_execution_plan_benchmark \
  --output benchmark-results/phase4.4/mac --samples 100 --repeats 10 --runs 3
```

ROG PowerShell (from the repository root, using its configured C++ toolchain):

```powershell
cmake -S . -B build-phase44-native -DCMAKE_BUILD_TYPE=Release -DTINYINFER_BUILD_TESTS=ON -DTINYINFER_BUILD_EXAMPLES=OFF -DTINYINFER_BUILD_BENCHMARKS=ON -DTINYINFER_ENABLE_NATIVE_ARCH=ON
cmake --build build-phase44-native --config Release --parallel 4
ctest --test-dir build-phase44-native -C Release --output-on-failure
# Multi-config generators place the executable under benchmarks/Release.
python benchmarks/run_phase44.py --executable build-phase44-native/benchmarks/Release/tinyinfer_cpu_execution_plan_benchmark.exe --output benchmark-results/phase4.4/rog --samples 100 --repeats 10 --runs 3
```

For a single-config ROG generator the executable is directly under `benchmarks/`.
Also run the full Release suite with `TINYINFER_ENABLE_NATIVE_ARCH=OFF` in a
separate build directory. Keep power mode, CPU/OS/compiler, revision and pending
source changes with the report. Investigate repeatable control regressions over
5%; compare ratios within a host. See the [Mac report](../docs/benchmarks/mac-cpu-phase4.4.md).
The completed [ROG report](../docs/benchmarks/rog-cpu-phase4.4.md) includes both
ISA configurations, preparation components, payload memory and all per-process
statistics.

## Phase 4.5 prepared CPU performance analysis

This benchmark measures the unchanged Phase 4.4 prepared path and direct
component controls. Build:

```bash
cmake -S . -B build-bench -DCMAKE_BUILD_TYPE=Release \
  -DTINYINFER_BUILD_BENCHMARKS=ON -DTINYINFER_ENABLE_NATIVE_ARCH=ON
cmake --build build-bench --target tinyinfer_cpu_performance_analysis_benchmark
./build-bench/benchmarks/tinyinfer_cpu_performance_analysis_benchmark \
  --m 16 --k 128 --n 128 --bias vector --activation relu \
  --warmup 20 --samples 100 --repeats 10 --format json
python3 benchmarks/run_phase45.py \
  --executable build-bench/benchmarks/tinyinfer_cpu_performance_analysis_benchmark \
  --output benchmark-results/phase4.5/mac-native --samples 100 --repeats 10 --runs 3
```

The runner executes 25 cases in three sequential processes each. Use separate
portable/native builds; do not run build/test jobs during latency measurement.
ROG PowerShell can use its existing Ninja configuration, or a new directory:

```powershell
cmake -S . -B build-phase45-native -G Ninja -DCMAKE_BUILD_TYPE=Release -DTINYINFER_BUILD_BENCHMARKS=ON -DTINYINFER_BUILD_EXAMPLES=OFF -DTINYINFER_ENABLE_NATIVE_ARCH=ON
cmake --build build-phase45-native --target tinyinfer_cpu_performance_analysis_benchmark --parallel 4
python benchmarks/run_phase45.py --executable build-phase45-native/benchmarks/tinyinfer_cpu_performance_analysis_benchmark.exe --output benchmark-results/phase4.5/rog-native --samples 100 --repeats 10 --runs 3
```

Run from a developer shell with the working MSVC toolchain/environment used in
Phase 4.4. With a Visual Studio multi-config generator, build `--config Release`
and select the executable under `benchmarks/Release`. Repeat in a separate
portable directory with native architecture OFF.

JSON contains full prepared execution, metadata bookkeeping control, captured
matrix-chain execution, input/output copies, per-matrix MatMul/epilogue/complete,
scalar-tail control and isolated ordinary operator registry dispatch. Pool resets
for epilogue are outside timers; each output is transformed once per sample.
`epilogue_pool_bytes` records the per-node buffer pool payload.
Every process checks prepared versus ordinary outputs, direct matrix outputs,
epilogue buffers, scalar tails and isolated ordinary outputs. Setup uses frozen
operands from ordinary execution and retains the public runtime unchanged.

Read the [analysis boundary](../docs/plans/phase4.5.md#450--analysis-boundary)
before interpreting ratios. Components have different cache/loop conditions;
they cannot be added into an exact graph latency or used to claim achieved
fusion speedup. Dynamic B still packs every production invocation; direct
captured-component timing excludes that packing. Tail control uses a strided
scalar kernel rather than instrumenting the production tail. The CLI also permits `--bias none --activation none` to inspect the no-op
epilogue floor; the standard suite uses no-bias/ReLU and vector-bias/None controls. Output extraction and input binding
copy payloads and are excluded from prepared-run latency.

## Phase 4.5 Gemm kernel fusion comparison

Use the same Release executable with `--fusion-comparison`. This measures three
reused prepared contexts: `Legacy`, direct broadcast `Specialized`, and final
K-block register `Fused`. Order rotates every sample; each sample averages
`repeats` runs. JSON reports each latency distribution and the median of paired
sample speedups (baseline time / candidate time). Setup, input binding and output
copy/checks are outside timers. All three plans retain the same graph, packing
and activation buffer policy. Dynamic B remains a fallback control.

```bash
python3 benchmarks/run_phase45_fusion.py \
  --executable build-bench/benchmarks/tinyinfer_cpu_performance_analysis_benchmark \
  --output benchmark-results/phase4.5-fusion/mac-native \
  --warmup 20 --samples 100 --repeats 10 --runs 3
```

The suite has 26 cases, including an extra no-bias/None no-op control. Run native
then portable sequentially after all builds/tests finish. Raw JSON, environment
and source hashes are retained in the output directory. Ratios above one mean
faster. Report process ranges and p95 alongside medians; inspect repeatable
regressions above 5% before making host acceptance claims.

ROG PowerShell, after building/testing both configurations:

```powershell
python benchmarks/run_phase45_fusion.py --executable build-phase45-native/benchmarks/tinyinfer_cpu_performance_analysis_benchmark.exe --output benchmark-results/phase4.5-fusion/rog-native --warmup 20 --samples 100 --repeats 10 --runs 3
python benchmarks/run_phase45_fusion.py --executable build-phase45-portable/benchmarks/tinyinfer_cpu_performance_analysis_benchmark.exe --output benchmark-results/phase4.5-fusion/rog-portable --warmup 20 --samples 100 --repeats 10 --runs 3
```

For Visual Studio generators use `benchmarks/Release` executables. Historical
Phase 4.4 and Phase 4.5 component analysis select Legacy explicitly; the default
production plan can therefore evolve without redefining those baselines.

The [local fusion report](../docs/benchmarks/mac-cpu-phase4.5-fusion.md) selects
Specialized as the production default. Fused is experimental: numerical gates
passed, but its initial SIMD implementation has performance regressions, especially
in Mac portable SSE2. [ROG testing](../docs/benchmarks/rog-cpu-phase4.5-fusion.md)
is complete: 41/41 tests per build, 150 analysis processes, 156 three-mode
processes and 18 independent control rechecks. Native specialization fails
the repeated no-bias/ReLU 5% regression gate; native fused no-op also fails.
Phase 4.5 remains open for performance remediation, not pending ROG execution.
The MSVC native-OFF build reports explicit kernel width 1 (scalar fallback),
not SSE2. Record actual `simd_width` rather than assuming OFF means width 4.
The [ROG analysis](../docs/benchmarks/rog-cpu-phase4.5.md) retains component/copy
diagnostics and buffer payloads; both reports link full per-process evidence.
