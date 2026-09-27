# Mac CPU — Phase 4.2 graph optimization acceptance

- Status: Accepted for the Phase 4.2 graph-simplification slice
- Measurement date: 2026-09-27
- Source commit: `7f1b89235c9b89ca35890c1c3739502fce898ac6` plus the benchmark and documentation changes in this report
- Branch: `main` working tree

## Scope and correctness

The benchmark imports `tests/fixtures/phase3_mlp_gemm.onnx`, a real opset-17
Gemm MLP. This model is the unchanged control: Constant Folding followed by
Dead Code Elimination reports no graph change. An acceptance model is then
constructed from that imported model's actual weights and the same input/output
contract. It adds three constant-only bias operations and a disconnected
four-node output head. The pipeline folds all three bias operations and deletes
the unused head, reducing the graph from **11 to 4 nodes** and **18 to 9 Values**.
The benchmark checks the imported model, its optimized control copy, and both
acceptance-model variants for three inputs with absolute tolerance `1e-5`,
checks the named model interface and pass statistics,
and checks that a second pass run changes nothing.

This is a deliberately optimizable model. It demonstrates the cost of work
that Phase 4.2 can remove, not an expected speedup for arbitrary exported
models. The unchanged imported ONNX model is separately timed to expose the
local measurement variation when the passes find no eligible work.

## Environment and method

| Item | Value |
| --- | --- |
| Machine | MacBookPro15,4, Intel Core i5 at 1.4 GHz, 4 cores |
| Operating system | macOS 15.4.1 (24E263) |
| Architecture | x86_64 |
| Compiler | Apple Clang 17.0.0 (`clang-1700.0.13.3`) |
| CMake | 4.4.3 |
| Build | Release, `TINYINFER_ENABLE_NATIVE_ARCH=ON` |
| Warm-up | 20 invocations per variant |
| Samples | 200 per variant in each of 3 processes |
| Repeats | 100 invocations per timed sample |

Power mode was not captured. The large run-to-run latency variation below is
another reason to treat the within-run ratio as the useful observation here.

The executable alternates before/after timing order each sample. Each variant
has its own reused `ExecutionContext` with input bound before timing. Model
loading, optimization, and context creation are excluded from warm-inference
latency. `MemoryPlan` is disabled in both variants. Raw JSON for the three
runs is saved locally under `benchmark-results/phase4.2/` and ignored by Git.

Reproduce with:

```bash
cmake -S . -B build-bench -DCMAKE_BUILD_TYPE=Release \
  -DTINYINFER_BUILD_TESTS=OFF -DTINYINFER_BUILD_EXAMPLES=OFF \
  -DTINYINFER_BUILD_BENCHMARKS=ON -DTINYINFER_ENABLE_NATIVE_ARCH=ON
cmake --build build-bench --target tinyinfer_graph_optimization_benchmark
./build-bench/benchmarks/tinyinfer_graph_optimization_benchmark \
  --warmup 20 --samples 200 --repeats 100 --format json
```

## Results

All timings below are microseconds per warm inference. Ratios divide the
before p50 by the after p50 within the same process.

| Run | Acceptance before p50 / p95 | Acceptance after p50 / p95 | p50 ratio | Unchanged ONNX before / after p50 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 87.846 / 146.766 | 32.015 / 59.472 | 2.744× | 30.855 / 30.649 |
| 2 | 275.604 / 438.123 | 104.927 / 177.460 | 2.627× | 129.813 / 128.939 |
| 3 | 235.499 / 321.499 | 86.786 / 133.613 | 2.714× | 96.873 / 99.750 |

The within-run ratio stays between **2.63× and 2.74×**. Absolute latency
varies substantially across runs, including the unchanged ONNX control, so
the table should not be used as a stable latency target. The control does not
show a systematic benefit when the graph has no foldable or dead work. The
acceptance model shows a consistent benefit from removing seven executed
nodes, with correctness verified before timing. This completes the Phase 4.2
before/after benchmark gate for the documented graph-simplification scope.

## Boundary

The Pass pipeline remains opt-in. This result does not validate automatic
optimization during ONNX import, a fusion kernel, a memory-planned runtime, or
other hardware. Later Phase 4 slices and hardware-specific performance claims
need their own acceptance measurements.
