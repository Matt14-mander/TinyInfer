# Mac CPU — Phase 4.5 prepared performance analysis

- Status: Local performance analysis complete; target ROG analysis reproduction pending
- Measurement date: 2026-10-07 (Asia/Shanghai)
- Source baseline: `f472304017d547def1dde58c804ea9082ea0af7c` plus the new analysis benchmark/runner
- Runtime: unchanged Phase 4.4 prepared execution; no new kernel implemented

## Environment and measurements

| Item | Value |
| --- | --- |
| Host | Intel x86_64 Mac workspace |
| OS | macOS 15.8.1 (24H32), from sw_vers |
| Compiler | Apple Clang 17.0.0 (`clang-1700.0.13.5`) |
| CMake | 4.4.3 |
| Build | Release; native ON and OFF in separate directories |
| SIMD | native AVX2 width 8; portable SSE2 width 4 |
| Volume | 25 cases × 3 processes × 2 builds = 150 checked processes |
| Warm-up / samples / repeats | 20 / 100 / 10 |

Builds and smoke checks finished before measurement. All benchmark processes
ran sequentially, native suite then portable suite. CPU frequency, power mode
and unrelated host activity were not controlled. Process ranges retain the
resulting variation; these timings are not stable latency targets. The OS and
compiler differ from previous Mac reports, so historical absolute timings are
not a before/after comparison. Python platform metadata reports a macOS 10.16
compatibility string on this host; sw_vers above is the authoritative OS value.

The baseline is a reused CpuExecutionContext with pre-bound inputs and weights
packed outside timers. Model import and graph optimization are also outside.
Matrix operands are captured from checked ordinary execution. Direct diagnostics
use those frozen operands, prepacked RHS and preallocated outputs. The captured
matrix chain does not propagate outputs between nodes and excludes ordinary
operators. Dynamic-B direct diagnostics also prepack the captured B, while
actual prepared fallback still packs every run.

Epilogue timing resets ten independent output buffers from the original MatMul
product outside each timed sample and transforms each buffer once. This avoids
repeated bias/ReLU application, but changes cache behavior compared with one
reused output. The largest per-node pool has 1,310,720 payload bytes. Per-workload
order rotates each sample. The empty std::function loop/timer floor is recorded.

Captured operands, extra packed objects and pools remain live during analysis,
so their working set can affect cache conditions of the prepared timer too.
Do not compare these absolute timings with a minimal standalone runner as an
optimization or regression.

Bookkeeping is a control with input checks, node/value lookup, intermediate
clear, output layout/slot preparation and releases, without numerical kernels.
It is not an instrumented execution partition. It also queries every input and
checks transA attributes, whereas the packed production path skips RHS lookup
and uses decoded descriptors. Ordinary operators use isolated
one-node planned registry dispatch, including lookup/slot overhead. The scalar
tail control uses a strided kernel over last N % SIMD_width columns, not the
actual in-kernel scalar loop timer. **Do not sum components or subtract medians
to claim exact graph attribution or an achieved future fusion speedup.**

## Full prepared execution and application copies

Each entry is the median of three process p50 values, microseconds. Input/output
copy timings are separate application costs, excluded from prepared-run latency.
They should guide integration decisions; adding these medians is not a measured
end-to-end application timer.

### native

| Case | Prepared | Bookkeeping control | Captured matrix chain | Bind copy | Output copy |
| --- | ---: | ---: | ---: | ---: | ---: |
| m1_k128_n128_t0 | 11.165 | 0.632 | 10.382 | 4.132 | 3.927 |
| m1_k128_n128_t1 | 11.070 | 0.616 | 10.305 | 4.014 | 3.836 |
| m16_k128_n128_t0 | 94.093 | 0.802 | 95.863 | 51.125 | 50.396 |
| m16_k128_n128_t1 | 88.293 | 0.579 | 83.928 | 41.597 | 43.161 |
| m64_k128_n128_t0 | 406.893 | 0.849 | 401.043 | 223.606 | 221.770 |
| m64_k128_n128_t1 | 416.584 | 0.855 | 397.014 | 213.446 | 214.050 |
| m1_k512_n512_t0 | 98.026 | 0.699 | 100.234 | 12.356 | 11.216 |
| m1_k512_n512_t1 | 102.536 | 0.747 | 103.791 | 13.046 | 12.493 |
| m16_k512_n512_t0 | 1201.377 | 0.915 | 1233.368 | 226.088 | 228.859 |
| m16_k512_n512_t1 | 1035.355 | 0.758 | 1016.092 | 191.464 | 188.831 |
| m64_k512_n512_t0 | 4456.206 | 0.844 | 4409.561 | 793.556 | 745.149 |
| m64_k512_n512_t1 | 3576.843 | 0.766 | 3552.599 | 664.934 | 630.385 |
| m3_k127_n131_t0 | 25.245 | 0.546 | 20.689 | 8.075 | 7.541 |
| m3_k127_n131_t1 | 27.584 | 0.725 | 26.597 | 11.039 | 10.943 |
| m16_k128_n128_bias_none | 45.959 | 0.647 | 42.400 | 43.807 | 42.965 |
| m16_k128_n128_bias_scalar | 76.418 | 0.586 | 76.279 | 43.813 | 49.894 |
| m16_k128_n128_bias_row | 105.476 | 0.797 | 101.692 | 51.343 | 55.714 |
| m16_k128_n128_bias_column | 105.039 | 0.800 | 99.240 | 53.926 | 56.694 |
| m16_k128_n128_bias_full | 83.707 | 0.620 | 76.988 | 40.576 | 41.673 |
| m16_k128_n128_no_relu | 88.126 | 0.626 | 86.951 | 45.413 | 44.613 |
| m16_k128_n128_dynamic | 470.219 | 0.686 | 80.438 | 413.220 | 45.031 |
| m3_k127_n128_aligned_n | 19.176 | 0.545 | 19.000 | 7.937 | 7.823 |
| m3_k128_n128_aligned_kn | 20.801 | 0.586 | 18.211 | 8.192 | 8.009 |
| phase3_mlp_gemm | 12.598 | 1.477 | 8.059 | 0.956 | 0.859 |
| rl_actor_mlp_tanh | 15.538 | 1.925 | 7.975 | 0.965 | 0.837 |

### portable

| Case | Prepared | Bookkeeping control | Captured matrix chain | Bind copy | Output copy |
| --- | ---: | ---: | ---: | ---: | ---: |
| m1_k128_n128_t0 | 15.103 | 0.581 | 14.681 | 3.494 | 3.288 |
| m1_k128_n128_t1 | 15.198 | 0.554 | 14.411 | 3.601 | 3.331 |
| m16_k128_n128_t0 | 183.047 | 0.671 | 182.724 | 43.502 | 42.646 |
| m16_k128_n128_t1 | 219.259 | 0.766 | 202.272 | 47.857 | 48.302 |
| m64_k128_n128_t0 | 1015.100 | 1.063 | 1024.799 | 238.608 | 232.024 |
| m64_k128_n128_t1 | 726.942 | 0.734 | 703.251 | 165.304 | 166.795 |
| m1_k512_n512_t0 | 195.119 | 0.665 | 194.090 | 14.395 | 10.269 |
| m1_k512_n512_t1 | 188.268 | 0.639 | 182.406 | 13.200 | 11.179 |
| m16_k512_n512_t0 | 2414.209 | 0.762 | 2465.021 | 169.747 | 176.468 |
| m16_k512_n512_t1 | 2737.759 | 0.787 | 2671.527 | 184.998 | 192.304 |
| m64_k512_n512_t0 | 8287.622 | 0.701 | 8306.344 | 577.744 | 571.709 |
| m64_k512_n512_t1 | 8777.727 | 0.738 | 8843.648 | 597.823 | 596.596 |
| m3_k127_n131_t0 | 26.015 | 0.380 | 25.691 | 5.807 | 5.909 |
| m3_k127_n131_t1 | 27.011 | 0.413 | 26.628 | 5.987 | 5.968 |
| m16_k128_n128_bias_none | 97.981 | 0.486 | 96.331 | 30.113 | 28.962 |
| m16_k128_n128_bias_scalar | 116.517 | 0.461 | 113.744 | 28.739 | 28.400 |
| m16_k128_n128_bias_row | 120.030 | 0.448 | 119.457 | 28.521 | 28.333 |
| m16_k128_n128_bias_column | 119.748 | 0.438 | 119.191 | 28.430 | 28.352 |
| m16_k128_n128_bias_full | 115.662 | 0.425 | 114.996 | 29.337 | 29.188 |
| m16_k128_n128_no_relu | 124.028 | 0.449 | 123.817 | 29.411 | 29.280 |
| m16_k128_n128_dynamic | 376.558 | 0.479 | 123.217 | 261.790 | 29.172 |
| m3_k127_n128_aligned_n | 26.227 | 0.383 | 25.863 | 6.025 | 6.011 |
| m3_k128_n128_aligned_kn | 26.169 | 0.387 | 25.742 | 6.026 | 5.977 |
| phase3_mlp_gemm | 9.176 | 1.049 | 6.097 | 0.677 | 0.641 |
| rl_actor_mlp_tanh | 11.849 | 1.447 | 6.180 | 0.730 | 0.635 |

## Matrix computation and existing epilogue

Entries are medians of process p50 values, microseconds. Epilogue / complete is
an isolated-time ratio under the different measurement boundaries above; it is
not an exact share of full execution. Complete includes packed MatMul and the
existing affine/ReLU pass over one reused output. Ratios can prioritize work,
but do not predict the realized benefit of a fused kernel.

| Build | Case | Packed MatMul | Epilogue | Direct complete | Epilogue / complete |
| --- | --- | ---: | ---: | ---: | ---: |
| native | m1_k128_n128_t1 | 3.017 | 6.221 | 9.944 | 62.6% |
| native | m16_k128_n128_t1 | 48.492 | 39.868 | 96.981 | 41.1% |
| native | m64_k128_n128_t1 | 201.722 | 165.675 | 374.741 | 44.2% |
| native | m1_k512_n512_t1 | 78.374 | 12.628 | 94.704 | 13.3% |
| native | m16_k512_n512_t1 | 1160.659 | 209.909 | 1411.227 | 14.9% |
| native | m64_k512_n512_t1 | 3059.367 | 528.143 | 3621.659 | 14.6% |
| portable | m1_k128_n128_t1 | 8.346 | 5.631 | 15.477 | 36.4% |
| portable | m16_k128_n128_t1 | 172.512 | 60.118 | 246.843 | 24.4% |
| portable | m64_k128_n128_t1 | 513.453 | 133.955 | 651.402 | 20.6% |
| portable | m1_k512_n512_t1 | 198.937 | 15.006 | 218.311 | 6.9% |
| portable | m16_k512_n512_t1 | 2400.893 | 146.433 | 2684.058 | 5.5% |
| portable | m64_k512_n512_t1 | 7521.995 | 475.964 | 8538.599 | 5.6% |

The native M16/K128/N128 vector-bias case measures 48.492 us packed MatMul,
39.868 us epilogue and 96.981 us direct complete (medians across processes).
The no-bias/ReLU epilogue control is 0.203 us. Keeping vector bias while removing
ReLU still measures 33.678 us epilogue. These separately measured controls support
prioritizing bias/layout work over ReLU arithmetic; workload/run variation means
they do not quantify an exact removable cost.

For native M64/K512/N512, packed MatMul is 3,059.367 us, epilogue 528.143 us
and direct complete 3,621.659 us. Matrix work dominates this direct diagnostic,
so epilogue specialization alone cannot address most of its computation.


## Bias and activation controls

The shapes are M16/K128/N128, transB=1. Default means vector bias + ReLU;
other bias controls retain ReLU, and no_relu retains vector bias. No-bias
execution bypasses TensorIterator. These controls change numerical outputs,
which are independently checked against the appropriate existing operation;
they are diagnostics, not semantically equivalent before/after optimizations.

| Build | Bias / activation | Prepared | MatMul | Epilogue | Direct complete |
| --- | --- | ---: | ---: | ---: | ---: |
| native | m16_k128_n128_t1 | 88.293 | 48.492 | 39.868 | 96.981 |
| native | m16_k128_n128_bias_none | 45.959 | 36.793 | 0.203 | 37.781 |
| native | m16_k128_n128_bias_scalar | 76.418 | 48.524 | 38.255 | 90.226 |
| native | m16_k128_n128_bias_row | 105.476 | 43.806 | 36.642 | 80.899 |
| native | m16_k128_n128_bias_column | 105.039 | 51.447 | 53.986 | 100.854 |
| native | m16_k128_n128_bias_full | 83.707 | 40.945 | 25.915 | 65.997 |
| native | m16_k128_n128_no_relu | 88.126 | 40.887 | 33.678 | 77.802 |
| portable | m16_k128_n128_t1 | 219.259 | 172.512 | 60.118 | 246.843 |
| portable | m16_k128_n128_bias_none | 97.981 | 98.911 | 0.291 | 100.998 |
| portable | m16_k128_n128_bias_scalar | 116.517 | 92.760 | 18.559 | 111.998 |
| portable | m16_k128_n128_bias_row | 120.030 | 92.183 | 26.384 | 119.103 |
| portable | m16_k128_n128_bias_column | 119.748 | 92.060 | 26.351 | 119.039 |
| portable | m16_k128_n128_bias_full | 115.662 | 95.348 | 19.801 | 115.918 |
| portable | m16_k128_n128_no_relu | 124.028 | 95.249 | 27.325 | 122.971 |

Code inspection explains a candidate cost: apply_gemm_epilogue constructs a
TensorIterator and calls operand_offset for every output when bias is present.
That function performs coordinate division/remainder for each iteration
dimension, even when the bias is a scalar or contiguous vector. No-bias ReLU
uses a direct contiguous loop. Bias controls plus this implementation support
investigating specialized common-bias handling. They do not separately measure
index arithmetic, bias memory accesses and arithmetic instructions.

## Tail and aligned-shape controls

N131 has three scalar columns in both current builds. N128 controls are aligned;
their outputs and operation count differ, so ratios cannot isolate scalar tail
cost. Scalar-tail control includes validation and a different RHS access loop.

| Build | Shape | Prepared | MatMul | Scalar-tail control |
| --- | --- | ---: | ---: | ---: |
| native | m3_k127_n131_t1 | 27.584 | 11.216 | 1.695 |
| native | m3_k127_n128_aligned_n | 19.176 | 7.500 | n/a |
| native | m3_k128_n128_aligned_kn | 20.801 | 7.839 | n/a |
| portable | m3_k127_n131_t1 | 27.011 | 18.458 | 1.262 |
| portable | m3_k127_n128_aligned_n | 26.227 | 18.195 | n/a |
| portable | m3_k128_n128_aligned_kn | 26.169 | 17.970 | n/a |

## Real model components

Each fixture uses the full explicit optimization pipeline. Matrix rows refer to
captured operands at that node; ordinary rows include isolated registry dispatch.
Use these values alongside full prepared model latency, not as additive slices.

| Build | Model / node | Shape | MatMul | Epilogue | Direct complete / registry dispatch |
| --- | --- | --- | ---: | ---: | ---: |
| native | phase3_mlp_gemm / __onnx_node_1 | 1×2×3 | 0.055 | 3.990 | 4.025 |
| native | phase3_mlp_gemm / __onnx_node_2 | 1×3×2 | 0.051 | 3.953 | 4.001 |
| native | phase3_mlp_gemm / __onnx_node_3 | ordinary | n/a | n/a | 2.993 |
| native | rl_actor_mlp_tanh / __onnx_node_0 | 1×4×5 | 0.066 | 3.903 | 3.969 |
| native | rl_actor_mlp_tanh / __onnx_node_2 | 1×5×2 | 0.057 | 3.782 | 3.889 |
| native | rl_actor_mlp_tanh / __onnx_node_1 | ordinary | n/a | n/a | 3.155 |
| native | rl_actor_mlp_tanh / __onnx_node_3 | ordinary | n/a | n/a | 3.004 |
| portable | phase3_mlp_gemm / __onnx_node_1 | 1×2×3 | 0.041 | 3.017 | 3.046 |
| portable | phase3_mlp_gemm / __onnx_node_2 | 1×3×2 | 0.038 | 3.004 | 3.046 |
| portable | phase3_mlp_gemm / __onnx_node_3 | ordinary | n/a | n/a | 2.268 |
| portable | rl_actor_mlp_tanh / __onnx_node_0 | 1×4×5 | 0.047 | 3.068 | 3.109 |
| portable | rl_actor_mlp_tanh / __onnx_node_2 | 1×5×2 | 0.041 | 3.033 | 3.073 |
| portable | rl_actor_mlp_tanh / __onnx_node_1 | ordinary | n/a | n/a | 2.374 |
| portable | rl_actor_mlp_tanh / __onnx_node_3 | ordinary | n/a | n/a | 2.333 |

For the native MLP fixture, each tiny matrix MatMul diagnostic is near the
loop/timer floor (0.051–0.055 us), while each epilogue is about 3.95–3.99 us.
Those matrix timings should not be used to estimate tiny-kernel GFLOP/s. The
full MLP prepared latency is 12.598 us and its bookkeeping control 1.477 us.
The actor's ordinary Tanh dispatches are also visible (about 3 us each). These
fixtures are dominated by small-operation metadata/dispatch and epilogue costs,
so a large-matrix throughput kernel is insufficient for their optimization.


## Verification and next implementation boundary

All 150 processes completed numerical checks before/after timing at
`1e-5 + 1e-5 * abs(reference)`: prepared versus ordinary outputs, direct matrix
outputs, once-transformed epilogue buffers, scalar tail results and isolated
ordinary operators. Native/portable smoke checks covered both ONNX fixtures and
the rectangular tail case before the suite. Static cases report zero runtime
packs; the dynamic control reports one per completed session run. No production
library, public API or graph semantics changed, so this analysis did not require
a new full runtime correctness gate. Phase 4.4's existing 40/40 portable/native
ROG tests remain the runtime acceptance baseline.

The proposed implementation order is:

1. Classify common bias layouts and add dedicated scalar/vector/row/full affine
   loops, retaining the current generic broadcast implementation for other
   layouts. Validate dynamic bias strides at execution time; shape alone cannot
   establish a contiguous fast path. Preserve alpha/beta and special values.
2. Compare this specialization separately against the Phase 4.4 epilogue, then
   integrate eligible epilogues into the final K-block output store. Apply
   affine/ReLU only after all K contributions. Keep graph/plan/packing unchanged.
3. Evaluate multi-row micro-kernels as a separate slice for compute-heavy shapes.
   Revisit dispatch and input/output-copy APIs for tiny-model/application costs
   rather than assuming an arithmetic kernel can remove them. Tensor's copy
   constructor currently computes a logical storage offset and copies one
   element at a time even for contiguous input. A separate contiguous bulk-copy
   optimization could preserve deep-copy ownership while reducing these costs;
   validate it with actual bind-run-extract application timing before acceptance.

These are implementation hypotheses; no optimized kernel speedup is claimed.
ROG Phase 4.5 measurements are not available yet. Reproduce the same suite on
ROG before finalizing its ISA-specific support policy. Phase 4.4 is already
closed; this pending reproduction belongs to the new Phase 4.5 analysis.

See the [Phase 4.5 plan](../plans/phase4.5.md),
[benchmark commands](../../benchmarks/README.md#phase-45-prepared-cpu-performance-analysis)
and [per-process evidence](mac-cpu-phase4.5-details.md). Raw JSON and executable/
source fingerprints remain in ignored benchmark-results/phase4.5/mac-native
and mac-portable. Runner metadata uses the measured source baseline plus explicit
hashes of the new analysis files, since those files were uncommitted during runs.
