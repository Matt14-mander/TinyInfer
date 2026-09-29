# Mac CPU — Phase 4.4 prepared execution

- Status: Local correctness and Mac performance gates passed; ROG gate pending
- Measurement date: 2026-09-29
- Source: `d3849e898ac9e9074f5aa15eb93b731988eaed48` plus the Phase 4.4 implementation in this working tree
- Baseline: Phase 4.3 optimized graph, Executor + MemoryPlan + combined Gemm epilogue

## Environment and method

| Item | Value |
| --- | --- |
| Host | Intel Mac workspace used for previous Mac reports; x86_64 |
| OS | macOS 15.4.1 (24E263), Darwin 24.4.0 |
| Compiler | Apple Clang 17.0.0 (`clang-1700.0.13.3`) |
| Build | Release, native architecture enabled |
| Warm-up / samples / repeats | 20 / 100 / 10 |
| Independent processes | 3 per case, 24 cases, 72 total |
| Preparation/component/first-run samples | 20 single invocations per process |

Power mode and current CPU clock were not captured. Previous host identification
is documented in the [Phase 4.2 report](mac-cpu-phase4.2.md). Builds and tests
finished before measurement; benchmark processes ran sequentially. Absolute
latencies reflect this environment and should not become cross-host targets.

Both variants execute the same graph with the same input values, planned buffer
policy, CPU kernel and separate reused contexts. Each sample alternates variant
order. Preparation, model import/optimization, input binding and copied output
extraction are outside inference timers. The synthetic graph has one
FusedGemmActivation node, alpha=beta=1 and static bias. ONNX fixtures run the
full optional optimization pipeline; the actor retains ordinary Tanh dispatch.

Every process checks numerical outputs before and after measurement using
`1e-5 + 1e-5 * abs(reference)`. New correctness tests also use independent
fixture references. First inference means execution in a freshly created and
bound context after plan preparation, with warmed process caches. Preparation
and context timers include temporary destruction; they measure full lifecycle
cost. Component timers are separate diagnostics and are never summed to predict
warm latency.

## Warm results

Latency columns below are the **median of the three process p50 values**, in
microseconds. The ratio interval is the minimum/maximum of the three process
medians of same-sample baseline/prepared ratios. It is not a confidence interval.
`t0`/`t1` denote transB=0/1. Dynamic controls use transB=1 unless named t0.
Preparation is likewise the median of three process p50 values.

| Case | Baseline p50 us | Prepared p50 us | Paired speedup range | Prep p50 us | Packed bytes |
| --- | ---: | ---: | --- | ---: | ---: |
| m1_k128_n128_t0 | 15.356 | 5.319 | 2.882–2.959x | 220.803 | 65536 |
| m1_k128_n128_t1 | 221.369 | 5.565 | 39.620–40.959x | 234.260 | 65536 |
| m1_k128_n128_dynamic | 218.891 | 211.817 | 1.028–1.034x | 27.570 | 0 |
| m16_k128_n128_t0 | 55.728 | 45.077 | 1.231–1.236x | 223.995 | 65536 |
| m16_k128_n128_t1 | 259.291 | 45.117 | 5.689–5.751x | 241.745 | 65536 |
| m16_k128_n128_dynamic | 259.077 | 251.294 | 1.031–1.033x | 27.209 | 0 |
| m64_k128_n128_t0 | 184.471 | 172.663 | 1.062–1.066x | 225.110 | 65536 |
| m64_k128_n128_t1 | 385.311 | 172.532 | 2.199–2.235x | 234.762 | 65536 |
| m64_k128_n128_dynamic | 387.639 | 380.183 | 1.016–1.026x | 27.321 | 0 |
| m1_k512_n512_t0 | 115.673 | 45.653 | 2.493–2.589x | 3258.799 | 1048576 |
| m1_k512_n512_t1 | 3641.756 | 48.942 | 72.815–76.001x | 3678.083 | 1048576 |
| m1_k512_n512_dynamic | 3637.580 | 3625.826 | 1.003–1.004x | 30.855 | 0 |
| m16_k512_n512_t0 | 590.658 | 507.312 | 1.152–1.161x | 3256.706 | 1048576 |
| m16_k512_n512_t1 | 4121.495 | 505.467 | 8.107–8.145x | 3613.356 | 1048576 |
| m16_k512_n512_dynamic | 4070.135 | 4071.584 | 0.995–1.013x | 31.201 | 0 |
| m64_k512_n512_t0 | 2158.868 | 2065.514 | 1.042–1.052x | 3225.990 | 1048576 |
| m64_k512_n512_t1 | 5655.069 | 2059.916 | 2.743–2.775x | 3636.476 | 1048576 |
| m64_k512_n512_dynamic | 5672.860 | 5662.175 | 1.001–1.002x | 31.216 | 0 |
| m3_k127_n131_t0 | 21.835 | 11.455 | 1.893–1.941x | 226.536 | 66548 |
| m3_k127_n131_t1 | 221.476 | 11.829 | 18.455–18.855x | 233.405 | 66548 |
| m3_k127_n131_dynamic | 230.877 | 222.974 | 1.030–1.089x | 27.811 | 0 |
| m16_k128_n128_dynamic_t0 | 55.184 | 47.955 | 1.140–1.149x | 27.157 | 0 |
| phase3_mlp_gemm | 23.457 | 7.325 | 3.226–3.308x | 52.026 | 48 |
| rl_actor_mlp_tanh | 27.765 | 9.194 | 3.018–3.042x | 59.734 | 120 |

All constant-weight cases show positive paired gains in each process. Gains
range from about 4% for the compute-heavy M64/K512/transB=0 case to much larger
values when repeated RHS transpose copying dominates the existing path. The
Gemm MLP improves 3.226–3.308x and the actor improves 3.018–3.042x in this run.
These ratios measure preparation reuse against the current Phase 4.3 runtime;
they do not indicate a comparable improvement in the arithmetic micro-kernel.

Dynamic controls show no repeatable regression over the 5% review threshold.
The M16/K512 control ranges 0.995–1.013x and is effectively unchanged. Smaller
controls benefit from cached ordering; their matrix packing still happens each
run. There is no universal speedup guarantee.

## Preparation, memory, and first execution

Entries are medians across the three process p50 values, in microseconds.
Amortization ranges charge **all mean preparation cost** against positive mean
warm savings within each process, rounded up. They exclude differences in
context creation and output extraction. `n/a` means at least one process has no
positive mean savings; these estimates are not measured startup latency.

| Case | Prep | Context baseline / prepared | First baseline / prepared | Activation bytes | Estimated calls |
| --- | ---: | ---: | ---: | ---: | --- |
| m1_k128_n128_t0 | 220.803 | 201.210 / 201.790 | 18.453 / 6.241 | 512 | 22–23 |
| m1_k128_n128_t1 | 234.260 | 201.555 / 200.759 | 221.289 / 6.824 | 512 | 2–2 |
| m1_k128_n128_dynamic | 27.570 | 9.563 / 9.499 | 217.569 / 210.209 | 512 | 4–6 |
| m16_k128_n128_t0 | 223.995 | 199.335 / 200.052 | 58.564 / 46.263 | 8192 | 22–23 |
| m16_k128_n128_t1 | 241.745 | 202.138 / 201.224 | 264.932 / 46.909 | 8192 | 2–2 |
| m16_k128_n128_dynamic | 27.209 | 9.582 / 9.457 | 257.530 / 250.090 | 8192 | 4–5 |
| m64_k128_n128_t0 | 225.110 | 200.240 / 199.835 | 186.164 / 173.994 | 32768 | 17–21 |
| m64_k128_n128_t1 | 234.762 | 199.290 / 199.361 | 384.572 / 172.833 | 32768 | 2–2 |
| m64_k128_n128_dynamic | 27.321 | 9.305 / 9.204 | 386.452 / 377.354 | 32768 | 4–4 |
| m1_k512_n512_t0 | 3258.799 | 3094.545 / 3092.581 | 182.756 / 104.457 | 2048 | 43–48 |
| m1_k512_n512_t1 | 3678.083 | 3108.369 / 3083.208 | 3758.610 / 114.007 | 2048 | 2–2 |
| m1_k512_n512_dynamic | 30.855 | 12.733 / 12.613 | 3720.372 / 3631.089 | 2048 | 2–4 |
| m16_k512_n512_t0 | 3256.706 | 3125.832 / 3121.316 | 656.519 / 560.018 | 32768 | 31–42 |
| m16_k512_n512_t1 | 3613.356 | 3076.173 / 3067.240 | 4206.016 / 555.516 | 32768 | 2–2 |
| m16_k512_n512_dynamic | 31.201 | 12.954 / 12.848 | 4150.118 / 4136.365 | 32768 | n/a |
| m64_k512_n512_t0 | 3225.990 | 3117.852 / 3083.633 | 2213.968 / 2108.602 | 131072 | 29–41 |
| m64_k512_n512_t1 | 3636.476 | 3116.311 / 3067.615 | 5775.060 / 2136.740 | 131072 | 1–2 |
| m64_k512_n512_dynamic | 31.216 | 13.742 / 13.539 | 5772.042 / 5716.052 | 131072 | 1–8 |
| m3_k127_n131_t0 | 226.536 | 202.161 / 237.916 | 25.064 / 12.610 | 1572 | 22–23 |
| m3_k127_n131_t1 | 233.405 | 205.704 / 240.576 | 226.447 / 14.437 | 1572 | 2–2 |
| m3_k127_n131_dynamic | 27.811 | 9.575 / 9.886 | 229.132 / 215.646 | 1572 | 2–7 |
| m16_k128_n128_dynamic_t0 | 27.157 | 9.565 / 9.410 | 57.602 / 49.887 | 8192 | 4–5 |
| phase3_mlp_gemm | 52.026 | 17.718 / 17.802 | 26.631 / 8.143 | 24 | 4–4 |
| rl_actor_mlp_tanh | 59.734 | 19.928 / 20.859 | 30.046 / 9.954 | 52 | 3–5 |

The static synthetic cases create one packed object; each fixture creates two.
All static cases report zero runtime packs for 1,021 completed session runs.
Dynamic controls create zero preparation packs and report 1,021 fallback matrix
packs. Same-weight sharing/different-transpose separation are checked separately
in the correctness suite.

Packed bytes in the warm table count float payload, not total process memory.
The snapshot still retains original constants and each context copies them.
Activation bytes include only MemoryPlan capacity. Inputs, output copies,
metadata, vector capacity and fallback temporaries are additional. Context
creation retains the old constant-copy cost; optimizing that would require a
separate ownership change.

## Component observations

The following are medians across process component p50 values (microseconds).
RHS packing uses precomputed effective operands outside the component timer;
transpose-copy includes the existing Tensor deep copy plus metadata transpose.

| Case | Topology | RHS pack | RHS transpose copy |
| --- | ---: | ---: | ---: |
| m16_k128_n128_t0 | 6.959 | 3.074 | 0.043 |
| m16_k128_n128_t1 | 6.962 | 13.261 | 189.876 |
| m1_k512_n512_t1 | 6.995 | 482.503 | 3075.696 |
| m64_k512_n512_t0 | 6.969 | 61.958 | 0.043 |
| phase3_mlp_gemm | 14.070 | 0.209 | 2.027 |
| rl_actor_mlp_tanh | 16.116 | 0.432 | 2.539 |

These observations support moving repeated sorting/packing/transpose preparation
out of warm execution. The particularly large transB=1 ratios are explained by
the existing deep-copy cost on each Gemm invocation. Without transB copying,
packing/order savings have a much smaller share at large M. Preparation also
copies the model once, so the cold preparation costs remain visible.

## Per-process latency evidence

Each latency cell is **p50 / p95 / mean**, microseconds per invocation. The
last column is the median same-sample ratio. These values preserve process
variation behind the aggregate table above.

| Case | Run | Baseline p50 / p95 / mean | Prepared p50 / p95 / mean | Paired ratio |
| --- | ---: | ---: | ---: | ---: |
| m1_k128_n128_t0 | 1 | 15.352 / 18.864 / 15.959 | 5.339 / 6.069 / 5.598 | 2.882x |
| m1_k128_n128_t0 | 2 | 15.356 / 17.321 / 15.630 | 5.170 / 6.823 / 5.344 | 2.959x |
| m1_k128_n128_t0 | 3 | 15.447 / 19.754 / 16.105 | 5.319 / 5.750 / 5.394 | 2.903x |
| m1_k128_n128_t1 | 1 | 219.158 / 241.308 / 222.194 | 5.274 / 6.769 / 5.580 | 40.959x |
| m1_k128_n128_t1 | 2 | 221.954 / 254.576 / 226.412 | 5.766 / 7.674 / 5.898 | 39.620x |
| m1_k128_n128_t1 | 3 | 221.369 / 238.906 / 223.567 | 5.565 / 6.409 / 5.778 | 40.406x |
| m1_k128_n128_dynamic | 1 | 218.891 / 240.067 / 221.548 | 211.817 / 226.263 / 213.476 | 1.033x |
| m1_k128_n128_dynamic | 2 | 220.821 / 250.638 / 224.833 | 212.735 / 276.985 / 219.935 | 1.028x |
| m1_k128_n128_dynamic | 3 | 217.775 / 233.097 / 219.896 | 210.802 / 226.691 / 212.282 | 1.034x |
| m16_k128_n128_t0 | 1 | 56.072 / 64.787 / 57.954 | 45.305 / 53.810 / 46.780 | 1.236x |
| m16_k128_n128_t0 | 2 | 55.728 / 64.780 / 56.663 | 45.077 / 55.050 / 46.149 | 1.233x |
| m16_k128_n128_t0 | 3 | 54.758 / 65.754 / 57.087 | 44.243 / 53.930 / 45.809 | 1.231x |
| m16_k128_n128_t1 | 1 | 259.170 / 278.920 / 262.168 | 44.627 / 52.388 / 46.060 | 5.738x |
| m16_k128_n128_t1 | 2 | 260.814 / 290.213 / 263.588 | 45.117 / 53.868 / 46.031 | 5.751x |
| m16_k128_n128_t1 | 3 | 259.291 / 291.152 / 262.147 | 45.579 / 55.234 / 47.107 | 5.689x |
| m16_k128_n128_dynamic | 1 | 258.525 / 277.457 / 260.333 | 251.294 / 273.466 / 254.262 | 1.031x |
| m16_k128_n128_dynamic | 2 | 259.846 / 282.844 / 262.244 | 251.272 / 266.755 / 253.174 | 1.032x |
| m16_k128_n128_dynamic | 3 | 259.077 / 280.013 / 262.184 | 251.985 / 272.819 / 254.306 | 1.033x |
| m64_k128_n128_t0 | 1 | 184.689 / 213.352 / 188.634 | 172.663 / 188.907 / 174.862 | 1.066x |
| m64_k128_n128_t0 | 2 | 184.471 / 205.910 / 187.876 | 172.886 / 192.157 / 175.579 | 1.065x |
| m64_k128_n128_t0 | 3 | 182.937 / 201.670 / 184.946 | 172.442 / 190.150 / 173.541 | 1.062x |
| m64_k128_n128_t1 | 1 | 385.174 / 409.578 / 388.522 | 172.532 / 186.454 / 174.028 | 2.235x |
| m64_k128_n128_t1 | 2 | 387.410 / 443.564 / 395.244 | 174.191 / 210.364 / 179.278 | 2.199x |
| m64_k128_n128_t1 | 3 | 385.311 / 407.744 / 387.368 | 172.361 / 185.803 / 174.173 | 2.230x |
| m64_k128_n128_dynamic | 1 | 386.411 / 428.196 / 390.992 | 380.183 / 405.325 / 383.531 | 1.016x |
| m64_k128_n128_dynamic | 2 | 388.503 / 427.177 / 394.026 | 378.932 / 421.398 / 386.536 | 1.026x |
| m64_k128_n128_dynamic | 3 | 387.639 / 407.876 / 390.281 | 380.546 / 396.575 / 382.822 | 1.021x |
| m1_k512_n512_t0 | 1 | 115.673 / 133.103 / 118.021 | 45.653 / 56.451 / 47.100 | 2.493x |
| m1_k512_n512_t0 | 2 | 115.446 / 126.036 / 116.082 | 44.748 / 50.703 / 44.742 | 2.589x |
| m1_k512_n512_t0 | 3 | 118.017 / 149.857 / 123.165 | 45.854 / 57.733 / 47.312 | 2.559x |
| m1_k512_n512_t1 | 1 | 3641.756 / 3786.780 / 3669.897 | 48.185 / 61.008 / 48.800 | 76.001x |
| m1_k512_n512_t1 | 2 | 3637.601 / 3874.823 / 3664.634 | 49.510 / 59.234 / 49.085 | 72.815x |
| m1_k512_n512_t1 | 3 | 3652.856 / 3884.963 / 3677.650 | 48.942 / 58.713 / 48.873 | 75.534x |
| m1_k512_n512_dynamic | 1 | 3637.580 / 3777.355 / 3654.179 | 3631.339 / 3770.124 / 3642.779 | 1.004x |
| m1_k512_n512_dynamic | 2 | 3643.573 / 3814.256 / 3658.827 | 3617.517 / 3728.375 / 3634.856 | 1.003x |
| m1_k512_n512_dynamic | 3 | 3633.226 / 3776.096 / 3646.462 | 3625.826 / 3745.631 / 3634.728 | 1.003x |
| m16_k512_n512_t0 | 1 | 590.658 / 689.441 / 608.123 | 507.312 / 583.657 / 519.679 | 1.161x |
| m16_k512_n512_t0 | 2 | 608.688 / 1311.153 / 756.322 | 525.321 / 1421.613 / 651.535 | 1.152x |
| m16_k512_n512_t0 | 3 | 586.106 / 687.633 / 599.021 | 506.479 / 560.686 / 516.490 | 1.157x |
| m16_k512_n512_t1 | 1 | 4122.262 / 4267.264 / 4136.221 | 505.803 / 553.600 / 522.905 | 8.145x |
| m16_k512_n512_t1 | 2 | 4121.495 / 4250.152 / 4130.966 | 505.467 / 548.812 / 511.198 | 8.138x |
| m16_k512_n512_t1 | 3 | 4097.148 / 4410.311 / 4136.462 | 503.085 / 552.084 / 509.809 | 8.107x |
| m16_k512_n512_dynamic | 1 | 4048.489 / 4228.757 / 4056.448 | 4070.780 / 4263.483 / 4079.089 | 0.995x |
| m16_k512_n512_dynamic | 2 | 4117.122 / 4337.361 / 4141.759 | 4071.584 / 4222.126 / 4085.270 | 1.013x |
| m16_k512_n512_dynamic | 3 | 4070.135 / 4500.762 / 4124.068 | 4099.733 / 4419.126 / 4143.517 | 0.997x |
| m64_k512_n512_t0 | 1 | 2175.886 / 2761.470 / 2260.460 | 2075.708 / 2561.807 / 2134.168 | 1.052x |
| m64_k512_n512_t0 | 2 | 2158.868 / 2259.136 / 2166.197 | 2057.020 / 2131.201 / 2060.666 | 1.051x |
| m64_k512_n512_t0 | 3 | 2154.655 / 2278.263 / 2170.125 | 2065.514 / 2297.276 / 2090.078 | 1.042x |
| m64_k512_n512_t1 | 1 | 5722.994 / 6068.224 / 5768.687 | 2065.396 / 2320.869 / 2091.409 | 2.775x |
| m64_k512_n512_t1 | 2 | 5655.069 / 5857.203 / 5678.056 | 2059.916 / 2165.325 / 2073.940 | 2.743x |
| m64_k512_n512_t1 | 3 | 5645.156 / 5933.542 / 5680.926 | 2052.815 / 2143.729 / 2060.822 | 2.759x |
| m64_k512_n512_dynamic | 1 | 5653.040 / 6394.784 / 5891.075 | 5618.514 / 7150.207 / 5795.532 | 1.001x |
| m64_k512_n512_dynamic | 2 | 5729.028 / 6520.487 / 5842.686 | 5728.219 / 6580.297 / 5803.822 | 1.002x |
| m64_k512_n512_dynamic | 3 | 5672.860 / 5827.107 / 5679.703 | 5662.175 / 5940.977 / 5675.437 | 1.002x |
| m3_k127_n131_t0 | 1 | 22.363 / 26.125 / 22.738 | 11.479 / 12.019 / 11.637 | 1.941x |
| m3_k127_n131_t0 | 2 | 21.835 / 30.377 / 23.224 | 11.455 / 16.605 / 12.480 | 1.893x |
| m3_k127_n131_t0 | 3 | 21.531 / 25.565 / 22.250 | 11.417 / 13.897 / 11.921 | 1.903x |
| m3_k127_n131_t1 | 1 | 220.523 / 284.480 / 228.906 | 11.528 / 19.058 / 12.480 | 18.855x |
| m3_k127_n131_t1 | 2 | 221.476 / 239.272 / 223.709 | 11.829 / 13.387 / 12.129 | 18.663x |
| m3_k127_n131_t1 | 3 | 223.413 / 268.221 / 234.283 | 12.383 / 20.127 / 13.334 | 18.455x |
| m3_k127_n131_dynamic | 1 | 221.089 / 238.598 / 222.362 | 213.606 / 240.702 / 217.261 | 1.030x |
| m3_k127_n131_dynamic | 2 | 323.543 / 439.930 / 322.868 | 284.800 / 408.104 / 297.297 | 1.089x |
| m3_k127_n131_dynamic | 3 | 230.877 / 374.052 / 249.625 | 222.974 / 345.726 / 240.433 | 1.039x |
| m16_k128_n128_dynamic_t0 | 1 | 55.056 / 61.675 / 56.157 | 47.947 / 52.082 / 49.045 | 1.140x |
| m16_k128_n128_dynamic_t0 | 2 | 55.846 / 77.591 / 58.341 | 48.853 / 58.726 / 49.852 | 1.149x |
| m16_k128_n128_dynamic_t0 | 3 | 55.184 / 61.811 / 56.734 | 47.955 / 58.180 / 49.635 | 1.145x |
| phase3_mlp_gemm | 1 | 23.269 / 24.809 / 23.631 | 7.187 / 7.572 / 7.258 | 3.226x |
| phase3_mlp_gemm | 2 | 23.457 / 28.462 / 24.546 | 7.325 / 8.051 / 7.574 | 3.256x |
| phase3_mlp_gemm | 3 | 24.150 / 27.062 / 24.456 | 7.351 / 8.539 / 7.480 | 3.308x |
| rl_actor_mlp_tanh | 1 | 27.195 / 32.849 / 28.498 | 8.907 / 11.013 / 9.398 | 3.042x |
| rl_actor_mlp_tanh | 2 | 27.914 / 35.042 / 29.039 | 9.194 / 11.264 / 9.489 | 3.018x |
| rl_actor_mlp_tanh | 3 | 27.765 / 29.860 / 28.332 | 9.270 / 9.944 / 9.428 | 3.040x |

## Correctness and acceptance boundary

- Portable Release: 40/40 tests passed.
- Native Release: 40/40 tests passed.
- ASan + UBSan Debug: 3/3 targeted MatMul, Gemm fusion and CPU execution plan tests passed.
- Release tests explicitly keep assertions enabled, so inherited checks remain active.
- New tests cover source independence, context lifetime/moves, separate activations,
  copied constant outputs, packing deduplication, all Gemm transpose and bias forms,
  changed A/B/bias across runs, strides, zero sizes, tails and special values.

Mac gates pass for the documented static FP32 CPU slice. ROG performance and
portable/native correctness execution remain pending; this report does not
close cross-host Phase 4.4 acceptance. No in-register epilogue or new kernel
arithmetic optimization was introduced.

Raw per-process JSON, all component statistics and environment metadata remain
locally in ignored `benchmark-results/phase4.4/mac/`. Reproduce the full suite:

```bash
python3 benchmarks/run_phase44.py \
  --executable build-bench/benchmarks/tinyinfer_cpu_execution_plan_benchmark \
  --output benchmark-results/phase4.4/mac --samples 100 --repeats 10 --runs 3
```

The [benchmark instructions](../../benchmarks/README.md#phase-44-prepared-cpu-benchmark)
include Release configuration, the SDK workaround, and ROG PowerShell commands.
