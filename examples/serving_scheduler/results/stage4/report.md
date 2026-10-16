# Stage 4 actual CPU engineering measurements

2026-10-15. Random-weight tiny decoder, FP32, D=16/V=32, single head/layer.
Three independent C++ worker processes; 30 fixed requests per configuration, 120 outputs.
Rows are medians of the three within-run metrics; P95 is not a pooled/global percentile.

| Configuration | TTFT p50 ms | TTFT p95 ms | Mean request TPOT ms | Request p95 ms | Output tokens/s |
|---|---:|---:|---:|---:|---:|
| static_full | 8.058 | 14.136 | 0.282 | 14.701 | 4896.130 |
| continuous_full | 4.708 | 9.303 | 0.378 | 10.518 | 6010.005 |
| static_chunked | 4.909 | 7.824 | 0.214 | 8.358 | 6696.641 |
| continuous_chunked | 4.687 | 10.223 | 0.334 | 10.952 | 5882.242 |

These are actual monotonic-clock observations including controller/IPC/scheduling.
They are not modeled costs, pretrained LLM throughput, a hardware-batching speedup
or a statistically validated SLA. Every step computes its requests serially in C++.
Warmup/model preparation/reference checking are outside the measurement window.
Chunking may add IPC/scheduling cost and worsen metrics for these tiny workloads;
no speedup target or automatic backend-selection rule is claimed.

All 12 reports pass full-prefix greedy comparisons, input-token/forward accounting,
logical capacity bounds and native KV cleanup. Request timestamps, rejection lists,
traces, actual KV counters, model metadata and source/binary hashes are retained
in cpu-benchmark.json. Native KV payload peaks range from 5120 to 5632 bytes.

See ../../STAGE4.md for exact timing and scope boundaries.
