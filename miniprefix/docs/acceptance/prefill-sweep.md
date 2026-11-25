# CPU prefill workload sweep

Run from the repository root with the [fixed FP32 model and Release worker](cpu-decode-batch.md).

```sh
python3 miniprefix/tools/verify_batch_pressure.py build/miniprefix/miniprefix_worker weights/model.mpf
python3 miniprefix/benchmarks/prefill_sweep.py build/miniprefix/miniprefix_worker weights/model.mpf --pairs 2
```

The pressure regression compares serial CPU against batched decode and prefill.
Seeds 17, 29 and 41 ran 9, 12 and 10 steps, reaching 11, 11 and 10 of 12
KV pages. Each exercised cancellation, an impossible admission, prefix eviction
and cleanup. On failure, replay the generated JSON using the same command plus
`--replay batch-pressure-failure.json`.

The timed sweep uses the eight ordinary arrivals from seed 17, with no
cancellation or oversized request. Both modes use batched decode, cache 16,
pool 64, and identical step-indexed arrivals. Two pairs alternate order,
warm each worker and compare every step's events. Each case changes one
factor from the base. The short-request P95 is nearest-rank across ten gaps
from request 0. RSS is maximum worker VmHWM including warmup.

| Long prompts / 8 | Chunk | Budget | Output tok/s | Short P95 gap ms | Peak RSS MiB |
|---:|---:|---:|---:|---:|---:|
| 3 | 4 | 13 | 8.94 / 9.70 | 577 / 532 | 1042.54 / 1042.73 |
| 5 | 4 | 13 | 7.15 / 8.62 | 719 / 580 | 1042.37 / 1042.38 |
| 3 | 2 | 13 | 7.36 / 6.96 | 393 / 451 | 1042.34 / 1042.62 |
| 3 | 4 | 4 | 8.51 / 8.78 | 394 / 311 | 1042.61 / 1042.52 |

Values are serial / batched prefill. The two-token chunk lost throughput and
increased tail latency on this trace. Keep CPU prefill opt-in. Timings include
IPC and scheduler effects; these small samples do not establish a general
chunk-size threshold. [Raw paired samples](prefill-sweep.json).
