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

## Inspecting compute groups

After each `NativeEngine.step()`, call `last_batch_stats()` (worker command
`BATCH_STATS`). Each group records phase, distinct request count and packed
token rows. A prepared batch counts once when preparation succeeds; a serial
append counts when it returns. On errors, computed rows can exceed committed
rows. A fresh step resets the counters, including an idle step.

`budget_splits` is 0 or 1 per step: 1 when the budget shortened a chunk below
its normal cap or excluded a later scheduled request. `prefix_restore_splits`
counts prefill group lookaheads stopped by an unrestored request. It measures
ordering boundaries rather than wall time spent waiting.

```sh
python3 miniprefix/benchmarks/inspect_batch_groups.py build/miniprefix/miniprefix_worker weights/model.mpf --chunk 4 --budget 13
```

Separate untimed replays checked that prefill/decode group rows sum to their
respective committed counters on every successful step.

| Chunk / budget | Steps | Prefill groups | Multi-request prefill groups | Budget-limited steps | Restore boundaries |
|---|---:|---:|---:|---:|---:|
| 4 / 13 | 10 | 11 | 2 | 0 | 5 |
| 2 / 13 | 14 | 18 | 5 | 0 | 6 |
| 4 / 4 | 16 | 16 | 0 | 9 | 5 |

With budget 4 every prefill group had one request, so the small throughput
difference in that timed pair cannot establish a cross-request compute gain.
Two-token chunks produced five multi-request groups but were slower in the
timed sweep.
