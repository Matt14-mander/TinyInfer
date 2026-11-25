# Mixed CPU chunked prefill and decode

Use the pinned FP32 export and Release CPU worker from [the decode benchmark](cpu-decode-batch.md).
The CPU worker enables prefill batching with `NativeEngine(..., batch_prefill=True)`.
The default is off. It packs chunks from consecutive scheduled requests once
their histories are resident, including chunks with different token counts.
The first request can restore its prefix immediately before packing. The
next unrestored request starts a new group so it sees preceding publications.
Each request attends only to its own committed history and causal chunk rows.
KV transactions still commit in scheduler order.

```sh
python3 miniprefix/benchmarks/mixed_prefill.py build/miniprefix/miniprefix_worker weights/model.mpf --pairs 3 --cache 16
```

Three pairs alternated serial-first and batched-first. Decode batching was
enabled in both modes. A two-token request decoded alongside 20- and 18-token
prompts; a 21-token request arrived before step 3. Chunk size was 4, step
budget 13, with four slots, 64 KV pages and a 16-page prefix cache. Each
sample used a fresh worker and cleared warmup prefixes. The benchmark asserts
identical per-step tokens, request IDs, order and finish reasons for each pair.

| Metric | Serial prefill | Batched prefill |
|---|---:|---:|
| Median output tokens/s | 4.71 | 5.32 |
| P95 arrival-to-first-token, ms | 3008 | 2654 |
| P95 consecutive output gap, ms | 640 | 546 |
| Short request P95 output gap, ms | 674 | 557 |
| Maximum worker peak RSS, MiB | 1042.45 | 1042.62 |

Times include worker IPC. Output gaps measure consecutive tokens of the
same request, including any intervening prefill work. TTFT is measured from
each request's submission, including the later arrival. Nearest-rank P95
uses 12 TTFT, 45 gap and 21 short-request gap observations per mode. With
this single trace and host these are workload-specific tail observations.
[Per-request timing samples](mixed-prefill.json).
