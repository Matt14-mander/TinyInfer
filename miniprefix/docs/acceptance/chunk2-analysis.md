# CPU chunk=2 investigation

Repeat the seed-17 case from [the sweep](prefill-sweep.md) with the two worker
arguments pointing to the same baseline executable:

```sh
python3 miniprefix/benchmarks/compare_small_chunk.py WORKER WORKER weights/model.mpf --serial-before --pairs 3
python3 miniprefix/benchmarks/inspect_batch_groups.py WORKER weights/model.mpf --chunk 2 --budget 13
```

The previous two pairs gave batched throughputs of 6.41 and 7.51 tok/s versus
7.42 and 7.29 serial. The worst short-request gap (451 ms) came from the first
batched run. The second batched run's worst gap was 359 ms. In a fresh three-pair
replay on the unchanged worker, the median and pooled nearest-rank P95 were
6.84 tok/s and 427 ms serial, 7.58 tok/s and 382 ms batched. The sign changed;
the original two-pair slowdown has no established reproducible cause.
[Fresh samples](chunk2-baseline.json).

The groups do identify a repeatable cost. With chunk=2 the trace runs 14 steps
and 18 prefill groups, including one three-request/five-row projection and
four two-request/four-row projections. The four latter groups all coincide
with two-row decode in steps 2–5 (zero-based), within request 0's five output
intervals. Budget splits are zero. With chunk=4 there are 10 steps and only two
multi-request prefill groups (eight and five rows). The SSE2 packed GEMM inner
loop loads RHS vectors separately for each output row. The four-row case can
reuse each loaded RHS vector for all four tokens while preserving the order of
each dot product. This is a kernel optimization hypothesis, not proof of the
historical latency regression.

## Four-row weight reuse

The non-fused SSE2 packed MatMul now loads eight RHS columns once per K for
four output rows. A fifth row, short column tails and blocks with fewer than
four rows use the existing loop. The AVX2 and ARM paths are unchanged.

For an isolated 4×576 by 576×1536 FP32 projection (SmolLM2 MLP gate/up shape)
on the Release CPU build, eight alternating old/new pairs gave median 1273 µs
before and 269 µs after. Each sample averaged 200 calls after 10 warmups with
an already packed RHS. This isolates the kernel from scheduler and IPC time.

To reproduce the end-to-end comparison, build and save the Release worker at
the diagnosis commit and at this revision, then run:

```sh
python3 miniprefix/benchmarks/compare_small_chunk.py OLD_WORKER NEW_WORKER weights/model.mpf --pairs 4
```

Both workers use batched prefill and decode with identical seed-17 requests,
chunk 2, budget 13, cache 16 and identical event traces. Median output throughput
was 7.49 / 8.05 tok/s and request 0's P95 gap was 420 / 378 ms (old / new).
Peak worker RSS was 1042.62 / 1042.55 MiB. The new worker led in three of four
throughput pairs; P95 uses 20 gaps per worker. These single-host results measure
this optimization on the trace, without attributing the original two-pair
regression to a single cause. [Raw paired samples](chunk2-optimized.json).

CPU CTest passed 56/56, including four/five rows with a transposed input,
multiple K blocks and a column tail. The three pressure traces passed against
serial CPU after the change.
