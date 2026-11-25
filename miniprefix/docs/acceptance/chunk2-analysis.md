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
