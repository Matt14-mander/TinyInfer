"""Compare CPU prefill modes with identical mixed arrivals and batched decode."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bindings/python"))
from client import NativeEngine

# (arrival step, request ID, prompt tokens, generation limit)
REQUESTS = [(0, "short", [4, 5], 8), (0, "long-a", [1, 2, 3, 4]*5, 4),
            (0, "long-b", [2, 3, 4]*6, 4), (2, "late", [1, 2, 3, 4]*5+[2], 3)]


def run(worker, model, batched, cache, requests=REQUESTS, chunk=4, budget=13):
    with NativeEngine(worker, model, context=64, batch_size=4, token_budget=budget,
                      chunk_size=chunk, block_tokens=4, kv_blocks=64, prefix_cache_blocks=cache,
                      batch_decode=True, batch_prefill=batched) as engine:
        engine.submit("warm", [31, 32], 2, None)
        list(engine.events())
        engine.clear_prefix_cache()
        times = {key: [] for _, key, _, _ in requests}
        arrivals, trace = {}, []
        start = time.perf_counter_ns()
        while len(trace) <= max(r[0] for r in requests) or engine.pending:
            for step, key, prompt, limit in requests:
                if step == len(trace):
                    arrivals[key] = time.perf_counter_ns()
                    engine.submit(key, prompt, limit, None)
            events = engine.step()
            for event in events:
                times[event["request_id"]].append(event["received_ns"])
            trace.append([(e["request_id"], e["token_id"], e["finish_reason"]) for e in events])
        end = time.perf_counter_ns()
        for _, key, _, limit in requests:
            stream = [e for step in trace for e in step if e[0] == key]
            if len(stream) != limit or stream[-1][2] != "length":
                raise RuntimeError(f"incomplete stream for {key}")
        status = Path(f"/proc/{engine.process.pid}/status").read_text().splitlines()
        peak = int(next(line.split()[1] for line in status if line.startswith("VmHWM:")))
        return trace, dict(tokens_s=sum(map(len,times.values()))*1e9/(end-start),
                           peak_mib=peak/1024,
                           ttft_ms={k:(v[0]-arrivals[k])/1e6 for k,v in times.items()},
                           gaps_ms={k:[(b-a)/1e6 for a,b in zip(v,v[1:])] for k,v in times.items()})


def p95(values):
    return sorted(values)[math.ceil(.95*len(values))-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("worker")
    parser.add_argument("model", type=Path)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--cache", type=int, default=16)
    args = parser.parse_args()
    if args.pairs < 1:
        parser.error("--pairs must be positive")
    digest = hashlib.sha256(args.model.read_bytes()).hexdigest()
    if digest != "fe826f487690854ac806197b60edffc2c76c6c446eede6b188e47763deae7aef":
        raise ValueError("unexpected model digest")
    result = dict(sha256=digest, cache=args.cache, pairs=args.pairs, modes={})
    modes = {False: [], True: []}
    for pair in range(args.pairs):
        expected = None
        for mode in (bool(pair%2), not bool(pair%2)):
            trace, sample = run(args.worker, args.model, mode, args.cache)
            if expected is not None and trace != expected:
                raise RuntimeError("serial/batched step events differ")
            expected = trace
            modes[mode].append(sample)
    for mode,samples in modes.items():
        result["modes"]["batched" if mode else "serial"] = dict(samples=samples,
            median_tokens_s=statistics.median(s["tokens_s"] for s in samples),
            p95_ttft_ms=p95([t for s in samples for t in s["ttft_ms"].values()]),
            p95_decode_gap_ms=p95([t for s in samples for gaps in s["gaps_ms"].values() for t in gaps]),
            short_p95_gap_ms=p95([t for s in samples for t in s["gaps_ms"]["short"]]),
            peak_mib=max(s["peak_mib"] for s in samples))
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
