"""Paired CPU decode measurements through the worker IPC client (Linux)."""
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


def run(worker, model, concurrency, batched):
    with NativeEngine(worker, model, context=64, batch_size=concurrency, token_budget=32,
                      chunk_size=8, block_tokens=4, kv_blocks=128, prefix_cache_blocks=0,
                      batch_decode=batched) as engine:
        # Load packed weights before measuring. No prefix state survives warmup.
        engine.submit("warm", [1, 2], 2, None)
        list(engine.events())
        streams = [[] for _ in range(concurrency)]
        times = [[] for _ in streams]
        start = time.perf_counter()
        for i in range(concurrency):
            engine.submit(str(i), [1, 500+i, 1000+i, 200+i], 6, None)
        for event in engine.events():
            i = int(event["request_id"])
            streams[i].append((event["token_id"], event["finish_reason"]))
            times[i].append(event["received_ns"] / 1e9)
        elapsed = time.perf_counter() - start
        if any(len(row) != 6 or row[-1][1] != "length" for row in streams):
            raise RuntimeError("incomplete generation")
        status = Path(f"/proc/{engine.process.pid}/status").read_text().splitlines()
        peak = int(next(line.split()[1] for line in status if line.startswith("VmHWM:")))
        return streams, dict(tokens_s=6*concurrency/elapsed, peak_mib=peak/1024,
                             ttft_ms=[(t[0]-start)*1000 for t in times],
                             tpot_ms=[(t[-1]-t[0])*1000/5 for t in times])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("worker")
    parser.add_argument("model", type=Path)
    parser.add_argument("--pairs", type=int, default=2)
    args = parser.parse_args()
    digest = hashlib.sha256(args.model.read_bytes()).hexdigest()
    if digest != "fe826f487690854ac806197b60edffc2c76c6c446eede6b188e47763deae7aef":
        raise ValueError("expected pinned SmolLM2-135M-Instruct FP32 export")
    report = {"sha256": digest, "pairs": args.pairs, "results": []}
    for n in (1, 2, 4, 8):
        modes = {False: [], True: []}
        for pair in range(args.pairs):
            expected = None
            for mode in (bool(pair % 2), not bool(pair % 2)):
                outputs, sample = run(args.worker, args.model, n, mode)
                if expected is not None and outputs != expected:
                    raise RuntimeError("greedy or terminal mismatch")
                expected = outputs
                modes[mode].append(sample)
        for mode, samples in modes.items():
            ttft = sorted(x for sample in samples for x in sample["ttft_ms"])
            tpot = [x for sample in samples for x in sample["tpot_ms"]]
            report["results"].append(dict(concurrency=n, batched=mode, samples=samples,
                median_tokens_s=statistics.median(s["tokens_s"] for s in samples),
                p95_ttft_ms=ttft[math.ceil(.95*len(ttft))-1], mean_tpot_ms=statistics.mean(tpot),
                max_peak_mib=max(s["peak_mib"] for s in samples)))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
