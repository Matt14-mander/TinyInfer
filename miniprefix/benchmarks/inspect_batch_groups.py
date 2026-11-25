"""Untimed per-step compute group inspection for the base CPU sweep."""
import argparse
import json

from mixed_prefill import NativeEngine
from prefill_sweep import requests


def inspect(worker, model, chunk, budget):
    jobs = requests(17, False)
    steps = []
    with NativeEngine(worker, model, context=64, batch_size=4, token_budget=budget,
                      chunk_size=chunk, block_tokens=4, kv_blocks=64,
                      prefix_cache_blocks=16, batch_decode=True, batch_prefill=True) as engine:
        while engine.pending or len(steps) <= max(j[0] for j in jobs):
            for arrival, key, prompt, limit in jobs:
                if arrival == len(steps):
                    engine.submit(key, prompt, limit, None)
            events = engine.step()
            if any(e["finish_reason"] == "error" for e in events):
                raise RuntimeError("unexpected request failure")
            stats = engine.last_batch_stats()
            work = engine.last_step_stats()
            for phase in ("decode", "prefill"):
                rows = sum(g["token_rows"] for g in stats["groups"] if g["phase"] == phase)
                if rows != work[phase+"_tokens"]:
                    raise AssertionError(f"{phase} group rows differ from committed work")
            steps.append(stats)
        engine.step()
        if engine.last_batch_stats() != dict(groups=[], budget_splits=0, prefix_restore_splits=0):
            raise AssertionError("idle step retained previous groups")
    return steps


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("worker")
    p.add_argument("model")
    p.add_argument("--chunk",type=int,default=4)
    p.add_argument("--budget",type=int,default=13)
    args = p.parse_args()
    print(json.dumps(inspect(args.worker,args.model,args.chunk,args.budget),indent=2))
