"""Replay seeded CPU scheduling and bounded-KV traces through NativeEngine."""
import argparse
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bindings/python"))
from client import NativeEngine


def workload(seed, long_count=3):
    rng = random.Random(seed)
    prefix = [rng.randrange(10, 300) for _ in range(5)]
    jobs = []
    for i in range(8):
        length = rng.randint(12, 19) if 1 <= i <= long_count else rng.randint(2, 5)
        prompt = (prefix + [rng.randrange(300, 600) for _ in range(length)])[:length]
        jobs.append(dict(id=str(i), arrival=0 if i < 3 else rng.randrange(1, 5),
                         prompt=prompt, limit=6 if i == 0 else rng.randint(3, 5)))
    # The final request exceeds the whole stress pool but fits the model context.
    jobs.append(dict(id="oversize", arrival=3, prompt=prefix*10, limit=4))
    return dict(seed=seed, jobs=jobs, cancel_id=str(rng.choice([1, 2])), cancel_step=1,
                config=dict(context=64, batch_size=4, token_budget=11, chunk_size=4,
                            block_tokens=4, kv_blocks=12, prefix_cache_blocks=6))


def replay(worker, model, trace, batched):
    history, terminals = [], {}
    jobs = trace["jobs"]
    with NativeEngine(worker, model, **trace["config"], batch_decode=batched,
                      batch_prefill=batched) as engine:
        for step in range(1 + max(j["arrival"] for j in jobs) + sum(len(j["prompt"])+j["limit"] for j in jobs)):
            for j in jobs:
                if j["arrival"] == step:
                    engine.submit(j["id"], j["prompt"], j["limit"], None)
            events = []
            if step == trace["cancel_step"]:
                events.extend(engine.cancel(trace["cancel_id"]))
            events.extend(engine.step())
            rows = [(e["request_id"], e["token_id"], e["finish_reason"]) for e in events]
            for key, _, reason in rows:
                if key in terminals:
                    raise AssertionError(f"event after terminal at {step}: {key}")
                if reason:
                    terminals[key] = reason
            history.append((rows,engine.last_step_stats()))
            if not engine.pending and step >= max(j["arrival"] for j in jobs):
                break
        expected = {j["id"]: "length" for j in jobs}
        expected.update({trace["cancel_id"]: "cancelled", "oversize": "error"})
        if terminals != expected:
            raise AssertionError(f"terminal mismatch {terminals}")
        stats = engine.stats()
        engine.clear_prefix_cache()
        final = engine.stats()
        if final["live_sequences"] or final["kv_live_blocks"] or final["live_kv_bytes"]:
            raise AssertionError("KV resources survived release and cache clear")
        return history, {k:stats[k] for k in ("forward_tokens","prefix_reused_tokens",
                                               "kv_peak_blocks","prefix_evictions")}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("worker")
    p.add_argument("model")
    p.add_argument("--seeds", nargs="+", type=int, default=[17,29,41])
    p.add_argument("--replay", type=Path)
    p.add_argument("--failure", type=Path, default=Path("batch-pressure-failure.json"))
    args = p.parse_args()
    traces = [json.loads(args.replay.read_text())] if args.replay else map(workload,args.seeds)
    for trace in traces:
        try:
            reference = replay(args.worker,args.model,trace,False)
            candidate = replay(args.worker,args.model,trace,True)
            if reference != candidate:
                raise AssertionError("serial and batched step events, accounting or KV stats differ")
        except Exception:
            args.failure.write_text(json.dumps(trace,indent=2)+"\n")
            raise
        print(trace["seed"],len(reference[0]),reference[1],flush=True)


if __name__ == "__main__":
    main()
