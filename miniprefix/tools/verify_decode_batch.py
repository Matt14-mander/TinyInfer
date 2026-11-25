"""Compare serial and batched scheduling through NativeEngine (CPU or CUDA)."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bindings/python"))
from client import NativeEngine


def replay(args, batched):
    with NativeEngine(args.worker, args.model, backend=args.backend, context=32,
                      batch_size=3, token_budget=8, chunk_size=3, block_tokens=4,
                      kv_blocks=32, prefix_cache_blocks=8, batch_decode=batched) as engine:
        for key, prompt in (("a", [1, 2, 3]), ("b", [1, 2, 3, 4]), ("c", [7, 8])):
            engine.submit(key, prompt, 4, None)
        trace = []
        while engine.pending:
            events = engine.step()
            if len(trace) == 1:
                events += engine.cancel("b")
            trace.append(([(e["request_id"], e["token_id"], e["finish_reason"]) for e in events],
                          engine.last_step_stats()))
        stats = engine.stats()
        if stats["live_sequences"] or not engine.cost_stats()["cow_pages"]:
            raise RuntimeError("expected COW and complete request cleanup")
        if max(step["decode_tokens"] for _, step in trace) != 3:
            raise RuntimeError("expected a three-row decode")
        return trace


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("worker")
    parser.add_argument("model")
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    serial = replay(args, False)
    if serial != replay(args, True):
        raise RuntimeError("per-step events or token accounting differ")
    print(f"{args.backend}: matched {len(serial)} steps with prefix reuse, COW and cancellation")
