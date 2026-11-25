"""Compare serial and batched scheduling through NativeEngine (CPU or CUDA)."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bindings/python"))
from client import NativeEngine


def replay(worker, model, backend, batched):
    with NativeEngine(worker, model, backend=backend, context=32,
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
    parser.add_argument("--cpu-worker", help="compare the complete trace to serial CPU")
    parser.add_argument("--mode", choices=("compare", "serial", "batched"), default="compare",
                        help="select one mode to isolate a profiler capture")
    parser.add_argument("--report", type=Path, help="save worker/model hashes and step traces as JSON")
    args = parser.parse_args()

    def digest(path):
        h = hashlib.sha256()
        with open(path, "rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()

    model_sha = digest(args.model)
    expected_model = "fe826f487690854ac806197b60edffc2c76c6c446eede6b188e47763deae7aef"
    if args.cpu_worker and model_sha != expected_model:
        parser.error("the CPU comparison requires the pinned SmolLM2 FP32 export")
    traces = {}
    if args.cpu_worker:
        traces["cpu_reference"] = replay(args.cpu_worker, args.model, "cpu", False)
    for mode in (("serial", "batched") if args.mode == "compare" else (args.mode,)):
        traces[mode] = replay(args.worker, args.model, args.backend, mode == "batched")
    reference = next(iter(traces.values()))
    if any(trace != reference for trace in traces.values()):
        raise RuntimeError("per-step events or token accounting differ")
    if args.report:
        report = dict(backend=args.backend, model_sha256=model_sha,
                      worker_sha256=digest(args.worker), traces=traces)
        if args.cpu_worker:
            report["cpu_worker_sha256"] = digest(args.cpu_worker)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{args.backend}: completed {', '.join(traces)}; {len(reference)} steps per trace")
