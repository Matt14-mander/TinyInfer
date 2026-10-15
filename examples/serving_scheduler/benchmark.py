"""Run the stage-one virtual scheduling comparison with Python standard library."""
import argparse
import json
from pathlib import Path
try:
    from .backend import Request, SimulatedBackend
    from .scheduler import simulate
    from .metrics import summarize
except ImportError:
    from backend import Request, SimulatedBackend
    from scheduler import simulate
    from metrics import summarize


def workloads(name):
    if name == "mixed":
        return [Request("long", 0, 8, 12)] + [Request("short%d" % i, 0, 4, 2) for i in range(7)]
    if name == "staggered":
        return [Request("r%d" % i, i * 1.5, 4 + i % 3, 2 + i % 5) for i in range(10)]
    if name == "long_prefill":
        return [Request("decode", 0, 2, 12), Request("large_prompt", 2, 200, 2), Request("short", 3, 2, 2)]
    raise ValueError("unknown workload")


def compare(requests, batch_size=4, backend=None, trace=False):
    requests = list(requests)
    backend = backend or SimulatedBackend()
    runs = [simulate(requests, backend, mode, batch_size) for mode in ("static", "continuous")]
    a, b = [{r.request.request_id: r.tokens for r in run.results} for run in runs]
    if a != b:
        raise RuntimeError("batching changed request output sequences")
    reports = []
    for run in runs:
        report = summarize(run)
        if trace:
            report["trace"] = run.trace
        reports.append(report)
    return {"stage": 1, "backend": "deterministic_simulation", "real_backend": False,
            "same_output_sequences": True,
            "cost_model": {"overhead_ms": backend.overhead_ms,
                           "prefill_ms_per_token": backend.prefill_ms_per_token,
                           "decode_ms_per_request": backend.decode_ms_per_request},
            "reports": reports}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=["mixed", "staggered", "long_prefill"], default="mixed")
    parser.add_argument("--requests", type=Path, help="JSON list of request_id, arrival_ms, prompt_tokens, output_tokens")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--format", choices=["text", "json"], default="text")
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    try:
        requests = ([Request(**r) for r in json.loads(args.requests.read_text())]
                    if args.requests else workloads(args.scenario))
        report = compare(requests, args.batch_size, trace=args.trace)
    except (ValueError, TypeError, OSError) as error:
        parser.error(str(error))
    if args.format == "json":
        print(json.dumps(report, indent=2, allow_nan=False))
    else:
        print("SIMULATION ONLY: virtual clock and modeled costs; no actual inference performance")
        print("Outputs identical across schedules:", report["same_output_sequences"])
        print("mode          steps  tokens/s   TTFT p95 ms  request p95 ms  slot utilization")
        for r in report["reports"]:
            fmt = lambda v: "n/a" if v is None else "%.3f" % v
            print("%-13s %5d %9.2f %12s %15s %16.1f%%" % (
                r["mode"], r["steps"], r["output_tokens_per_second"], fmt(r["ttft_p95_ms"]),
                fmt(r["request_latency_p95_ms"]), r["slot_utilization"] * 100))
            if args.trace:
                for step in r["trace"]:
                    print("  %.3f -> %.3f %s" % (step["start_ms"], step["end_ms"],
                          ", ".join(i["request_id"] + ":" + i["phase"] for i in step["batch"])))


if __name__ == "__main__":
    main()
