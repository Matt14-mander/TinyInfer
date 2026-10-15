"""Run the virtual scheduling comparison with Python standard library."""
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
    if name == "kv_pressure":
        return [Request("long", 0, 4, 6), Request("peer", 0, 4, 2),
                Request("small", 0, 1, 1), Request("oversized", .5, 40, 2)]
    raise ValueError("unknown workload")


def compare(requests, batch_size=4, backend=None, trace=False,
            kv_capacity_blocks=None, kv_block_size=1, kv_policy="wait"):
    requests = list(requests)
    backend = backend or SimulatedBackend()
    runs = [simulate(requests, backend, mode, batch_size, kv_capacity_blocks, kv_block_size, kv_policy)
            for mode in ("static", "continuous")]
    outputs = [{r.request.request_id: r.tokens for r in run.results} for run in runs]
    shared_ids = outputs[0].keys() & outputs[1].keys()
    if any(outputs[0][key] != outputs[1][key] for key in shared_ids):
        raise RuntimeError("batching changed a shared request's output sequence")
    # Rejection sets can legitimately differ. Every completed request must also
    # match its independent, unconstrained single-request reference.
    reference = {}
    for request in requests:
        if any(request.request_id in output for output in outputs):
            alone = simulate([request], backend, "static", 1)
            reference[request.request_id] = alone.results[0].tokens
    for output in outputs:
        if any(tokens != reference[key] for key, tokens in output.items()):
            raise RuntimeError("completed request differs from unconstrained reference")
    reports = []
    for run in runs:
        report = summarize(run)
        if trace:
            report["trace"] = run.trace
            report["kv_events"] = run.kv["events"]
            report["admission_events"] = run.admission_events
        reports.append(report)
    return {"stage": 2, "backend": "deterministic_simulation", "real_backend": False,
            "same_output_sequences": True,
            "same_completed_request_ids": outputs[0].keys() == outputs[1].keys(),
            "reference_verified_requests": len(reference),
            "cost_model": {"overhead_ms": backend.overhead_ms,
                           "prefill_ms_per_token": backend.prefill_ms_per_token,
                           "decode_ms_per_request": backend.decode_ms_per_request},
            "reports": reports}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=["mixed", "staggered", "long_prefill", "kv_pressure"], default="mixed")
    parser.add_argument("--requests", type=Path, help="JSON list of request_id, arrival_ms, prompt_tokens, output_tokens")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--kv-capacity-blocks", type=int, default=None, help="None means unlimited; zero rejects all requests")
    parser.add_argument("--kv-block-size", type=int, default=1, help="token slots per block")
    parser.add_argument("--kv-policy", choices=["wait", "reject"], default="wait")
    parser.add_argument("--format", choices=["text", "json"], default="text")
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    try:
        requests = ([Request(**r) for r in json.loads(args.requests.read_text())]
                    if args.requests else workloads(args.scenario))
        report = compare(requests, args.batch_size, trace=args.trace,
                         kv_capacity_blocks=args.kv_capacity_blocks, kv_block_size=args.kv_block_size,
                         kv_policy=args.kv_policy)
    except (ValueError, TypeError, OSError) as error:
        parser.error(str(error))
    if args.format == "json":
        print(json.dumps(report, indent=2, allow_nan=False))
    else:
        print("SIMULATION ONLY: virtual clock and modeled costs; no actual inference performance")
        print("Completed outputs match their independent references:", report["same_output_sequences"])
        print("Same completed request IDs:", report["same_completed_request_ids"])
        print("Reference-verified request IDs:", report["reference_verified_requests"])
        print("mode          steps  tokens/s   TTFT p95 ms  request p95 ms  slot utilization")
        for r in report["reports"]:
            fmt = lambda v: "n/a" if v is None else "%.3f" % v
            print("%-13s %5d %9.2f %12s %15s %16.1f%%" % (
                r["mode"], r["steps"], r["output_tokens_per_second"], fmt(r["ttft_p95_ms"]),
                fmt(r["request_latency_p95_ms"]), r["slot_utilization"] * 100))
            print("  completed=%d rejected=%d KV peak=%d/%s blocks; final=%d" % (
                r["completed_requests"], r["rejected_requests"], r["kv"]["peak_reserved_blocks"],
                "unlimited" if r["kv"]["capacity_blocks"] is None else r["kv"]["capacity_blocks"],
                r["kv"]["final_reserved_blocks"]))
            for rejection in r["rejections"]:
                print("  rejected %s: %s at %.3f ms" % (rejection["request_id"], rejection["reason"], rejection["decided_ms"]))
            if args.trace:
                for step in r["trace"]:
                    print("  %.3f -> %.3f KV start=%d blocks; %s" % (step["start_ms"], step["end_ms"], step["kv_reserved_blocks"],
                          ", ".join(i["request_id"] + ":" + i["phase"] for i in step["batch"])))
                for event in r["kv_events"]:
                    print("  KV %.3f %s %s blocks=%d used=%d" % (event["time_ms"], event["event"], event["request_id"], event["blocks"], event["used_blocks"]))
                for event in r["admission_events"]:
                    print("  admission %.3f %s %s: %s" % (event["time_ms"], event["event"], event["request_id"], event.get("reason", "kv_capacity_unavailable")))


if __name__ == "__main__":
    main()
