"""All reported times and rates are simulated, not measured model performance."""
from collections import Counter
import math
import statistics


def percentile(values, probability):
    if not values:
        return None
    return sorted(values)[max(0, math.ceil(len(values) * probability) - 1)]


def summarize(run):
    requests = []
    for result in run.results:
        r = result.request
        requests.append({"request_id": r.request_id, "arrival_ms": r.arrival_ms,
                         "admitted_ms": result.admitted_ms, "finished_ms": result.finished_ms,
                         "queue_ms": result.admitted_ms - r.arrival_ms,
                         "ttft_ms": result.token_times_ms[0] - r.arrival_ms,
                         "tpot_ms": ((result.finished_ms - result.token_times_ms[0]) / (len(result.tokens) - 1)
                                     if len(result.tokens) > 1 else None),
                         "latency_ms": result.finished_ms - r.arrival_ms,
                         "tokens": result.tokens, "token_times_ms": result.token_times_ms})
    rejections = [{"request_id": r.request.request_id, "arrival_ms": r.request.arrival_ms,
                   "decided_ms": r.decided_ms, "rejection_delay_ms": r.decided_ms - r.request.arrival_ms,
                   "reason": r.reason, "required_blocks": r.required_blocks} for r in run.rejections]
    total = len(requests) + len(rejections)
    gaps = [b - a for result in run.results for a, b in zip(result.token_times_ms, result.token_times_ms[1:])]
    duration = run.finished_ms - run.started_ms
    # Time-weighted number of live request slots occupied during service.
    busy = sum((step["end_ms"] - step["start_ms"]) * step.get("active_requests", len(step["batch"])) for step in run.trace)
    scheduled = sum((step["end_ms"] - step["start_ms"]) * len(step["batch"]) for step in run.trace)
    service = sum(step["end_ms"] - step["start_ms"] for step in run.trace)
    values = lambda key: [r[key] for r in requests if r[key] is not None]
    return {"mode": run.mode, "clock": "simulated_ms", "requests": requests,
            "submitted_requests": total, "completed_requests": len(requests), "rejected_requests": len(rejections),
            "rejection_rate": len(rejections) / total if total else 0.0,
            "rejections": rejections, "rejections_by_reason": dict(Counter(r["reason"] for r in rejections)),
            "rejection_delay_p95_ms": percentile([r["rejection_delay_ms"] for r in rejections], .95),
            "kv_policy": run.kv_policy, "kv": {k: v for k, v in run.kv.items() if k != "events"},
            "kv_waited_requests": len({e["request_id"] for e in run.admission_events if e["event"] == "kv_wait"}),
            "token_budget": run.token_budget, "prefill_chunk_size": run.prefill_chunk_size,
            "prefill_input_tokens": sum(s["prefill_tokens"] for s in run.trace),
            "decode_input_tokens": sum(s["decode_tokens"] for s in run.trace),
            "peak_step_input_tokens": max((s["input_tokens"] for s in run.trace), default=0),
            "inter_token_p95_ms": percentile(gaps, .95), "max_inter_token_ms": max(gaps, default=None),
            "scheduled_slot_utilization": scheduled / (duration * run.max_batch_size) if duration else 0.0,
            "steps": len(run.trace), "measurement_window_ms": duration,
            "generated_tokens": sum(len(r.tokens) for r in run.results),
            "output_tokens_per_second": sum(len(r.tokens) for r in run.results) * 1000 / duration if duration else 0.0,
            "requests_per_second": len(requests) * 1000 / duration if duration else 0.0,
            "ttft_p50_ms": percentile(values("ttft_ms"), .5),
            "ttft_p95_ms": percentile(values("ttft_ms"), .95),
            "tpot_mean_ms": statistics.mean(values("tpot_ms")) if values("tpot_ms") else None,
            "request_latency_p95_ms": percentile(values("latency_ms"), .95),
            "slot_utilization": busy / (duration * run.max_batch_size) if duration else 0.0,
            "service_slot_utilization": busy / (service * run.max_batch_size) if service else 0.0}
