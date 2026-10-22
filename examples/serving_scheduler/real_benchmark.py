"""Real CPU engineering benchmark, not pretrained LLM or GPU batch performance."""
import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path
try:
    from .backend import Request
    from .tinyinfer_backend import TinyInferBackend
    from .realtime import serve
    from .metrics import summarize
except ImportError:
    from backend import Request
    from tinyinfer_backend import TinyInferBackend
    from realtime import serve
    from metrics import summarize


def run(executable, runs=3, kv_layout="contiguous", attention_mode="direct"):
    if type(runs) is not int or runs < 1:
        raise ValueError("runs must be positive")
    pattern = [(0, 6, 4), (0, 8, 5), (.5, 12, 3), (1, 4, 6), (1.5, 7, 4), (2, 5, 2)]
    requests = [Request("r%d" % i, (i // 6) * 2 + pattern[i % 6][0],
                        pattern[i % 6][1], pattern[i % 6][2]) for i in range(30)]
    configurations = [("static_full", "static", None, None), ("continuous_full", "continuous", None, None),
                      ("static_chunked", "static", 8, 4), ("continuous_chunked", "continuous", 8, 4)]
    reports = []
    root = Path(__file__).parent
    for index in range(runs):
        prepared = time.perf_counter_ns()
        with TinyInferBackend(executable, block_size=4, kv_layout=kv_layout,
                              kv_capacity_blocks=32 if kv_layout == "paged" else None,
                              attention_mode=attention_mode) as backend:
            preparation_ms = (time.perf_counter_ns() - prepared) / 1000000.0
            # Independent worker per run, warm each scheduling configuration.
            order = configurations[index % len(configurations):] + configurations[:index % len(configurations)]
            for name, mode, budget, chunk in order:
                serve([Request("warmup", 0, 4, 3)], backend, mode, 3, 32, 4, "wait", budget, chunk)
                result = serve(requests, backend, mode, 3, 32, 4, "wait", budget, chunk)
                report = summarize(result)
                report.update(configuration=name, run=index + 1, worker_preparation_ms=preparation_ms,
                              trace=result.trace, kv_events=result.kv["events"], admission_events=result.admission_events)
                if report["completed_requests"] != len(requests) or report["rejected_requests"]:
                    raise RuntimeError("unexpected rejection in the fixed benchmark workload")
                for completed in result.results:
                    if completed.tokens != backend.reference(completed.request):
                        raise RuntimeError("real cached greedy differs from full-prefix reference")
                stats = report["backend_stats"]
                if stats["live_requests"] or stats["live_kv_bytes"] or report["kv"]["final_reserved_blocks"]:
                    raise RuntimeError("KV resources leaked")
                if stats["forward_tokens"] != report["prefill_input_tokens"] + report["decode_input_tokens"]:
                    raise RuntimeError("real forward/token accounting differs")
                if (stats["batch_steps"] != len(result.trace)
                        or stats["linear_matmul_calls"] != 7 * stats["batch_steps"]
                        or stats["linear_matmul_rows"] != 7 * stats["forward_tokens"]):
                    raise RuntimeError("packed linear batching accounting differs")
                if stats["peak_kv_bytes"] > 32 * 4 * 2 * 16 * 4:
                    raise RuntimeError("native KV exceeded logical capacity")
                if kv_layout == "paged":
                    query_key = "paged_attention_queries" if attention_mode == "direct" else "gather_attention_queries"
                    if (stats[query_key] != stats["forward_tokens"] or stats["live_physical_blocks"]
                            or stats["free_physical_blocks"] != 32
                            or stats["pool_kv_bytes"] != 32 * 4 * 2 * 16 * 4
                            or stats["peak_physical_blocks"] > report["kv"]["peak_reserved_blocks"]):
                        raise RuntimeError("paged KV/attention accounting differs")
                report["reference_correct"] = True
                reports.append(report)
    aggregates = {}
    for name, _, _, _ in configurations:
        group = [r for r in reports if r["configuration"] == name]
        aggregates[name] = {key: statistics.median(r[key] for r in group) for key in
                            ("ttft_p50_ms", "ttft_p95_ms", "tpot_mean_ms", "request_latency_p95_ms",
                             "output_tokens_per_second", "requests_per_second", "max_inter_token_ms")}
    return {"stage": 4, "revision": "paged_attention" if kv_layout == "paged" else "tensor_batching", "clock": "wall_ms", "backend": "tinyinfer_cpu_single_layer_decoder",
            "kv_layout": kv_layout, "attention_mode": attention_mode,
            "real_backend": True, "pretrained": False, "hardware_batched": True,
            "batching": "packed CPU token rows for projections/FFN/head; per-query causal attention",
            "model": {"seed": 20261015, "dimension": 16, "vocabulary": 32, "layers": 1,
                      "attention_heads": 1, "ff_dimension": 32, "dtype": "float32", "max_context": 4096},
            "measurement": "controller arrival to completed batch token publication, includes IPC/scheduling; worker preparation and warmup excluded",
            "percentile": "nearest rank ceil(n*p); 30 requests per configuration; median of per-run percentiles, descriptive engineering result only",
            "aggregates": aggregates, "reports": reports,
            "environment": {"platform": platform.platform(), "python": platform.python_version(),
                            "worker_sha256": hashlib.sha256(Path(executable).read_bytes()).hexdigest(),
                            "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in
                                              ("tinyinfer_worker.cpp", "native/paged_kv_cache.h", "tinyinfer_backend.py", "scheduler.py", "realtime.py", "real_benchmark.py")}},
            "arrivals": [{"id": r.request_id, "arrival_ms": r.arrival_ms, "prompt_tokens": r.prompt_tokens, "output_tokens": r.output_tokens} for r in requests],
            "prompts": {r.request_id: TinyInferBackend.prompt(r) for r in requests}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--kv-layout", choices=("contiguous", "paged"), default="contiguous")
    parser.add_argument("--attention-mode", choices=("direct", "gather"), default="direct")
    args = parser.parse_args()
    result = run(args.worker, args.runs, args.kv_layout, args.attention_mode)
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__": main()
