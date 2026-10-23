"""Sequential fresh-process FP32 CPU/CUDA engine baseline with checked outputs.

The timed endpoint is tokenized scheduled arrival -> controller token event.
Tokenizer, model startup, warmup, stats, reference and JSON are outside warm
windows. This is a dense-KV, serial-per-sequence runner baseline, not a prefix
cache or cross-request matrix-batching speedup claim.
"""
import argparse
import gc
import hashlib
import json
import math
import platform
from pathlib import Path
import statistics
import subprocess
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from miniprefix.bindings.python.client import NativeEngine
from miniprefix.tools.verify_real_model import sha256


def percentile(values, probability):
    if not values:
        return None
    return sorted(values)[max(0, math.ceil(len(values) * probability) - 1)]


def gpu_snapshot():
    command = ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,pstate,clocks.sm,temperature.gpu,power.draw", "--format=csv,noheader"]
    try:
        return {"command": command, "value": subprocess.check_output(command, text=True, timeout=10).strip()}
    except (OSError, subprocess.SubprocessError) as error:
        return {"command": command, "unavailable": str(error)}


def prepare_workload(model_dir):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(4)
    tokenizer_start = time.perf_counter_ns()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    texts = ["What is the capital of France?", "Complete this sentence: The sun rises in the",
             "Give a short explanation of why water freezes. Use simple words.",
             "Read these words: red, green, blue, yellow, white, black, orange, purple. Name one color."]
    prompts = [tokenizer.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True) for text in texts]
    tokenization_ms = (time.perf_counter_ns() - tokenizer_start) / 1e6
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, torch_dtype=torch.float32,
                                               attn_implementation="eager").eval()
    expected = []
    for prompt in prompts:
        prefix, tokens = list(prompt), []
        # EOS deliberately disabled for the fixed output-length benchmark.
        for _ in range(8):
            with torch.inference_mode():
                ids = torch.tensor([prefix], dtype=torch.long)
                logits = model(ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[0, -1]
                token = int(logits.argmax())
            tokens.append(token);prefix.append(token)
        expected.append(tokens)
    workload = [{"request_id": "r" + str(i), "arrival_ms": (i // 2) * 10.0,
                 "text": texts[i % 4], "prompt_tokens": prompts[i % 4], "max_new_tokens": 8,
                 "eos_enabled": False, "expected_tokens": expected[i % 4],
                 "expected_text": tokenizer.decode(expected[i % 4], skip_special_tokens=True)} for i in range(8)]
    import tokenizers
    import safetensors
    import numpy
    versions = {"torch": torch.__version__, "transformers": transformers.__version__,
                "numpy": numpy.__version__, "tokenizers": tokenizers.__version__, "safetensors": safetensors.__version__,
                "reference": "FP32/eager CPU full-prefix, no KV reuse, argmax per output", "tokenizer_preparation_ms": tokenization_ms}
    del model
    gc.collect()
    return workload, versions


def measure(engine, workload):
    if not workload:
        raise ValueError("empty performance workload")
    if engine.pending:
        raise RuntimeError("measurement requires idle engine")
    origin = time.perf_counter_ns()
    records = {r["request_id"]: {"request_id": r["request_id"], "arrival_ms": r["arrival_ms"],
                                "submitted_ms": None, "token_times_ms": [], "tokens": [], "finish_reason": None} for r in workload}
    ordered = sorted(workload, key=lambda r: r["arrival_ms"])
    index, steps = 0, []
    while index < len(ordered) or engine.pending:
        now = (time.perf_counter_ns() - origin) / 1e6
        if not engine.pending and index < len(ordered) and ordered[index]["arrival_ms"] > now:
            time.sleep((ordered[index]["arrival_ms"] - now) / 1000)
            now = (time.perf_counter_ns() - origin) / 1e6
        while index < len(ordered) and ordered[index]["arrival_ms"] <= now:
            r = ordered[index];index += 1
            records[r["request_id"]]["submitted_ms"] = (time.perf_counter_ns() - origin) / 1e6
            engine.submit(r["request_id"], r["prompt_tokens"], r["max_new_tokens"], None)
            now = (time.perf_counter_ns() - origin) / 1e6
        if not engine.pending:
            continue
        start = (time.perf_counter_ns() - origin) / 1e6
        events = engine.step()
        end = (time.perf_counter_ns() - origin) / 1e6
        steps.append({"start_ms": start, "end_ms": end, "events": [dict(e, received_ms=(e["received_ns"] - origin) / 1e6) for e in events]})
        for e in events:
            r = records[e["request_id"]]
            if r["finish_reason"]:
                raise RuntimeError("event after terminal result")
            if e["token_id"] is not None:
                r["tokens"].append(e["token_id"]);r["token_times_ms"].append((e["received_ns"] - origin) / 1e6)
            if e["finish_reason"]:
                r["finish_reason"] = e["finish_reason"]
    # Freeze the warm window before stats queries, output verification or IO.
    end = max((r["token_times_ms"][-1] for r in records.values()), default=0.0)
    result = list(records.values())
    for r, source in zip(result, workload):
        if r["tokens"] != source["expected_tokens"] or r["finish_reason"] != "length":
            raise RuntimeError("independent-reference mismatch in " + r["request_id"])
        r["controller_submit_delay_ms"] = r["submitted_ms"] - r["arrival_ms"]
        r["ttft_ms"] = r["token_times_ms"][0] - r["arrival_ms"]
        r["latency_ms"] = r["token_times_ms"][-1] - r["arrival_ms"]
        r["tpot_ms"] = ((r["token_times_ms"][-1] - r["token_times_ms"][0]) / (len(r["tokens"]) - 1)
                        if len(r["tokens"]) > 1 else None)
    gaps = [b-a for r in result for a,b in zip(r["token_times_ms"], r["token_times_ms"][1:])]
    tpots = [r["tpot_ms"] for r in result if r["tpot_ms"] is not None]
    return {"requests": result, "steps": steps, "measurement_window_ms": end,
            "output_tokens": sum(len(r["tokens"]) for r in result), "completed_requests": len(result),
            "ttft_p50_ms": percentile([r["ttft_ms"] for r in result], .5),
            "ttft_p95_ms": percentile([r["ttft_ms"] for r in result], .95),
            "tpot_mean_ms": statistics.mean(tpots) if tpots else None,
            "itl_p95_ms": percentile(gaps, .95), "request_latency_p95_ms": percentile([r["latency_ms"] for r in result], .95),
            "output_tokens_per_second": sum(len(r["tokens"]) for r in result) * 1000 / end,
            "reference_correct": True}


def run(model_dir, workers, verification_path, runs=5):
    if type(runs) is not int or runs < 3:
        raise ValueError("baseline requires at least three independent runs; default is five")
    root = Path(model_dir)
    verified = json.loads(Path(verification_path).read_text(encoding="utf-8"))
    fingerprint = sha256(root / "model.mpf")
    if not verified.get("passed") or verified.get("native_model_sha256") != fingerprint:
        raise ValueError("a passing verification of this exact model is required")
    worker_hashes = {backend: sha256(path) for backend,path in workers.items()}
    if any(verified["worker_sha256"].get(k) != v for k,v in worker_hashes.items()):
        raise ValueError("worker changed since correctness verification")
    before_gpu = gpu_snapshot()
    workload, reference = prepare_workload(root)
    profiles = [("cpu_b1", "cpu", 1)]
    if "cuda" in workers:
        profiles += [("cuda_b1", "cuda", 1), ("cuda_b2", "cuda", 2)]
    reports = []
    for run_index in range(runs):
        order = profiles[run_index % len(profiles):] + profiles[:run_index % len(profiles)]
        for name, backend, batch_size in order:
            print("measuring " + name + " run " + str(run_index + 1), file=sys.stderr, flush=True)
            start = time.perf_counter_ns()
            with NativeEngine(workers[backend], root / "model.mpf", backend, 256, batch_size, 32, 16) as engine:
                startup_ms = (time.perf_counter_ns() - start) / 1e6
                warmup_samples = []
                for warmup in range(3):
                    warm_start = time.perf_counter_ns()
                    engine.submit("warmup" + str(warmup), workload[warmup]["prompt_tokens"], 4, None)
                    warm_events = list(engine.events())
                    if any(e["finish_reason"] == "error" for e in warm_events):
                        raise RuntimeError("warmup failed")
                    first = next(e["received_ns"] for e in warm_events if e["token_id"] is not None)
                    warmup_samples.append({"ttft_ms": (first-warm_start)/1e6, "end_ms": (time.perf_counter_ns()-warm_start)/1e6,
                                           "first_token_since_worker_start_ms": (first-start)/1e6})
                before = engine.stats()
                if before["live_sequences"] or before["live_kv_bytes"]:
                    raise RuntimeError("warmup leaked KV")
                report = measure(engine, workload)
                after = engine.stats()
                delta = {key: after[key] - before[key] for key in before if key not in ("peak_kv_bytes", "device_workspace_bytes")}
                expected_inputs = sum(len(r["prompt_tokens"]) + r["max_new_tokens"] - 1 for r in workload)
                if after["live_sequences"] or after["live_kv_bytes"] or delta["forward_tokens"] != expected_inputs:
                    raise RuntimeError("KV lifecycle or token accounting failure")
                if backend == "cuda":
                    if (delta["weight_uploaded_bytes"] or delta["kv_uploaded_bytes"]
                            or delta["token_uploaded_bytes"] != expected_inputs * 4
                            or delta["device_allocations"] != 2 * len(workload)
                            or before["device_workspace_bytes"] != after["device_workspace_bytes"]):
                        raise RuntimeError("warm CUDA residency/allocation regression")
                report.update(profile=name, backend=backend, batch_size=batch_size, run=run_index + 1,
                              worker_pid=engine.process.pid, worker_startup_ms=startup_ms,
                              fresh_process_first_token_ms=warmup_samples[0]["first_token_since_worker_start_ms"],
                              warmup_requests=3, warmup_samples=warmup_samples,
                              before_stats=before, after_stats=after, counter_delta=delta)
                reports.append(report)
    aggregates = {}
    keys = ("worker_startup_ms", "fresh_process_first_token_ms", "ttft_p50_ms", "ttft_p95_ms", "tpot_mean_ms", "itl_p95_ms", "output_tokens_per_second", "measurement_window_ms")
    for name,_,_ in profiles:
        group = [r for r in reports if r["profile"] == name]
        aggregates[name] = {key: {"median": statistics.median(r[key] for r in group),
                                 "min": min(r[key] for r in group), "max": max(r[key] for r in group),
                                 "sample_stddev": statistics.stdev(r[key] for r in group)} for key in keys}
    source = Path(__file__).resolve().parents[1]
    build = {}
    for backend, worker in workers.items():
        cache = Path(worker).resolve().parent.parent / "CMakeCache.txt"
        if cache.exists():
            wanted = ("CMAKE_BUILD_TYPE", "CMAKE_CXX_COMPILER", "CMAKE_CXX_FLAGS", "CMAKE_CXX_FLAGS_RELEASE",
                      "TINYINFER_ENABLE_NATIVE_ARCH", "CMAKE_CUDA_COMPILER", "CMAKE_CUDA_ARCHITECTURES", "CMAKE_CUDA_FLAGS")
            build[backend] = {"cache_sha256": sha256(cache), "flags": {line.split(":",1)[0]: line.split("=",1)[1]
                              for line in cache.read_text().splitlines() if ":" in line and "=" in line and line.split(":",1)[0] in wanted}}
    return {"date": "2026-10-23", "dtype": "float32", "model": json.loads((root / "manifest.json").read_text()),
            "command": sys.argv, "build": build,
            "native_model_sha256": fingerprint, "worker_sha256": worker_hashes, "correctness_sha256": sha256(verification_path),
            "source_sha256": {str(p.relative_to(source)): sha256(p) for p in source.rglob("*") if p.suffix in (".py", ".cpp", ".cu", ".h")},
            "environment": {"platform": platform.platform(), "python": sys.version, "gpu_before": before_gpu, "gpu_after": gpu_snapshot()},
            "reference": reference, "execution": "serial per sequence; native continuous scheduling; dense per-layer KV; no prefix cache",
            "measurement": "scheduled tokenized arrival to controller token-event delivery; includes queue, IPC, forward, sampling and native KV release",
            "excluded": "tokenizer/model startup, warmup, stats, oracle, decoding and JSON IO; startup is reported separately",
            "startup_semantics": "new process per sample, OS file cache and driver caches are not flushed",
            "percentiles": "nearest rank per 8-request run; median/min/max across independent runs; no SLA or confidence interval claim",
            "memory_semantics": "owned FP32 weights/KV/workspace payload only; not process RSS, total VRAM or cuBLAS internal allocations",
            "workload": workload, "runs_per_profile": runs, "reports": reports, "aggregates": aggregates}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--cpu-worker", required=True)
    parser.add_argument("--cuda-worker")
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    workers = {"cpu": args.cpu_worker}
    if args.cuda_worker:
        workers["cuda"] = args.cuda_worker
    result = run(args.model_dir, workers, args.verification, args.runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["aggregates"], indent=2))


if __name__ == "__main__":
    main()
