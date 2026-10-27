"""Independent-process, oracle-checked off/cold/warm exact-prefix ablation.

The endpoint and arrival accounting reuse baseline.measure. No per-step stats
IPC enters the timed window. Hot seed cost and retained payload are explicit.
"""
import argparse
from datetime import datetime, timedelta, timezone
import gc
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from miniprefix.benchmarks.baseline import measure, gpu_snapshot
from miniprefix.bindings.python.client import NativeEngine
from miniprefix.tools.verify_real_model import sha256

MODES=("off","cold","warm")
SCENARIOS=("shared_96","mixed_50pct","no_shared")
CONFIG=dict(context=256,batch_size=2,token_budget=32,chunk_size=16,block_tokens=16,kv_blocks=128)
CACHE_BLOCKS=48


def distribution(values):
    return dict(median=statistics.median(values),min=min(values),max=max(values),
                sample_stddev=statistics.stdev(values) if len(values)>1 else None)


def sampling_order(round_index):
    scenarios=SCENARIOS[round_index%3:]+SCENARIOS[:round_index%3]
    result=[]
    for scenario in scenarios:
        # Use stable scenario identity: its rotating *position* would cancel
        # the round rotation and accidentally fix each scenario's mode order.
        offset=(round_index+SCENARIOS.index(scenario))%3
        result.extend((scenario,mode) for mode in MODES[offset:]+MODES[:offset])
    return result


def validate_counters(before,after,workload,backend,mode):
    if (after["live_sequences"] or after["block_table_device_bytes"] or
            after["kv_live_blocks"]!=after["kv_cache_only_blocks"] or
            after["kv_cached_blocks"]!=after["kv_cache_only_blocks"]):
        raise RuntimeError("unfinished request or leaked non-cache page")
    delta={key:after[key]-before[key] for key in before}
    full=sum(len(r["prompt_tokens"])+r["max_new_tokens"]-1 for r in workload)
    prompt_tokens=sum(len(r["prompt_tokens"]) for r in workload)
    reused=delta["prefix_reused_tokens"]
    if not 0<=reused<=prompt_tokens or delta["forward_tokens"]!=full-reused:
        raise RuntimeError("cache hit did not skip actual forward tokens")
    if (after["prefix_cache_capacity"]!=(0 if mode=="off" else CACHE_BLOCKS) or
            after["kv_cached_blocks"]>after["prefix_cache_capacity"]):
        raise RuntimeError("wrong cache configuration")
    if mode=="off" and (reused or after["kv_live_blocks"] or delta["prefix_hits"] or delta["prefix_misses"]):
        raise RuntimeError("off control retained cache state")
    if mode!="off" and delta["prefix_hits"]+delta["prefix_misses"]!=len(workload):
        raise RuntimeError("prefix lookup count differs from admitted requests")
    if delta["kv_reserved_bytes"] or before["device_workspace_bytes"]!=after["device_workspace_bytes"]:
        raise RuntimeError("steady workspace/pool grew")
    if backend=="cuda" and (delta["weight_uploaded_bytes"] or delta["kv_uploaded_bytes"] or
            delta["token_uploaded_bytes"]!=delta["forward_tokens"]*4 or
            delta["device_allocations"]!=len(workload)):
        raise RuntimeError("warm GPU residency/metadata allocation contract violated")
    block_bytes=after["kv_reserved_bytes"]//after["kv_total_blocks"]
    return dict(counter_delta=delta,prompt_tokens=prompt_tokens,uncached_forward_tokens=full,
                actual_forward_tokens=delta["forward_tokens"],reused_tokens=reused,
                reused_prompt_fraction=reused/prompt_tokens,request_hit_fraction=delta["prefix_hits"]/len(workload),
                retained_cache_payload_bytes=after["kv_cached_blocks"]*block_bytes,
                evictions=delta["prefix_evictions"],
                process_peak_kv_payload_bytes=after["peak_kv_bytes"])


def paired_comparisons(reports):
    result={}
    for scenario in SCENARIOS:
        selected=[r for r in reports if r["scenario"]==scenario]
        rounds=sorted({r["run"] for r in selected})
        by_key={(r["run"],r["mode"]):r for r in selected}
        comparisons={}
        for mode in ("cold","warm"):
            pairs=[]
            for run in rounds:
                off=by_key[run,"off"];on=by_key[run,mode]
                pairs.append(dict(run=run,ttft_p50_speedup=off["ttft_p50_ms"]/on["ttft_p50_ms"],
                    throughput_speedup=on["output_tokens_per_second"]/off["output_tokens_per_second"],
                    prime_inclusive_throughput_speedup=on["prime_inclusive_tokens_per_second"]/off["output_tokens_per_second"],
                    tpot_latency_ratio=on["tpot_mean_ms"]/off["tpot_mean_ms"]))
            comparisons[mode]=dict(pairs=pairs,summary={k:distribution([p[k] for p in pairs]) for k in
                ("ttft_p50_speedup","throughput_speedup","prime_inclusive_throughput_speedup","tpot_latency_ratio")})
        result[scenario]=comparisons
    return result


def prepare_workloads(root):
    import torch
    import transformers
    import numpy
    import tokenizers
    import safetensors
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(4)
    tokenizer=AutoTokenizer.from_pretrained(root,local_files_only=True)
    passage=("The public library opens every morning. Readers borrow books, study together, and ask the staff for help. "
             "The reading room is quiet, and returned books are sorted by subject. ")*12
    common=tokenizer.encode(passage,add_special_tokens=False)[:96]
    words=("Apple","Bridge","Candle","Desert","Eagle","Forest","Garden","Harbor")
    roots=[tokenizer.encode(word+". "+passage,add_special_tokens=False)[:96] for word in words]
    if len({r[0] for r in roots})!=8 or common[0] in {r[0] for r in roots}:
        raise RuntimeError("no-shared control must differ in its first token")
    suffixes=[]
    suffix_texts=[]
    for i in range(8):
        text=("\nQuestion "+str(i)+": Describe the reading room and explain one useful service at this library. "
              "Answer clearly using the passage. Do not add unsupported facts. ")*3
        suffixes.append(tokenizer.encode(text,add_special_tokens=False)[:24]);suffix_texts.append(text)
    if len(common)!=96 or any(len(r)!=96 for r in roots) or any(len(s)!=24 for s in suffixes):
        raise RuntimeError("fixed prompt segment lengths were not produced")
    workloads={}
    for scenario in SCENARIOS:
        rows=[]
        for i in range(8):
            shared=scenario=="shared_96" or scenario=="mixed_50pct" and i<4
            prompt=(common if shared else roots[i])+suffixes[i]
            rows.append(dict(request_id="r"+str(i),arrival_ms=(i//2)*25.0,prompt_tokens=prompt,
                             max_new_tokens=8,eos_enabled=False,shared_root=shared))
        workloads[scenario]=rows
    model=AutoModelForCausalLM.from_pretrained(root,local_files_only=True,torch_dtype=torch.float32,
                                              attn_implementation="eager").eval()
    expected={}
    for rows in workloads.values():
        for r in rows:
            key=tuple(r["prompt_tokens"])
            if key not in expected:
                prefix=list(key);output=[]
                for _ in range(8):
                    with torch.inference_mode():
                        ids=torch.tensor([prefix],dtype=torch.long)
                        token=int(model(ids,attention_mask=torch.ones_like(ids),use_cache=False).logits[0,-1].argmax())
                    output.append(token);prefix.append(token)
                expected[key]=output
            r["expected_tokens"]=expected[key]
            r["decoded_prompt"]=tokenizer.decode(r["prompt_tokens"])
            r["expected_text"]=tokenizer.decode(r["expected_tokens"],skip_special_tokens=True)
    # Different first suffix token; only the first six full pages are published.
    # Identical seed work in every arm warms identical compute paths. Only warm
    # retains it, and its actual preparation cost is added to warm's amortized rate.
    seed=common+[tokenizer.eos_token_id]
    versions=dict(torch=torch.__version__,transformers=transformers.__version__,numpy=numpy.__version__,
                  tokenizers=tokenizers.__version__,safetensors=safetensors.__version__,
                  oracle="FP32 CPU eager full-prefix Greedy, no KV reuse",unique_prompts=len(expected))
    sources=dict(passage_text=passage,root_words=list(words),question_texts=suffix_texts,
                 construction="raw tokenized prose, fixed first 96 root tokens + first 24 question tokens; no chat template")
    del model;gc.collect()
    return workloads,seed,versions,sources


def verify_gate(root,worker,backend,verification):
    evidence=json.loads(verification.read_text(encoding="utf-8"))
    if (not evidence.get("passed") or not evidence.get("logit_checks") or
            not any(c.get("case")=="long" and c.get("backend")==backend for c in evidence.get("checks",[])) or
            evidence.get("native_model_sha256")!=sha256(root/"model.mpf") or
            evidence.get("worker_sha256",{}).get(backend)!=sha256(worker)):
        raise ValueError("passing incremental-prefill numerical acceptance for this exact worker/model is required")


def run(root,worker,backend,verification,runs):
    if runs<3:raise ValueError("at least three independent processes per arm; five recommended")
    verify_gate(root,worker,backend,verification)
    native_hash=sha256(root/"model.mpf");worker_hash=sha256(worker)
    source=Path(__file__).resolve().parents[1]
    source_hashes={str(p.relative_to(source)):sha256(p) for p in source.rglob("*") if p.suffix in (".py",".cpp",".cu",".h")}
    workloads,seed,reference,sources=prepare_workloads(root)
    environment=dict(platform=platform.platform(),python=sys.version,processor=platform.processor(),
                     logical_cpu_count=os.cpu_count(),gpu_before=gpu_snapshot())
    reports=[]
    for round_index in range(runs):
        for scenario,mode in sampling_order(round_index):
            print(f"measuring {backend}/{scenario}/{mode} round {round_index+1}/{runs}",file=sys.stderr,flush=True)
            start=time.perf_counter_ns()
            with NativeEngine(worker,root/"model.mpf",backend=backend,
                    prefix_cache_blocks=0 if mode=="off" else CACHE_BLOCKS,**CONFIG) as engine:
                startup_ms=(time.perf_counter_ns()-start)/1e6
                if engine.backend!=backend:raise RuntimeError("backend fallback cannot count as GPU acceptance")
                for i in range(3):
                    engine.submit("warmup-"+str(i),workloads[scenario][i]["prompt_tokens"],4,None)
                    if any(e["finish_reason"]=="error" for e in engine.events()):raise RuntimeError("warmup error")
                engine.clear_prefix_cache()
                prime_start=time.perf_counter_ns()
                engine.submit("cache-seed",seed,1,None);events=list(engine.events())
                prime_ms=(time.perf_counter_ns()-prime_start)/1e6
                if len(events)!=1 or events[0]["finish_reason"]!="length":raise RuntimeError("seed failure")
                if mode!="warm":engine.clear_prefix_cache()
                before=engine.stats()
                if before["kv_cached_blocks"]!=(6 if mode=="warm" else 0):raise RuntimeError("wrong initial cache state")
                report=measure(engine,workloads[scenario])
                after=engine.stats()
                report.update(validate_counters(before,after,workloads[scenario],backend,mode))
                if scenario=="no_shared" and report["reused_tokens"]:
                    raise RuntimeError("unrelated causal prefix falsely reused")
                if mode=="warm":
                    required=(8 if scenario=="shared_96" else 4 if scenario=="mixed_50pct" else 0)*96
                    if report["reused_tokens"]!=required:raise RuntimeError("hot root reuse differs from fixed workload")
                inclusive_ms=report["measurement_window_ms"]+(prime_ms if mode=="warm" else 0)
                report.update(backend=backend,scenario=scenario,mode=mode,run=round_index+1,
                    worker_pid=engine.process.pid,worker_startup_ms=startup_ms,warmup_requests=3,
                    cache_seed_ms=prime_ms,cache_seed_forward_tokens=len(seed),
                    prime_inclusive_window_ms=inclusive_ms,prime_inclusive_tokens_per_second=report["output_tokens"]*1000/inclusive_ms,
                    before_stats=before,after_stats=after)
                engine.clear_prefix_cache();cleared=engine.stats()
                if cleared["kv_live_blocks"] or cleared["block_table_device_bytes"]:raise RuntimeError("clear leaked pages")
                report["after_clear_stats"]=cleared;reports.append(report)
    keys=("ttft_p50_ms","ttft_p95_ms","tpot_mean_ms","itl_p95_ms","output_tokens_per_second",
          "measurement_window_ms","prime_inclusive_tokens_per_second","cache_seed_ms","actual_forward_tokens",
          "reused_tokens","reused_prompt_fraction","request_hit_fraction","retained_cache_payload_bytes","evictions")
    aggregates={scenario:{mode:{k:distribution([r[k] for r in reports if r["scenario"]==scenario and r["mode"]==mode])
        for k in keys} for mode in MODES} for scenario in SCENARIOS}
    if sha256(worker)!=worker_hash or sha256(root/"model.mpf")!=native_hash:
        raise RuntimeError("worker/model changed during sampling")
    for rel,digest in source_hashes.items():
        if sha256(source/rel)!=digest:raise RuntimeError("source changed during sampling: "+rel)
    cache=Path(worker).resolve().parent.parent/"CMakeCache.txt"
    flags={}
    if cache.exists():
        wanted=("CMAKE_BUILD_TYPE","CMAKE_CXX_COMPILER","CMAKE_CXX_FLAGS_RELEASE","TINYINFER_ENABLE_NATIVE_ARCH",
                "CMAKE_CUDA_COMPILER","CMAKE_CUDA_ARCHITECTURES","CMAKE_CUDA_FLAGS")
        flags={line.split(":",1)[0]:line.split("=",1)[1] for line in cache.read_text().splitlines()
               if ":" in line and "=" in line and line.split(":",1)[0] in wanted}
    environment["gpu_after"]=gpu_snapshot()
    return dict(date=datetime.now(timezone(timedelta(hours=8))).date().isoformat(),passed=True,
        model=json.loads((root/"manifest.json").read_text()),native_model_sha256=native_hash,
        worker_sha256={backend:worker_hash},verification_sha256=sha256(verification),source_sha256=source_hashes,
        command=sys.argv,config=CONFIG,cache_blocks=CACHE_BLOCKS,runs_per_arm=runs,environment=environment,
        build=dict(flags=flags,cache_sha256=sha256(cache) if cache.exists() else None),reference=reference,
        workloads=workloads,source_texts=sources,cache_seed_tokens=seed,reports=reports,aggregates=aggregates,
        paired_comparisons=paired_comparisons(reports),
        protocol="serial sampling; fresh worker per arm/workload/round; rotating workload/arm order; 8 requests, arrivals 0/0/25/25/50/50/75/75 ms, 120 prompt + 8 output tokens, EOS disabled",
        measurement="planned tokenized arrival -> controller token event; includes submission delay/queue/IPC/forward/Greedy/release; no per-step stats query",
        exclusions="tokenizer, oracle, startup, warmup, cache seed, clear and stats outside main window; warm prime-inclusive rate additionally charges seed cost",
        memory="owned KV/cache/table/weights/workspace payload, not RSS or total VRAM; peak_kv_bytes is since worker start including warmup/seed",
        percentiles="nearest rank over 8 requests/run, P95=max; five-run median/min/max/stdev and paired ratios, no SLA/CI claim")


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--model-dir",type=Path,required=True)
    p.add_argument("--worker",required=True);p.add_argument("--backend",choices=("cpu","cuda"),default="cuda")
    p.add_argument("--verification",type=Path,required=True);p.add_argument("--runs",type=int,default=5)
    p.add_argument("--output",type=Path,required=True);args=p.parse_args()
    result=run(args.model_dir,args.worker,args.backend,args.verification,args.runs)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    for scenario,modes in result["aggregates"].items():
        for mode,m in modes.items():
            print(scenario,mode,json.dumps({k:m[k]["median"] for k in
                ("ttft_p50_ms","ttft_p95_ms","tpot_mean_ms","output_tokens_per_second","reused_tokens","retained_cache_payload_bytes","evictions")}))


if __name__=="__main__":main()
