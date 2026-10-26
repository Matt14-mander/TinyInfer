"""Real CPU/CUDA exact-prefix verification and a small paired warm-engine experiment.

No downloads. Golden logits/tokens come from independent FP32 Transformers
full-prefix recomputation, prepared before the timed flows. This experiment
uses one warm process per arm, not independent-process production percentiles.
"""
import argparse
from datetime import datetime, timedelta, timezone
import gc
import json
from pathlib import Path
import platform
import statistics
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from miniprefix.bindings.python.client import NativeEngine
from miniprefix.tools.verify_real_model import sha256
from miniprefix.benchmarks.baseline import gpu_snapshot


def verify(model_dir, workers, runs=5):
    import numpy as np
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if runs < 3:
        raise ValueError("at least three paired flows required")
    root = Path(model_dir)
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(root, local_files_only=True,
        torch_dtype=torch.float32, attn_implementation="eager").eval()
    prompt = tokenizer.apply_chat_template([{"role":"user","content":"What is the capital of France?"}], add_generation_prompt=True)
    branch = prompt[:8]+[(prompt[8]+17)%model.config.vocab_size]+prompt[9:]
    different = [(prompt[0]+1)%model.config.vocab_size]+prompt[1:]
    cases = {"base":prompt, "branch":branch, "different":different, "aligned":prompt[:32], "short":[1]}
    golden, golden_logits = {}, {}
    for name, tokens in cases.items():
        prefix, output = list(tokens), []
        for step in range(4):
            with torch.inference_mode():
                ids = torch.tensor([prefix], dtype=torch.long)
                logits = model(ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[0,-1].numpy()
            if step==0:
                golden_logits[name] = logits.copy()
            token = int(logits.argmax());output.append(token);prefix.append(token)
        golden[name] = output
    del model
    gc.collect()
    gpu_before = gpu_snapshot()
    checks, logit_checks, samples, summaries = [], [], [], {}

    def compare_logits(actual, name, label):
        a = np.asarray(actual, dtype=np.float32);expected = golden_logits[name]
        diff = np.abs(a-expected)
        if (a.shape!=expected.shape or not np.isfinite(a).all() or
                not np.all(diff<=2e-3+2e-4*np.abs(expected)) or int(a.argmax())!=int(expected.argmax())):
            raise AssertionError("prefix logits differ: "+label)
        logit_checks.append({"case":label,"checked_logits":len(a),"max_abs_error":float(diff.max())})

    def generate(engine, key, name="base", reused=None):
        tokens = cases[name];before = engine.stats()
        start = time.perf_counter_ns()
        engine.submit(key, tokens, 4, None)
        events = list(engine.events())
        after = engine.stats()
        actual = [e["token_id"] for e in events if e["token_id"] is not None]
        if actual!=golden[name] or events[-1]["finish_reason"]!="length":
            raise AssertionError("independent generation mismatch: "+key)
        skipped = after["prefix_reused_tokens"]-before["prefix_reused_tokens"]
        forward = after["forward_tokens"]-before["forward_tokens"]
        if reused is not None and skipped!=reused:
            raise AssertionError("unexpected exact prefix hit: "+key+" got "+str(skipped))
        if forward!=len(tokens)+3-skipped:
            raise AssertionError("skipped prefix was still computed")
        if after["live_sequences"] or after["block_table_device_bytes"]:
            raise AssertionError("finished request resources leaked")
        if after["kv_live_blocks"]!=after["kv_cache_only_blocks"]:
            raise AssertionError("idle physical pages must be cache-only")
        if engine.backend=="cuda":
            if (after["weight_uploaded_bytes"]!=before["weight_uploaded_bytes"] or
                    after["kv_uploaded_bytes"]!=before["kv_uploaded_bytes"] or
                    after["token_uploaded_bytes"]-before["token_uploaded_bytes"]!=4*forward):
                raise AssertionError("prefix reuse uploaded weights/KV or cached input tokens")
        delivered = [(e["received_ns"]-start)/1e6 for e in events if e["token_id"] is not None]
        return {"request_id":key,"case":name,"prompt_tokens":tokens,"output_tokens":actual,
                "ttft_ms":delivered[0],"elapsed_ms":delivered[-1],"token_times_ms":delivered,
                "reused_tokens":skipped,"forward_tokens":forward,"before":before,"after":after}

    for backend, worker in workers.items():
        print("checking prefix cache: "+backend, file=sys.stderr, flush=True)
        config = dict(backend=backend, context=128, batch_size=2, token_budget=16,
                      chunk_size=8, block_tokens=4, kv_blocks=64)
        with NativeEngine(worker, root/"model.mpf", **config) as off, \
             NativeEngine(worker, root/"model.mpf", prefix_cache_blocks=16, **config) as on:
            for name in cases:
                on.clear_prefix_cache()
                generate(off, "off-"+name, name, 0)
                cold = generate(on, "cold-"+name, name, 0)
                skip = (len(cases[name])-1)//4*4
                warm = generate(on, "warm-"+name, name, skip)
                checks.append({"backend":backend,"case":name,"cold_forward":cold["forward_tokens"],
                               "warm_forward":warm["forward_tokens"],"reused_tokens":skip})
            # A common first eight tokens must not cause later divergent pages
            # to match. Conversely, identical later blocks do not bypass the
            # dependency on all earlier positions.
            on.clear_prefix_cache();generate(on,"branch-source",reused=0)
            generate(on,"branch-query","branch",8)
            generate(on,"different-query","different",0)

            # Raw logits across shared full pages, including aligned prompts.
            for name in ("base","aligned","branch"):
                on.clear_prefix_cache()
                compare_logits(on.forward("source-"+name,cases[name]),name,backend+"/cold/"+name)
                on.cache_prefix("source-"+name);on.release("source-"+name)
                skip = on.reuse_prefix("reuse-"+name,cases[name])
                if skip!=(len(cases[name])-1)//4*4:
                    raise AssertionError("aligned full hit failed logits boundary")
                compare_logits(on.forward("reuse-"+name,cases[name][skip:]),name,backend+"/warm/"+name)
                on.release("reuse-"+name)

            # Two active requests hold the same immutable pages; one cancellation
            # and an explicit cache clear must not invalidate the survivor.
            on.clear_prefix_cache();generate(on,"shared-source",reused=0)
            before = on.stats()
            on.submit("shared-a",prompt,4,None);on.submit("shared-b",prompt,4,None)
            first = on.step();middle = on.stats()
            skip = (len(prompt)-1)//4*4
            if middle["prefix_reused_tokens"]-before["prefix_reused_tokens"]!=2*skip:
                raise AssertionError("concurrent request reuse missing")
            if middle["kv_live_blocks"]!=len(prompt)//4+2:
                raise AssertionError("shared pages were duplicated or tails aliased")
            cancelled = on.cancel("shared-a")
            if len(cancelled)!=1 or cancelled[0]["finish_reason"]!="cancelled":
                raise AssertionError("shared cancellation boundary")
            on.clear_prefix_cache()
            events = first+list(on.events())
            b_events = [e for e in events if e["request_id"]=="shared-b"]
            if [e["token_id"] for e in b_events]!=golden["base"] or b_events[-1]["finish_reason"]!="length":
                raise AssertionError("active lease invalidated by cache clear/cancel")
            if on.stats()["kv_live_blocks"]:
                raise AssertionError("cleared shared request leaked pages")
            checks.append({"backend":backend,"case":"concurrent-sharing/cancel/cache-clear","middle_stats":middle})

            # Warm both arms outside timing; all reference compute is complete.
            generate(off,"warmup-off",reused=0);generate(on,"warmup-on",reused=0)
            for run in range(runs):
                # Rotate the off arm before/after the dependent cold -> warm pair.
                if run%2==0:
                    s=generate(off,"sample-off-"+str(run),reused=0);s.update(backend=backend,mode="off",run=run+1);samples.append(s)
                on.clear_prefix_cache()
                for mode,expected in (("cold",0),("warm",skip)):
                    s=generate(on,"sample-"+mode+"-"+str(run),reused=expected)
                    s.update(backend=backend,mode=mode,run=run+1);samples.append(s)
                if run%2:
                    s=generate(off,"sample-off-"+str(run),reused=0);s.update(backend=backend,mode="off",run=run+1);samples.append(s)
            on.clear_prefix_cache()
            if on.stats()["kv_live_blocks"] or off.stats()["kv_live_blocks"]:
                raise AssertionError("cache clear did not return all pages")

        # Capacity LRU and automatic eviction under physical pool pressure.
        for capacity, pages in ((2,12),(9,10)):
            with NativeEngine(worker,root/"model.mpf",backend,128,2,16,8,4,pages,capacity) as engine:
                generate(engine,"lru-prime",reused=0)
                generate(engine,"lru-repeat",reused=min(skip,capacity*4))
                generate(engine,"lru-other","different",0)
                after = engine.stats()
                if not after["prefix_evictions"] or after["kv_cached_blocks"]>capacity:
                    raise AssertionError("bounded LRU/pressure eviction missing")
                engine.clear_prefix_cache()
                if engine.stats()["kv_live_blocks"]:
                    raise AssertionError("LRU cleanup leaked pages")
                checks.append({"backend":backend,"case":"capacity-pressure","capacity":capacity,"stats":after})

        # Cache leases cannot evict active leases: fail safely when those alone
        # exhaust capacity, with exactly one terminal event per unfinished ID.
        with NativeEngine(worker,root/"model.mpf",backend,128,2,16,8,4,10,9) as engine:
            generate(engine,"exhaustion-prime",reused=0)
            for key in ("a","b","waiting"):
                engine.submit(key,prompt,4,None)
            events = list(engine.events());stats = engine.stats()
            terminals = [e for e in events if e["finish_reason"]]
            if len(terminals)!=3 or any(e["finish_reason"]!="error" for e in terminals):
                raise AssertionError("shared exhaustion terminal events")
            if stats["kv_live_blocks"] or stats["kv_cached_blocks"] or stats["block_table_device_bytes"]:
                raise AssertionError("failed engine leaked cache leases/pages")
            checks.append({"backend":backend,"case":"active-capacity-failure-cleanup","stats":stats})
        summaries[backend] = {}
        for mode in ("off","cold","warm"):
            group=[s for s in samples if s["backend"]==backend and s["mode"]==mode]
            summaries[backend][mode]={k:{"median":statistics.median(s[k] for s in group),
                "min":min(s[k] for s in group),"max":max(s[k] for s in group)} for k in ("ttft_ms","elapsed_ms","forward_tokens","reused_tokens")}
    source=Path(__file__).resolve().parents[1]
    return {"date":datetime.now(timezone(timedelta(hours=8))).date().isoformat(),"passed":True,
            "model":json.loads((root/"manifest.json").read_text()),"native_model_sha256":sha256(root/"model.mpf"),
            "worker_sha256":{k:sha256(v) for k,v in workers.items()},
            "source_sha256":{str(p.relative_to(source)):sha256(p) for p in source.rglob("*") if p.suffix in (".py",".cpp",".cu",".h")},
            "platform":platform.platform(),"python":sys.version,"command":sys.argv,
            "torch":torch.__version__,"transformers":transformers.__version__,
            "gpu_before":gpu_before,"gpu_after":gpu_snapshot(),"golden_tokens":golden,
            "checks":checks,"logit_checks":logit_checks,"summaries":summaries,"samples":samples,
            "tolerance":"abs(a-b) <= 2e-3 + 2e-4*abs(b), identical argmax",
            "protocol":"FP32, block 4, pool 64, cache 16; 37-token prompt/4 outputs, EOS disabled; one warm process per arm/backend; five paired flows by default",
            "measurement":"tokenized submit to controller event delivery; reference/stats/clear/startup/warmup/tokenizer outside; no SLA/independent-process confidence claim"}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir",type=Path,required=True)
    parser.add_argument("--cpu-worker",required=True)
    parser.add_argument("--cuda-worker")
    parser.add_argument("--runs",type=int,default=5)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();workers={"cpu":args.cpu_worker}
    if args.cuda_worker:workers["cuda"]=args.cuda_worker
    result=verify(args.model_dir,workers,args.runs)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({"passed":result["passed"],"checks":len(result["checks"]),"logit_checks":len(result["logit_checks"]),"summaries":result["summaries"]},indent=2))


if __name__=="__main__":
    main()
