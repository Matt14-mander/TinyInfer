"""Check prefix-hit suffix execution, chunk positions/budgets and actual logits.

Uses local SmolLM2 weights and an independent Transformers full-prefix oracle.
This is correctness/accounting acceptance; per-step stats IPC is not a benchmark.
"""
import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from miniprefix.bindings.python.client import NativeEngine
from miniprefix.tools.verify_real_model import sha256


def verify(root, workers):
    import numpy as np
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(4)
    tokenizer=AutoTokenizer.from_pretrained(root,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(root,local_files_only=True,
        torch_dtype=torch.float32,attn_implementation="eager").eval()
    shared=tokenizer.encode("Please read this passage carefully and answer the question. "*12,add_special_tokens=False)[:48]
    suffix=tokenizer.encode("The small library opens every morning. Readers can borrow books and ask questions. "*12,add_special_tokens=False)[:65]
    assert len(shared)==48 and len(suffix)==65
    seed=shared+[28]
    long=shared+suffix
    branch=long.copy();branch[17]=(branch[17]+17)%model.config.vocab_size
    prompts={"long":long,"aligned":shared,"full-hit":seed,"branch":branch}
    def reference(tokens):
        with torch.inference_mode():
            ids=torch.tensor([tokens],dtype=torch.long)
            return model(ids,attention_mask=torch.ones_like(ids),use_cache=False).logits[0,-1].numpy().copy()
    golden={}
    for name,prompt in prompts.items():
        prefix=list(prompt);output=[]
        for _ in range(6 if name=="full-hit" else 3):
            token=int(reference(prefix).argmax());output.append(token);prefix.append(token)
        golden[name]=output
    ends=[];cursor=48
    while cursor<len(long):
        cursor=min(len(long),cursor+(1,7,3,11)[len(ends)%4]);ends.append(cursor)
    oracle={end:reference(long[:end]) for end in ends}
    checks=[];logits=[]
    def close(actual,expected,label):
        actual=np.asarray(actual,dtype=np.float32);diff=np.abs(actual-expected)
        if (actual.shape!=expected.shape or not np.isfinite(actual).all() or
                not np.all(diff<=2e-3+2e-4*np.abs(expected)) or int(actual.argmax())!=int(expected.argmax())):
            raise AssertionError("incremental logits mismatch: "+label)
        logits.append({"case":label,"checked_logits":len(actual),"max_abs_error":float(diff.max())})
    for backend,worker in workers.items():
        print("checking incremental prefill: "+backend,file=sys.stderr,flush=True)
        config=dict(backend=backend,context=256,batch_size=2,token_budget=5,chunk_size=7,
                    block_tokens=16,kv_blocks=64)
        with NativeEngine(worker,root/"model.mpf",prefix_cache_blocks=8,**config) as engine:
            def prime():
                engine.clear_prefix_cache();engine.forward("seed",seed)
                engine.cache_prefix("seed");engine.release("seed")
            prime()
            reused=engine.reuse_prefix("diagnostic",long)
            if reused!=48:raise AssertionError("expected 48-token page-aligned prefix")
            cursor=reused
            for end in ends:
                close(engine.forward("diagnostic",long[cursor:end]),oracle[end],backend+"/suffix/"+str(end))
                cursor=end
            engine.release("diagnostic")

            for name,skip in (("long",48),("aligned",32),("full-hit",48),("branch",16)):
                prime();prompt=prompts[name];key="request-"+name;before=engine.stats()
                engine.submit(key,prompt,3,None)
                steps=[];output=[];prefilled=0;decoded=0;reused=0;first_step=None
                while engine.pending:
                    events=engine.step();stats=engine.last_step_stats()
                    steps.append({"stats":stats,"events":events})
                    if stats["prefill_tokens"]+stats["decode_tokens"]>5:
                        raise AssertionError("reused tokens consumed budget or suffix exceeded budget")
                    prefilled+=stats["prefill_tokens"];decoded+=stats["decode_tokens"];reused+=stats["prefix_reused_tokens"]
                    if stats["prefix_reused_tokens"]!=(skip if len(steps)==1 else 0):
                        raise AssertionError("prefix restore repeated during incremental prefill")
                    if prefilled<len(prompt)-skip and events:
                        raise AssertionError("output before the complete suffix was computed")
                    for event in events:
                        if first_step is None:first_step=len(steps)
                        if event["token_id"] is not None:output.append(event["token_id"])
                    if any(e["finish_reason"]=="error" for e in events):
                        raise AssertionError("suffix execution failed")
                after=engine.stats()
                if (output!=golden[name][:3] or prefilled!=len(prompt)-skip or decoded!=2 or reused!=skip
                        or first_step!=(len(prompt)-skip+4)//5):
                    raise AssertionError("suffix progress/first event/accounting mismatch: "+name)
                if after["forward_tokens"]-before["forward_tokens"]!=prefilled+decoded:
                    raise AssertionError("runner/engine forward token counts disagree")
                if backend=="cuda" and (after["token_uploaded_bytes"]-before["token_uploaded_bytes"]!=4*(prefilled+decoded)
                        or after["weight_uploaded_bytes"]!=before["weight_uploaded_bytes"]
                        or after["kv_uploaded_bytes"]!=before["kv_uploaded_bytes"]):
                    raise AssertionError("cached KV/input was re-uploaded")
                if events[-1]["finish_reason"]!="length":raise AssertionError("length boundary")
                checks.append({"backend":backend,"case":name,"prompt_tokens":len(prompt),"reused_tokens":skip,
                    "prefill_tokens":prefilled,"decode_tokens":decoded,"first_token_step":first_step,
                    "output_tokens":output,"steps":steps,"before":before,"after":after})

            prime();engine.submit("decoder",seed,6,None)
            initial=engine.step();decoder=[e["token_id"] for e in initial]
            engine.submit("interleaved",long,3,None)
            prefilled=decoded=0;output=[];steps=[]
            while engine.pending:
                events=engine.step();stats=engine.last_step_stats();steps.append({"stats":stats,"events":events})
                if stats["prefill_tokens"]+stats["decode_tokens"]>5:raise AssertionError("mixed budget")
                prefilled+=stats["prefill_tokens"];decoded+=stats["decode_tokens"]
                for e in events:
                    if e["token_id"] is not None:
                        if e["request_id"]=="decoder":decoder.append(e["token_id"])
                        else:
                            if prefilled!=65:raise AssertionError("early suffix event during mixed decode")
                            output.append(e["token_id"])
            if decoder!=golden["full-hit"] or output!=golden["long"] or prefilled!=65 or decoded!=7:
                raise AssertionError("interleaved decode/prefill output mismatch")
            if steps[0]["stats"]!={"prefix_reused_tokens":48,"prefill_tokens":4,"decode_tokens":1}:
                raise AssertionError("decode did not receive first token budget")
            checks.append({"backend":backend,"case":"decode-first-interleaving","steps":steps,
                           "prefill_tokens":prefilled,"decode_tokens":decoded})

            prime();before=engine.stats();engine.submit("cancel-prefill",long,3,None)
            for _ in range(2):
                if engine.step():raise AssertionError("partial suffix emitted output")
            cancel=engine.cancel("cancel-prefill");after=engine.stats()
            if len(cancel)!=1 or cancel[0]["finish_reason"]!="cancelled" or cancel[0]["token_id"] is not None:
                raise AssertionError("prefill cancellation boundary")
            if after["kv_cached_blocks"]!=before["kv_cached_blocks"] or after["live_sequences"]:
                raise AssertionError("incomplete suffix published or request leaked")
            engine.clear_prefix_cache()
            if engine.stats()["kv_live_blocks"] or engine.stats()["block_table_device_bytes"]:
                raise AssertionError("cancelled suffix pages leaked")
            checks.append({"backend":backend,"case":"cancel-mid-suffix","before":before,"after":after})
        # Cache-off control uses the same long prompt and independent oracle.
        with NativeEngine(worker,root/"model.mpf",**config) as engine:
            engine.submit("off",long,3,None);output=[];prefilled=decoded=reused=0
            while engine.pending:
                events=engine.step();stats=engine.last_step_stats()
                prefilled+=stats["prefill_tokens"];decoded+=stats["decode_tokens"];reused+=stats["prefix_reused_tokens"]
                output.extend(e["token_id"] for e in events if e["token_id"] is not None)
            if output!=golden["long"] or (prefilled,decoded,reused)!=(113,2,0):
                raise AssertionError("cache-off suffix control")
            checks.append({"backend":backend,"case":"cache-off","prefill_tokens":prefilled,"decode_tokens":decoded})
    source=Path(__file__).resolve().parents[1]
    return {"date":datetime.now(timezone(timedelta(hours=8))).date().isoformat(),"passed":True,
        "model":json.loads((root/"manifest.json").read_text()),"native_model_sha256":sha256(root/"model.mpf"),
        "worker_sha256":{k:sha256(v) for k,v in workers.items()},"torch":torch.__version__,"transformers":transformers.__version__,
        "source_sha256":{str(p.relative_to(source)):sha256(p) for p in source.rglob("*") if p.suffix in (".py",".cpp",".cu",".h")},
        "checks":checks,"logit_checks":logits,"prompts":prompts,"golden_tokens":golden,
        "scope":"FP32, block 16/pool 64/cache 8, token budget 5/chunk 7; correctness and accounting, not performance"}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--model-dir",type=Path,required=True)
    p.add_argument("--cpu-worker",required=True);p.add_argument("--cuda-worker");p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();workers={"cpu":args.cpu_worker}
    if args.cuda_worker:workers["cuda"]=args.cuda_worker
    result=verify(args.model_dir,workers);args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({"passed":result["passed"],"checks":len(result["checks"]),"logit_checks":len(result["logit_checks"]),
                      "max_abs_error":max(x["max_abs_error"] for x in result["logit_checks"])}))


if __name__=="__main__":main()
