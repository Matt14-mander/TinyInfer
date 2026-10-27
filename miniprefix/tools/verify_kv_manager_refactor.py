"""Exact before/after worker differential for KVManager extraction (no timing)."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import struct
import sys

if __package__ in (None, ""):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from miniprefix.bindings.python.client import NativeEngine
from miniprefix.tools.verify_real_model import sha256


def verify(root,workers):
    records=[]
    for backend,(before,after) in workers.items():
        print("comparing KVManager workers: "+backend,file=sys.stderr,flush=True)
        with NativeEngine(before,root/"model.mpf",backend,64,2,8,3,3,12,6) as old, \
             NativeEngine(after,root/"model.mpf",backend,64,2,8,3,3,12,6) as new:
            def stats(label):
                a,b=old.stats(),new.stats()
                if a!=b:raise AssertionError("stats changed at "+backend+"/"+label+": "+str({k:(a[k],b[k]) for k in a if a[k]!=b[k]}))
                records.append(dict(backend=backend,case=label,kind="stats",stats=b))
            def both(method,*args):
                a=getattr(old,method)(*args);b=getattr(new,method)(*args)
                if a!=b:raise AssertionError("result changed: "+backend+"/"+method)
                return b
            def events(label,method="step",*args):
                a=getattr(old,method)(*args);b=getattr(new,method)(*args)
                def normalized(values):return [{k:v for k,v in e.items() if k!="received_ns"} for e in values]
                if normalized(a)!=normalized(b):raise AssertionError("event trace changed: "+label)
                if method=="step" and old.last_step_stats()!=new.last_step_stats():raise AssertionError("step accounting changed")
                records.append(dict(backend=backend,case=label,kind="events",events=normalized(b)))
                stats(label)
                return b
            def forward(label,key,tokens):
                a=old.forward(key,tokens);b=new.forward(key,tokens)
                packed_a=struct.pack("<"+str(len(a))+"f",*a);packed_b=struct.pack("<"+str(len(b))+"f",*b)
                if packed_a!=packed_b:raise AssertionError("FP32 logits changed at "+label)
                records.append(dict(backend=backend,case=label,kind="logits",checked_logits=len(b),
                                    max_abs_error=0.0,logits_sha256=hashlib.sha256(packed_b).hexdigest()))
                stats(label)
            stats("startup")
            a=[1,504,3575];tail=[4649,314,7042,30,2,28,198]
            forward("first-page","a",a);forward("interleaved-owner","b",[28,21781,198,504])
            forward("fragmented-append","a",tail);a+=tail
            both("cache_prefix","a");both("release","b");stats("publish/release")
            target=a+[504,3575,282,30]
            skip=both("reuse_prefix","c",target)
            if skip!=9:raise AssertionError("expected shared three pages")
            stats("prefix-restore")
            forward("reused-hole-append","c",target[skip:]);both("release","a")
            both("clear_prefix_cache");stats("clear-with-active-owner")
            forward("cross-another-page","c",[504,30])
            for key,tokens in (("c",[old.vocabulary]),("exhausted",[1]*40)):
                errors=[]
                for engine in (old,new):
                    try:engine.forward(key,tokens)
                    except RuntimeError as error:errors.append(str(error))
                    else:raise AssertionError("invalid/capacity input was accepted")
                if errors[0]!=errors[1]:raise AssertionError("error diagnostic changed")
                stats("rejected-"+key)
            both("release","exhausted");both("release","c");stats("all-diagnostics-released")
            both("submit","seed",a,3,None)
            step=0
            while old.pending:
                events("seed/"+str(step));step+=1
                if step>100:raise AssertionError("seed did not terminate")
            both("submit","shared-a",a,4,None);both("submit","shared-b",a+[28,198],4,None)
            both("submit","waiting",a,3,None)
            events("shared-first-step");events("cancel-shared-owner","cancel","shared-a")
            both("clear_prefix_cache");stats("clear-cache-during-generation")
            step=0
            while old.pending:
                events("refill/"+str(step));step+=1
                if step>100:raise AssertionError("requests did not terminate")
            both("clear_prefix_cache");stats("final-clear")
            if new.stats()["kv_live_blocks"] or new.pending:raise AssertionError("differential leaked pages")
    source=Path(__file__).resolve().parents[1]
    return dict(date=datetime.now(timezone(timedelta(hours=8))).date().isoformat(),passed=True,
        native_model_sha256=sha256(root/"model.mpf"),
        before_worker_sha256={k:sha256(v[0]) for k,v in workers.items()},
        after_worker_sha256={k:sha256(v[1]) for k,v in workers.items()},
        source_sha256={str(p.relative_to(source)):sha256(p) for p in source.rglob("*") if p.suffix in (".py",".cpp",".cu",".h")},
        protocol="same inputs/order/config; exact parsed FP32 logit bytes, event/step traces, errors and all 25 STATS fields; no performance claim",
        records=records)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--model-dir",type=Path,required=True)
    p.add_argument("--before-cpu-worker",required=True);p.add_argument("--cpu-worker",required=True)
    p.add_argument("--before-cuda-worker");p.add_argument("--cuda-worker");p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();workers={"cpu":(args.before_cpu_worker,args.cpu_worker)}
    if bool(args.before_cuda_worker)!=bool(args.cuda_worker):p.error("both CUDA worker paths are required")
    if args.cuda_worker:workers["cuda"]=(args.before_cuda_worker,args.cuda_worker)
    result=verify(args.model_dir,workers);args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps(dict(passed=result["passed"],records=len(result["records"]),
        logits=sum(r.get("checked_logits",0) for r in result["records"]))))


if __name__=="__main__":main()
