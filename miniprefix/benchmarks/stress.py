"""Run bounded native soak tests and summarize measured wall-clock components."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import time


def percentile(values, fraction):
    return sorted(values)[max(0, math.ceil(len(values)*fraction)-1)] if values else None


def summarize(rows):
    if not rows or rows[-1].get("passed") is not True:
        raise ValueError("missing successful completion marker")
    waves=rows[:-1]
    if not waves or [w["wave"] for w in waves]!=list(range(len(waves))):
        raise ValueError("missing/reordered samples")
    if rows[-1]["waves"]!=len(waves) or rows[-1]["requests"]!=8*len(waves):
        raise ValueError("inconsistent completion counts")
    if rows[-1]["tokens"]!=sum(w["tokens"] for w in waves):
        raise ValueError("inconsistent token counts")
    for wave in waves:
        reasons=[r["reason"] for r in wave["requests"]]
        if reasons!=[2,1,1,1,1,2,0,3]:
            raise ValueError("unexpected terminal outcomes")
        if wave["wall_ms"]<=0 or wave["step_ms"]<0 or wave["step_ms"]>wave["wall_ms"]:
            raise ValueError("invalid timings")
    # First ten waves remain in raw data; exclude them only from steady-state summary.
    steady=waves[10:] or waves
    timing={}
    for wave in steady:
        for key,value in wave["timings"].items():
            entry=timing.setdefault(key,dict(ms=0,calls=0))
            entry["ms"]+=value["ms"];entry["calls"]+=value["calls"]
    wall=sum(w["wall_ms"] for w in steady)
    for value in timing.values():
        value["wall_percent"]=100*value["ms"]/wall
        value["mean_call_ms"]=value["ms"]/value["calls"] if value["calls"] else None
    requests=[r for w in steady for r in w["requests"]]
    completed=[r for r in requests if r["reason"] in (0,1)]
    ttft=[r["ttft_ms"] for r in completed]
    queue=[r["queue_ms"] for r in completed]
    tpot=[(r["last_ms"]-r["ttft_ms"])/(r["output_tokens"]-1) for r in completed if r["output_tokens"]>1]
    n=min(20,max(1,len(steady)//4))
    memory=lambda sample:statistics.median(w["private_bytes"] for w in sample)
    return dict(passed=True,waves=len(waves),requests=rows[-1]["requests"],elapsed_seconds=rows[-1]["elapsed_ms"]/1000,
                steady_waves=len(steady),excluded_warmup_waves=len(waves)-len(steady),
                steady_output_tokens_per_second=sum(w["tokens"] for w in steady)*1000/wall,
                ttft_p50_ms=percentile(ttft,.5),ttft_p95_ms=percentile(ttft,.95),queue_p95_ms=percentile(queue,.95),
                mean_request_tpot_ms=statistics.mean(tpot),timing=timing,
                unattributed_wall_percent=100*(wall-sum(v["ms"] for v in timing.values()))/wall,
                initial_private_bytes=memory(steady[:n]),final_private_bytes=memory(steady[-n:]),
                private_growth_bytes=memory(steady[-n:])-memory(steady[:n]),
                max_private_bytes=max(w["private_bytes"] for w in waves),
                max_rss_bytes=max(w["rss_bytes"] for w in waves),
                workspace_sizes=sorted({w["workspace_bytes"] for w in steady}),
                reused_tokens=sum(w["reused_tokens"] for w in waves),evictions=sum(w["evictions"] for w in waves))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--executable",type=Path,required=True);p.add_argument("--model",type=Path,required=True)
    p.add_argument("--backend",choices=("cpu","cuda"),required=True)
    p.add_argument("--seconds",type=float,default=600);p.add_argument("--min-waves",type=int,default=128)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if not math.isfinite(a.seconds) or a.seconds<=0 or a.min_waves<1:p.error("positive finite bounds required")
    a.output.parent.mkdir(parents=True,exist_ok=True)
    executable_sha=sha(a.executable);model_sha=sha(a.model)
    command=[str(a.executable.resolve()),str(a.model.resolve()),a.backend,str(a.seconds),str(a.min_waves)]
    start=time.monotonic()
    # The entire run, including reference setup, is bounded. Keep partial output on timeout/failure.
    with a.output.with_suffix(".jsonl").open("w",encoding="utf-8") as out, a.output.with_suffix(".stderr.log").open("w",encoding="utf-8") as err:
        try:
            result=subprocess.run(command,stdout=out,stderr=err,text=True,timeout=max(a.seconds*3,180))
            returncode=result.returncode
        except subprocess.TimeoutExpired:
            returncode=-1;err.write("\nwatchdog timeout")
    raw=a.output.with_suffix(".jsonl").read_text(encoding="utf-8")
    error=a.output.with_suffix(".stderr.log").read_text(encoding="utf-8")
    if returncode:raise RuntimeError(f"native soak failed ({returncode}): {error}")
    summary=summarize([json.loads(line) for line in raw.splitlines()])
    if summary["elapsed_seconds"]<a.seconds or summary["waves"]<a.min_waves:raise ValueError("run too short")
    if sha(a.executable)!=executable_sha or sha(a.model)!=model_sha:raise ValueError("binary/model changed during run")
    report=dict(backend=a.backend,command=command,platform=platform.platform(),
                executable_sha256=executable_sha,model_sha256=model_sha,raw_sha256=sha(a.output.with_suffix(".jsonl")),
                total_process_seconds=time.monotonic()-start,summary=summary,
                scope="native synchronous wall time; excludes worker IPC/tokenizer; same-backend uncached oracle; first 10 waves excluded only from steady summary; no speedup claim")
    a.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report))

if __name__=="__main__":main()
