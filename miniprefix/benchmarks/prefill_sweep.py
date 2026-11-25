"""One-factor paired sweep with the arrival trace from the scheduler regression."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

from mixed_prefill import run, p95
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from verify_batch_pressure import workload


def requests(seed, extra_long):
    jobs = workload(seed)["jobs"][:-1]
    return [(j["arrival"], j["id"], j["prompt"]*4 if extra_long and i in (4,5) else j["prompt"],
             j["limit"]) for i,j in enumerate(jobs)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("worker")
    p.add_argument("model", type=Path)
    p.add_argument("--pairs", type=int, default=2)
    args = p.parse_args()
    if args.pairs < 1:
        p.error("--pairs must be positive")
    with args.model.open("rb") as src:
        digest = hashlib.file_digest(src,"sha256").hexdigest()
    if digest != "fe826f487690854ac806197b60edffc2c76c6c446eede6b188e47763deae7aef":
        raise ValueError("unexpected FP32 model")
    report = dict(sha256=digest,seed=17,pairs=args.pairs,results=[])
    for label,extra_long,chunk,budget in (("base",False,4,13),("more-long",True,4,13),
                                         ("small-chunk",False,2,13),("tight-budget",False,4,4)):
        jobs = requests(17,extra_long)
        samples = {False:[],True:[]}
        for pair in range(args.pairs):
            reference = None
            for mode in (bool(pair%2),not bool(pair%2)):
                events, sample = run(args.worker,args.model,mode,16,jobs,chunk,budget)
                if reference is not None and events != reference:
                    raise AssertionError(f"events differ in {label}")
                reference = events
                samples[mode].append(sample)
        for mode,runs in samples.items():
            report["results"].append(dict(case=label,chunk=chunk,budget=budget,batched=mode,
                lengths=[len(j[2]) for j in jobs],samples=runs,
                tokens_s=statistics.median(s["tokens_s"] for s in runs),
                short_p95_ms=p95([gap for s in runs for gap in s["gaps_ms"]["0"]]),
                peak_mib=max(s["peak_mib"] for s in runs)))
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
