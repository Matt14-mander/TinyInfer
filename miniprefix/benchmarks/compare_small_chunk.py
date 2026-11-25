"""Alternating paired workers on the fixed chunk=2 prefill sweep case."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

from mixed_prefill import run, p95
from prefill_sweep import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before")
    parser.add_argument("after")
    parser.add_argument("model",type=Path)
    parser.add_argument("--serial-before",action="store_true")
    parser.add_argument("--pairs",type=int,default=3)
    args = parser.parse_args()
    if args.pairs < 1:
        parser.error("--pairs must be positive")
    with args.model.open("rb") as source:
        digest = hashlib.file_digest(source,"sha256").hexdigest()
    if digest != "fe826f487690854ac806197b60edffc2c76c6c446eede6b188e47763deae7aef":
        raise ValueError("unexpected model digest")
    samples = {"before":[], "after":[]}
    jobs = requests(17,False)
    for i in range(args.pairs):
        expected = None
        for mode in (("after","before") if i%2 else ("before","after")):
            worker = args.before if mode=="before" else args.after
            batch = mode=="after" or not args.serial_before
            trace,sample = run(worker,args.model,batch,16,jobs,2,13)
            if expected is not None and expected != trace:
                raise AssertionError("per-step events differ")
            expected = trace
            samples[mode].append(sample)
    report = dict(model_sha256=digest,serial_before=args.serial_before,results={})
    for mode,runs in samples.items():
        path = Path(args.before if mode=="before" else args.after)
        report["results"][mode] = dict(worker_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            tokens_s=statistics.median(r["tokens_s"] for r in runs),
            short_p95_ms=p95([t for r in runs for t in r["gaps_ms"]["0"]]),samples=runs)
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
