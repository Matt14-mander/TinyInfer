"""Compare complete protocol-v3 stdout/stderr and exit status before/after refactoring."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sha(data):
    return hashlib.sha256(data).hexdigest()


def verify(before, after, model, backend):
    commands = [
        "", "step", "UNKNOWN 1", "CANCEL", "CANCEL 2147483648",
        "SUBMIT 0 0 -2", "SUBMIT 0 0 -2 0", "SUBMIT 1 1 -2 0",
        "SUBMIT 1 1 -2 1", "SUBMIT 1 1 -2 1 0", "FORWARD 0 0",
        "REUSE 0 1", "REUSE 0 1 0", "FORWARD 1 131073", "FORWARD 1 -1",
        "STATS ignored", "STEP ignored", "STEP_STATS",
        "FORWARD 1 1 65536", "FORWARD 1 3 1 504 3575", "CACHE 1",
        "REUSE 2 4 1 504 3575 2", "FORWARD 2 1 2", "REUSE 2 1 1",
        "SUBMIT +1 2 -1 2 1 1 trailing", "SUBMIT 2 1 -1 1 1", "STEP",
        "STEP_STATS", "CANCEL 1x", "CANCEL 2", "STEP", "CACHE -2",
        "RELEASE 0", "RELEASE 1", "RELEASE 1", "FORWARD 1 1 1",
        "CLEAR_PREFIX", "STATS", "QUIT ignored", "UNKNOWN after quit",
    ]
    cases = {
        "mixed-and-quit": "\n".join(commands) + "\n",
        "eof-with-pending-request": "SUBMIT 1 3 -1 3 1 1 1\nSTEP\nSTEP_STATS",
    }
    records = []
    for name, payload in cases.items():
        results = []
        for worker in (before, after):
            args = [str(worker.resolve()), str(model.resolve()), backend, "64", "2", "8", "3", "3", "12", "6"]
            results.append(subprocess.run(args, input=payload.encode(), capture_output=True, timeout=180))
        old, new = results
        if (old.returncode, old.stdout, old.stderr) != (new.returncode, new.stdout, new.stderr):
            raise AssertionError(f"protocol changed: {backend}/{name}")
        if new.returncode or not new.stdout.startswith(b"READY MINIPREFIX 3 "):
            raise AssertionError(f"worker did not execute: {backend}/{name}")
        records.append(dict(backend=backend, case=name, returncode=new.returncode,
                            stdout_sha256=sha(new.stdout), stdout_bytes=len(new.stdout),
                            response_lines=len(new.stdout.splitlines()), stderr_sha256=sha(new.stderr)))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-worker", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = verify(args.before_worker, args.worker, args.model, args.backend)
    result = dict(passed=True, before_sha256=sha(args.before_worker.read_bytes()),
                  after_sha256=sha(args.worker.read_bytes()), model_sha256=sha(args.model.read_bytes()), records=records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(passed=True, records=len(records))))


if __name__ == "__main__":
    main()
