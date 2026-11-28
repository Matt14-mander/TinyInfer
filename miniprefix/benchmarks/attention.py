"""Paired old/new paged attention stream-time sweep, with numerical validation."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from miniprefix.benchmarks.baseline import gpu_snapshot


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args()
    if args.repeats<1:parser.error('repeats must be positive')
    header=dict(binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),gpu=gpu_snapshot(),
                repeats=args.repeats,timing='CUDA event stream time / 20 launches; excludes uploads, oracle, downloads; may include host launch gaps',
                shapes=144,reference='pre-optimization 256-thread FP32 kernel and dense FP64 oracle on first/last query rows')
    groups={}
    with gzip.open(args.output,'wt',encoding='utf-8') as output:
        output.write(json.dumps(header)+'\n')
        for repeat in range(args.repeats):
            run=subprocess.run([str(args.binary),'--require-device'],capture_output=True,text=True,check=True,timeout=300)
            rows=[json.loads(line) for line in run.stdout.splitlines()]
            if len(rows)!=144:raise ValueError('incomplete sweep')
            seen=set()
            for row in rows:
                key=tuple(row[k] for k in ('dim','length','requests','chunk'))
                if key in seen or not row['passed'] or len(row['before_us'])!=3 or len(row['after_us'])!=3:
                    raise ValueError('invalid case')
                seen.add(key);row['repeat']=repeat
                groups.setdefault(key,[]).append(row)
                output.write(json.dumps(row)+'\n')
            output.flush();print(f'Validated sweep {repeat+1}/{args.repeats}',flush=True)
    cells=[]
    for key,rows in sorted(groups.items()):
        before=[statistics.median(r['before_us']) for r in rows]
        after=[statistics.median(r['after_us']) for r in rows]
        cells.append(dict(zip(('dim','length','requests','chunk'),key),
                          before_us=statistics.median(before),after_us=statistics.median(after),
                          speedup=statistics.median(a/b for a,b in zip(before,after)),
                          before_range=[min(before),max(before)],after_range=[min(after),max(after)],
                          max_difference=max(r['max_difference'] for r in rows),
                          oracle_error=max(r['oracle_error'] for r in rows)))
    args.output.with_suffix('.summary.json').write_text(json.dumps(dict(header=header,cells=cells),indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
