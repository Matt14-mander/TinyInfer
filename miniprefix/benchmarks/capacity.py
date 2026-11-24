"""Paired admission-policy sweep using the same native binary and arrival schedule."""
import argparse
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import random
import statistics
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from miniprefix.benchmarks.baseline import percentile, gpu_snapshot
from miniprefix.benchmarks.sustained import window_metrics


def summarize(lines):
    if not lines or not lines[-1].get('passed'):
        raise ValueError('incomplete native run')
    summary, rows = dict(lines[-1]), lines[:-1]
    if len(rows) != summary['count'] or [r['request'] for r in rows] != list(range(len(rows))):
        raise ValueError('missing or duplicated requests')
    for row in rows:
        times = row['token_times_ms']
        if (len(times) != 8 or times != sorted(times) or
                not row['arrival_ms'] <= row['submit_ms'] <= row['admitted_ms'] <= times[0] or
                row['capacity_wait_ms'] < 0 or row['capacity_wait_ms'] > row['admitted_ms']-row['submit_ms']+1e-6):
            raise ValueError('invalid request timeline')
        row.update(ttft_ms=times[0]-row['arrival_ms'], tpot_ms=(times[-1]-times[0])/7,
                   latency_ms=times[-1]-row['arrival_ms'], queue_ms=row['admitted_ms']-row['submit_ms'],
                   submit_delay_ms=row['submit_ms']-row['arrival_ms'])
    if summary['final_blocks'] or summary['final_committed'] or summary['peak_committed'] > summary['pool_pages']:
        raise ValueError('invalid capacity accounting')
    for name in ('ttft_ms', 'tpot_ms', 'latency_ms', 'queue_ms', 'capacity_wait_ms', 'submit_delay_ms'):
        summary[name] = {key: percentile([r[name] for r in rows], q)
                         for key, q in (('p50', .5), ('p95', .95), ('p99', .99))}
    lo, hi = len(rows)//5, 4*len(rows)//5
    finishes = sorted(r['token_times_ms'][-1] for r in rows)
    summary['arriving_tok_s'], summary['arriving_backlog'] = window_metrics(rows, rows[lo]['arrival_ms'], rows[hi]['arrival_ms'], 4)
    summary['service_tok_s'], summary['service_backlog'] = window_metrics(rows, finishes[lo], finishes[hi], 4)
    summary['full_tok_s'] = len(rows)*8*1000/summary['duration_ms']
    summary['wait_attempts'] = sum(r['wait_attempts'] for r in rows)
    summary['waited_requests'] = sum(r['wait_attempts'] > 0 for r in rows)
    summary['drain_ms'] = finishes[-1]-rows[-1]['arrival_ms']
    summary['itl_p99_ms'] = percentile([b-a for r in rows for a, b in zip(r['token_times_ms'], r['token_times_ms'][1:])], .99)
    return summary


def aggregate(runs):
    cells = []
    keys = sorted({(r.get('variant', 'current'), r['pool_pages'], r['rate'], r['mode'], r['policy']) for r in runs})
    for variant, pages, rate, mode, policy in keys:
        group = [r for r in runs if (r.get('variant', 'current'), r['pool_pages'], r['rate'], r['mode'], r['policy']) == (variant, pages, rate, mode, policy)]
        cell = dict(variant=variant, rate=rate, mode=mode, policy=policy, repeats=len(group))
        for key, value in group[0].items():
            if isinstance(value, dict):
                cell[key] = {q: statistics.median(r[key][q] for r in group) for q in value}
            elif isinstance(value, (int, float)) and key not in ('seed', 'passed'):
                cell[key] = statistics.median(r[key] for r in group)
        cell['arriving_range'] = [min(r['arriving_tok_s'] for r in group), max(r['arriving_tok_s'] for r in group)]
        cell['service_range'] = [min(r['service_tok_s'] for r in group), max(r['service_tok_s'] for r in group)]
        cells.append(cell)
    return cells


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--before-binary', type=Path)
    parser.add_argument('--policies', nargs='+', choices=('conservative','shared'), default=['conservative','shared'])
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--backend', choices=('cpu', 'cuda'), required=True)
    parser.add_argument('--rates', type=float, nargs='+', default=[10, 40, 100])
    parser.add_argument('--requests', type=int, default=96)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--pages', type=int, default=20)
    parser.add_argument('--pools', type=int, nargs='+', help='optional interleaved physical-pool diagnostic')
    parser.add_argument('--cache-pages', type=int)
    parser.add_argument('--modes', nargs='+', choices=('off', 'cold', 'hot'), default=['off', 'cold', 'hot'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.requests < 8 or args.repeats < 1 or any(r <= 0 for r in args.rates):
        parser.error('invalid run bounds')
    header = dict(binary_sha256=digest(args.binary), model_sha256=digest(args.model), backend=args.backend,
                  rates=args.rates, requests=args.requests, repeats=args.repeats, pages=args.pages, pools=args.pools, cache_pages=args.cache_pages, modes=args.modes,
                  gpu=gpu_snapshot(), timing='native instrumented; planned arrival to event polling; model load/warmup/prime excluded',
                  config=dict(batch_size=4, token_budget=32, chunk=16, block_tokens=8, shared_prefix_tokens=64,
                              prompt_lengths=[65, 68, 71, 74], output_tokens=8))
    header['before_binary_sha256'] = digest(args.before_binary) if args.before_binary else None
    header['policies'] = args.policies
    variants = ['before', 'after'] if args.before_binary else ['current']
    summaries = []
    with gzip.open(args.output, 'wt', encoding='utf-8') as output:
        output.write(json.dumps(header)+'\n')
        for repeat in range(args.repeats):
            cases = list(itertools.product(variants, args.pools or [args.pages], args.rates, args.modes, args.policies))
            random.Random(918+repeat).shuffle(cases)
            for variant, pages, rate, mode, policy in cases:
                binary=args.before_binary if variant=='before' else args.binary
                result = subprocess.run([str(binary), str(args.model), args.backend, policy, mode,
                                         str(args.requests), str(rate), str(100+repeat), str(pages),
                                         str(args.cache_pages if args.cache_pages is not None else pages)],
                                        capture_output=True, text=True, timeout=900, check=True)
                rows = [json.loads(line) for line in result.stdout.splitlines()]
                summary = summarize(rows)
                summary['repeat'] = repeat
                summary['variant'] = variant
                summaries.append(summary)
                output.write(json.dumps(dict(summary=summary, requests=rows[:-1]))+'\n');output.flush()
                print(json.dumps(summary), flush=True)
    args.output.with_suffix('.summary.json').write_text(json.dumps(dict(header=header, cells=aggregate(summaries)), indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
