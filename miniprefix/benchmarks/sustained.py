"""Open-loop CUDA serving benchmark. All arrivals are scheduled before the first STEP."""
import argparse
import gzip
import json
import math
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from miniprefix.benchmarks.baseline import gpu_snapshot, measure, percentile, prepare_workload
from miniprefix.benchmarks.partial_prefix import generate
from miniprefix.bindings.python.client import NativeEngine
from miniprefix.tools.verify_real_model import sha256


def arrivals(prompts, count, rate, seed):
    rng = random.Random(seed)
    gap = 1000 / rate
    # One arrival in each slot, with reproducible jitter and no completion feedback.
    return [dict(prompts[i % len(prompts)], request_id=f'r{i}',
                 arrival_ms=(i + rng.uniform(0, .8)) * gap) for i in range(count)]


def window_metrics(rows, start, end, batch):
    delivered = sum(start < t <= end for r in rows for t in r['token_times_ms'])
    changes = sorted([(r['arrival_ms'], 1) for r in rows] +
                     [(r['token_times_ms'][-1], -1) for r in rows])
    pending, last, busy = 0, start, 0.0
    for t, delta in changes:
        if t > end:
            break
        if t >= start:
            if pending > batch:
                busy += t - last
            last = t
        pending += delta
    if pending > batch:
        busy += end - last
    duration = end - start
    return delivered * 1000 / duration, min(1.0, busy / duration)


def metrics(sample, batch):
    rows = sample['requests']
    finishes = sorted(r['token_times_ms'][-1] for r in rows)
    lo, hi = len(rows) // 5, 4 * len(rows) // 5
    speed, backlog = window_metrics(rows, finishes[lo], finishes[hi], batch)
    return dict(interior_tok_s=speed, backlogged_fraction=backlog,
                drain_ms=finishes[-1]-rows[-1]['arrival_ms'],
                full_tok_s=sample['output_tokens_per_second'],
                step_ms=sum(s['end_ms']-s['start_ms'] for s in sample['steps']))


def summarize(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        header = json.loads(next(stream))
        runs = [json.loads(line) for line in stream]
    cells = []
    for rate in header['rates']:
        for mode in ('off', 'cold', 'hot'):
            group = [r for r in runs if (r['rate'], r['mode']) == (rate, mode)]
            if sorted(r['repeat'] for r in group) != list(range(header['repeats'])):
                raise ValueError('missing or duplicate repeat')
            for r in group:
                requests = r['requests']
                lo, hi = len(requests) // 5, 4 * len(requests) // 5
                speed, backlog = window_metrics(requests, requests[lo]['arrival_ms'],
                                                requests[hi]['arrival_ms'], header['config']['batch_size'])
                r['metrics'].update(arriving_tok_s=speed, arriving_backlogged_fraction=backlog)
            rows = [q for r in group for q in r['requests']]
            cell = dict(rate=rate, mode=mode)
            for key in ('ttft_ms', 'tpot_ms', 'latency_ms', 'controller_submit_delay_ms'):
                values = [q[key] for q in rows]
                cell[key] = {label: percentile(values, p) for label, p in
                             (('p50', .5), ('p95', .95), ('p99', .99))}
            gaps = [b - a for q in rows for a, b in zip(q['token_times_ms'], q['token_times_ms'][1:])]
            cell['itl_p99_ms'] = percentile(gaps, .99)
            for key in group[0]['metrics']:
                cell[key] = statistics.median(r['metrics'][key] for r in group)
            cell['interior_range'] = [min(r['metrics']['interior_tok_s'] for r in group),
                                      max(r['metrics']['interior_tok_s'] for r in group)]
            speed = [r['metrics']['arriving_tok_s'] for r in group]
            cell['arriving_range'] = [min(speed), max(speed)]
            cell['first_four_ttft_ms'] = statistics.median(q['ttft_ms'] for r in group for q in r['requests'][:4])
            quarter = header['requests'] // 4
            cell['ttft_growth_ms'] = statistics.median(
                statistics.median(q['ttft_ms'] for q in r['requests'][-quarter:]) -
                statistics.median(q['ttft_ms'] for q in r['requests'][:quarter]) for r in group)
            cell['reuse_per_request'] = sum(r['counters']['prefix_reused_tokens'] for r in group) / len(rows)
            if header['profile']:
                for key in ('admission_plan_ns', 'forward_download_ns', 'host_forward_ns', 'stream_forward_ns'):
                    cell[key + '_pct_step'] = statistics.median(
                        100 * r['execution_ns'][key] / (1e6 * r['metrics']['step_ms']) for r in group)
            cells.append(cell)
    return cells


def counters(engine):
    return {**engine.stats(), **engine.prefill_batch_stats(),
            **{'decode_' + k: v for k, v in engine.decode_batch_stats().items()}}


def run(args):
    workload, oracle = prepare_workload(args.model_dir)
    unique = list({tuple(r['prompt_tokens']): r for r in workload}.values())
    model = args.model_dir / 'model.mpf'
    config = dict(context=256, batch_size=4, token_budget=32, chunk_size=16, block_tokens=16, kv_blocks=128)
    header = dict(config=config, prefix_capacity=48, rates=args.rates, requests=args.requests,
                  repeats=args.repeats, seed=args.seed, oracle=oracle, prompts=unique,
                  model_sha256=sha256(model), worker_sha256=sha256(args.worker), gpu=gpu_snapshot(),
                  profile=args.profile)
    # Each line after the header is an independently verified run.
    with gzip.open(args.output, 'wt', encoding='utf-8') as output:
        print(json.dumps(header), file=output)
        for rep in range(args.repeats):
            cells = [(rate, mode) for rate in args.rates for mode in ('off', 'cold', 'hot')]
            random.Random(args.seed+rep).shuffle(cells)
            for rate, mode in cells:
                requests = arrivals(unique, args.requests, rate, args.seed+rep)
                capacity = 0 if mode == 'off' else header['prefix_capacity']
                with NativeEngine(args.worker, model, backend='cuda', prefix_cache_blocks=capacity, **config) as e:
                    warm = [dict(r, request_id=f'w{i}', arrival_ms=0) for i, r in enumerate(workload)]
                    measure(e, warm)
                    e.clear_prefix_cache()
                    if mode == 'hot':
                        for i, r in enumerate(unique):
                            if generate(e, f'prime{i}', r['prompt_tokens']) != r['expected_tokens']:
                                raise RuntimeError('incorrect hot-cache priming output')
                    if args.profile:
                        e.set_execution_timing(True)
                    before = counters(e)
                    sample = measure(e, requests)
                    after = counters(e)
                    delta = {k: after[k] - v for k, v in before.items()}
                    expected = sum(len(r['prompt_tokens']) + r['max_new_tokens'] - 1 for r in requests)
                    if delta['forward_tokens'] + delta['prefix_reused_tokens'] != expected:
                        raise RuntimeError('forward and reuse token accounting mismatch')
                    if (mode == 'off' and delta['prefix_reused_tokens']) or delta['weight_uploaded_bytes'] or delta['kv_uploaded_bytes']:
                        raise RuntimeError('unexpected prefix reuse or device upload')
                    if after['live_sequences'] or after['block_table_device_bytes']:
                        raise RuntimeError('live request resources after drain')
                    e.clear_prefix_cache()
                    if e.stats()['kv_live_blocks']:
                        raise RuntimeError('live KV pages after cache clear')
                    record = dict(repeat=rep, rate=rate, mode=mode, metrics=metrics(sample, config['batch_size']),
                                  counters=delta, requests=sample['requests'],
                                  steps=[[s['start_ms'], s['end_ms']] for s in sample['steps']])
                    if args.profile:
                        profile = e.execution_stats()
                        if not (profile['forwards'] and profile['stream_forward_ns'] and profile['admission_plan_ns']):
                            raise RuntimeError('missing native timing samples')
                        record['execution_ns'] = profile
                    print(json.dumps(record, separators=(',', ':')), file=output, flush=True)
                    print(rep, rate, mode, round(record['metrics']['interior_tok_s'], 1), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summarize', type=Path, help='print metrics from a completed .jsonl.gz run')
    p.add_argument('--worker', type=Path)
    p.add_argument('--model-dir', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--rates', type=float, nargs='+', default=[10, 24, 48, 96])
    p.add_argument('--requests', type=int, default=96)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--seed', type=int, default=1741)
    p.add_argument('--profile', action='store_true')
    args = p.parse_args()
    if args.summarize:
        print(json.dumps(summarize(args.summarize), indent=2))
        sys.exit(0)
    if not all((args.worker, args.model_dir, args.output)):
        p.error('measurement requires --worker, --model-dir and --output')
    if args.requests < 8 or args.repeats < 1 or any(not math.isfinite(r) or r <= 0 for r in args.rates):
        p.error('require at least eight requests, positive rates and repeats')
    if len(set(args.rates)) != len(args.rates):
        p.error('rates must be distinct')
    run(args)
