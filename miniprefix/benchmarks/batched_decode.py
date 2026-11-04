"""Paired serial/CUDA batched-decode comparison using baseline.measure."""
import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from miniprefix.benchmarks.baseline import measure, prepare_workload, gpu_snapshot
from miniprefix.benchmarks.partial_prefix import generate
from miniprefix.bindings.python.client import NativeEngine


def summarize(samples):
    result = []
    for batch in (1, 4):
        for mode in ('off', 'cold', 'hot'):
            selected = [s for s in samples if s['batch'] == batch and s['mode'] == mode]
            metrics = ('ttft_p50_ms', 'tpot_mean_ms', 'output_tokens_per_second')
            values = {v: {k: dict(median=statistics.median(s[k] for s in selected if s['version'] == v),
                                 min=min(s[k] for s in selected if s['version'] == v),
                                 max=max(s[k] for s in selected if s['version'] == v)) for k in metrics}
                      for v in ('before', 'after')}
            pairs = []
            for repeat in sorted({s['repeat'] for s in selected}):
                p = {s['version']: s for s in selected if s['repeat'] == repeat}
                pairs.append(dict(ttft_speedup=p['before']['ttft_p50_ms']/p['after']['ttft_p50_ms'],
                    tpot_speedup=p['before']['tpot_mean_ms']/p['after']['tpot_mean_ms'],
                    throughput_speedup=p['after']['output_tokens_per_second']/p['before']['output_tokens_per_second']))
            result.append(dict(batch=batch, mode=mode, **values, pairs=pairs,
                               paired_median={k: statistics.median(p[k] for p in pairs) for k in pairs[0]}))
    return result


def run(a):
    if a.repeats < 3:
        raise ValueError('at least three paired rounds required')
    workload, reference = prepare_workload(a.model_dir)
    # Fixed all-at-once arrivals isolate compute batching from wall-clock-driven
    # admission differences. Keep the existing four prompts and oracle outputs.
    workload = [dict(r, arrival_ms=0.0) for r in workload]
    config = dict(context=256, token_budget=32, chunk_size=16, block_tokens=16, kv_blocks=128)
    samples = []
    for repeat in range(a.repeats):
        modes = ['off', 'cold', 'hot'];modes = modes[repeat % 3:] + modes[:repeat % 3]
        for batch in ((1, 4) if repeat % 2 == 0 else (4, 1)):
            for mode in modes:
                pair = {}
                for version in (('before', 'after') if repeat % 2 == 0 else ('after', 'before')):
                    with NativeEngine(getattr(a, version), a.model_dir/'model.mpf', backend='cuda',
                                      batch_size=batch, prefix_cache_blocks=0 if mode == 'off' else 48, **config) as e:
                        unique = {tuple(r['prompt_tokens']): r['expected_tokens'] for r in workload}
                        def seed(label):
                            e.clear_prefix_cache()
                            if mode == 'hot':
                                for i, (prompt, expected) in enumerate(unique.items()):
                                    assert generate(e, f'{label}-prime-{i}', prompt) == expected
                        seed('warm')
                        measure(e, [dict(r, request_id='warm-'+r['request_id']) for r in workload])
                        seed('measure')
                        before, costs = e.stats(), e.cost_stats()
                        batches = e.decode_batch_stats() if version == 'after' else dict(batches=0, rows=0, fallbacks=0)
                        row = measure(e, workload)
                        after = e.stats();delta = {k: after[k]-before[k] for k in before}
                        decode = e.decode_batch_stats() if version == 'after' else batches
                        decode = {k: decode[k]-batches[k] for k in batches}
                        assert delta['forward_tokens'] == sum(len(r['prompt_tokens'])+7 for r in workload)-delta['prefix_reused_tokens']
                        assert after['live_sequences'] == after['block_table_device_bytes'] == 0
                        assert after['kv_live_blocks'] == after['kv_cache_only_blocks']
                        assert delta['weight_uploaded_bytes'] == delta['kv_uploaded_bytes'] == delta['device_workspace_bytes'] == 0
                        if version == 'after':
                            assert (decode['batches'] > 0) == (batch > 1)
                            assert decode['rows'] <= 56 and decode['fallbacks'] == 0
                        if mode == 'off':assert delta['prefix_reused_tokens'] == after['kv_live_blocks'] == 0
                        row.update(batch=batch, mode=mode, version=version, repeat=repeat, before=before, after=after,
                                   counters=delta, decode=decode, costs={k: v-costs[k] for k, v in e.cost_stats().items()})
                        pair[version] = row;samples.append(row)
                        e.clear_prefix_cache();assert e.stats()['kv_live_blocks'] == 0
                    a.output.write_text(json.dumps(dict(complete=False, samples=samples), indent=2)+'\n')
                    print(f'{repeat=} {batch=} {mode} {version} passed', flush=True)
                def trace(r):
                    return [[(e['request_id'], e['token_id'], e['finish_reason']) for e in step['events']] for step in r['steps']]
                assert trace(pair['before']) == trace(pair['after']), 'scheduler/event trace changed'
                for field in ('forward_tokens', 'prefix_hits', 'prefix_misses', 'prefix_reused_tokens', 'prefix_evictions'):
                    assert pair['before']['counters'][field] == pair['after']['counters'][field], field
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    result = dict(complete=True, passed=True, config=config, cache_blocks=48, repeats=a.repeats,
        scope='Eight fixed all-at-once requests; arrival-to-delivery TTFT, per-request mean TPOT, aggregate output throughput; no startup/warmup/prime/stats inside timing',
        model_sha256=sha(a.model_dir/'model.mpf'), workers={v: sha(getattr(a,v)) for v in ('before','after')},
        script_sha256=sha(__file__), baseline_sha256=sha(Path(__file__).with_name('baseline.py')),
        reference=reference, gpu=gpu_snapshot(), workload=workload, samples=samples, summaries=summarize(samples))
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, samples=len(samples))))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--before', required=True);p.add_argument('--after', required=True)
    p.add_argument('--repeats', type=int, default=5);p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())
