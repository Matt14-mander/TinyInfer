"""Interleave old/new CUDA workers using the existing partial-prefix baseline."""
import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from miniprefix.benchmarks.partial_prefix import workloads, generate
from miniprefix.bindings.python.client import NativeEngine


def summarize(rows):
    summaries = []
    for case in dict.fromkeys(r['case'] for r in rows):
        for mode in ('off', 'cold', 'hot'):
            values = {}
            for version in ('before', 'after'):
                group = [r for r in rows if (r['case'], r['mode'], r['version']) == (case, mode, version)]
                timed = [r for r in group if not r['profiling']]
                profiles = [r for r in group if r['profiling']]
                values[version] = {k: dict(median=statistics.median(r[k] for r in timed),
                                          min=min(r[k] for r in timed), max=max(r[k] for r in timed))
                                   for k in ('ttft_ms', 'tpot_ms', 'output_tokens_per_second', 'elapsed_ms')}
                values[version]['copy_ms'] = dict(median=statistics.median(r['costs']['cow_copy_ns']/1e6 for r in profiles),
                                                  samples=[r['costs']['cow_copy_ns']/1e6 for r in profiles])
                values[version]['cow_bytes'] = group[0]['costs']['cow_bytes']
                values[version]['cow_pages'] = group[0]['costs']['cow_pages']
            # Within-round comparisons, not a ratio of separately pooled medians.
            pairs = {}
            for r in rows:
                if r['case'] == case and r['mode'] == mode and not r['profiling']:
                    pairs.setdefault(r['repeat'], {})[r['version']] = r
            ratios = {k: [p['before'][k]/p['after'][k] for p in pairs.values()]
                      for k in ('ttft_ms', 'tpot_ms')}
            ratios['throughput'] = [p['after']['output_tokens_per_second']/p['before']['output_tokens_per_second'] for p in pairs.values()]
            summaries.append(dict(case=case, mode=mode, **values, paired=ratios,
                                  paired_median={k: statistics.median(v) for k, v in ratios.items()}))
    return summaries


def run(a):
    if a.repeats < 3 or a.profiles < 1:
        raise ValueError('need >=3 measured repeats and >=1 profiling repeat')
    config = dict(context=128, batch_size=1, token_budget=16, chunk_size=16, block_tokens=16, kv_blocks=64)
    model = a.model_dir/'model.mpf'
    cases, golden = workloads(a.cpu_worker, model, config)
    workers = dict(before=a.before, after=a.after)
    rows = []
    for profiling, repeats in ((False, a.repeats), (True, a.profiles)):
        for repeat in range(repeats):
            modes = ['off', 'cold', 'hot']
            modes = modes[repeat % 3:]+modes[:repeat % 3]
            for mode in modes:
                versions = ['before', 'after'] if repeat % 2 == 0 else ['after', 'before']
                for version in versions:
                    with NativeEngine(workers[version], model, backend='cuda',
                                      prefix_cache_blocks=0 if mode == 'off' else 32, **config) as e:
                        e.set_cost_timing(profiling)
                        ordered = cases[repeat % 4:]+cases[:repeat % 4]
                        for name, prime, query, expected in ordered:
                            assert generate(e, name+'-warmup', query) == golden[name]
                            e.clear_prefix_cache()
                            if mode == 'hot':
                                generate(e, name+'-prime', prime)
                            r = generate(e, name+'-measure', query, True)
                            assert r['output'] == golden[name]
                            assert r['reused_tokens'] == (expected if mode == 'hot' else 0)
                            c = r['costs']
                            assert c['lookup_calls'] == 1 and c['publish_calls'] == 8
                            if profiling:
                                assert c['cow_timed_copies'] == c['cow_pages']
                            else:
                                assert c['cow_copy_ns'] == c['publish_ns'] == c['lookup_ns'] == c['cow_timed_copies'] == 0
                            if mode == 'off':
                                assert c['cow_bytes'] == c['cow_pages'] == 0
                            r.update(version=version, repeat=repeat, profiling=profiling, mode=mode, case=name)
                            rows.append(r)
                            e.clear_prefix_cache();assert e.stats()['kv_live_blocks'] == 0
                    a.output.write_text(json.dumps(dict(complete=False, samples=rows), indent=2)+'\n')
                    print(f'{profiling=} {repeat=} {mode} {version} passed', flush=True)
    for name, _, _, _ in cases:
        for mode in ('off', 'cold', 'hot'):
            group = [r for r in rows if r['case'] == name and r['mode'] == mode]
            for field in ('cow_pages', 'cow_bytes', 'lookup_calls', 'publish_calls'):
                assert len({r['costs'][field] for r in group}) == 1, (name, mode, field)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    result = dict(complete=True, passed=True, config=config, repeats=a.repeats, profiles=a.profiles,
                  platform=platform.platform(), model_sha256=sha(model),
                  worker_sha256={k: sha(v) for k, v in workers.items()}, cpu_worker_sha256=sha(a.cpu_worker),
                  script_sha256=sha(__file__), baseline_script_sha256=sha(Path(__file__).with_name('partial_prefix.py')),
                  gpu=subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv'], text=True),
                  samples=rows, summaries=summarize(rows))
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, samples=len(rows))))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', type=Path, required=True);p.add_argument('--cpu-worker', required=True)
    p.add_argument('--before', required=True);p.add_argument('--after', required=True)
    p.add_argument('--repeats', type=int, default=6);p.add_argument('--profiles', type=int, default=3)
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())
