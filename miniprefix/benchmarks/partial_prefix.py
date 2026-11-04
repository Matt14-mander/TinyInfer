"""Paired partial-prefix cost/benefit experiment; model preparation is explicit."""
import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from miniprefix.bindings.python.client import NativeEngine


def metrics(start, timestamps):
    if len(timestamps) < 2 or timestamps[-1] <= start:
        raise ValueError("at least two output tokens and positive duration required")
    if timestamps[0] < start or any(b < a for a, b in zip(timestamps, timestamps[1:])):
        raise ValueError("non-monotonic delivery timestamps")
    return dict(ttft_ms=(timestamps[0]-start)/1e6,
                tpot_ms=(timestamps[-1]-timestamps[0])/(len(timestamps)-1)/1e6,
                elapsed_ms=(timestamps[-1]-start)/1e6,
                output_tokens_per_second=len(timestamps)*1e9/(timestamps[-1]-start))


def delta(after, before):
    return {k: after[k]-before[k] for k in before}


def write_report(result, path):
    lines = ['# 非整页前缀复用成本与收益', '',
             f"每个组合 {result['repeats']} 次无耗时插桩测量，另有 1 次 profiling；模型 FP32，页长 16，输出 8 tokens。",
             '时间包含 submit/step IPC 和事件交付，不包含加载、tokenization、warmup、prime、统计查询与 clear。',
             '吞吐是单请求输出 token/s，不代表并发饱和吞吐。CPU/CUDA 顺序执行；下表为中位数，原始 JSON 保留范围与逐次结果。', '',
             '|后端|负载|模式|复用 tokens|TTFT ms|TPOT ms|输出 tok/s|COW 页|COW MiB|复制 ms*|查找 µs*|发布 µs*|',
             '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in result['summaries']:
        m, c = s['metrics'], s['costs']
        lines.append(f"|{s['backend']}|{s['case']}|{s['mode']}|{s['reused_tokens']}|"
                     f"{m['ttft_ms']['median']:.3f}|{m['tpot_ms']['median']:.3f}|"
                     f"{m['output_tokens_per_second']['median']:.2f}|{c['cow_pages']}|"
                     f"{c['cow_bytes']/2**20:.3f}|{c['cow_copy_ns']/1e6:.3f}|"
                     f"{c['lookup_ns']/1e3:.2f}|{c['publish_ns']/1e3:.2f}|")
    lines += ['', '*耗时来自单独 profiling 轮次，不能与无插桩延迟直接相减；CUDA 复制区间由同一 stream 的 event 测量，可能包含提交间隙。',
              '查找/发布为 KVManager 主机时间，包含对应元数据及引用操作；复制字节为所有层 K+V 的有效部分，不包含未使用尾部。', '',
              '|后端|负载|off/hot TTFT 配对加速比中位数|hot/off 吞吐配对加速比中位数|',
              '|---|---|---:|---:|']
    for p in result['paired']:
        lines.append(f"|{p['backend']}|{p['case']}|{p['median']['ttft_speedup']:.3f}|{p['median']['throughput_speedup']:.3f}|")
    lines += ['', '加速比小于 1 表示退化。hot 的预填充/生成 prime 成本未计入，热缓存收益需要后续请求摊销。',
              'cold 相对 off 同时反映缓存发布、COW 和索引开销；本实验没有旧版整页缓存对照，不能将全部热缓存收益归因于非整页新增能力。',
              f"\n原始数据：[JSON]({path.with_suffix('.json').name})。模型 SHA256：`{result['model_sha256']}`。", '']
    path.write_text('\n'.join(lines), encoding='utf-8')


def generate(engine, key, prompt, observe=False):
    if observe:
        before, cost_before = engine.stats(), engine.cost_stats()
    start = time.perf_counter_ns()
    engine.submit(key, prompt, 8, None)
    events = []
    for _ in range(256):
        if not engine.pending:
            break
        events.extend(engine.step())
    if engine.pending:
        raise AssertionError("bounded progress failed")
    tokens = [e['token_id'] for e in events if e['token_id'] is not None]
    assert len(tokens) == 8 and [e['finish_reason'] for e in events if e['finish_reason']] == ['length']
    if not observe:
        return tokens
    times = [e['received_ns'] for e in events if e['token_id'] is not None]
    after, cost_after = engine.stats(), engine.cost_stats()
    reused = after['prefix_reused_tokens']-before['prefix_reused_tokens']
    forward = after['forward_tokens']-before['forward_tokens']
    assert forward == len(prompt)+7-reused
    assert after['live_sequences'] == after['block_table_device_bytes'] == 0
    assert after['kv_uploaded_bytes'] == before['kv_uploaded_bytes']
    assert after['weight_uploaded_bytes'] == before['weight_uploaded_bytes']
    return dict(output=tokens, prompt=prompt, received_ns=times, submit_ns=start,
                **metrics(start, times), reused_tokens=reused, forward_tokens=forward,
                costs=delta(cost_after, cost_before), before=before, after=after)


def workloads(cpu_worker, model, config):
    seed = ([1, 504, 3575, 282, 260, 905, 30, 198]*4)[:31]
    long_seed = (seed*3)[:65]
    # CPU no-cache reference, outside every measured interval.
    with NativeEngine(cpu_worker, model, prefix_cache_blocks=0, **config) as ref:
        history = generate(ref, 'history', seed)
        divergent = seed.copy(); divergent[17] += 7
        unrelated = seed.copy(); unrelated[0] += 7
        cases = [('identical-tail', seed, seed, 30),
                 ('multi-turn', seed, seed+history+[30, 504], 38),
                 ('in-page-branch', long_seed, long_seed[:17]+divergent[17:], 17),
                 ('zero-sharing', seed, unrelated, 0)]
        golden = {name: generate(ref, name, query) for name, _, query, _ in cases}
    return cases, golden


def run(args):
    model = args.model_dir/'model.mpf'
    workers = {'cpu': args.cpu_worker, 'cuda': args.cuda_worker}
    selected = args.backends.split(',')
    assert args.repeats >= 3 and all(b in workers for b in selected)
    config = dict(context=128, batch_size=1, token_budget=16, chunk_size=16,
                  block_tokens=16, kv_blocks=64)
    cases, golden = workloads(args.cpu_worker, model, config)
    rows = []
    for backend in selected:
        for profile in (False, True):
            for repeat in range(1 if profile else args.repeats):
                modes = ['off', 'cold', 'hot']
                modes = modes[repeat % 3:]+modes[:repeat % 3]
                for mode in modes:
                    with NativeEngine(workers[backend], model, backend=backend,
                                      prefix_cache_blocks=0 if mode == 'off' else 32, **config) as e:
                        e.set_cost_timing(profile)
                        ordered = cases[repeat % len(cases):]+cases[:repeat % len(cases)]
                        for name, prime, query, expected in ordered:
                            # Warm weights/workspace/kernel path; then discard its cache.
                            assert generate(e, name+'-warmup', query) == golden[name]
                            e.clear_prefix_cache()
                            if mode == 'hot':
                                generate(e, name+'-prime', prime)
                            record = generate(e, name+'-measured', query, True)
                            assert record['output'] == golden[name]
                            assert record['reused_tokens'] == (expected if mode == 'hot' else 0)
                            c = record['costs']
                            assert c['lookup_calls'] == 1 and c['publish_calls'] == 8
                            if mode == 'off':
                                assert c['cow_pages'] == c['cow_bytes'] == 0
                            if not profile:
                                assert c['lookup_ns'] == c['publish_ns'] == c['cow_copy_ns'] == c['cow_timed_copies'] == 0
                            else:
                                assert c['cow_timed_copies'] == c['cow_pages']
                            record.update(backend=backend, mode=mode, case=name, repeat=repeat,
                                          profiling=profile, prime_prompt=prime)
                            rows.append(record)
                            e.clear_prefix_cache()
                            assert e.stats()['kv_live_blocks'] == 0
                    print(f'{backend} profile={profile} repeat={repeat} {mode} passed', flush=True)
                    args.output.write_text(json.dumps({'complete': False, 'samples': rows}, indent=2)+'\n')
    summaries = []
    for backend in selected:
        for name, _, _, _ in cases:
            for mode in ('off', 'cold', 'hot'):
                group = [r for r in rows if (r['backend'], r['case'], r['mode'], r['profiling']) == (backend, name, mode, False)]
                prof = next(r for r in rows if (r['backend'], r['case'], r['mode'], r['profiling']) == (backend, name, mode, True))
                summary = {k: {'median': statistics.median(r[k] for r in group),
                               'min': min(r[k] for r in group), 'max': max(r[k] for r in group)}
                           for k in ('ttft_ms', 'tpot_ms', 'output_tokens_per_second', 'elapsed_ms')}
                summaries.append(dict(backend=backend, case=name, mode=mode, metrics=summary,
                                      reused_tokens=group[0]['reused_tokens'], forward_tokens=group[0]['forward_tokens'],
                                      costs=prof['costs']))
    paired = []
    for backend in selected:
        for name, _, _, _ in cases:
            ratios = []
            for repeat in range(args.repeats):
                pair = {r['mode']: r for r in rows if r['backend'] == backend and r['case'] == name and r['repeat'] == repeat and not r['profiling']}
                ratios.append(dict(ttft_speedup=pair['off']['ttft_ms']/pair['hot']['ttft_ms'],
                                   throughput_speedup=pair['hot']['output_tokens_per_second']/pair['off']['output_tokens_per_second']))
            paired.append(dict(backend=backend, case=name, ratios=ratios,
                               median={k: statistics.median(r[k] for r in ratios) for k in ratios[0]}))
    def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    try:
        gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        gpu = 'unavailable'
    result = dict(complete=True, passed=True, config=config, repeats=args.repeats, output_tokens=8,
                  cache_capacity=32, model_sha256=sha(model), workers={b: sha(workers[b]) for b in selected},
                  script_sha256=sha(__file__), platform=platform.platform(), python=sys.version,
                  gpu=gpu, command=sys.argv, golden=golden, samples=rows, summaries=summaries, paired=paired,
                  scope='Sequential single-request worker IPC delivery latency; no tokenization/model load/priming; profiling runs excluded from latency summaries; not saturated serving throughput')
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    write_report(result, args.output.with_suffix('.md'))
    print(json.dumps({'passed': True, 'samples': len(rows)}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--cpu-worker', required=True); p.add_argument('--cuda-worker', required=True)
    p.add_argument('--backends', default='cpu,cuda'); p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())
