#!/usr/bin/env python3
"""ROG CUDA frozen performance protocol: sequential processes, no sample filtering."""
import argparse, datetime, hashlib, json, math, os, pathlib, platform, statistics, subprocess

WORKLOADS = ('phase3_mlp_gemm', 'rl_actor_mlp_tanh', 'control', 'target', 'fallback')

def command(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=60)
        return {'command': args, 'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {'command': args, 'error': str(error)}

def environment():
    return {
        'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'gpu': command(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.used,pstate,temperature.gpu,power.draw,clocks.sm,utilization.gpu', '--format=csv']),
        'host': command(['powershell', '-NoProfile', '-Command',
            '[pscustomobject]@{cpu=@(Get-CimInstance Win32_Processor | Select-Object Name,LoadPercentage);battery=@(Get-CimInstance Win32_Battery | Select-Object BatteryStatus,EstimatedChargeRemaining);os=(Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,FreePhysicalMemory);power=@(powercfg /getactivescheme)} | ConvertTo-Json -Depth 5']),
    }

def percentile(values,p):
    values=sorted(values)
    return values[max(0,math.ceil(len(values)*p)-1)] # nearest-rank, no interpolation

def analyze(reports):
    rows=[]
    for round_id in (1,2):
        for name in ('phase3_mlp_gemm','rl_actor_mlp_tanh','control','target','fallback'):
            matched=[]
            for entry in reports:
                if entry['round']!=round_id:continue
                case=next(c for c in entry['report']['cases'] if c['workload']==name)
                cpu,gpu=case['results']
                c=cpu['timings']['warm_e2e_us'];g=gpu['timings']['warm_e2e_us']
                matched.append({'process':entry['process'],'cpu_p50_us':percentile(c,.5),'cpu_p95_us':percentile(c,.95),
                    'cuda_p50_us':percentile(g,.5),'cuda_p95_us':percentile(g,.95),
                    'p50_speedup':percentile(c,.5)/percentile(g,.5), 'p95_ratio':percentile(g,.95)/percentile(c,.95),
                    'stages':{r['backend']:{k:{'p50_us':percentile(v,.5),'p95_us':percentile(v,.95)} for k,v in r['timings'].items()} for r in case['results']},
                    'memory':{r['backend']:{k:v for k,v in r.items() if k.endswith('bytes')} for r in case['results']}})
            rows.append({'round':round_id,'workload':name,'processes':matched,
                'median_p50_speedup':statistics.median(x['p50_speedup'] for x in matched),
                'median_p95_ratio':statistics.median(x['p95_ratio'] for x in matched),
                'median_p50_latency_ratio':statistics.median(x['cuda_p50_us']/x['cpu_p50_us'] for x in matched)})
    target=[r for r in rows if r['workload']=='target'];fallback=[r for r in rows if r['workload']=='fallback']
    gates={'target_p50':all(r['median_p50_speedup']>=1.2 for r in target),
           'target_p95':all(r['median_p95_ratio']<=1.05 for r in target),
           'fallback_p50':all(r['median_p50_latency_ratio']<=1.05 for r in fallback)}
    return {'percentile':'nearest rank ceil(n*p)', 'rows':rows,'gates':gates,'performance_accepted':all(gates.values())}


def validate(report, samples, warmup):
    if report.get('profiling', True) or report.get('samples') != samples or report.get('warmup') != warmup:
        raise RuntimeError('unexpected timing protocol')
    if set(c['workload'] for c in report['cases']) != set(WORKLOADS) or len(report['cases']) != 5:
        raise RuntimeError('unexpected workloads')
    for case in report['cases']:
        if not case['correctness'] or [r['backend'] for r in case['results']] != ['cpu_auto', 'cuda']:
            raise RuntimeError('correctness/backend gate failed')
        gpu = case['results'][1]
        if gpu['effective'] != ('cpu' if case['workload'] == 'fallback' else 'cuda'):
            raise RuntimeError('unexpected effective device')
        for result in case['results']:
            required = {'prepare_us','context_create_us','first_inference_us','warm_e2e_us',
                        'bind_host_input_us','resident_run_completed_us','copy_host_output_us','resident_only_us'}
            if set(result['timings']) != required:
                raise RuntimeError('missing/unexpected stage timers')
            for name, values in result['timings'].items():
                count = 1 if name in ('prepare_us', 'context_create_us', 'first_inference_us') else samples
                if len(values) != count or not all(math.isfinite(v) and v > 0 for v in values):
                    raise RuntimeError('invalid timing samples: ' + name)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--binary', required=True); p.add_argument('--output', required=True)
    p.add_argument('--samples', type=int, default=31); p.add_argument('--warmup', type=int, default=5)
    a = p.parse_args()
    if a.samples < 31 or a.warmup < 5:
        p.error('acceptance requires samples >=31 and warmup >=5')
    if os.environ.get('CUDA_LAUNCH_BLOCKING', '0') not in ('', '0'):
        p.error('CUDA_LAUNCH_BLOCKING must be disabled for timing')
    binary = pathlib.Path(a.binary).resolve()
    output = pathlib.Path(a.output)
    output.mkdir(parents=True, exist_ok=False)
    metadata = {
        'protocol': 'docs/plans/phase5.4-performance.md',
        'baseline': command(['git', 'rev-parse', 'HEAD']),
        'host': platform.platform(), 'started_at': environment(),
        'samples': a.samples, 'warmup': a.warmup, 'rounds': 2, 'processes_per_round': 3,
        'target': {'M':128,'K':1024,'N':1024,'p50_speedup_min':1.2,'p95_ratio_max':1.05,'fallback_p50_ratio_max':1.05},
        'commands': [command(args) for args in (['git','status','--short'],['git','diff'],['cmake','--version'],['nvcc','--version'])],
        'selected_environment': {key:os.environ.get(key) for key in ('CUDA_VISIBLE_DEVICES','CUDA_LAUNCH_BLOCKING','CUDA_MODULE_LOADING','NVIDIA_TF32_OVERRIDE')},
        'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
        'source_sha256': {str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in map(pathlib.Path, (
            'docs/plans/phase5.4-performance.md','benchmarks/cuda_model_benchmark.cpp','tools/run_phase54_cuda.py',
            'benchmarks/CMakeLists.txt','CMakeLists.txt','src/backend/cuda/execution_plan.cpp',
            'src/backend/cuda/cuda_runtime.cpp','src/backend/cuda/kernels.cu','src/backend/cuda/kernels.h',
            'src/backend/cuda/operator_contract.cpp','src/backend/cpu/execution_plan.cpp'))},
        'cmake_cache': (binary.parents[1]/'CMakeCache.txt').read_text(),
    }
    (output/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    reports = []
    for round_id in (1,2):
        for process in (1,2,3):
            args = [str(binary),'--samples',str(a.samples),'--warmup',str(a.warmup)]
            if (round_id+process)%2: args.append('--gpu-first')
            print(f'round {round_id}, process {process}', flush=True)
            before = environment()
            result = subprocess.run(args, capture_output=True, text=True, timeout=600)
            stem = output/f'round{round_id}-process{process}'
            stem.with_suffix('.environment.json').write_text(json.dumps({'before':before,'after':environment(),'command':args,'returncode':result.returncode},indent=2)+'\n', encoding='utf-8')
            stem.with_suffix('.stderr.log').write_text(result.stderr, encoding='utf-8')
            stem.with_suffix('.json').write_text(result.stdout, encoding='utf-8')
            if result.returncode:
                raise RuntimeError(f'{stem}: failed ({result.returncode}): {result.stderr}')
            report = json.loads(result.stdout); validate(report,a.samples,a.warmup)
            reports.append({'round':round_id,'process':process,'report':report})
    summary = analyze(reports)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n', encoding='utf-8')
    result = subprocess.run([str(binary),'--memory-only'],capture_output=True,text=True,timeout=180)
    (output/'memory.stderr.log').write_text(result.stderr,encoding='utf-8')
    (output/'memory.json').write_text(result.stdout,encoding='utf-8')
    if result.returncode: raise RuntimeError('memory diagnostic failed: '+result.stderr)
    memory = json.loads(result.stdout)
    if not memory.get('memory_only') or len(memory['cases']) != 5:
        raise RuntimeError('invalid memory diagnostic')
    (output/'finished-environment.json').write_text(json.dumps(environment(),indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary['gates']), flush=True)
    print('performance_accepted='+str(summary['performance_accepted']), flush=True)

if __name__ == '__main__': main()
