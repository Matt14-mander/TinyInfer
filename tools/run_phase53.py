#!/usr/bin/env python3
"""Sequential actual-device runs and frozen Phase 5.3 gates; no third-party deps."""
import argparse, json, math, pathlib, platform, statistics, subprocess, datetime, os, hashlib

def command(args):
    result = subprocess.run(args, capture_output=True, text=True)
    return {'command':args,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr}

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
                    'metal_p50_us':percentile(g,.5),'metal_p95_us':percentile(g,.95),
                    'p50_speedup':percentile(c,.5)/percentile(g,.5), 'p95_ratio':percentile(g,.95)/percentile(c,.95),
                    'stages':{r['backend']:{k:{'p50_us':percentile(v,.5),'p95_us':percentile(v,.95)} for k,v in r['timings'].items()} for r in case['results']},
                    'memory':{r['backend']:{k:v for k,v in r.items() if k.endswith('bytes')} for r in case['results']}})
            rows.append({'round':round_id,'workload':name,'processes':matched,
                'median_p50_speedup':statistics.median(x['p50_speedup'] for x in matched),
                'median_p95_ratio':statistics.median(x['p95_ratio'] for x in matched),
                'median_p50_latency_ratio':statistics.median(x['metal_p50_us']/x['cpu_p50_us'] for x in matched)})
    target=[r for r in rows if r['workload']=='target'];fallback=[r for r in rows if r['workload']=='fallback']
    gates={'target_p50':all(r['median_p50_speedup']>=1.2 for r in target),
           'target_p95':all(r['median_p95_ratio']<=1.05 for r in target),
           'fallback_p50':all(r['median_p50_latency_ratio']<=1.05 for r in fallback)}
    return {'percentile':'nearest rank ceil(n*p)', 'rows':rows,'gates':gates,'performance_accepted':all(gates.values())}

def main():
    p=argparse.ArgumentParser();p.add_argument('--binary',required=True);p.add_argument('--output',required=True)
    p.add_argument('--samples',type=int,default=31);p.add_argument('--warmup',type=int,default=5)
    a=p.parse_args()
    if a.samples<31 or a.warmup<5:p.error('acceptance requires samples >=31 and warmup >=5')
    output=pathlib.Path(a.output);output.mkdir(parents=True,exist_ok=False)
    # Capture configured SDK/host state; no GPU validation layer during performance runs.
    metadata={'date':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y-%m-%d Asia/Shanghai'),'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'host':platform.platform(),'samples':a.samples,'warmup':a.warmup,'rounds':2,'processes_per_round':3,
              'target':{'M':128,'K':1024,'N':1024,'operation':'Gemm vector bias ReLU','p50_speedup_min':1.2,'p95_ratio_max':1.05,'fallback_ratio_max':1.05},
              'commands':[command(args) for args in (["git","rev-parse","HEAD"],["git","status","--short"],["git","diff"],["sw_vers"],["system_profiler","SPDisplaysDataType"],["xcrun","--show-sdk-version"],["c++","--version"],["cmake","--version"],["uptime"],["pmset","-g","batt"],["sysctl","machdep.cpu.brand_string"]) ]}
    metadata['binary_sha256']=hashlib.sha256(pathlib.Path(a.binary).read_bytes()).hexdigest()
    metadata['source_sha256']={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in
        [pathlib.Path('benchmarks/metal_model_benchmark.cpp'),pathlib.Path('tools/run_phase53.py'),pathlib.Path('src/backend/metal/execution_plan.cpp'),pathlib.Path('src/backend/metal/metal_runtime.mm'),pathlib.Path('src/backend/metal/kernels.h')]}
    cache=pathlib.Path(a.binary).resolve().parents[1]/'CMakeCache.txt'
    metadata['cmake_cache']=cache.read_text()
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    env=os.environ.copy();env.pop('MTL_DEBUG_LAYER',None);env.pop('MTL_SHADER_VALIDATION',None)
    reports=[]
    for round_id in (1,2):
        for process in (1,2,3):
            args=[str(pathlib.Path(a.binary).resolve()),'--samples',str(a.samples),'--warmup',str(a.warmup)]
            if (round_id+process)%2:args.append('--gpu-first')
            print(f'round {round_id}, process {process}',flush=True)
            process_started=datetime.datetime.now(datetime.timezone.utc).isoformat()
            result=subprocess.run(args,capture_output=True,text=True,env=env)
            stem=output/f'round{round_id}-process{process}'
            stem.with_suffix('.environment.json').write_text(json.dumps({'started_at_utc':process_started,
                'finished_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'commands':[command(["uptime"]),command(["pmset","-g","batt"])]},indent=2)+'\n')
            stem.with_suffix('.stderr.log').write_text(result.stderr)
            stem.with_suffix('.json').write_text(result.stdout)
            if result.returncode:raise RuntimeError(f'{stem}: benchmark failed ({result.returncode}): {result.stderr}')
            report=json.loads(result.stdout)
            if report.get('profiling',False):raise RuntimeError('profiled runs cannot count as acceptance')
            if report['samples']!=a.samples or report['warmup']!=a.warmup or len(report['cases'])!=5:
                raise RuntimeError('unexpected benchmark protocol')
            for case in report['cases']:
                if not case['correctness']:raise RuntimeError('correctness gate failed')
                for result in case['results']:
                    values=result['timings']['warm_e2e_us']
                    if len(values)!=a.samples or not all(math.isfinite(v) and v>0 for v in values):
                        raise RuntimeError('invalid end-to-end samples')
            reports.append({'round':round_id,'process':process,'report':report})
    summary=analyze(reports)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary['gates']),flush=True)
    print('performance_accepted='+str(summary['performance_accepted']),flush=True)

if __name__=='__main__':main()
