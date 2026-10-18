#!/usr/bin/env python3
"""Six isolated fallback-first processes; CUDA hidden, no device probe before timing."""
import argparse
import datetime
import hashlib
import json
import os
import pathlib
import statistics
import subprocess
import run_phase54_cuda as protocol


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--binary',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    binary=pathlib.Path(args.binary).resolve()
    output=pathlib.Path(args.output); output.mkdir(parents=True,exist_ok=False)
    metadata={'protocol':'six fresh processes, fallback only, five warmups/31 warm samples; no CUDA enumeration in child',
              'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'baseline':protocol.command(['git','rev-parse','HEAD']),
              'status':protocol.command(['git','status','--short']),
              'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),
              'source_sha256':{name:hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest() for name in
                  ('src/backend/cuda/execution_plan.cpp','benchmarks/cuda_model_benchmark.cpp',
                   'tests/cuda_static_fallback_test.cpp','tools/run_phase54_cuda_cold_fallback.py')},
              'cold_diagnostic_target_us':5000,
              'note':'5 ms is a local regression check, not a general startup SLA; process loading is outside API timers'}
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']='-1'
    rows=[]
    for index in range(1,7):
        command=[str(binary),'--workload','fallback','--samples','31','--warmup','5']
        if index%2==0:command.append('--gpu-first')
        print(f'cold fallback process {index}/6',flush=True)
        result=subprocess.run(command,capture_output=True,text=True,env=env,timeout=120)
        stem=output/f'process{index}'
        stem.with_suffix('.json').write_text(result.stdout,encoding='utf-8')
        stem.with_suffix('.stderr.log').write_text(result.stderr,encoding='utf-8')
        if result.returncode:raise RuntimeError(f'{stem}: {result.stderr}')
        report=json.loads(result.stdout)
        if len(report['cases'])!=1 or report['cases'][0]['workload']!='fallback':raise RuntimeError('not isolated fallback')
        case=report['cases'][0]; cpu,wrapper=case['results']
        protocol.validate(report,31,5,workloads=('fallback',))
        if wrapper['effective']!='cpu' or case['logical_allocations'] or case['dispatches'] or case['uploaded_bytes'] or case['downloaded_bytes']:
            raise RuntimeError('fallback touched CUDA model resources')
        rows.append({'process':index,'wrapper_first':index%2==0,'cuda_visible_devices':'-1',
                     'cpu_prepare_us':cpu['timings']['prepare_us'][0],
                     'wrapper_prepare_us':wrapper['timings']['prepare_us'][0],
                     'cpu_first_inference_us':cpu['timings']['first_inference_us'][0],
                     'wrapper_first_inference_us':wrapper['timings']['first_inference_us'][0],
                     'warm_p50_ratio':protocol.percentile(wrapper['timings']['warm_e2e_us'],.5)/protocol.percentile(cpu['timings']['warm_e2e_us'],.5)})
    values=[row['wrapper_prepare_us'] for row in rows]
    summary={'rows':rows,'wrapper_prepare_us':{'min':min(values),'median':statistics.median(values),'max':max(values)},
             'all_below_5ms':all(value<5000 for value in values),'all_passed_without_visible_cuda_device':True}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary['wrapper_prepare_us']),flush=True)
    if not summary['all_below_5ms']:raise RuntimeError('cold diagnostic exceeded 5 ms; retain all results and investigate')


if __name__=='__main__':main()
