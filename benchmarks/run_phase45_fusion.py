#!/usr/bin/env python3
"""Compare legacy, specialized, fused and automatic Gemm; retain raw JSON."""
import argparse
import hashlib
import json
import platform
import statistics
from datetime import datetime, timezone
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--samples', type=int, default=100)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--runs', type=int, default=4)
    parser.add_argument('--controls-only', action='store_true',
                        help='independent no-bias, identity and dynamic-B rechecks')
    parser.add_argument('--recheck-only', action='store_true',
                        help='independent selected targets plus the three fallback controls')
    args = parser.parse_args()
    if min(args.samples, args.repeats, args.warmup, args.runs) < 1:
        parser.error('measurement counts must be positive')
    if args.runs % 4:
        parser.error('--runs must be a multiple of 4 to balance allocation positions')
    if args.controls_only and args.recheck_only:
        parser.error('choose either --controls-only or --recheck-only')
    root = Path(__file__).resolve().parents[1]
    exe = args.executable.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    diff = subprocess.check_output(['git', 'diff', 'HEAD'], cwd=root)
    environment = dict(measurement_started_utc=datetime.now(timezone.utc).isoformat(),
                       executable_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
                       platform=platform.platform(), machine=platform.machine(),
                       processor=platform.processor(), source_revision=revision,
                       tracked_diff_sha256=hashlib.sha256(diff).hexdigest(),
                       git_status=subprocess.check_output(['git', 'status', '--short'], cwd=root, text=True),
                       arguments=vars(args).copy())
    environment['analysis_source_sha256'] = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ('benchmarks/cpu_performance_analysis_benchmark.cpp',
                     'benchmarks/run_phase45_fusion.py', 'src/backend/cpu/matmul_kernel.cpp',
                     'src/backend/cpu/gemm_epilogue.cpp', 'src/backend/cpu/execution_plan.cpp',
                     'include/tinyinfer/backend/cpu/gemm_epilogue.h',
                     'include/tinyinfer/backend/cpu/execution_plan.h',
                     'include/tinyinfer/backend/cpu/matmul_kernel.h',
                     'src/ops/gemm_internal.h', 'CMakeLists.txt', 'benchmarks/CMakeLists.txt')}
    environment['arguments'] = {k: str(v) if isinstance(v, Path) else v
                                for k, v in environment['arguments'].items()}
    (args.output / 'environment.json').write_text(json.dumps(environment, indent=2) + '\n')
    shapes = [(m, size, size) for size in (128, 512) for m in (1, 16, 64)] + [(3, 127, 131)]
    cases = []
    for m, k, n in shapes:
        for trans in (0, 1):
            cases.append((f'm{m}_k{k}_n{n}_t{trans}',
                          ['--m', str(m), '--k', str(k), '--n', str(n), '--trans-b', str(trans)]))
    for bias in ('none', 'scalar', 'row', 'column', 'full'):
        cases.append((f'm16_k128_n128_bias_{bias}', ['--bias', bias]))
    cases.append(('m16_k128_n128_no_relu', ['--activation', 'none']))
    cases.append(('m16_k128_n128_no_epilogue', ['--bias', 'none', '--activation', 'none']))
    cases.append(('m16_k128_n128_dynamic', ['--dynamic-b']))
    cases.append(('m3_k127_n128_aligned_n', ['--m', '3', '--k', '127', '--n', '128']))
    cases.append(('m3_k128_n128_aligned_kn', ['--m', '3', '--k', '128', '--n', '128']))
    for fixture in ('phase3_mlp_gemm', 'rl_actor_mlp_tanh'):
        cases.append((fixture, ['--model', str(root / 'tests' / 'fixtures' / (fixture + '.onnx'))]))
    if args.controls_only:
        cases = [(name, options) for name, options in cases
                 if name in ('m16_k128_n128_bias_none', 'm16_k128_n128_no_epilogue',
                             'm16_k128_n128_dynamic')]
    elif args.recheck_only:
        cases = [(name, options) for name, options in cases
                 if name.startswith(('m1_k128_n128_', 'm16_k128_n128_', 'm64_k128_n128_'))]
    results = {}
    for name, options in cases:
        runs = []
        for run in range(1, args.runs + 1):
            command = [str(exe), '--fusion-comparison', '--allocation-order', str((run - 1) % 4),
                       *options, '--warmup', str(args.warmup),
                       '--samples', str(args.samples), '--repeats', str(args.repeats), '--format', 'json']
            result = json.loads(subprocess.check_output(command, cwd=root, text=True))
            if result['build_type'] != 'Release':
                raise RuntimeError('analysis requires Release')
            result['command'] = command
            (args.output / f'{name}_run{run}.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
            runs.append(result)
        results[name] = runs
        p50 = [run['fused']['p50_us'] for run in runs]
        print(f'{name}: fused p50 {min(p50):.3f}–{max(p50):.3f} us', flush=True)
    (args.output / 'summary.json').write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    fields = ('legacy', 'specialized', 'fused', 'auto')
    ratios = ('specialized_vs_legacy_paired_p50', 'fused_vs_legacy_paired_p50',
              'fused_vs_specialized_paired_p50', 'auto_vs_specialized_paired_p50')
    lines = ['| Case | Legacy p50 us | Specialized p50 us | Fused p50 us | Auto p50 us | Specialized / legacy speedup | Fused / legacy speedup | Fused / specialized speedup | Auto / specialized speedup |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, runs in results.items():
        values = [statistics.median(r[field]['p50_us'] for r in runs) for field in fields]
        values += [statistics.median(r[field] for r in runs) for field in ratios]
        lines.append('| ' + name + ' | ' + ' | '.join(f'{v:.3f}' for v in values) + ' |')
    (args.output / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
