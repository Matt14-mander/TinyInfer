#!/usr/bin/env python3
"""Run Phase 4.4 acceptance cases sequentially; retain raw per-process JSON."""
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
    parser.add_argument('--runs', type=int, default=3)
    args = parser.parse_args()
    if min(args.samples, args.repeats, args.warmup, args.runs) < 1:
        parser.error('measurement counts must be positive')
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
    environment['arguments'] = {k: str(v) if isinstance(v, Path) else v
                                for k, v in environment['arguments'].items()}
    (args.output / 'environment.json').write_text(json.dumps(environment, indent=2) + '\n')
    shapes = [(m, size, size) for size in (128, 512) for m in (1, 16, 64)] + [(3, 127, 131)]
    cases = []
    for m, k, n in shapes:
        for trans in (0, 1):
            cases.append((f'm{m}_k{k}_n{n}_t{trans}',
                          ['--m', str(m), '--k', str(k), '--n', str(n), '--trans-b', str(trans)]))
        cases.append((f'm{m}_k{k}_n{n}_dynamic',
                      ['--m', str(m), '--k', str(k), '--n', str(n), '--dynamic-b']))
    cases.append(('m16_k128_n128_dynamic_t0',
                  ['--m', '16', '--k', '128', '--n', '128', '--trans-b', '0', '--dynamic-b']))
    for fixture in ('phase3_mlp_gemm', 'rl_actor_mlp_tanh'):
        cases.append((fixture, ['--model', str(root / 'tests' / 'fixtures' / (fixture + '.onnx'))]))
    results = {}
    for name, options in cases:
        results[name] = []
        for run in range(1, args.runs + 1):
            command = [str(exe), *options, '--warmup', str(args.warmup),
                       '--samples', str(args.samples), '--repeats', str(args.repeats), '--format', 'json']
            result = json.loads(subprocess.check_output(command, cwd=root, text=True))
            if result['build_type'] != 'Release':
                raise RuntimeError('acceptance benchmark requires Release')
            result['command'] = command
            path = args.output / f'{name}_run{run}.json'
            path.write_text(json.dumps(result, indent=2) + '\n')
            results[name].append(result)
        ratios = [r['paired_speedup_p50'] for r in results[name]]
        print(f'{name}: paired speedup {min(ratios):.3f}–{max(ratios):.3f}x', flush=True)
    (args.output / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
    lines = ['| Case | Baseline p50 us | Prepared p50 us | Paired speedup range | Prep p50 us | Packed bytes |',
             '| --- | ---: | ---: | --- | ---: | ---: |']
    for name, runs in results.items():
        med = lambda field: statistics.median(r[field] for r in runs)
        ratios = [r['paired_speedup_p50'] for r in runs]
        lines.append(f'| {name} | {med("baseline_warm_p50_us"):.3f} | {med("prepared_warm_p50_us"):.3f} | '
                     f'{min(ratios):.3f}–{max(ratios):.3f}x | {med("preparation_p50_us"):.3f} | {int(med("packed_weight_bytes"))} |')
    (args.output / 'summary.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
