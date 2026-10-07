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
    sections = [lines]
    for title, fields in (
        ('Preparation components (median process p50, us)',
         ['model_snapshot_p50_us', 'topological_order_p50_us',
          'baseline_memory_plan_p50_us', 'baseline_rhs_pack_p50_us',
          'baseline_rhs_transpose_copy_p50_us']),
        ('Context and first inference (median process p50, us)',
         ['baseline_context_init_p50_us', 'prepared_context_init_p50_us',
          'baseline_first_inference_p50_us', 'prepared_first_inference_p50_us']),
        ('Memory payload and packing counts (bytes / counts)',
         ['snapshot_constant_bytes', 'context_input_bytes', 'activation_bytes',
          'packed_weight_bytes', 'plan_accounted_payload_bytes',
          'context_accounted_payload_bytes', 'pack_count',
          'runtime_pack_count', 'session_run_count'])):
        table = ['## ' + title, '', '| Case | ' + ' | '.join(fields) + ' |',
                 '| --- | ' + ' | '.join(['---:'] * len(fields)) + ' |']
        for name, runs in results.items():
            values = [statistics.median(r[field] for r in runs) for field in fields]
            table.append('| ' + name + ' | ' + ' | '.join(f'{v:.3f}' for v in values) + ' |')
        sections.append(table)
    table = ['## Per-process latency and amortization', '',
             '| Case | Run | Baseline p50 / p95 / mean us | Prepared p50 / p95 / mean us | Paired ratio | Full prep estimated calls |',
             '| --- | ---: | --- | --- | ---: | ---: |']
    for name, runs in results.items():
        for index, run in enumerate(runs, 1):
            before = ' / '.join(f'{run["baseline_warm_" + field + "_us"]:.3f}'
                                for field in ('p50', 'p95', 'mean'))
            after = ' / '.join(f'{run["prepared_warm_" + field + "_us"]:.3f}'
                               for field in ('p50', 'p95', 'mean'))
            calls = run.get('full_preparation_amortization_calls', 'n/a')
            table.append(f'| {name} | {index} | {before} | {after} | '
                         f'{run["paired_speedup_p50"]:.3f}x | {calls} |')
    sections.append(table)
    (args.output / 'detailed-summary.md').write_text(
        '\n\n'.join('\n'.join(section) for section in sections) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
