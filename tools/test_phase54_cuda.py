import copy
import unittest
import run_phase54_cuda as bench


def report():
    stages = ('prepare_us', 'context_create_us', 'first_inference_us', 'warm_e2e_us',
              'bind_host_input_us', 'resident_run_completed_us', 'copy_host_output_us', 'resident_only_us')
    cases = []
    for name in bench.WORKLOADS:
        results = []
        for backend in ('cpu_auto', 'cuda'):
            value = 2.0 if backend == 'cpu_auto' else 1.0
            if name == 'fallback': value = 2.0
            times = {key: [value] * (1 if key in stages[:3] else 31) for key in stages}
            results.append({'backend': backend, 'effective': 'cpu' if backend == 'cpu_auto' or name == 'fallback' else 'cuda', 'timings': times})
        cases.append({'workload': name, 'correctness': True, 'results': results})
    return {'profiling': False, 'samples': 31, 'warmup': 5, 'cases': cases}


class ProtocolTests(unittest.TestCase):
    def test_nearest_rank(self):
        self.assertEqual(bench.percentile(list(range(31)), .5), 15)
        self.assertEqual(bench.percentile(list(range(31)), .95), 29)

    def test_all_rounds_required(self):
        rows = [{'round': r, 'process': p, 'report': report()} for r in (1, 2) for p in (1, 2, 3)]
        self.assertTrue(bench.analyze(rows)['performance_accepted'])
        for entry in rows:
            if entry['round'] == 2:
                target = next(c for c in entry['report']['cases'] if c['workload'] == 'target')
                target['results'][1]['timings']['warm_e2e_us'] = [3.0] * 31
        result = bench.analyze(rows)
        self.assertFalse(result['performance_accepted'])
        self.assertFalse(result['gates']['target_p50'])
        self.assertFalse(result['gates']['target_p95'])

    def test_validation(self):
        good = report(); bench.validate(good, 31, 5)
        mutations = [lambda r: r.update(profiling=True),
                     lambda r: r['cases'][0].update(correctness=False),
                     lambda r: r['cases'][0]['results'][1].update(effective='cpu'),
                     lambda r: r['cases'][0]['results'][0]['timings'].pop('prepare_us'),
                     lambda r: r['cases'][0]['results'][0]['timings'].update(warm_e2e_us=[float('nan')]*31),
                     lambda r: r['cases'][0]['results'][0]['timings'].update(warm_e2e_us=[1.0]*30)]
        for mutate in mutations:
            bad = copy.deepcopy(good); mutate(bad)
            with self.assertRaises(RuntimeError): bench.validate(bad, 31, 5)

    def test_isolated_fallback_validation(self):
        isolated=report()
        isolated['cases']=[c for c in isolated['cases'] if c['workload']=='fallback']
        bench.validate(isolated,31,5,workloads=('fallback',))
        with self.assertRaises(RuntimeError): bench.validate(isolated,31,5)
        isolated['cases'][0]['results'][1]['effective']='cuda'
        with self.assertRaises(RuntimeError): bench.validate(isolated,31,5,workloads=('fallback',))


if __name__ == '__main__': unittest.main()
