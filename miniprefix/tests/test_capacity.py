import copy
import unittest
from miniprefix.benchmarks.capacity import summarize, aggregate


class CapacityMetricsTests(unittest.TestCase):
    def fixture(self):
        rows = [dict(request=i, arrival_ms=i*10, submit_ms=i*10+1, admitted_ms=i*10+3,
                     capacity_wait_ms=2, wait_attempts=1, token_times_ms=[i*10+5+j for j in range(8)]) for i in range(10)]
        return rows+[dict(passed=True, count=10, duration_ms=102, final_blocks=0, final_committed=0,
                          pool_pages=20, peak_committed=18)]

    def test_timing_and_capacity_wait(self):
        result = summarize(self.fixture())
        self.assertEqual(result['ttft_ms']['p50'], 5)
        self.assertEqual(result['tpot_ms']['p50'], 1)
        self.assertEqual(result['queue_ms']['p50'], 2)
        self.assertEqual(result['waited_requests'], 10)
        self.assertEqual(result['drain_ms'], 12)

    def test_pool_diagnostic_stays_separate(self):
        a=summarize(self.fixture());a.update(rate=40, mode='hot', policy='shared')
        b=copy.deepcopy(a);b['pool_pages']=64
        cells=aggregate([a,b])
        self.assertEqual([x['pool_pages'] for x in cells], [20,64])
        self.assertEqual([x['repeats'] for x in cells], [1,1])

    def test_before_after_are_not_aggregated_together(self):
        a=summarize(self.fixture());a.update(rate=40,mode='hot',policy='shared',variant='before')
        b=copy.deepcopy(a);b['variant']='after';b['service_tok_s']=2*a['service_tok_s']
        cells=aggregate([a,b])
        self.assertEqual(len(cells),2)
        self.assertEqual({x['variant']:x['service_tok_s'] for x in cells},
                         {'before':a['service_tok_s'],'after':b['service_tok_s']})

    def test_mixed_output_lengths(self):
        rows=self.fixture()
        for i, row in enumerate(rows[:-1]):
            row['output_tokens']=4 if i%2==0 else 12
            row['token_times_ms']=[row['arrival_ms']+5+j for j in range(row['output_tokens'])]
        result=summarize(rows)
        self.assertEqual(result['tpot_ms']['p50'],1)
        self.assertAlmostEqual(result['full_tok_s'],80000/102)
        rows[0]['output_tokens']=8
        with self.assertRaises(ValueError): summarize(rows)

    def test_batch_and_workload_dimensions_stay_separate(self):
        base=summarize(self.fixture());base.update(rate=40,mode='hot',policy='shared')
        cases=[base]
        for field, value in [('batch_policy','adaptive'),('workload','mixed'),('shared_prefix_tokens',0),('timing','profile')]:
            row=copy.deepcopy(base);row[field]=value;cases.append(row)
        self.assertEqual(len(aggregate(cases)),5)

    def test_reject_incomplete_duplicate_and_invalid_timeline(self):
        rows = self.fixture()
        with self.assertRaises(ValueError): summarize(rows[:-1])
        for field, value in [('request', 1), ('admitted_ms', 7), ('capacity_wait_ms', 4)]:
            damaged = copy.deepcopy(rows);damaged[0][field] = value
            with self.assertRaises(ValueError): summarize(damaged)
        for field, value in [('final_blocks', 1), ('peak_committed', 21)]:
            damaged = copy.deepcopy(rows);damaged[-1][field] = value
            with self.assertRaises(ValueError): summarize(damaged)


if __name__ == '__main__':
    unittest.main()
