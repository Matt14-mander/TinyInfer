import unittest
from miniprefix.benchmarks.partial_prefix import metrics

class Metrics(unittest.TestCase):
    def test_delivery_metrics(self):
        actual=metrics(1000000,[3000000,6000000,9000000])
        self.assertEqual(actual['ttft_ms'],2)
        self.assertEqual(actual['tpot_ms'],3)
        self.assertEqual(actual['output_tokens_per_second'],375)
        self.assertEqual(actual['elapsed_ms'],8)
    def test_invalid(self):
        for times in ([1],[1,0],[0,1]):
            with self.assertRaises(ValueError): metrics(1,times)

class CowComparison(unittest.TestCase):
    def test_paired_ratios_and_profiling_exclusion(self):
        from miniprefix.benchmarks.compare_cuda_cow import summarize
        rows=[]
        for mode in ('off','cold','hot'):
            for version,latency,throughput in [('before',10,100),('after',5,200)]:
                for profile in (False,True):
                    rows.append(dict(case='test',mode=mode,version=version,repeat=0,profiling=profile,
                        ttft_ms=999 if profile else latency,tpot_ms=999 if profile else latency,
                        elapsed_ms=999 if profile else latency,output_tokens_per_second=0 if profile else throughput,
                        costs=dict(cow_copy_ns=1000000 if version=='before' else 100000,cow_bytes=192,cow_pages=1)))
        for result in summarize(rows):
            self.assertEqual(result['after']['ttft_ms']['median'],5)
            self.assertEqual(result['after']['copy_ms']['median'],0.1)
            self.assertEqual(result['paired_median'],dict(ttft_ms=2,tpot_ms=2,throughput=2))
