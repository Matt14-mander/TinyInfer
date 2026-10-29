import unittest
from miniprefix.benchmarks.stress import summarize, percentile

class StressMetricsTests(unittest.TestCase):
    def test_nearest_rank(self):
        self.assertEqual(percentile([4,1,3,2],.5),2)
        self.assertEqual(percentile([4,1,3,2],.95),4)
    def fixture(self):
        requests=[dict(reason=r,ttft_ms=2,last_ms=8,queue_ms=1,output_tokens=4) for r in (2,1,1,1,1,2,0,3)]
        return [dict(wave=0,wall_ms=10,step_ms=9,tokens=20,requests=requests,private_bytes=100,rss_bytes=200,
                     workspace_bytes=8,reused_tokens=4,evictions=1,timings={"prefill":dict(ms=6,calls=2)}),
                dict(passed=True,waves=1,requests=8,tokens=20,elapsed_ms=10)]
    def test_attribution(self):
        result=summarize(self.fixture())
        self.assertEqual(result["timing"]["prefill"]["wall_percent"],60)
        self.assertEqual(result["unattributed_wall_percent"],40)
        self.assertEqual(result["steady_output_tokens_per_second"],2000)
        self.assertEqual(result["mean_request_tpot_ms"],2)
    def test_reject_partial_and_wrong_counts(self):
        rows=self.fixture()
        with self.assertRaises(ValueError):summarize(rows[:-1])
        rows[-1]["requests"]=7
        with self.assertRaises(ValueError):summarize(rows)
    def test_reject_wrong_terminal(self):
        rows=self.fixture();rows[0]["requests"][0]["reason"]=1
        with self.assertRaises(ValueError):summarize(rows)

if __name__=="__main__":unittest.main()
