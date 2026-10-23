"""Hand-calculated controller timestamps and rejection of incorrect samples."""
import unittest
from unittest.mock import patch
from miniprefix.benchmarks import baseline


class Clock:
    now = 0
    def perf_counter_ns(self):
        return self.now
    def sleep(self, seconds):
        self.now += round(seconds * 1e9)


class FixtureEngine:
    def __init__(self, clock):
        self.clock, self.pending, self.calls = clock, set(), 0
    def submit(self, key, prompt, count, eos):
        self.pending.add(key)
    def step(self):
        self.calls += 1
        self.clock.now += (1, 2, 1)[self.calls-1] * 1000000
        plan = (("a",10,None),) if self.calls==1 else (("a",11,None),) if self.calls==2 else (("a",12,"length"),("b",20,"length"))
        events=[]
        for key,token,reason in plan:
            if reason:self.pending.remove(key)
            events.append(dict(request_id=key,token_id=token,finish_reason=reason,received_ns=self.clock.now))
        return events


class BaselineTests(unittest.TestCase):
    def workload(self):
        return [dict(request_id="a",arrival_ms=0,prompt_tokens=[1],max_new_tokens=3,expected_tokens=[10,11,12]),
                dict(request_id="b",arrival_ms=2,prompt_tokens=[2],max_new_tokens=1,expected_tokens=[20])]
    def measure(self, workload):
        clock=Clock()
        with patch.object(baseline, "time", clock):
            return baseline.measure(FixtureEngine(clock),workload)
    def test_hand_calculated_delivery_and_single_token_tpot(self):
        result=self.measure(self.workload())
        a,b=result["requests"]
        self.assertEqual(a["token_times_ms"],[1,3,4])
        self.assertEqual(b["submitted_ms"],3)
        self.assertEqual(b["controller_submit_delay_ms"],1)
        self.assertEqual(b["ttft_ms"],2)
        self.assertEqual(result["ttft_p95_ms"],2)
        self.assertEqual(result["tpot_mean_ms"],1.5)
        self.assertIsNone(b["tpot_ms"])
        self.assertEqual(result["measurement_window_ms"],4)
        self.assertEqual(result["output_tokens_per_second"],1000)
    def test_wrong_reference_is_not_a_performance_sample(self):
        workload=self.workload();workload[0]["expected_tokens"][0]=999
        with self.assertRaisesRegex(RuntimeError,"reference mismatch"):
            self.measure(workload)
    def test_empty_workload_and_nearest_rank(self):
        with self.assertRaises(ValueError):
            self.measure([])
        self.assertIsNone(baseline.percentile([], .95))
        self.assertEqual(baseline.percentile(list(range(1,9)),.95),8)


if __name__=="__main__":
    unittest.main()
