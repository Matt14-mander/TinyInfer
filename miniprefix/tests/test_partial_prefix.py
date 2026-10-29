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
