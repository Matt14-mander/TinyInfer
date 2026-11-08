import unittest
from miniprefix.benchmarks.batched_decode import summarize
from miniprefix.bindings.python.client import NativeEngine


class BatchedDecodeMetrics(unittest.TestCase):
    def test_pair_before_pooling(self):
        samples = []
        for batch in (1, 4):
            for mode in ('off', 'cold', 'hot'):
                for repeat, (before, after) in enumerate(((10, 5), (100, 10))):
                    for version, latency in (('before', before), ('after', after)):
                        throughput = ((100, 200) if version == 'before' else (200, 100))[repeat]
                        samples.append(dict(batch=batch, mode=mode, repeat=repeat, version=version,
                            ttft_p50_ms=latency, tpot_mean_ms=latency, output_tokens_per_second=throughput))
        for result in summarize(samples):
            self.assertEqual(result['paired_median'], dict(ttft_speedup=6, tpot_speedup=6, throughput_speedup=1.25))
            self.assertEqual(result['after']['ttft_p50_ms'], dict(median=7.5, min=5, max=10))

    def test_decode_stats_protocol(self):
        engine = NativeEngine.__new__(NativeEngine)
        def command(fields):
            self.assertEqual(fields, ['DECODE_STATS'])
            return [2, 7, 1]
        engine._command = command
        self.assertEqual(engine.decode_batch_stats(), dict(batches=2, rows=7, fallbacks=1))
        engine._command = lambda _: [2, 7]
        with self.assertRaises(RuntimeError):
            engine.decode_batch_stats()

    def test_prefill_stats_protocol(self):
        engine = NativeEngine.__new__(NativeEngine)
        def command(fields):
            self.assertEqual(fields, ['PREFILL_STATS'])
            return [2, 3, 12, 4, 1]
        engine._command = command
        self.assertEqual(engine.prefill_batch_stats(), dict(prefill_batches=2, mixed_batches=3,
                         prefill_tokens=12, decode_tokens=4, fallbacks=1))
        engine._command = lambda _: [2, 3]
        with self.assertRaises(RuntimeError):
            engine.prefill_batch_stats()
