"""Real CPU packed batching: numerical, causal, identity and failure contracts."""
import math
import os
import random
import unittest

try:
    from .backend import Request, WorkItem
    from .tinyinfer_backend import TinyInferBackend
    from .realtime import serve
except ImportError:
    from backend import Request, WorkItem
    from tinyinfer_backend import TinyInferBackend
    from realtime import serve


class TensorBatchingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.worker = os.environ.get("TINYINFER_SERVING_WORKER")
        if not cls.worker:
            raise unittest.SkipTest("actual worker required for tensor batching acceptance")

    def assert_logits(self, actual, expected):
        self.assertEqual(len(actual), 32)
        self.assertEqual(len(expected), 32)
        for a, b in zip(actual, expected):
            self.assertTrue(math.isfinite(a) and math.isfinite(b))
            self.assertLessEqual(abs(a - b), 2e-5 + 2e-5 * abs(b))

    def checked_step(self, backend, work):
        result = backend.step(work)
        logits = backend.last_step_logits()
        self.assertEqual(len(logits), len(work))
        for item, actual, token in zip(work, logits, result.tokens):
            prefix = backend.prompt(item.request)
            if item.is_prefill:
                prefix = prefix[:item.prefill_start + item.input_tokens]
            else:
                # step() has already appended the new output; it is not in KV yet.
                prefix += backend.outputs[item.request.request_id][:-1]
            expected = backend.reference_logits(prefix)
            self.assert_logits(actual, expected)
            self.assertEqual(token, max(range(32), key=expected.__getitem__) if item.emits_token else None)
        return result

    def test_ragged_chunks_mixed_decode_reorder_and_refill(self):
        a, b, c = Request("a", 0, 5, 3), Request("b", 0, 2, 2), Request("c", 0, 3, 1)
        with TinyInferBackend(self.worker, 4) as backend:
            self.checked_step(backend, [WorkItem(a, 0, 0, 2), WorkItem(b, 0)])
            self.checked_step(backend, [WorkItem(b, 1, 2), WorkItem(a, 0, 2, 2)])
            backend.release("b")
            self.checked_step(backend, [WorkItem(c, 0), WorkItem(a, 0, 4, 1)])
            backend.release("c")
            self.checked_step(backend, [WorkItem(a, 1, 5)])
            self.checked_step(backend, [WorkItem(a, 2, 5)])
            backend.release("a")
            stats = backend.stats()
            self.assertEqual(stats["forward_tokens"], 13)
            self.assertEqual(stats["batch_steps"], 5)
            self.assertEqual(stats["linear_matmul_calls"], 35)
            self.assertEqual(stats["linear_matmul_rows"], 7 * 13)
            self.assertEqual(stats["peak_batch_requests"], 2)
            self.assertEqual(stats["peak_batch_tokens"], 4)
            self.assertEqual(stats["live_kv_bytes"], 0)

    def test_many_decode_requests_share_seven_linear_calls(self):
        requests = [Request("r%d" % i, 0, i + 1, 2) for i in range(8)]
        with TinyInferBackend(self.worker) as backend:
            self.checked_step(backend, [WorkItem(r, 0) for r in requests])
            before = backend.stats()
            self.checked_step(backend, [WorkItem(r, 1, r.prompt_tokens) for r in reversed(requests)])
            after = backend.stats()
            self.assertTrue(backend.hardware_batched)
            self.assertEqual(after["linear_matmul_calls"] - before["linear_matmul_calls"], 7)
            self.assertEqual(after["linear_matmul_rows"] - before["linear_matmul_rows"], 7 * 8)
            self.assertEqual(after["forward_tokens"] - before["forward_tokens"], 8)
            for r in requests:
                backend.release(r.request_id)

    def test_batch_composition_does_not_change_partial_prefill_logits(self):
        a, b = Request("a", 0, 9, 1), Request("other", 0, 11, 1)
        with TinyInferBackend(self.worker) as backend:
            self.checked_step(backend, [WorkItem(a, 0, 0, 3)])
            alone = backend.last_step_logits()[0]
            backend.reset()
            self.checked_step(backend, [WorkItem(b, 0), WorkItem(a, 0, 0, 3)])
            self.assert_logits(backend.last_step_logits()[1], alone)
            backend.clear()
            self.assertEqual(backend.stats()["live_requests"], 0)

    def test_invalid_batch_clears_kv_and_can_restart(self):
        with TinyInferBackend(self.worker) as backend:
            a = Request("a", 0, 3, 2)
            backend.step([WorkItem(a, 0, 0, 1)])
            before = backend.stats()["batch_steps"]
            # A valid first item must not execute when a later item is invalid.
            with self.assertRaisesRegex(RuntimeError, "invalid token"):
                backend._command(["STEP", 2, 1, 0, 1, 1, 3, 2, 5, 7,
                                  2, 0, 0, 1, 1, 1, 2, 32])
            self.assertEqual(backend.stats()["live_kv_bytes"], 0)
            self.assertEqual(backend.stats()["batch_steps"], before)
            self.assertEqual(backend.last_step_logits(), [])
            backend.reset()
            with self.assertRaisesRegex(RuntimeError, "duplicate batch request"):
                backend.step([WorkItem(a, 0), WorkItem(a, 0)])
            self.assertEqual(backend.stats()["live_requests"], 0)
            backend.reset()
            self.checked_step(backend, [WorkItem(a, 0)])
            backend.clear()

    def test_random_workloads_modes_budgets_and_resource_accounting(self):
        rng = random.Random(20261022)
        requests = [Request("random%d" % i, i % 3 * .1, rng.randint(1, 17), rng.randint(1, 5))
                    for i in range(12)]
        with TinyInferBackend(self.worker, 4) as backend:
            expected = {r.request_id: backend.reference(r) for r in requests}
            for mode in ("static", "continuous"):
                for budget, chunk in ((None, None), (1, 1), (9, 3)):
                    result = serve(requests, backend, mode, 4, 32, 4, "wait", budget, chunk)
                    self.assertEqual(len(result.results), len(requests))
                    for item in result.results:
                        self.assertEqual(item.tokens, expected[item.request.request_id])
                    stats = result.backend_stats
                    tokens = sum(r.prompt_tokens + r.output_tokens - 1 for r in requests)
                    self.assertEqual(stats["forward_tokens"], tokens)
                    self.assertEqual(stats["linear_matmul_calls"], 7 * len(result.trace))
                    self.assertEqual(stats["linear_matmul_rows"], 7 * tokens)
                    self.assertEqual(stats["live_kv_bytes"], 0)
                    self.assertEqual(result.kv["final_reserved_blocks"], 0)


if __name__ == "__main__":
    unittest.main()
