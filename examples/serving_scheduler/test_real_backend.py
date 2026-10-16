"""Set TINYINFER_SERVING_WORKER to run actual C++ CPU integration tests."""
import os
import time
import unittest
try:
    from .backend import Request, StepResult, SimulatedBackend
    from .tinyinfer_backend import TinyInferBackend
    from .realtime import serve
    from .metrics import summarize
except ImportError:
    from backend import Request, StepResult, SimulatedBackend
    from tinyinfer_backend import TinyInferBackend
    from realtime import serve
    from metrics import summarize


class RealBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.worker = os.environ.get("TINYINFER_SERVING_WORKER")
        if not cls.worker:
            raise unittest.SkipTest("actual worker path required; skip is not stage-four acceptance")

    def test_greedy_reference_both_modes_and_chunk_boundaries(self):
        requests = [Request("a", 0, 7, 4), Request("b", 0, 3, 3), Request("c", .2, 5, 2)]
        with TinyInferBackend(self.worker, 4) as backend:
            expected = {r.request_id: backend.reference(r) for r in requests}
            for mode in ("static", "continuous"):
                for budget, chunk in ((None, None), (1, 1), (5, 2), (64, None)):
                    result = serve(requests, backend, mode, 3, 20, 4, "wait", budget, chunk)
                    self.assertEqual(result.clock, "wall_ms")
                    for r in result.results:
                        self.assertEqual(r.tokens, expected[r.request.request_id])
                        self.assertGreaterEqual(r.admitted_ms, r.request.arrival_ms)
                        self.assertGreater(r.token_times_ms[0], r.admitted_ms)
                    self.assertEqual(result.backend_stats["live_kv_bytes"], 0)
                    self.assertEqual(result.kv["final_reserved_blocks"], 0)
                    self.assertGreater(result.backend_stats["peak_kv_bytes"], 0)

    def test_real_arrivals_and_ignore_cost_formula(self):
        with TinyInferBackend(self.worker) as backend:
            original = backend.step
            def exaggerated(batch):
                value = original(batch)
                return StepResult(1e12, value.tokens)
            backend.step = exaggerated
            start = time.perf_counter()
            result = serve([Request("late", 20, 2, 2)], backend, token_budget=1)
            self.assertGreaterEqual(time.perf_counter() - start, .02)
            self.assertLess(result.finished_ms, 1e12)
            self.assertGreaterEqual(result.results[0].admitted_ms, 20)
            self.assertGreater(summarize(result)["tpot_mean_ms"], 0)

    def test_kv_rejection_and_all_rejected_no_forward(self):
        with TinyInferBackend(self.worker, 4) as backend:
            result = serve([Request("a", 0, 3, 2)], backend, kv_capacity_blocks=0, kv_block_size=4)
            self.assertEqual(len(result.rejections), 1)
            self.assertEqual(result.backend_stats["forward_tokens"], 0)
            self.assertEqual(result.backend_stats["live_kv_bytes"], 0)
            requests = [Request("a", 0, 4, 4), Request("b", 0, 4, 4)]
            result = serve(requests, backend, kv_capacity_blocks=2, kv_block_size=4, kv_policy="reject")
            self.assertEqual(len(result.results), 1)
            self.assertEqual(len(result.rejections), 1)
            self.assertEqual(result.backend_stats["peak_kv_bytes"], 8 * 16 * 4 * 2)

    def test_backend_failure_clears_native_kv_and_reusable(self):
        with TinyInferBackend(self.worker) as backend:
            original = backend.step
            calls = [0]
            def fail(batch):
                calls[0] += 1
                if calls[0] == 2: raise RuntimeError("injected failure after partial real prefill")
                return original(batch)
            backend.step = fail
            with self.assertRaises(RuntimeError):
                serve([Request("a", 0, 6, 2)], backend, token_budget=2)
            self.assertEqual(backend.stats()["live_kv_bytes"], 0)
            backend.step = original
            result = serve([Request("a", 0, 6, 2)], backend)
            self.assertEqual(len(result.results), 1)

    def test_virtual_backend_rejected_and_limits_fail_explicitly(self):
        with self.assertRaises(ValueError): serve([], SimulatedBackend())
        with TinyInferBackend(self.worker) as backend:
            with self.assertRaises(ValueError):
                serve([], backend, kv_block_size=4)
            with self.assertRaises(RuntimeError):
                serve([Request("oversized", 0, 4096, 1)], backend)
            self.assertEqual(backend.stats()["live_requests"], 0)


if __name__ == "__main__": unittest.main()
