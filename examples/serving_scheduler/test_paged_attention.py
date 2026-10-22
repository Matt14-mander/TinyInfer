"""Physical paging, direct attention and scheduler integration against CPU oracles."""
from contextlib import ExitStack
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


class PagedAttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.worker = os.environ.get("TINYINFER_SERVING_WORKER")
        if not cls.worker:
            raise unittest.SkipTest("actual worker required for paged attention acceptance")

    def paged(self, size=4, capacity=32, mode="direct"):
        return TinyInferBackend(self.worker, size, "paged", capacity, mode)

    def assert_logits(self, actual, expected):
        self.assertEqual(len(actual), 32)
        self.assertEqual(len(expected), 32)
        for a, b in zip(actual, expected):
            self.assertTrue(math.isfinite(a) and math.isfinite(b))
            self.assertLessEqual(abs(a - b), 2e-5 + 2e-5 * abs(b))

    def compare_step(self, backends, work):
        results = [backend.step(work).tokens for backend in backends]
        for result in results[1:]:
            self.assertEqual(result, results[0])
        for item_index, item in enumerate(work):
            for backend in backends:
                prefix = backend.prompt(item.request)
                if item.is_prefill:
                    prefix = prefix[:item.prefill_start + item.input_tokens]
                else:
                    prefix += backend.outputs[item.request.request_id][:-1]
                actual = backend.last_step_logits()[item_index]
                self.assert_logits(actual, backend.reference_logits(prefix))
                self.assert_logits(actual, backends[0].last_step_logits()[item_index])

    def test_three_paths_fragmentation_growth_reorder_and_reuse(self):
        a, b, c = Request("a", 0, 7, 3), Request("b", 0, 2, 1), Request("c", 0, 3, 2)
        with ExitStack() as stack:
            backends = [stack.enter_context(TinyInferBackend(self.worker, 2)),
                        stack.enter_context(self.paged(2, 10, "gather")),
                        stack.enter_context(self.paged(2, 10))]
            self.compare_step(backends, [WorkItem(a, 0, 0, 2), WorkItem(b, 0)])
            self.compare_step(backends, [WorkItem(a, 0, 2, 2)])
            for backend in backends[1:]:
                self.assertEqual(backend.block_table("a"), [0, 2])
                self.assertEqual(backend.block_table("b"), [1])
            for backend in backends:
                backend.release("b")
            self.compare_step(backends, [WorkItem(c, 0), WorkItem(a, 0, 4, 3)])
            for backend in backends[1:]:
                self.assertIn(1, backend.block_table("c"))
                self.assertFalse(set(backend.block_table("a")) & set(backend.block_table("c")))
            self.compare_step(backends, [WorkItem(a, 1, 7), WorkItem(c, 1, 3)])
            for backend in backends:
                backend.release("c")
            self.compare_step(backends, [WorkItem(a, 2, 7)])
            for backend in backends:
                backend.release("a")
                self.assertEqual(backend.stats()["live_kv_bytes"], 0)
            for backend in backends[1:]:
                self.assertEqual(backend.stats()["free_physical_blocks"], 10)
                self.assertEqual(backend.stats()["live_physical_blocks"], 0)
            self.assertEqual(backends[2].stats()["paged_attention_queries"], 15)
            self.assertEqual(backends[2].stats()["gather_attention_queries"], 0)
            self.assertEqual(backends[1].stats()["gather_attention_queries"], 15)

    def test_block_and_tail_boundaries(self):
        for size in (1, 2, 3, 4, 16):
            with ExitStack() as stack:
                backends = [stack.enter_context(TinyInferBackend(self.worker, size)),
                            stack.enter_context(self.paged(size))]
                requests = [Request("s%d-p%d" % (size, length), 0, length, 2)
                            for length in sorted(set((max(1, size - 1), size, size + 1)))]
                self.compare_step(backends, [WorkItem(r, 0) for r in requests])
                self.compare_step(backends, [WorkItem(r, 1, r.prompt_tokens) for r in reversed(requests)])
                for backend in backends:
                    backend.clear()
                self.assertEqual(backends[1].stats()["free_physical_blocks"], 32)

    def test_allocates_for_actual_cached_tokens_and_retains_pool(self):
        r = Request("growing", 0, 9, 4)
        with self.paged(4, 6) as backend:
            pool_bytes = 6 * 4 * 16 * 2 * 4
            for work, expected_blocks in ((WorkItem(r, 0, 0, 2), 1),
                                          (WorkItem(r, 0, 2, 3), 2),
                                          (WorkItem(r, 0, 5, 4), 3),
                                          (WorkItem(r, 1, 9), 3),
                                          (WorkItem(r, 2, 9), 3),
                                          (WorkItem(r, 3, 9), 3)):
                backend.step([work])
                stats = backend.stats()
                self.assertEqual(stats["live_physical_blocks"], expected_blocks)
                self.assertEqual(stats["live_kv_bytes"], expected_blocks * 4 * 128)
                self.assertEqual(stats["pool_kv_bytes"], pool_bytes)
            self.assertEqual(backend.stats()["forward_tokens"], 12)
            self.assertEqual(backend.outputs[r.request_id], backend.reference(r))
            backend.release(r.request_id)
            self.assertEqual(backend.stats()["live_kv_bytes"], 0)
            self.assertEqual(backend.stats()["pool_kv_bytes"], pool_bytes)
            self.assertEqual(backend.stats()["peak_physical_blocks"], 3)
            backend.reset()
            self.assertEqual(backend.stats()["peak_physical_blocks"], 0)
            self.assertEqual(backend.stats()["paged_attention_queries"], 0)
            self.assertEqual(backend.stats()["pool_kv_bytes"], pool_bytes)

    def test_long_context_across_many_blocks_and_chunks(self):
        r = Request("long", 0, 257, 3)
        with ExitStack() as stack:
            backends = [stack.enter_context(TinyInferBackend(self.worker, 7)),
                        stack.enter_context(self.paged(7, 40, "gather")),
                        stack.enter_context(self.paged(7, 40))]
            start = 0
            for count in (5, 67, 121, 64):
                self.compare_step(backends, [WorkItem(r, 0, start, count)])
                start += count
            self.compare_step(backends, [WorkItem(r, 1, 257)])
            self.compare_step(backends, [WorkItem(r, 2, 257)])
            self.assertEqual(len(backends[2].block_table("long")), 37)
            for backend in backends:
                backend.release("long")
                self.assertEqual(backend.stats()["live_kv_bytes"], 0)

    def test_batch_capacity_failure_and_validation_cleanup(self):
        a, b = Request("a", 0, 5, 2), Request("b", 0, 4, 1)
        with self.paged(2, 3) as backend:
            backend.step([WorkItem(a, 0, 0, 2)])
            with self.assertRaisesRegex(RuntimeError, "capacity exhausted"):
                backend.step([WorkItem(a, 0, 2, 2), WorkItem(b, 0)])
            stats = backend.stats()
            self.assertEqual(stats["forward_tokens"], 2)
            self.assertEqual(stats["linear_matmul_calls"], 7)
            self.assertEqual(stats["live_requests"], 0)
            self.assertEqual(stats["free_physical_blocks"], 3)
            self.assertEqual(stats["peak_physical_blocks"], 1)
            self.assertEqual(backend.last_step_logits(), [])
            backend.reset()
            backend.step([WorkItem(b, 0)])
            with self.assertRaisesRegex(RuntimeError, "duplicate batch request"):
                backend.step([WorkItem(a, 0, 0, 1), WorkItem(a, 0, 0, 1)])
            self.assertEqual(backend.stats()["live_physical_blocks"], 0)
            backend.reset()
            backend.step([WorkItem(a, 0)])
            backend.clear()
            self.assertEqual(backend.stats()["free_physical_blocks"], 3)
            original = backend.step
            calls = [0]

            def fail_after_write(work):
                result = original(work)
                calls[0] += 1
                if calls[0] == 2:
                    raise RuntimeError("injected failure after paged append")
                return result

            backend.step = fail_after_write
            request = Request("fails", 0, 3, 2)
            with self.assertRaisesRegex(RuntimeError, "after paged append"):
                serve([request], backend, kv_capacity_blocks=3, kv_block_size=2, token_budget=1)
            self.assertEqual(backend.stats()["live_physical_blocks"], 0)
            backend.step = original
            result = serve([request], backend, kv_capacity_blocks=3, kv_block_size=2, token_budget=1)
            self.assertEqual(result.results[0].tokens, backend.reference(request))
            self.assertEqual(result.kv["final_reserved_blocks"], 0)

    def test_scheduler_modes_budgets_pressure_and_zero_capacity(self):
        rng = random.Random(20261022)
        requests = [Request("random%d" % i, i % 3 * .1, rng.randint(1, 13), rng.randint(1, 5))
                    for i in range(12)]
        requests.append(Request("oversized", 0, 25, 2))
        with self.paged(3, 8) as backend:
            expected = {r.request_id: backend.reference(r) for r in requests}
            for mode in ("static", "continuous"):
                for budget, chunk in ((None, None), (1, 1), (9, 3)):
                    for policy in ("wait", "reject"):
                        result = serve(requests, backend, mode, 4, 8, 3, policy, budget, chunk)
                        self.assertEqual(len(result.results) + len(result.rejections), len(requests))
                        if policy == "wait":
                            self.assertEqual(len(result.results), 12)
                        for item in result.results:
                            self.assertEqual(item.tokens, expected[item.request.request_id])
                        stats = result.backend_stats
                        self.assertEqual(stats["paged_attention_queries"], stats["forward_tokens"])
                        self.assertEqual(stats["linear_matmul_calls"], 7 * len(result.trace))
                        self.assertLessEqual(stats["peak_physical_blocks"], result.kv["peak_reserved_blocks"])
                        self.assertEqual(stats["live_physical_blocks"], 0)
                        self.assertEqual(stats["free_physical_blocks"], 8)
                        self.assertEqual(result.kv["final_reserved_blocks"], 0)
        with self.paged(4, 0) as backend:
            result = serve([requests[0]], backend, kv_capacity_blocks=0, kv_block_size=4)
            self.assertEqual(len(result.rejections), 1)
            self.assertEqual(result.backend_stats["pool_kv_bytes"], 0)
            self.assertEqual(result.backend_stats["paged_attention_queries"], 0)

    def test_configuration_and_scheduler_capacity_mismatch(self):
        for options in ({"kv_layout": "unknown"}, {"kv_layout": "paged"},
                        {"kv_layout": "paged", "kv_capacity_blocks": -1},
                        {"kv_layout": "paged", "kv_capacity_blocks": True},
                        {"kv_layout": "paged", "kv_capacity_blocks": 2, "block_size": 4097},
                        {"attention_mode": "gather"}, {"kv_capacity_blocks": 2}):
            with self.assertRaises(ValueError):
                TinyInferBackend(self.worker, **options)
        with self.paged(4, 4) as backend:
            for capacity in (None, 2, 8):
                with self.assertRaisesRegex(ValueError, "physical capacity"):
                    serve([], backend, kv_block_size=4, kv_capacity_blocks=capacity)
            self.assertEqual(backend.stats()["batch_steps"], 0)


if __name__ == "__main__":
    unittest.main()
