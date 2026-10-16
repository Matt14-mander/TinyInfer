import unittest
from unittest import mock
try:
    from .kv_cache import KVCache
    from .backend import Request, SimulatedBackend, StepResult
    from .benchmark import compare
    from .metrics import summarize
    from . import scheduler as scheduler_module
except ImportError:
    from kv_cache import KVCache
    from backend import Request, SimulatedBackend, StepResult
    from benchmark import compare
    from metrics import summarize
    import scheduler as scheduler_module
simulate = scheduler_module.simulate


class KVCacheTests(unittest.TestCase):
    def test_rounding_exact_fit_and_atomic_failed_reserve(self):
        cache = KVCache(3, 4)
        self.assertEqual(cache.required_blocks(5), 2)
        self.assertTrue(cache.reserve("a", 5, 0))
        before = cache.summary()
        self.assertFalse(cache.reserve("b", 5, 0))
        self.assertEqual(cache.summary(), before)
        self.assertTrue(cache.reserve("b", 4, 0))
        self.assertEqual(cache.used_blocks, 3)
        cache.release("a", 1)
        cache.release("b", 1)
        self.assertEqual(cache.used_blocks, 0)
        self.assertEqual(cache.peak_blocks, 3)
        with self.assertRaises(ValueError):
            cache.release("a", 2)

    def test_invalid_and_duplicate_reservations(self):
        for cap, block in ((-1, 1), (True, 1), (1.5, 1), (1, 0), (1, True)):
            with self.assertRaises(ValueError): KVCache(cap, block)
        cache = KVCache()
        cache.reserve("a", 1, 0)
        with self.assertRaises(ValueError): cache.reserve("a", 1, 0)
        for budget in (0, -1, True, 1.5):
            with self.assertRaises(ValueError): cache.required_blocks(budget)

    def test_reservation_snapshot_not_mutable(self):
        cache = KVCache(2)
        cache.reserve("a", 2, 0)
        cache.reservations.clear()
        self.assertEqual(cache.used_blocks, 2)
        self.assertIn("a", cache.reservations)


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.backend = SimulatedBackend(1, 0, 0)

    def run_policy(self, requests, mode="continuous", capacity=4, block=1, policy="wait", batch=4):
        return simulate(requests, self.backend, mode, batch, capacity, block, policy)

    def test_reserve_entire_budget_not_only_prompt(self):
        run = self.run_policy([Request("large", 0, 1, 4)], capacity=4)
        self.assertEqual(run.trace, [])
        self.assertEqual(run.rejections[0].required_blocks, 5)
        self.assertEqual(run.rejections[0].reason, "request_exceeds_kv_capacity")

    def test_zero_capacity_and_all_rejected_metrics(self):
        run = self.run_policy([Request("a", 2, 1, 1), Request("b", 5, 1, 1)], capacity=0)
        m = summarize(run)
        self.assertEqual((m["submitted_requests"], m["completed_requests"], m["rejected_requests"]), (2, 0, 2))
        self.assertEqual(m["measurement_window_ms"], 3)
        self.assertEqual(m["output_tokens_per_second"], 0)
        self.assertEqual(m["rejection_rate"], 1)
        self.assertIsNone(m["ttft_p95_ms"])
        self.assertIsNone(m["request_latency_p95_ms"])
        self.assertEqual(m["kv"]["final_reserved_blocks"], 0)

    def test_rounding_can_reject_raw_token_fit(self):
        run = self.run_policy([Request("a", 0, 4, 1)], capacity=1, block=4)
        self.assertEqual(run.rejections[0].required_blocks, 2)
        run = self.run_policy([Request("a", 0, 3, 1)], capacity=1, block=4)
        self.assertEqual(len(run.results), 1)
        self.assertEqual(run.kv["peak_reserved_blocks"], 1)

    def test_temporarily_full_waits_then_releases(self):
        requests = [Request("a", 0, 1, 3), Request("b", 0, 1, 1)]
        for mode in ("static", "continuous"):
            run = self.run_policy(requests, mode, capacity=4)
            self.assertFalse(run.rejections)
            self.assertEqual([r.admitted_ms for r in run.results], [0, 3])
            self.assertEqual(run.kv["final_reserved_blocks"], 0)
            self.assertEqual(summarize(run)["kv_waited_requests"], 1)
            reserve_b = next(e for e in run.kv["events"] if e["event"] == "reserve" and e["request_id"] == "b")
            release_a = next(e for e in run.kv["events"] if e["event"] == "release" and e["request_id"] == "a")
            self.assertEqual(reserve_b["time_ms"], release_a["time_ms"])

    def test_fifo_head_blocking_without_bypass(self):
        requests = [Request("a", 0, 1, 3), Request("head", 0, 1, 2), Request("small", 0, 1, 1)]
        run = self.run_policy(requests, capacity=6)
        self.assertEqual([r.admitted_ms for r in run.results], [0, 3, 3])
        self.assertTrue(any(e["event"] == "kv_wait" and e["request_id"] == "head" for e in run.admission_events))
        self.assertEqual(run.kv["peak_reserved_blocks"], 5)

    def test_reject_pressure_does_not_block_smaller_requests(self):
        requests = [Request("a", 0, 1, 3), Request("head", 0, 1, 2), Request("small", 0, 1, 1)]
        run = self.run_policy(requests, capacity=6, policy="reject")
        self.assertEqual([r.request.request_id for r in run.results], ["a", "small"])
        self.assertEqual([r.admitted_ms for r in run.results], [0, 0])
        self.assertEqual(run.rejections[0].reason, "kv_capacity_unavailable")
        self.assertEqual(run.kv["peak_reserved_blocks"], 6)

    def test_static_cohort_keeps_empty_slot_despite_free_cache(self):
        requests = [Request("a", 0, 1, 3), Request("b", 0, 1, 1), Request("c", 0, 1, 1)]
        static = self.run_policy(requests, "static", capacity=6, batch=2)
        continuous = self.run_policy(requests, "continuous", capacity=6, batch=2)
        self.assertEqual(static.results[-1].admitted_ms, 3)
        self.assertEqual(continuous.results[-1].admitted_ms, 1)

    def test_oversized_rejection_during_static_cohort(self):
        requests = [Request("a", 0, 1, 3), Request("too_big", .5, 10, 1)]
        run = self.run_policy(requests, "static", capacity=4)
        self.assertEqual(run.rejections[0].decided_ms, 1)
        self.assertEqual(run.rejections[0].reason, "request_exceeds_kv_capacity")

    def test_batch_full_is_queued_not_kv_rejected(self):
        run = self.run_policy([Request("a", 0, 1, 1), Request("b", 0, 1, 1)], capacity=2, policy="reject", batch=1)
        self.assertFalse(run.rejections)
        self.assertEqual([r.admitted_ms for r in run.results], [0, 1])

    def test_different_acceptance_sets_are_not_false_correctness_failure(self):
        requests = [Request("long", 0, 1, 4), Request("peer", 0, 1, 1), Request("later", 1, 1, 3)]
        report = compare(requests, 2, self.backend, True, 7, 1, "reject")
        self.assertTrue(report["same_output_sequences"])
        self.assertFalse(report["same_completed_request_ids"])
        self.assertEqual(report["reference_verified_requests"], 3)
        self.assertEqual([r["completed_requests"] for r in report["reports"]], [3, 2])

    def test_no_overcommit_no_leak_and_all_requests_terminal(self):
        requests = [Request("r%d" % i, i / 2, 1 + i % 7, 1 + i % 4) for i in range(16)]
        for mode in ("static", "continuous"):
            for policy in ("wait", "reject"):
                for capacity in (0, 1, 3, 8, None):
                    run = self.run_policy(requests, mode, capacity=capacity, block=3, policy=policy)
                    self.assertEqual(len(run.results) + len(run.rejections), len(requests))
                    self.assertEqual(run.kv["final_reserved_blocks"], 0)
                    self.assertEqual(run.kv["final_reservations"], 0)
                    live = {}
                    for event in run.kv["events"]:
                        key = event["request_id"]
                        if event["event"] == "reserve":
                            self.assertNotIn(key, live)
                            live[key] = event["blocks"]
                        else:
                            self.assertEqual(live.pop(key), event["blocks"])
                        self.assertEqual(sum(live.values()), event["used_blocks"])
                        if capacity is not None:
                            self.assertTrue(0 <= event["used_blocks"] <= capacity)
                    self.assertFalse(live)
                    for result in run.results:
                        single = simulate([result.request], self.backend)
                        self.assertEqual(result.tokens, single.results[0].tokens)

    def test_backend_failure_and_invalid_output_cleanup(self):
        class Broken:
            def step(self, batch): raise RuntimeError("backend failed")
        class Invalid:
            def step(self, batch): return StepResult(0, [1] * len(batch))
        for backend, error in ((Broken(), RuntimeError), (Invalid(), ValueError)):
            pool = KVCache(4)
            with mock.patch.object(scheduler_module, "KVCache", return_value=pool):
                with self.assertRaises(error):
                    simulate([Request("a", 0, 1, 2)], backend, kv_capacity_blocks=4)
            self.assertEqual(pool.used_blocks, 0)
            self.assertEqual(pool.reservations, {})
            self.assertEqual(pool.events[-1]["reason"], "abort_cleanup")

    def test_failure_after_partial_completion_releases_only_live_requests(self):
        class FailLater:
            def __init__(self): self.calls = 0
            def step(self, batch):
                self.calls += 1
                if self.calls == 2: raise RuntimeError("later failure")
                return StepResult(1, [1] * len(batch))
        pool = KVCache(6)
        with mock.patch.object(scheduler_module, "KVCache", return_value=pool):
            with self.assertRaises(RuntimeError):
                simulate([Request("short", 0, 1, 1), Request("long", 0, 1, 3)],
                         FailLater(), kv_capacity_blocks=6)
        releases = [e for e in pool.events if e["event"] == "release"]
        self.assertEqual([(e["request_id"], e["reason"]) for e in releases],
                         [("short", "completed"), ("long", "abort_cleanup")])
        self.assertEqual(pool.used_blocks, 0)

    def test_invalid_policy_and_empty_with_capacity(self):
        with self.assertRaises(ValueError):
            simulate([], self.backend, kv_policy="skip")
        run = simulate([], self.backend, kv_capacity_blocks=0)
        self.assertEqual(summarize(run)["rejection_rate"], 0)
        self.assertEqual(run.kv["final_reserved_blocks"], 0)


if __name__ == "__main__":
    unittest.main()
