import unittest
from unittest import mock
try:
    from .backend import Request, SimulatedBackend, StepResult, WorkItem
    from .scheduler import simulate
    from .metrics import summarize
    from .benchmark import compare, workloads
    from .kv_cache import KVCache
    from . import scheduler as scheduler_module
except ImportError:
    from backend import Request, SimulatedBackend, StepResult, WorkItem
    from scheduler import simulate
    from metrics import summarize
    from benchmark import compare, workloads
    from kv_cache import KVCache
    import scheduler as scheduler_module


class ChunkedPrefillTests(unittest.TestCase):
    def test_partial_prefill_no_early_token_and_exact_progress(self):
        run = simulate([Request("a", 0, 5, 2)], SimulatedBackend(1, 1, 1),
                       token_budget=2, prefill_chunk_size=2, kv_capacity_blocks=7)
        self.assertEqual(run.results[0].token_times_ms, [8, 10])
        self.assertEqual(run.results[0].prefilled_tokens, 5)
        work = [s["batch"][0] for s in run.trace]
        self.assertEqual([(w["prefill_start"], w["prefill_end"]) for w in work[:3]], [(0, 2), (2, 4), (4, 5)])
        self.assertEqual([w["emits_token"] for w in work], [False, False, True, True])
        self.assertEqual([e["event"] for e in run.kv["events"]], ["reserve", "release"])
        self.assertEqual(run.kv["events"][-1]["time_ms"], 10)
        self.assertEqual(summarize(run)["prefill_input_tokens"], 5)
        self.assertEqual(summarize(run)["decode_input_tokens"], 1)

    def test_decode_priority_hand_calculated_latency_tradeoff(self):
        requests = [Request("a", 0, 1, 4), Request("b", 2, 6, 1)]
        backend = SimulatedBackend(1, 1, 0)
        full = simulate(requests, backend, max_batch_size=2)
        chunked = simulate(requests, backend, max_batch_size=2, token_budget=3, prefill_chunk_size=2)
        self.assertEqual(full.results[0].token_times_ms, [2, 9, 10, 11])
        self.assertEqual(chunked.results[0].token_times_ms, [2, 5, 8, 11])
        self.assertEqual(full.results[1].token_times_ms, [9])
        self.assertEqual(chunked.results[1].token_times_ms, [11])
        self.assertEqual(summarize(full)["max_inter_token_ms"], 7)
        self.assertEqual(summarize(chunked)["max_inter_token_ms"], 3)
        for step in chunked.trace[1:]:
            self.assertEqual(step["batch"][0]["phase"], "decode")
            self.assertEqual(step["input_tokens"], 3)
        self.assertEqual([r.tokens for r in full.results], [r.tokens for r in chunked.results])

    def test_round_robin_prefill_short_request_gets_progress(self):
        run = simulate([Request("long", 0, 10, 1), Request("short", 0, 1, 1)],
                       SimulatedBackend(1, 0, 0), max_batch_size=2, token_budget=2, prefill_chunk_size=2)
        self.assertEqual(run.trace[0]["batch"][0]["request_id"], "long")
        self.assertEqual(run.trace[1]["batch"][0]["request_id"], "short")
        self.assertEqual(run.results[1].finished_ms, 2)
        self.assertLess(run.results[1].finished_ms, run.results[0].finished_ms)

    def test_budget_smaller_than_batch_progress_and_residency(self):
        run = simulate([Request(str(i), 0, 1, 3) for i in range(4)], SimulatedBackend(1, 0, 0),
                       max_batch_size=4, token_budget=1, kv_capacity_blocks=16)
        self.assertEqual(len(run.results), 4)
        self.assertEqual([r.token_times_ms[0] for r in run.results], [1, 4, 7, 10])
        self.assertTrue(all(s["input_tokens"] == 1 for s in run.trace))
        self.assertEqual(run.trace[0]["active_requests"], 4)
        self.assertEqual(len(run.trace[0]["deferred_requests"]), 3)
        self.assertEqual(run.kv["peak_reserved_blocks"], 16)
        m = summarize(run)
        self.assertGreater(m["slot_utilization"], m["scheduled_slot_utilization"])

    def test_large_budget_reproduces_unchunked_results(self):
        for mode in ("static", "continuous"):
            baseline = simulate(workloads("staggered"), SimulatedBackend(), mode)
            chunked = simulate(workloads("staggered"), SimulatedBackend(), mode, token_budget=10000)
            self.assertEqual(summarize(baseline)["requests"], summarize(chunked)["requests"])
            self.assertEqual(baseline.finished_ms, chunked.finished_ms)

    def test_all_modes_budgets_kv_policies_and_token_conservation(self):
        for scenario in ("mixed", "staggered", "long_prefill", "kv_pressure"):
            for policy in ("wait", "reject"):
                for budget in (1, 3, 16):
                    report = compare(workloads(scenario), 4, trace=True, kv_capacity_blocks=64,
                                     kv_block_size=4, kv_policy=policy, token_budget=budget, prefill_chunk_size=3)
                    self.assertTrue(report["same_output_sequences"])
                    for run in report["reports"]:
                        self.assertEqual(run["submitted_requests"], run["completed_requests"] + run["rejected_requests"])
                        self.assertEqual(run["kv"]["final_reserved_blocks"], 0)
                        requests = {r.request_id: r for r in workloads(scenario)}
                        progress = {}
                        emitted = {}
                        for step in run["trace"]:
                            self.assertTrue(0 < step["input_tokens"] <= budget)
                            self.assertEqual(step["input_tokens"], step["prefill_tokens"] + step["decode_tokens"])
                            self.assertLessEqual(len(step["batch"]), 4)
                            for w in step["batch"]:
                                key = w["request_id"]
                                if w["phase"] == "prefill":
                                    self.assertEqual(w["prefill_start"], progress.get(key, 0))
                                    self.assertLessEqual(w["input_tokens"], 3)
                                    progress[key] = w["prefill_end"]
                                    self.assertEqual(w["emits_token"], progress[key] == requests[key].prompt_tokens)
                                else:
                                    self.assertEqual(progress[key], requests[key].prompt_tokens)
                                if w["emits_token"]:
                                    emitted[key] = emitted.get(key, 0) + 1
                        for result in run["requests"]:
                            key = result["request_id"]
                            self.assertEqual(progress[key], requests[key].prompt_tokens)
                            self.assertEqual(emitted[key], requests[key].output_tokens)

    def test_backend_rejects_invalid_chunk_ranges(self):
        backend = SimulatedBackend()
        request = Request("a", 0, 4, 1)
        for item in (WorkItem(request, 0, 0, 0), WorkItem(request, 0, -1, 1),
                     WorkItem(request, 0, 3, 2), WorkItem(request, 0, 0, True)):
            with self.assertRaises(ValueError): backend.step([item])

    def test_no_early_or_missing_tokens_and_error_cleanup(self):
        class Wrong:
            def __init__(self, token): self.token = token
            def step(self, batch): return StepResult(1, [self.token] * len(batch))
        for prompt, token in ((4, 1), (1, None)):
            pool = KVCache(10)
            with mock.patch.object(scheduler_module, "KVCache", return_value=pool):
                with self.assertRaises(ValueError):
                    simulate([Request("a", 0, prompt, 2)], Wrong(token),
                             kv_capacity_blocks=10, token_budget=2)
            self.assertEqual(pool.used_blocks, 0)

    def test_failure_after_partial_prefill_cleanup(self):
        class Fail:
            def __init__(self): self.calls = 0
            def step(self, batch):
                self.calls += 1
                if self.calls == 2: raise RuntimeError("partial failure")
                return SimulatedBackend().step(batch)
        pool = KVCache(10)
        with mock.patch.object(scheduler_module, "KVCache", return_value=pool):
            with self.assertRaises(RuntimeError):
                simulate([Request("a", 0, 6, 2)], Fail(), kv_capacity_blocks=10, token_budget=2)
        self.assertEqual(pool.used_blocks, 0)
        self.assertEqual(pool.events[-1]["reason"], "abort_cleanup")

    def test_invalid_configuration_empty_and_zero_capacity(self):
        for kwargs in ({"token_budget": 0}, {"token_budget": True}, {"token_budget": 1.5},
                       {"prefill_chunk_size": 2}, {"token_budget": 2, "prefill_chunk_size": -1}):
            with self.assertRaises(ValueError): simulate([], SimulatedBackend(), **kwargs)
        empty = simulate([], SimulatedBackend(), token_budget=1)
        self.assertEqual(summarize(empty)["peak_step_input_tokens"], 0)
        run = simulate([Request("a", 0, 4, 2)], SimulatedBackend(), token_budget=1, kv_capacity_blocks=0)
        self.assertEqual(run.trace, [])
        self.assertEqual(len(run.rejections), 1)


if __name__ == "__main__": unittest.main()
