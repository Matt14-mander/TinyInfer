import math
import unittest
try:
    from .backend import Request, SimulatedBackend, StepResult
    from .scheduler import simulate
    from .metrics import summarize
    from .benchmark import compare, workloads
except ImportError:
    from backend import Request, SimulatedBackend, StepResult
    from scheduler import simulate
    from metrics import summarize
    from benchmark import compare, workloads


class SchedulerTests(unittest.TestCase):
    def test_hand_calculated_trace(self):
        requests = [Request("a", 0, 1, 3), Request("b", 0, 1, 1), Request("c", 0, 1, 1)]
        backend = SimulatedBackend(1, 0, 0)
        a = simulate(requests, backend, "static", 2)
        b = simulate(requests, backend, "continuous", 2)
        self.assertEqual([[i["request_id"] for i in s["batch"]] for s in a.trace], [["a", "b"], ["a"], ["a"], ["c"]])
        self.assertEqual([[i["request_id"] for i in s["batch"]] for s in b.trace], [["a", "b"], ["a", "c"], ["a"]])
        self.assertEqual([r.finished_ms for r in a.results], [3, 1, 4])
        self.assertEqual([r.finished_ms for r in b.results], [3, 1, 2])
        self.assertEqual(summarize(a)["ttft_p95_ms"], 4)
        self.assertEqual(summarize(b)["ttft_p95_ms"], 2)
        self.assertAlmostEqual(summarize(a)["slot_utilization"], 5 / 8)
        self.assertAlmostEqual(summarize(b)["slot_utilization"], 5 / 6)

    def test_admission_only_at_boundaries(self):
        run = simulate([Request("a", 0, 1, 2), Request("b", .5, 1, 1)], SimulatedBackend(1, 0, 0), "continuous", 2)
        self.assertEqual(run.results[1].admitted_ms, 1)
        self.assertEqual(run.results[1].finished_ms, 2)
        self.assertEqual(run.trace[1]["batch"][1]["phase"], "prefill")
        self.assertEqual(run.trace[1]["batch"][0]["phase"], "decode")

    def test_arrival_exact_boundary(self):
        run = simulate([Request("a", 0, 1, 2), Request("b", 1, 1, 1)], SimulatedBackend(1, 0, 0), "continuous", 2)
        self.assertEqual(run.results[1].admitted_ms, 1)

    def test_idle_window_and_single_token_tpot(self):
        run = simulate([Request("b", 10, 1, 1), Request("a", 5, 1, 1)], SimulatedBackend(1, 0, 0), "continuous", 4)
        metrics = summarize(run)
        self.assertEqual([r.request.request_id for r in run.results], ["a", "b"])
        self.assertEqual(metrics["measurement_window_ms"], 6)
        self.assertEqual(metrics["ttft_p95_ms"], 1)
        self.assertIsNone(metrics["tpot_mean_ms"])
        self.assertAlmostEqual(metrics["output_tokens_per_second"], 2000 / 6)

    def test_tpot_with_prefill_stall(self):
        run = simulate([Request("a", 0, 1, 3), Request("b", 2, 10, 1)], SimulatedBackend(1, 1, 0), "continuous", 2)
        a = run.results[0]
        self.assertEqual(a.token_times_ms, [2, 13, 14])
        self.assertEqual(summarize(run)["requests"][0]["tpot_ms"], 6)

    def test_fifo_at_equal_arrival(self):
        run = simulate([Request(n, 0, 1, 1) for n in ("z", "a", "m")], SimulatedBackend(1, 0, 0), "continuous", 1)
        self.assertEqual([r.admitted_ms for r in run.results], [0, 1, 2])

    def test_single_slot_equivalence(self):
        r = compare(workloads("staggered"), 1)
        a, b = r["reports"]
        self.assertEqual(a["requests"], b["requests"])

    def test_outputs_and_capacity(self):
        for scenario in ("mixed", "staggered", "long_prefill"):
            for batch in (1, 2, 4, 20):
                report = compare(workloads(scenario), batch, trace=True)
                self.assertTrue(report["same_output_sequences"])
                for r in report["reports"]:
                    self.assertEqual(len(r["requests"]), len(workloads(scenario)))
                    for step in r["trace"]:
                        self.assertTrue(0 < len(step["batch"]) <= batch)
                        self.assertEqual(len({i["request_id"] for i in step["batch"]}), len(step["batch"]))
                    for request in r["requests"]:
                        self.assertGreaterEqual(request["admitted_ms"], request["arrival_ms"])
                        self.assertEqual(request["token_times_ms"], sorted(request["token_times_ms"]))

    def test_output_independent_of_peers(self):
        requests = workloads("mixed")
        batched = simulate(requests, SimulatedBackend(), "continuous", 4)
        for result in batched.results:
            alone = simulate([result.request], SimulatedBackend(), "static", 1)
            self.assertEqual(result.tokens, alone.results[0].tokens)
            self.assertEqual(len(result.tokens), result.request.output_tokens)

    def test_empty(self):
        result = summarize(simulate([], SimulatedBackend()))
        self.assertEqual(result["steps"], 0)
        self.assertEqual(result["output_tokens_per_second"], 0)
        self.assertIsNone(result["request_latency_p95_ms"])

    def test_invalid_requests(self):
        for kwargs in ({"arrival_ms": math.nan}, {"arrival_ms": -1}, {"output_tokens": 0}, {"prompt_tokens": True}, {"output_tokens": 1.5}):
            params = dict(request_id="a", arrival_ms=0, prompt_tokens=1, output_tokens=1)
            params.update(kwargs)
            with self.assertRaises(ValueError):
                Request(**params)
        r = Request("a", 0, 1, 1)
        with self.assertRaises(ValueError):
            simulate([r, r], SimulatedBackend())
        for batch in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                simulate([r], SimulatedBackend(), max_batch_size=batch)
        with self.assertRaises(ValueError):
            simulate([r], SimulatedBackend(), mode="unknown")

    def test_invalid_costs_and_backend(self):
        for kwargs in ({"overhead_ms": -1}, {"decode_ms_per_request": math.inf}, {"overhead_ms": 0, "decode_ms_per_request": 0}):
            with self.assertRaises(ValueError):
                SimulatedBackend(**kwargs)
        class Broken:
            def __init__(self, result): self.result = result
            def step(self, batch): return self.result
        for value in (StepResult(0, [1]), StepResult(math.nan, [1]), StepResult(1, []), StepResult(1, [True])):
            with self.assertRaises(ValueError):
                simulate([Request("a", 0, 1, 1)], Broken(value))


if __name__ == "__main__":
    unittest.main()
