import unittest
from miniprefix.benchmarks.sustained import arrivals, window_metrics
from miniprefix.bindings.python.client import NativeEngine


class SustainedMeasurements(unittest.TestCase):
    def test_same_open_loop_schedule_across_cache_modes(self):
        prompts = [{'prompt_tokens': [1]}, {'prompt_tokens': [2]}]
        requests = arrivals(prompts, 10, 20, 9)
        self.assertEqual(requests, arrivals(prompts, 10, 20, 9))
        for i, request in enumerate(requests):
            self.assertTrue(i*50 <= request['arrival_ms'] < (i+1)*50)
            self.assertEqual(request['prompt_tokens'], prompts[i%2]['prompt_tokens'])

    def test_service_window_and_backlog(self):
        rows = [dict(arrival_ms=t, token_times_ms=times) for t, times in
                [(0, [4, 6]), (1, [5, 8]), (2, [7, 10])]]
        speed, queued = window_metrics(rows, 3, 9, 1)
        self.assertAlmostEqual(speed, 5000/6)
        self.assertAlmostEqual(queued, 5/6)
        # No arrival/completion inside the window. Count its trailing backlog.
        self.assertEqual(window_metrics(rows, 4, 5, 1), (1000, 1))

    def test_execution_timers_wire_format(self):
        engine = NativeEngine.__new__(NativeEngine)
        commands = []
        def respond(fields):
            commands.append(fields)
            return [4, 90, 60, 80, 1]
        engine._command = respond
        engine.set_execution_timing(True)
        self.assertEqual(engine.execution_stats(), dict(forwards=4, forward_download_ns=90,
                         host_forward_ns=60, stream_forward_ns=80, admission_plan_ns=1))
        self.assertEqual(commands, [['EXEC_TIMING', 1], ['EXEC_STATS']])
        engine._command = lambda _: []
        with self.assertRaises(RuntimeError):
            engine.execution_stats()
