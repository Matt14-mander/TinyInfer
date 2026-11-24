import unittest
from collections import deque
from unittest.mock import patch
from miniprefix.bindings.python.client import NativeEngine


class LifecycleClientTests(unittest.TestCase):
    def engine(self):
        e = NativeEngine.__new__(NativeEngine)
        e.ids, e.pending, e.history = {}, set(), deque()
        e.history_capacity, e.history_ttl_ms, e.next_handle = 2, 10, 1
        e.eos = 2
        e.commands = []
        e._command = lambda fields: e.commands.append(fields) or []
        return e

    def test_timeout_and_bounded_history(self):
        e = self.engine()
        with patch('miniprefix.bindings.python.client.time.monotonic', return_value=0):
            e.submit('a', [1], 2, None, timeout_ms=5)
            self.assertEqual(e.commands[-1], ['SUBMIT_TIMEOUT', 1, 2, -1, 5, 1, 1])
            self.assertEqual(e._events([1, 1, -1, 4])[0]['finish_reason'], 'timeout')
            with self.assertRaises(ValueError): e.submit('a', [1])
        with patch('miniprefix.bindings.python.client.time.monotonic', return_value=.01):
            e.submit('a', [1], 1, None)
            self.assertEqual(e.ids['a'], 2)  # Never recycle a wire handle.
            e._events([1, 2, 2, 1])
            for i in range(20):
                key=str(i);e.submit(key, [1], 1, None)
                e._events([1, e.ids[key], 2, 1])
                self.assertLessEqual(len(e.ids), 2)

    def test_active_pin_and_rejected_submission(self):
        e = self.engine()
        e.history_capacity=0
        e.submit('a', [1], 2, None)
        with patch('miniprefix.bindings.python.client.time.monotonic', return_value=100):
            with self.assertRaises(ValueError): e.submit('a', [1])
        before=e.next_handle
        def reject(_): raise RuntimeError('waiting queue full')
        e._command=reject
        with self.assertRaises(RuntimeError): e.submit('b', [1])
        self.assertNotIn('b', e.ids)
        self.assertEqual(e.next_handle, before)
        events=e._events([2, 1, 2, -1, 1, -1, 4])
        self.assertEqual(len(events), 2)
        self.assertFalse(e.ids)
        self.assertFalse(e.pending)


if __name__ == '__main__':
    unittest.main()
