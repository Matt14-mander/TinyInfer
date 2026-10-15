"""Wall-clock driver: actual arrival waits and completed backend token publication."""
import time
try:
    from .scheduler import _schedule
except ImportError:
    from scheduler import _schedule


class WallClock:
    def __init__(self):
        self.origin = time.perf_counter_ns()

    def now_ms(self):
        return (time.perf_counter_ns() - self.origin) / 1000000.0

    def wait_until(self, arrival_ms):
        while True:
            delay = arrival_ms - self.now_ms()
            if delay <= 0:
                return
            time.sleep(delay / 1000.0)


def serve(requests, backend, mode="continuous", max_batch_size=4,
          kv_capacity_blocks=None, kv_block_size=1, kv_policy="wait",
          token_budget=None, prefill_chunk_size=None):
    if not getattr(backend, "real_backend", False):
        raise ValueError("serve requires a completed real inference backend")
    if getattr(backend, "block_size", kv_block_size) != kv_block_size:
        raise ValueError("backend allocation block size must match scheduler KV block size")
    # Model startup and reset are outside the request latency window.
    backend.reset()
    try:
        return _schedule(requests, backend, mode, max_batch_size, kv_capacity_blocks,
                         kv_block_size, kv_policy, token_budget, prefill_chunk_size, WallClock())
    except BaseException:
        try:
            backend.clear()
        except Exception:
            backend.close()
        raise
