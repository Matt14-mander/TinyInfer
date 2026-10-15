"""FIFO static cohorts and iteration-boundary continuous batching."""
from dataclasses import dataclass, field
import math
from typing import List
try:
    from .backend import Backend, Request, WorkItem
except ImportError:
    from backend import Backend, Request, WorkItem


@dataclass
class RequestResult:
    request: Request
    admitted_ms: float
    token_times_ms: List[float] = field(default_factory=list)
    tokens: List[int] = field(default_factory=list)

    @property
    def finished_ms(self):
        return self.token_times_ms[-1]


@dataclass
class Simulation:
    mode: str
    max_batch_size: int
    started_ms: float
    finished_ms: float
    results: List[RequestResult]
    trace: list


def simulate(requests, backend: Backend, mode="continuous", max_batch_size=4):
    if mode not in ("static", "continuous"):
        raise ValueError("mode must be static or continuous")
    if type(max_batch_size) is not int or max_batch_size < 1:
        raise ValueError("max_batch_size must be a positive integer")
    incoming = list(requests)
    if any(not isinstance(r, Request) for r in incoming):
        raise ValueError("requests must contain Request objects")
    if len({r.request_id for r in incoming}) != len(incoming):
        raise ValueError("duplicate request ID")
    # Python's stable sort preserves supplied FIFO order at equal arrival times.
    incoming.sort(key=lambda r: r.arrival_ms)
    if not incoming:
        return Simulation(mode, max_batch_size, 0.0, 0.0, [], [])
    now = started = float(incoming[0].arrival_ms)
    pending_index = 0
    active = []
    completed = []
    trace = []
    while pending_index < len(incoming) or active:
        if not active and incoming[pending_index].arrival_ms > now:
            now = float(incoming[pending_index].arrival_ms)
        # Static cohorts keep holes until the entire cohort has finished.
        if mode == "continuous" or not active:
            while (len(active) < max_batch_size and pending_index < len(incoming)
                   and incoming[pending_index].arrival_ms <= now):
                active.append(RequestResult(incoming[pending_index], now))
                pending_index += 1
        if not active:
            raise RuntimeError("scheduler failed to make progress")
        work = [WorkItem(r.request, len(r.tokens)) for r in active]
        step = backend.step(tuple(work))
        if (isinstance(step.duration_ms, bool) or not isinstance(step.duration_ms, (int, float))
                or not math.isfinite(step.duration_ms) or step.duration_ms <= 0):
            raise ValueError("backend must return finite positive step duration")
        if len(step.tokens) != len(active) or any(type(t) is not int or t < 0 for t in step.tokens):
            raise ValueError("backend must return one nonnegative integer token per item")
        end = now + step.duration_ms
        if not math.isfinite(end) or end <= now:
            raise ValueError("virtual clock overflow or insufficient precision")
        trace.append({"start_ms": now, "end_ms": end,
                      "batch": [{"request_id": w.request.request_id,
                                 "phase": "prefill" if w.is_prefill else "decode",
                                 "token_index": w.generated} for w in work]})
        survivors = []
        for result, token in zip(active, step.tokens):
            result.tokens.append(token)
            result.token_times_ms.append(end)
            if len(result.tokens) == result.request.output_tokens:
                completed.append(result)
            else:
                survivors.append(result)
        active = survivors
        now = end
    # Return in arrival/FIFO order, not completion order.
    by_id = {r.request.request_id: r for r in completed}
    return Simulation(mode, max_batch_size, started, now,
                      [by_id[r.request_id] for r in incoming], trace)
