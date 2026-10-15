"""FIFO static cohorts and iteration-boundary continuous batching."""
from collections import deque
from dataclasses import dataclass, field
import math
from typing import List
try:
    from .backend import Backend, Request, WorkItem
    from .kv_cache import KVCache
except ImportError:
    from backend import Backend, Request, WorkItem
    from kv_cache import KVCache


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
class RejectedRequest:
    request: Request
    decided_ms: float
    reason: str
    required_blocks: int


@dataclass
class Simulation:
    mode: str
    max_batch_size: int
    started_ms: float
    finished_ms: float
    results: List[RequestResult]
    trace: list
    rejections: list = field(default_factory=list)
    kv: dict = field(default_factory=dict)
    admission_events: list = field(default_factory=list)
    kv_policy: str = "wait"


def simulate(requests, backend: Backend, mode="continuous", max_batch_size=4,
             kv_capacity_blocks=None, kv_block_size=1, kv_policy="wait"):
    if mode not in ("static", "continuous"):
        raise ValueError("mode must be static or continuous")
    if type(max_batch_size) is not int or max_batch_size < 1:
        raise ValueError("max_batch_size must be a positive integer")
    if kv_policy not in ("wait", "reject"):
        raise ValueError("kv_policy must be wait or reject")
    cache = KVCache(kv_capacity_blocks, kv_block_size)
    incoming = list(requests)
    if any(not isinstance(r, Request) for r in incoming):
        raise ValueError("requests must contain Request objects")
    if len({r.request_id for r in incoming}) != len(incoming):
        raise ValueError("duplicate request ID")
    incoming.sort(key=lambda r: r.arrival_ms)
    if not incoming:
        return Simulation(mode, max_batch_size, 0.0, 0.0, [], [], kv=cache.summary(), kv_policy=kv_policy)
    now = started = float(incoming[0].arrival_ms)
    pending_index = 0
    waiting = deque()
    active, completed, rejected, trace, decisions = [], [], [], [], []

    def budget(request):
        return request.prompt_tokens + request.output_tokens

    def reject(request, reason):
        blocks = cache.required_blocks(budget(request))
        rejected.append(RejectedRequest(request, now, reason, blocks))
        decisions.append({"time_ms": now, "event": "reject", "request_id": request.request_id,
                          "reason": reason, "required_blocks": blocks, "available_blocks": cache.available_blocks})

    try:
        while pending_index < len(incoming) or waiting or active:
            if not active and not waiting and incoming[pending_index].arrival_ms > now:
                now = float(incoming[pending_index].arrival_ms)
            # Observe arrivals even while a static cohort is still active. A
            # permanently oversized request is rejected independently of slots.
            while pending_index < len(incoming) and incoming[pending_index].arrival_ms <= now:
                request = incoming[pending_index]
                pending_index += 1
                if not cache.can_ever_fit(budget(request)):
                    reject(request, "request_exceeds_kv_capacity")
                else:
                    waiting.append(request)
            if mode == "continuous" or not active:
                while waiting and len(active) < max_batch_size:
                    request = waiting[0]
                    if cache.reserve(request.request_id, budget(request), now):
                        waiting.popleft()
                        active.append(RequestResult(request, now))
                    elif kv_policy == "reject":
                        waiting.popleft()
                        reject(request, "kv_capacity_unavailable")
                    else:
                        decisions.append({"time_ms": now, "event": "kv_wait", "request_id": request.request_id,
                                          "required_blocks": cache.required_blocks(budget(request)),
                                          "available_blocks": cache.available_blocks})
                        break  # Strict FIFO: no bypass around a temporarily blocked head.
            if not active:
                if waiting:
                    raise RuntimeError("KV admission deadlock: feasible request cannot run in an empty cache")
                continue  # Rejections may drain this arrival group without a backend call.
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
            trace.append({"start_ms": now, "end_ms": end, "kv_reserved_blocks": cache.used_blocks,
                          "batch": [{"request_id": w.request.request_id,
                                     "phase": "prefill" if w.is_prefill else "decode",
                                     "token_index": w.generated} for w in work]})
            survivors = []
            for result, token in zip(active, step.tokens):
                result.tokens.append(token)
                result.token_times_ms.append(end)
                if len(result.tokens) == result.request.output_tokens:
                    completed.append(result)
                    cache.release(result.request.request_id, end)
                else:
                    survivors.append(result)
            active = survivors
            now = end
    finally:
        # Virtual reservations are released even if the backend or validation
        # fails. Real device resources will need terminal completion in stage 4.
        for request_id in list(cache.reservations):
            cache.release(request_id, now, reason="abort_cleanup")
    completed_by_id = {r.request.request_id: r for r in completed}
    rejected_by_id = {r.request.request_id: r for r in rejected}
    return Simulation(mode, max_batch_size, started, now,
                      [completed_by_id[r.request_id] for r in incoming if r.request_id in completed_by_id], trace,
                      [rejected_by_id[r.request_id] for r in incoming if r.request_id in rejected_by_id],
                      cache.summary(), decisions, kv_policy)
