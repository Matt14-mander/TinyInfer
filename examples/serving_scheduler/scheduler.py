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
    prefilled_tokens: int = 0

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
    token_budget: int = None
    prefill_chunk_size: int = None


def simulate(requests, backend: Backend, mode="continuous", max_batch_size=4,
             kv_capacity_blocks=None, kv_block_size=1, kv_policy="wait",
             token_budget=None, prefill_chunk_size=None):
    if mode not in ("static", "continuous"):
        raise ValueError("mode must be static or continuous")
    if type(max_batch_size) is not int or max_batch_size < 1:
        raise ValueError("max_batch_size must be a positive integer")
    if kv_policy not in ("wait", "reject"):
        raise ValueError("kv_policy must be wait or reject")
    for name, value in (("token_budget", token_budget), ("prefill_chunk_size", prefill_chunk_size)):
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError(name + " must be a positive integer or None")
    if prefill_chunk_size is not None and token_budget is None:
        raise ValueError("prefill_chunk_size requires token_budget")
    cache = KVCache(kv_capacity_blocks, kv_block_size)
    incoming = list(requests)
    if any(not isinstance(r, Request) for r in incoming):
        raise ValueError("requests must contain Request objects")
    if len({r.request_id for r in incoming}) != len(incoming):
        raise ValueError("duplicate request ID")
    incoming.sort(key=lambda r: r.arrival_ms)
    if not incoming:
        return Simulation(mode, max_batch_size, 0.0, 0.0, [], [], kv=cache.summary(), kv_policy=kv_policy,
                          token_budget=token_budget, prefill_chunk_size=prefill_chunk_size)
    now = started = float(incoming[0].arrival_ms)
    pending_index = 0
    waiting = deque()
    active, completed, rejected, trace, decisions = [], [], [], [], []

    last_prefill_id = None

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
            # Every existing decoder gets one token before prefill uses the
            # remaining budget. New decoders only emerge from budgeted prefill,
            # so their count cannot exceed the next iteration's token budget.
            decoders = [r for r in active if r.tokens]
            prefills = [r for r in active if not r.tokens]
            work_results = list(decoders)
            work = [WorkItem(r.request, len(r.tokens), r.prefilled_tokens) for r in decoders]
            remaining = None if token_budget is None else token_budget - len(work)
            if remaining is not None and remaining < 0:
                raise RuntimeError("decode count exceeds token budget invariant")
            if token_budget is not None:
                ids = [r.request.request_id for r in prefills]
                if last_prefill_id in ids:
                    start = ids.index(last_prefill_id) + 1
                    prefills = prefills[start:] + prefills[:start]
            for result in prefills:
                if remaining == 0:
                    break
                count = result.request.prompt_tokens - result.prefilled_tokens
                if remaining is not None:
                    count = min(count, remaining)
                if prefill_chunk_size is not None:
                    count = min(count, prefill_chunk_size)
                work.append(WorkItem(result.request, 0, result.prefilled_tokens, count))
                work_results.append(result)
                last_prefill_id = result.request.request_id
                if remaining is not None:
                    remaining -= count
            if not work:
                raise RuntimeError("token scheduling failed to make progress")
            step = backend.step(tuple(work))
            if (isinstance(step.duration_ms, bool) or not isinstance(step.duration_ms, (int, float))
                    or not math.isfinite(step.duration_ms) or step.duration_ms <= 0):
                raise ValueError("backend must return finite positive step duration")
            if len(step.tokens) != len(work):
                raise ValueError("backend must return one result per work item")
            for item, token in zip(work, step.tokens):
                if item.emits_token:
                    if type(token) is not int or token < 0:
                        raise ValueError("finished prefill/decode must return a nonnegative integer token")
                elif token is not None:
                    raise ValueError("unfinished prefill must return None, not an early token")
            end = now + step.duration_ms
            if not math.isfinite(end) or end <= now:
                raise ValueError("virtual clock overflow or insufficient precision")
            trace.append({"start_ms": now, "end_ms": end, "kv_reserved_blocks": cache.used_blocks,
                          "active_requests": len(active),
                          "deferred_requests": [r.request.request_id for r in active if r not in work_results],
                          "input_tokens": sum(w.input_tokens for w in work),
                          "decode_tokens": len(decoders),
                          "prefill_tokens": sum(w.input_tokens for w in work if w.is_prefill),
                          "batch": [{"request_id": w.request.request_id,
                                     "phase": "prefill" if w.is_prefill else "decode",
                                     "token_index": w.generated,
                                     "input_tokens": w.input_tokens,
                                     "prefill_start": w.prefill_start if w.is_prefill else None,
                                     "prefill_end": w.prefill_start + w.input_tokens if w.is_prefill else None,
                                     "emits_token": w.emits_token} for w in work]})
            finished_ids = set()
            for result, item, token in zip(work_results, work, step.tokens):
                if item.is_prefill:
                    result.prefilled_tokens += item.input_tokens
                if token is not None:
                    result.tokens.append(token)
                    result.token_times_ms.append(end)
                if len(result.tokens) == result.request.output_tokens:
                    completed.append(result)
                    finished_ids.add(result.request.request_id)
                    cache.release(result.request.request_id, end)
            active = [r for r in active if r.request.request_id not in finished_ids]
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
                      cache.summary(), decisions, kv_policy, token_budget, prefill_chunk_size)
