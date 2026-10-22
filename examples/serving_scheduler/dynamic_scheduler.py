"""Finite-workload continuous scheduling with deterministic preempt/recompute.

The oldest active request is protected; younger requests release native KV and
replay their previously cached prefix, without re-delivering sampled tokens.
Replay rows share the same packed worker STEP and input token budget.
"""
from collections import deque
from dataclasses import dataclass
import sys
try:
    from .backend import Request, WorkItem
    from .scheduler import RequestResult, RejectedRequest, Simulation
except ImportError:
    from backend import Request, WorkItem
    from scheduler import RequestResult, RejectedRequest, Simulation


@dataclass(frozen=True)
class ReplayWork:
    request: Request
    prefill_start: int
    input_tokens: int
    prefix_length: int
    generated: int = 0
    is_prefill: bool = False
    is_replay: bool = True
    emits_token: bool = False


def schedule_dynamic(requests, backend, max_batch_size, capacity, block_size,
                     token_budget, prefill_chunk_size, clock):
    if type(max_batch_size) is not int or max_batch_size < 1:
        raise ValueError("max_batch_size must be positive")
    for name, value in (("token_budget", token_budget), ("prefill_chunk_size", prefill_chunk_size)):
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError(name + " must be positive")
    if prefill_chunk_size is not None and token_budget is None:
        raise ValueError("prefill_chunk_size requires token_budget")
    incoming = list(requests)
    if any(not isinstance(r, Request) for r in incoming):
        raise ValueError("requests must contain Request objects")
    if len({r.request_id for r in incoming}) != len(incoming):
        raise ValueError("duplicate request ID")
    incoming.sort(key=lambda r: r.arrival_ms)
    started = float(incoming[0].arrival_ms) if incoming else 0.0
    index = 0
    waiting, active, completed, rejected = deque(), [], [], []
    native, replay = set(), {}
    trace, events = [], []
    preemptions = replay_input_tokens = peak_blocks = 0

    def key(result):
        return result.request.request_id

    def cached_length(result):
        return (result.request.prompt_tokens + len(result.tokens) - 1
                if result.tokens else result.prefilled_tokens)

    def make_work():
        ready = [r for r in active if r.tokens and key(r) not in replay]
        work = [(r, WorkItem(r.request, len(r.tokens), r.prefilled_tokens)) for r in ready]
        remaining = None if token_budget is None else token_budget - len(work)
        if remaining is not None and remaining < 0:
            raise RuntimeError("decode count exceeds dynamic token budget")
        # Restore previously admitted work before serving new prefills.
        candidates = [r for r in active if key(r) in replay] + [r for r in active if not r.tokens and key(r) not in replay]
        for r in candidates:
            if remaining == 0:
                break
            if key(r) in replay:
                target, progress = replay[key(r)]
                count = target - progress
            else:
                progress = r.prefilled_tokens
                count = r.request.prompt_tokens - progress
            if remaining is not None:
                count = min(count, remaining)
            if prefill_chunk_size is not None:
                count = min(count, prefill_chunk_size)
            item = (ReplayWork(r.request, progress, count, target) if key(r) in replay
                    else WorkItem(r.request, 0, progress, count))
            work.append((r, item))
            if remaining is not None:
                remaining -= count
        return work

    try:
        while index < len(incoming) or waiting or active:
            now = clock.now_ms()
            if not active and not waiting and index < len(incoming) and incoming[index].arrival_ms > now:
                clock.wait_until(incoming[index].arrival_ms)
                now = clock.now_ms()
            while index < len(incoming) and incoming[index].arrival_ms <= now:
                request = incoming[index]
                index += 1
                # Bound each request's worst-case footprint so the protected
                # oldest request can always finish by itself.
                blocks = (request.prompt_tokens + request.output_tokens + block_size - 1) // block_size
                if blocks > capacity:
                    rejected.append(RejectedRequest(request, now, "request_exceeds_kv_capacity", blocks))
                else:
                    waiting.append(RequestResult(request, None))
            while waiting and len(active) < max_batch_size:
                result = waiting.popleft()
                if result.admitted_ms is None:
                    result.admitted_ms = now
                active.append(result)
            if not active:
                continue
            while True:
                scheduled = make_work()
                if not scheduled:
                    raise RuntimeError("dynamic scheduler made no progress")
                if backend.can_step([item for _, item in scheduled]):
                    break
                if len(active) == 1:
                    raise RuntimeError("feasible protected request cannot fit in paged pool")
                victim = active.pop()  # Deterministic newest admission first.
                victim_id = key(victim)
                had_native = victim_id in native
                if had_native:
                    backend.release(victim_id)
                    native.remove(victim_id)
                target = cached_length(victim)
                if target:
                    replay[victim_id] = (target, 0)
                waiting.appendleft(victim)
                preemptions += int(had_native)
                events.append({"time_ms": clock.now_ms(), "event": "preempt" if had_native else "kv_wait", "request_id": victim_id,
                               "cached_tokens": target, "delivered_tokens": len(victim.tokens)})
            start = clock.now_ms()
            work = [item for _, item in scheduled]
            step = backend.step(work)
            backend.synchronize()
            if len(step.tokens) != len(work):
                raise ValueError("worker result arity")
            for item, token in zip(work, step.tokens):
                if item.emits_token:
                    if type(token) is not int or token < 0:
                        raise ValueError("missing dynamic output token")
                elif token is not None:
                    raise ValueError("replay/unfinished prefill emitted a token")
            end = clock.now_ms()
            stats = backend.stats()
            peak_blocks = max(peak_blocks, stats["peak_physical_blocks"])
            trace.append({"start_ms": start, "end_ms": end,
                          "kv_reserved_blocks": 0, "kv_physical_blocks": stats["live_physical_blocks"], "active_requests": len(active),
                          "input_tokens": sum(w.input_tokens for w in work),
                          "decode_tokens": sum(not w.is_prefill and not getattr(w, "is_replay", False) for w in work),
                          "prefill_tokens": sum(w.input_tokens for w in work if w.is_prefill),
                          "recompute_tokens": sum(w.input_tokens for w in work if getattr(w, "is_replay", False)),
                          "deferred_requests": [key(r) for r in active if all(r is not x for x, _ in scheduled)],
                          "batch": [{"request_id": w.request.request_id,
                                     "phase": "recompute" if getattr(w, "is_replay", False) else "prefill" if w.is_prefill else "decode",
                                     "input_tokens": w.input_tokens, "token_index": w.generated,
                                     "prefill_start": w.prefill_start, "emits_token": w.emits_token} for w in work]})
            finished = set()
            for (result, item), token in zip(scheduled, step.tokens):
                request_id = key(result)
                native.add(request_id)
                if getattr(item, "is_replay", False):
                    target, progress = replay[request_id]
                    progress += item.input_tokens
                    replay_input_tokens += item.input_tokens
                    if progress == target:
                        del replay[request_id]
                        events.append({"time_ms": end, "event": "resumed", "request_id": request_id})
                    else:
                        replay[request_id] = (target, progress)
                elif item.is_prefill:
                    result.prefilled_tokens += item.input_tokens
                if token is not None:
                    result.tokens.append(token)
                    result.token_times_ms.append(end)
                if len(result.tokens) == result.request.output_tokens:
                    backend.release(request_id)
                    native.remove(request_id)
                    finished.add(request_id)
                    completed.append(result)
            active = [r for r in active if key(r) not in finished]
    finally:
        failing = sys.exc_info()[0] is not None
        try:
            backend.clear()
        except Exception:
            if not failing:
                raise
    stats = backend.stats()
    stats.update(preemptions=preemptions, replay_input_tokens=replay_input_tokens)
    by_id = {key(r): r for r in completed}
    rejects = {r.request.request_id: r for r in rejected}
    return Simulation("continuous", max_batch_size, started, clock.now_ms() if incoming else 0.0,
                      [by_id[r.request_id] for r in incoming if r.request_id in by_id], trace,
                      [rejects[r.request_id] for r in incoming if r.request_id in rejects],
                      {"allocation": "dynamic", "capacity_blocks": capacity, "block_size_tokens": block_size,
                       "peak_reserved_blocks": 0, "peak_physical_blocks": peak_blocks,
                       "final_reserved_blocks": 0, "final_reservations": 0, "events": []},
                      events, "wait", token_budget, prefill_chunk_size, "wall_ms", stats)
