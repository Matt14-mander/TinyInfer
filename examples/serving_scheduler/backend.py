"""Deterministic virtual backend; no model, wall-clock timing or KV allocation."""
from dataclasses import dataclass
import math
from abc import ABC, abstractmethod
from typing import Optional, Sequence


def _finite_nonnegative(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value) and value >= 0
    except OverflowError:
        return False


@dataclass(frozen=True)
class Request:
    request_id: str
    arrival_ms: float
    prompt_tokens: int
    output_tokens: int

    def __post_init__(self):
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("request_id must be a nonempty string")
        if not _finite_nonnegative(self.arrival_ms):
            raise ValueError("arrival_ms must be finite and nonnegative")
        for value in (self.prompt_tokens, self.output_tokens):
            if type(value) is not int or value < 1:
                raise ValueError("token counts must be positive integers")


@dataclass(frozen=True)
class WorkItem:
    request: Request
    generated: int
    prefill_start: int = 0
    prefill_tokens: Optional[int] = None

    @property
    def is_prefill(self):
        return self.generated == 0

    @property
    def input_tokens(self):
        if not self.is_prefill:
            return 1
        return self.request.prompt_tokens - self.prefill_start if self.prefill_tokens is None else self.prefill_tokens

    @property
    def emits_token(self):
        return not self.is_prefill or self.prefill_start + self.input_tokens == self.request.prompt_tokens


@dataclass(frozen=True)
class StepResult:
    duration_ms: float
    tokens: Sequence[Optional[int]]


class Backend(ABC):
    @abstractmethod
    def step(self, batch: Sequence[WorkItem]) -> StepResult:
        """One result per item; unfinished prefill returns None, otherwise one token."""
        ...


@dataclass(frozen=True)
class SimulatedBackend(Backend):
    overhead_ms: float = 1.0
    prefill_ms_per_token: float = 0.1
    decode_ms_per_request: float = 0.2

    def __post_init__(self):
        costs = (self.overhead_ms, self.prefill_ms_per_token, self.decode_ms_per_request)
        if any(not _finite_nonnegative(v) for v in costs):
            raise ValueError("costs must be finite and nonnegative")
        if self.overhead_ms + self.prefill_ms_per_token <= 0 or self.overhead_ms + self.decode_ms_per_request <= 0:
            raise ValueError("prefill and decode steps must have positive duration")

    def step(self, batch):
        if not batch:
            raise ValueError("backend cannot execute an empty batch")
        duration = self.overhead_ms
        tokens = []
        for item in batch:
            if type(item.generated) is not int or not 0 <= item.generated < item.request.output_tokens:
                raise ValueError("invalid generation position")
            if item.is_prefill and (type(item.prefill_start) is not int or item.prefill_start < 0
                                    or type(item.input_tokens) is not int or item.input_tokens < 1
                                    or item.prefill_start + item.input_tokens > item.request.prompt_tokens):
                raise ValueError("invalid prefill range")
            duration += (item.input_tokens * self.prefill_ms_per_token
                         if item.is_prefill else self.decode_ms_per_request)
            if not item.emits_token:
                tokens.append(None)
                continue
            # Stable across processes and independent of batching/admission order.
            identity = sum((i + 1) * ord(c) for i, c in enumerate(item.request.request_id))
            tokens.append(1 + (identity + item.request.prompt_tokens * 17 + item.generated * 31) % 32000)
        return StepResult(duration, tokens)
