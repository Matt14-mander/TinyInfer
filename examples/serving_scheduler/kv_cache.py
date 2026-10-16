"""Block reservation accounting only; no tensor, device memory or real KV data."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Reservation:
    request_id: str
    token_budget: int
    blocks: int


class KVCache:
    def __init__(self, capacity_blocks=None, block_size=1):
        if capacity_blocks is not None and (type(capacity_blocks) is not int or capacity_blocks < 0):
            raise ValueError("capacity_blocks must be a nonnegative integer or None")
        if type(block_size) is not int or block_size < 1:
            raise ValueError("block_size must be a positive integer")
        self.capacity_blocks = capacity_blocks
        self.block_size = block_size
        self.used_blocks = 0
        self.peak_blocks = 0
        self._reservations = {}
        self.events = []

    @property
    def reservations(self):
        return dict(self._reservations)

    @property
    def available_blocks(self):
        return None if self.capacity_blocks is None else self.capacity_blocks - self.used_blocks

    def required_blocks(self, token_budget):
        if type(token_budget) is not int or token_budget < 1:
            raise ValueError("token_budget must be a positive integer")
        return (token_budget + self.block_size - 1) // self.block_size

    def can_ever_fit(self, token_budget):
        blocks = self.required_blocks(token_budget)
        return self.capacity_blocks is None or blocks <= self.capacity_blocks

    def _event(self, kind, reservation, now, reason=None):
        self.events.append({"time_ms": now, "event": kind, "request_id": reservation.request_id,
                            "token_budget": reservation.token_budget, "blocks": reservation.blocks,
                            "used_blocks": self.used_blocks, "available_blocks": self.available_blocks,
                            "reason": reason})

    def reserve(self, request_id, token_budget, now):
        if not isinstance(request_id, str) or not request_id or request_id in self._reservations:
            raise ValueError("reservation ID must be nonempty and unique")
        blocks = self.required_blocks(token_budget)
        if self.capacity_blocks is not None and blocks > self.available_blocks:
            return False
        reservation = Reservation(request_id, token_budget, blocks)
        self._reservations[request_id] = reservation
        self.used_blocks += blocks
        self.peak_blocks = max(self.peak_blocks, self.used_blocks)
        self._event("reserve", reservation, now)
        return True

    def release(self, request_id, now, reason="completed"):
        if request_id not in self._reservations:
            raise ValueError("cannot release an unknown reservation")
        reservation = self._reservations.pop(request_id)
        self.used_blocks -= reservation.blocks
        self._event("release", reservation, now, reason)

    def summary(self):
        return {"capacity_blocks": self.capacity_blocks, "block_size_tokens": self.block_size,
                "capacity_tokens": None if self.capacity_blocks is None else self.capacity_blocks * self.block_size,
                "peak_reserved_blocks": self.peak_blocks, "final_reserved_blocks": self.used_blocks,
                "final_reservations": len(self._reservations), "events": list(self.events)}
