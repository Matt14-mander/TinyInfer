"""Independent Target Greedy oracle and a prefix-dependent toy token model."""
from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

NextToken = Callable[[Sequence[int]], int]


@dataclass(frozen=True)
class GreedyResult:
    tokens: Tuple[int, ...]  # Includes the original prompt.
    generated: Tuple[int, ...]
    target_calls: int


def greedy_decode(target: NextToken, prompt: Sequence[int], max_new_tokens: int,
                  eos_token: int = 0) -> GreedyResult:
    """Decode one target token at a time, including EOS in the returned output."""
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be nonnegative")
    tokens = list(prompt)
    start = len(tokens)
    calls = 0
    if not tokens or tokens[-1] != eos_token:
        for _ in range(max_new_tokens):
            token = target(tuple(tokens))
            calls += 1
            tokens.append(token)
            if token == eos_token:
                break
    return GreedyResult(tuple(tokens), tuple(tokens[start:]), calls)


@dataclass(frozen=True)
class ToyTarget:
    """Deterministic simulated argmax; uses the entire prefix, not a stored answer.

    EOS=0; other tokens are 1..vocab_size-1. stop_after counts prompt + output.
    No logits, neural network, random sampling or KV cache is involved.
    """
    vocab_size: int = 17
    stop_after: int = 25
    seed: int = 7

    def __post_init__(self):
        if self.vocab_size < 2 or self.stop_after < 0:
            raise ValueError("invalid toy vocabulary or stop length")

    def __call__(self, prefix: Sequence[int]) -> int:
        if len(prefix) >= self.stop_after:
            return 0
        state = self.seed
        for position, token in enumerate(prefix):
            state = (state * 33 + (position + 1) * (token + 3)) % 65521
        return 1 + (state + 11 * len(prefix)) % (self.vocab_size - 1)
