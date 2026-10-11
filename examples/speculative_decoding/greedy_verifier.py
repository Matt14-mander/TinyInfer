"""Exact greedy speculation: keep the matching prefix, correct the first error."""
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

if __package__:
    from .draft_proposer import DraftProposer
    from .greedy_baseline import NextToken
else:
    from draft_proposer import DraftProposer
    from greedy_baseline import NextToken


@dataclass(frozen=True)
class Verification:
    emitted: Tuple[int, ...]
    accepted: int
    checked: int
    mismatch_index: Optional[int] = None  # 0-based position in this draft block.
    correction: Optional[int] = None
    bonus: Optional[int] = None

    @property
    def target_calls(self) -> int:
        return self.checked + int(self.bonus is not None)


def verify_candidates(target: NextToken, prefix: Sequence[int],
                      candidates: Sequence[int], remaining: int,
                      eos_token: int = 0) -> Verification:
    """Sequential verification only; one call per checked token/bonus.

    The target is queried using the committed prefix plus accepted candidates.
    After a mismatch, later draft tokens are never queried or committed.
    """
    if remaining < 1 or not candidates or len(candidates) > remaining:
        raise ValueError("need a nonempty draft within the remaining token budget")
    committed = list(prefix)
    emitted = []
    accepted = 0
    for index, candidate in enumerate(candidates):
        expected = target(tuple(committed))
        if candidate != expected:
            emitted.append(expected)
            return Verification(tuple(emitted), accepted, index + 1,
                                mismatch_index=index, correction=expected)
        emitted.append(candidate)
        committed.append(candidate)
        accepted += 1
        if candidate == eos_token:
            return Verification(tuple(emitted), accepted, index + 1)
    # Full agreement permits one additional target token, if budget remains.
    bonus = None
    if len(emitted) < remaining:
        bonus = target(tuple(committed))
        emitted.append(bonus)
    return Verification(tuple(emitted), accepted, len(candidates), bonus=bonus)


@dataclass(frozen=True)
class RoundTrace:
    prefix_length: int
    candidates: Tuple[int, ...]
    verification: Verification


@dataclass(frozen=True)
class SpeculativeResult:
    tokens: Tuple[int, ...]
    generated: Tuple[int, ...]
    proposed_tokens: int
    accepted_tokens: int
    checked_tokens: int
    target_calls: int
    mismatches: int
    discarded_tokens: int  # Unchecked suffix, including tokens after accepted EOS.
    bonus_tokens: int
    trace: Tuple[RoundTrace, ...]

    @property
    def verification_rounds(self) -> int:
        return len(self.trace)

    @property
    def acceptance_rate(self) -> float:
        # Includes wasted proposals in the denominator; bonus/correction excluded.
        return self.accepted_tokens / self.proposed_tokens if self.proposed_tokens else 0.0


def speculative_decode(target: NextToken, draft: DraftProposer,
                       prompt: Sequence[int], max_new_tokens: int,
                       k: int, eos_token: int = 0) -> SpeculativeResult:
    if k < 1 or max_new_tokens < 0:
        raise ValueError("k must be positive and max_new_tokens nonnegative")
    if draft.eos_token != eos_token:
        raise ValueError("draft and verifier must share EOS")
    tokens = list(prompt)
    start = len(tokens)
    proposed = accepted = checked = calls = mismatches = discarded = bonus = 0
    trace = []
    while len(tokens) - start < max_new_tokens and (not tokens or tokens[-1] != eos_token):
        remaining = max_new_tokens - (len(tokens) - start)
        candidates = draft.propose(tuple(tokens), min(k, remaining))
        verification = verify_candidates(target, tokens, candidates, remaining, eos_token)
        trace.append(RoundTrace(len(tokens), candidates, verification))
        tokens.extend(verification.emitted)
        proposed += len(candidates)
        accepted += verification.accepted
        checked += verification.checked
        calls += verification.target_calls
        mismatches += int(verification.mismatch_index is not None)
        discarded += len(candidates) - verification.checked
        bonus += int(verification.bonus is not None)
    return SpeculativeResult(tuple(tokens), tuple(tokens[start:]), proposed, accepted,
                             checked, calls, mismatches, discarded, bonus, tuple(trace))
