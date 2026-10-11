"""Generate a draft autoregressively on its own tentative prefix."""
from dataclasses import dataclass
from typing import Sequence, Tuple

if __package__:
    from .greedy_baseline import NextToken, ToyTarget
else:
    from greedy_baseline import NextToken, ToyTarget


@dataclass(frozen=True)
class DraftProposer:
    next_token: NextToken
    eos_token: int = 0

    def propose(self, prefix: Sequence[int], k: int) -> Tuple[int, ...]:
        if k < 1:
            raise ValueError("k must be positive")
        tentative = list(prefix)
        candidates = []
        for _ in range(k):
            token = self.next_token(tuple(tentative))
            candidates.append(token)
            tentative.append(token)
            if token == self.eos_token:
                break
        return tuple(candidates)


def make_toy_draft(target_spec: ToyTarget, scenario: str) -> DraftProposer:
    """An independent toy draft with controlled mistakes, for experiments only.

    Matching rules are intentionally shared with the toy target. This is not a
    cheaper model and does not establish realistic acceptance or speedup.
    """
    if scenario not in ("exact", "mixed", "first_wrong"):
        raise ValueError("unknown draft scenario")
    model = ToyTarget(target_spec.vocab_size, target_spec.stop_after, target_spec.seed)

    def predict(prefix):
        token = model(prefix)
        wrong = scenario == "first_wrong" or (
            scenario == "mixed" and (len(prefix) + sum(prefix)) % 4 == 0)
        return (token + 1) % model.vocab_size if wrong else token

    # first_wrong is wrong at every prefix, so every round fails at position 0.
    return DraftProposer(predict)
