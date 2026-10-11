"""K=1/2/4 structural experiments and standalone correctness tests (stdlib only)."""
import argparse
from dataclasses import asdict
import json
import random
import unittest

if __package__:
    from .draft_proposer import DraftProposer, make_toy_draft
    from .greedy_baseline import ToyTarget, greedy_decode
    from .greedy_verifier import speculative_decode, verify_candidates
else:
    from draft_proposer import DraftProposer, make_toy_draft
    from greedy_baseline import ToyTarget, greedy_decode
    from greedy_verifier import speculative_decode, verify_candidates


class CorrectnessTests(unittest.TestCase):
    def assert_equivalent(self, target, draft, prompt, budget, k):
        original = tuple(prompt)
        baseline = greedy_decode(target, prompt, budget)
        result = speculative_decode(target, draft, prompt, budget, k)
        self.assertEqual(result.tokens, baseline.tokens)
        self.assertEqual(result.generated, baseline.generated)
        self.assertEqual(tuple(prompt), original, "caller prompt mutated")
        self.assertLessEqual(len(result.generated), budget)
        self.assertEqual(result.target_calls, baseline.target_calls)
        self.assertEqual(result.accepted_tokens + result.mismatches + result.discarded_tokens,
                         result.proposed_tokens)
        self.assertEqual(result.checked_tokens, result.accepted_tokens + result.mismatches)
        self.assertEqual(len(result.generated),
                         result.accepted_tokens + result.mismatches + result.bonus_tokens)
        self.assertTrue(all(t.verification.emitted for t in result.trace))
        return result

    def test_k_1_2_4_and_three_draft_scenarios(self):
        target = ToyTarget(stop_after=24)
        for k in (1, 2, 4):
            for scenario in ("exact", "mixed", "first_wrong"):
                with self.subTest(k=k, scenario=scenario):
                    self.assert_equivalent(target, make_toy_draft(target, scenario), [1], 24, k)

    def test_first_prediction_wrong(self):
        target = lambda _: 1
        # No draft EOS here: each round proposes the entire remaining K block.
        result = self.assert_equivalent(target, DraftProposer(lambda _: 2), [1], 8, 4)
        self.assertEqual(result.accepted_tokens, 0)
        self.assertEqual(result.verification_rounds, 8)
        first = result.trace[0]
        self.assertEqual(first.verification.mismatch_index, 0)
        self.assertEqual(first.verification.emitted, (target([1]),))
        self.assertEqual(first.verification.checked, 1)
        self.assertEqual(first.verification.bonus, None)
        self.assertEqual(result.discarded_tokens, 18)  # Budgets 8..1, min(K, remaining).

    def test_every_mismatch_position_and_rejected_suffix_is_not_queried(self):
        target = ToyTarget(stop_after=100)
        prompt = [2, 3]
        for k in (1, 2, 4):
            good = DraftProposer(target).propose(prompt, k)
            for wrong_at in range(k):
                calls = []

                def tracked(prefix):
                    calls.append(tuple(prefix))
                    return target(prefix)

                bad = list(good)
                bad[wrong_at] = (bad[wrong_at] + 1) % target.vocab_size
                checked = verify_candidates(tracked, prompt, bad, 12)
                self.assertEqual(checked.accepted, wrong_at)
                self.assertEqual(checked.mismatch_index, wrong_at)
                self.assertEqual(checked.emitted, good[:wrong_at + 1])
                self.assertEqual(len(calls), wrong_at + 1)
                self.assertEqual(calls[-1], tuple(prompt) + good[:wrong_at])
                self.assertIsNone(checked.bonus)

    def test_draft_uses_its_tentative_prefix(self):
        prefixes = []

        def draft(prefix):
            prefixes.append(tuple(prefix))
            return len(prefix) + 1

        self.assertEqual(DraftProposer(draft).propose([9], 3), (2, 3, 4))
        self.assertEqual(prefixes, [(9,), (9, 2), (9, 2, 3)])

    def test_all_match_bonus(self):
        target = ToyTarget(stop_after=100)
        prompt = [1]
        candidates = DraftProposer(target).propose(prompt, 4)
        result = verify_candidates(target, prompt, candidates, 9)
        self.assertEqual(result.accepted, 4)
        self.assertEqual(result.bonus, target(tuple(prompt) + candidates))
        self.assertEqual(result.emitted, candidates + (result.bonus,))
        self.assertEqual(result.target_calls, 5)

    def test_budget_and_partial_final_block(self):
        target = ToyTarget(stop_after=100)
        for budget in (0, 1, 2, 3, 4, 5, 7):
            for k in (1, 2, 4):
                result = self.assert_equivalent(target, DraftProposer(target), [], budget, k)
                self.assertEqual(len(result.generated), budget)
        candidates = DraftProposer(target).propose([], 4)
        self.assertIsNone(verify_candidates(target, [], candidates, 4).bonus)

    def test_accepted_eos_and_no_post_eos_work(self):
        calls = []

        def target(prefix):
            calls.append(tuple(prefix))
            return 0

        result = verify_candidates(target, [1], [0, 7, 8], 4)
        self.assertEqual(result.emitted, (0,))
        self.assertEqual(result.accepted, 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(DraftProposer(target).propose([1], 4), (0,))

    def test_rejected_draft_eos(self):
        target = ToyTarget(stop_after=50)
        result = self.assert_equivalent(target, DraftProposer(lambda _: 0), [1], 7, 4)
        self.assertEqual(result.accepted_tokens, 0)
        self.assertEqual(result.mismatches, 7)

    def test_correction_eos_and_bonus_eos(self):
        correction = verify_candidates(lambda _: 0, [1], [7, 8], 4)
        self.assertEqual(correction.emitted, (0,))
        self.assertEqual(correction.mismatch_index, 0)
        target = ToyTarget(stop_after=3)
        candidates = DraftProposer(target).propose([1], 2)
        bonus = verify_candidates(target, [1], candidates, 4)
        self.assertEqual(bonus.bonus, 0)
        self.assertEqual(bonus.emitted[-1], 0)
        for k in (1, 2, 4):
            self.assert_equivalent(target, make_toy_draft(target, "first_wrong"), [1], 20, k)

    def test_finished_prompt(self):
        def forbidden(_):
            raise AssertionError("model called after prompt EOS")

        result = self.assert_equivalent(forbidden, DraftProposer(forbidden), [2, 0], 8, 4)
        self.assertEqual(result.verification_rounds, 0)
        self.assertEqual(result.acceptance_rate, 0)

    def test_prefix_sensitive_seeded_random_models(self):
        # Distinct deterministic functions over *all* history, with naturally
        # occurring EOS. Exercise wrong-context suffixes and varied budgets.
        rng = random.Random(20261011)
        for case in range(200):
            vocab = rng.randrange(2, 12)
            target_seed, draft_seed = rng.randrange(65521), rng.randrange(65521)

            def model(seed):
                def predict(prefix):
                    state = seed
                    for i, token in enumerate(prefix):
                        state = (state * 37 + (i + 5) * (token + 1)) % 65521
                    return (state + len(prefix) * 13) % vocab
                return predict

            prompt = [rng.randrange(1, vocab) for _ in range(rng.randrange(4))]
            budget = rng.randrange(21)
            for k in (1, 2, 4):
                with self.subTest(case=case, k=k):
                    self.assert_equivalent(model(target_seed), DraftProposer(model(draft_seed)),
                                           prompt, budget, k)

    def test_invalid_parameters(self):
        target = ToyTarget()
        for k, budget in ((0, 5), (-1, 5), (4, -1)):
            with self.assertRaises(ValueError):
                speculative_decode(target, DraftProposer(target), [], budget, k)
        with self.assertRaises(ValueError):
            greedy_decode(target, [], -1)
        for candidates, remaining in (([], 1), ([1, 2], 1), ([1], 0)):
            with self.assertRaises(ValueError):
                verify_candidates(target, [], candidates, remaining)
        with self.assertRaises(ValueError):
            speculative_decode(target, DraftProposer(target, eos_token=99), [], 4, 2)
        with self.assertRaises(ValueError):
            DraftProposer(target).propose([], 0)


def experiment(prompt, budget, seed, scenario, trace=False):
    target = ToyTarget(stop_after=len(prompt) + max(0, budget - 1), seed=seed)
    baseline = greedy_decode(target, prompt, budget)
    rows = []
    scenarios = ("exact", "mixed", "first_wrong") if scenario == "all" else (scenario,)
    for name in scenarios:
        for k in (1, 2, 4):
            result = speculative_decode(target, make_toy_draft(target, name), prompt, budget, k)
            if result.tokens != baseline.tokens:
                raise RuntimeError("speculative output differs from Target Greedy")
            row = dict(scenario=name, k=k, exact_match=True,
                       generated_tokens=len(result.generated), proposed_tokens=result.proposed_tokens,
                       accepted_tokens=result.accepted_tokens, acceptance_rate=result.acceptance_rate,
                       checked_tokens=result.checked_tokens, mismatches=result.mismatches,
                       discarded_tokens=result.discarded_tokens, bonus_tokens=result.bonus_tokens,
                       verification_rounds=result.verification_rounds,
                       target_calls=result.target_calls, draft_calls=result.proposed_tokens)
            if trace:
                row['trace'] = [asdict(item) for item in result.trace]
            rows.append(row)
    return dict(stage=1, verification="sequential", prompt=prompt,
                max_new_tokens=budget, seed=seed, baseline=asdict(baseline), results=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test', action='store_true', help='run correctness tests only')
    parser.add_argument('--tokens', type=int, default=24, help='maximum new tokens (EOS included)')
    parser.add_argument('--prompt', nargs='*', type=int, default=[1], help='toy token IDs; EOS=0')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--scenario', choices=('all', 'exact', 'mixed', 'first_wrong'), default='all')
    parser.add_argument('--trace', action='store_true', help='show round-level decisions')
    parser.add_argument('--format', choices=('table', 'json'), default='table')
    args = parser.parse_args()
    if args.test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(CorrectnessTests)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if result.wasSuccessful() else 1
    if args.tokens < 0 or any(token < 0 or token >= 17 for token in args.prompt):
        parser.error('tokens must be nonnegative; prompt IDs must be in 0..16')
    report = experiment(args.prompt, args.tokens, args.seed, args.scenario, args.trace)
    if args.format == 'json':
        print(json.dumps(report, indent=2))
        return 0
    print('Stage 1: sequential toy verification; rounds are NOT batched target forward passes.')
    print('Target Greedy generated:', report['baseline']['generated'])
    print('Baseline target calls:', report['baseline']['target_calls'])
    print(f"{'scenario':<12} {'K':>2} {'match':>5} {'accepted/proposed':>18} {'rate':>8} {'rounds':>7} {'target':>7} {'draft':>7} {'mismatch':>8} {'discard':>7} {'bonus':>5}")
    for row in report['results']:
        ratio = f"{row['accepted_tokens']}/{row['proposed_tokens']}"
        print(f"{row['scenario']:<12} {row['k']:>2} {'yes':>5} {ratio:>18} "
              f"{row['acceptance_rate']:>7.1%} {row['verification_rounds']:>7} "
              f"{row['target_calls']:>7} {row['draft_calls']:>7} {row['mismatches']:>8} "
              f"{row['discarded_tokens']:>7} {row['bonus_tokens']:>5}")
        if args.trace:
            for index, item in enumerate(row['trace'], 1):
                v = item['verification']
                print(f"  round={index} prefix_len={item['prefix_length']} candidates={item['candidates']} "
                      f"accepted={v['accepted']} mismatch={v['mismatch_index']} "
                      f"correction={v['correction']} bonus={v['bonus']} emitted={v['emitted']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
