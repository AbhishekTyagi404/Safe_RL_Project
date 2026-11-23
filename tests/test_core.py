"""Unit tests for the confidence bounds, the environment's exact evaluation and the update rules.

Run from the project folder with either:
    python -m pytest tests
    python tests/test_core.py
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safe_rl.gridworld import GridWorld                                      # noqa: E402
from safe_rl.train import summarize                                          # noqa: E402
from safe_rl.updates import (BOUNDS, bernstein_bounds, clopper_pearson_bounds,  # noqa: E402
                             hoeffding_bounds, make_rule, wilson_bounds)

DELTA = 0.05
VALID = ["hoeffding", "bernstein", "clopper_pearson"]      # bounds with a finite sample guarantee


# ---------------- confidence bounds: published reference values ----------------
def test_clopper_pearson_zero_and_all_successes():
    # Closed forms for x = 0 and x = n: U = 1 - (delta/2)^(1/n), L = (delta/2)^(1/n).
    n = 10
    L, U = clopper_pearson_bounds(0.0, n, DELTA)
    assert L == 0.0 and math.isclose(U, 1 - (DELTA / 2) ** (1 / n), abs_tol=1e-6)
    L, U = clopper_pearson_bounds(1.0, n, DELTA)
    assert U == 1.0 and math.isclose(L, (DELTA / 2) ** (1 / n), abs_tol=1e-6)


def test_clopper_pearson_textbook_value():
    # 10 successes in 100 trials, 95% exact interval: [0.0490, 0.1762].
    L, U = clopper_pearson_bounds(0.10, 100, DELTA)
    assert math.isclose(L, 0.0490, abs_tol=5e-4) and math.isclose(U, 0.1762, abs_tol=5e-4)


def test_clopper_pearson_matches_beta_quantiles():
    # The exact interval equals Beta quantiles; check against scipy when it is installed.
    try:
        from scipy.stats import beta
    except ImportError:
        return
    for n, x in [(50, 1), (250, 7), (1000, 12), (1000, 240)]:
        L, U = clopper_pearson_bounds(x / n, n, 0.001)
        assert math.isclose(L, beta.ppf(0.0005, x, n - x + 1), abs_tol=1e-6)
        assert math.isclose(U, beta.ppf(0.9995, x + 1, n - x), abs_tol=1e-6)


def test_wilson_textbook_value():
    # 10 successes in 100 trials, 95% Wilson interval: [0.0552, 0.1744].
    L, U = wilson_bounds(0.10, 100, DELTA)
    assert math.isclose(L, 0.0552, abs_tol=5e-4) and math.isclose(U, 0.1744, abs_tol=5e-4)


def test_hoeffding_formula():
    L, U = hoeffding_bounds(0.3, 400, DELTA)
    eps = math.sqrt(math.log(2 / DELTA) / 800)
    assert math.isclose(U - 0.3, eps) and math.isclose(0.3 - L, eps)


def test_bernstein_constants():
    # Maurer and Pontil (2009), Theorem 4, applied to each side with delta/2:
    # eps = sqrt(2 V ln(4/delta) / n) + 7 ln(4/delta) / (3 (n - 1)), V the unbiased sample variance.
    n, p = 200, 0.05
    V = p * (1 - p) * n / (n - 1)
    eps = math.sqrt(2 * V * math.log(4 / DELTA) / n) + 7 * math.log(4 / DELTA) / (3 * (n - 1))
    L, U = bernstein_bounds(p, n, DELTA)
    assert math.isclose(U, p + eps) and math.isclose(L, max(0.0, p - eps))
    # With no collisions the variance term vanishes and only the second term is left.
    L, U = bernstein_bounds(0.0, n, DELTA)
    assert L == 0.0 and math.isclose(U, 7 * math.log(4 / DELTA) / (3 * (n - 1)))


# ---------------- confidence bounds: general properties ----------------
def test_bounds_are_ordered_and_inside_unit_interval():
    for name, f in BOUNDS.items():
        for n in [2, 5, 50, 1000]:
            for x in sorted({0, 1, n // 2, n - 1, n}):
                L, U = f(x / n, n, DELTA)
                assert 0.0 <= L <= x / n + 1e-12 <= U + 1e-12 <= 1.0 + 1e-12, (name, n, x, L, U)


def test_bounds_narrow_as_n_grows():
    for name, f in BOUNDS.items():
        widths = [np.subtract(*f(0.1, n, DELTA)[::-1]) for n in [50, 200, 1000, 5000]]
        assert all(a > b for a, b in zip(widths, widths[1:])), name


def test_valid_bounds_reach_their_coverage():
    # Simulated coverage must be at least 1 - delta (small slack for simulation noise).
    rng = np.random.default_rng(0)
    for p in [0.01, 0.1, 0.24]:
        for n in [50, 250]:
            xs = rng.binomial(n, p, 20000)
            values, counts = np.unique(xs, return_counts=True)       # each count x needs one bound
            for name in VALID:
                inside = [BOUNDS[name](x / n, n, DELTA)[0] <= p <= BOUNDS[name](x / n, n, DELTA)[1]
                          for x in values]
                cover = np.sum(counts * np.array(inside)) / len(xs)
                assert cover >= 1 - DELTA - 0.005, (name, p, n, cover)


# ---------------- environment ----------------
def test_transition_rows_sum_to_one():
    env = GridWorld()
    assert np.allclose(env.P.sum(axis=2), 1.0)


def test_exact_evaluation_matches_simulation():
    # Exact collision probability and expected return must match a large simulation.
    env = GridWorld()
    rng = np.random.default_rng(1)
    policies = [rng.integers(0, 4, env.n_states) for _ in range(3)] + [env.solve(0), env.solve(40)]
    for pol in policies:
        p, _, R = env.evaluate(pol)
        C, G = env.rollout(pol, 200000, rng)
        assert abs(C.mean() - p) < 4 * math.sqrt(p * (1 - p) / len(C)) + 1e-9
        assert abs(G.mean() - R) < 4 * G.std() / math.sqrt(len(G)) + 1e-9


def test_reference_policies():
    env = GridWorld()
    p_short, _, ret_short = env.evaluate(env.solve(0))
    p_safe, _, ret_safe = env.evaluate(env.solve(40))
    assert p_short > 0.2 > 0.1 > 0.01 > p_safe          # shortcut breaks the budget, safe route does not
    assert ret_short > ret_safe                          # the shortcut pays more, so there is a tradeoff


# ---------------- update rules ----------------
class _FakeEnv:
    """Returns a fixed number of collisions out of n evaluation episodes."""

    def __init__(self, collisions):
        self.collisions = collisions

    def rollout(self, policy, n, rng):
        C = np.zeros(n)
        C[: self.collisions] = 1.0
        return C, np.zeros(n)


def _gated(lam0=10.0):
    return make_rule({"type": "gated", "bound": "clopper_pearson", "beta_up": 20, "beta_down": 2,
                      "n_eval": 1000, "delta": 0.05, "lam0": lam0}, budget=0.1)


def test_gated_moves_up_only_on_clear_violation():
    rule = _gated()
    info = rule.update(1, None, None, _FakeEnv(300), None)           # p_hat = 0.30, clearly over
    assert info["moved"] == 1 and rule.lam > 10.0


def test_gated_moves_down_only_on_clear_safety():
    rule = _gated()
    info = rule.update(1, None, None, _FakeEnv(5), None)             # p_hat = 0.005, clearly under
    assert info["moved"] == -1 and rule.lam < 10.0


def test_gated_holds_when_interval_contains_budget():
    rule = _gated()
    info = rule.update(1, None, None, _FakeEnv(100), None)           # p_hat = 0.10, right at the budget
    assert info["moved"] == 0 and rule.lam == 10.0


def test_standard_uses_asymmetric_steps():
    rule = make_rule({"type": "standard", "beta_up": 20, "beta_down": 2, "lam0": 10.0}, budget=0.1)
    rule.update(1, np.full(50, 0.2), None, None, None)               # +20 * 0.1 = +2
    assert math.isclose(rule.lam, 12.0)
    rule.update(2, np.zeros(50), None, None, None)                   # -2 * 0.1 = -0.2
    assert math.isclose(rule.lam, 11.8)


def test_wrong_direction_metric():
    out = {"p_true": np.array([0.2, 0.05, 0.2, 0.05]), "moved": np.array([1, 1, -1, -1]),
           "lam": np.zeros(4), "return_true": np.zeros(4), "train_C": np.zeros(10),
           "eval_episodes": np.zeros(4), "eval_collisions": np.zeros(4)}
    # up while over (right), up while under (wrong), down while over (wrong), down while under (right)
    assert summarize(out, 50, 0.1)["wrong_direction_moves"] == 2


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, f in tests:
        f()
        print(f"passed  {name}")
    print(f"\nAll {len(tests)} tests passed.")
