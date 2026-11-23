"""Rules for updating the penalty weight lambda every K episodes.

Every rule has the same interface:
    rule.lam                                   current lambda
    rule.update(k, train_C, policy, env, rng)  returns an info dict and changes rule.lam

k        index of this check (1, 2, 3, ...)
train_C  collision indicators of the last K training episodes
policy   current greedy policy (used by rules that run evaluation episodes)
"""

import math
from functools import lru_cache
from statistics import NormalDist

import numpy as np


# ---------------- confidence bounds ----------------
# Each returns (L, U) for the mean p of n Bernoulli samples with empirical mean p_hat,
# such that P(p outside [L, U]) <= delta (exactly for all but Wilson, which is asymptotic).

def hoeffding_bounds(p_hat, n, delta):
    """Hoeffding (1963). Valid for any n; ignores that the variance p(1-p) is small near p = 0."""
    eps = np.sqrt(np.log(2.0 / delta) / (2.0 * n))
    return max(0.0, p_hat - eps), min(1.0, p_hat + eps)


def bernstein_bounds(p_hat, n, delta):
    """Empirical Bernstein (Maurer and Pontil, 2009), two sided with delta/2 per side.
    Valid for any n; uses the sample variance, so it is much tighter than Hoeffding when p is small.
    """
    var = p_hat * (1.0 - p_hat) * n / (n - 1)          # unbiased sample variance of 0/1 data
    log_term = np.log(4.0 / delta)
    eps = np.sqrt(2.0 * var * log_term / n) + 7.0 * log_term / (3.0 * (n - 1))
    return max(0.0, p_hat - eps), min(1.0, p_hat + eps)


def wilson_bounds(p_hat, n, delta):
    """Wilson score interval. Tight, but its coverage is only approximate (normal approximation),
    so the wrong direction guarantee does not formally hold for it.
    """
    z = NormalDist().inv_cdf(1.0 - delta / 2.0)
    denom = 1.0 + z * z / n
    center = (p_hat + z * z / (2 * n)) / denom
    half = z / denom * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n))
    return max(0.0, center - half), min(1.0, center + half)


@lru_cache(maxsize=None)
def _log_binom_coef(n):
    return np.array([math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) for k in range(n + 1)])


def _binom_cdf(x, n, p):
    """P(Bin(n, p) <= x)."""
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 1.0 if x >= n else 0.0
    k = np.arange(x + 1)
    logpmf = _log_binom_coef(n)[: x + 1] + k * math.log(p) + (n - k) * math.log1p(-p)
    m = logpmf.max()
    return float(min(1.0, math.exp(m) * np.exp(logpmf - m).sum()))


def clopper_pearson_bounds(p_hat, n, delta, iters=60):
    """Exact binomial interval (Clopper and Pearson, 1934). Valid for any n and as tight as an exact
    interval can be, so it keeps the guarantee while matching Wilson's width.
    """
    x = int(round(p_hat * n))
    a = delta / 2.0
    if x == 0:
        L = 0.0
    else:                                   # smallest p with P(Bin >= x) >= a
        lo, hi = 0.0, x / n
        for _ in range(iters):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if 1.0 - _binom_cdf(x - 1, n, mid) < a else (lo, mid)
        L = hi
    if x == n:
        U = 1.0
    else:                                   # largest p with P(Bin <= x) >= a
        lo, hi = x / n, 1.0
        for _ in range(iters):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if _binom_cdf(x, n, mid) >= a else (lo, mid)
        U = lo
    return L, U


BOUNDS = {
    "hoeffding": hoeffding_bounds,
    "bernstein": bernstein_bounds,
    "wilson": wilson_bounds,
    "clopper_pearson": clopper_pearson_bounds,
}


# ---------------- rules ----------------
class FixedLambda:
    """Baseline: a hand picked penalty that never changes."""

    def __init__(self, lam, budget, **_):
        self.lam, self.budget = float(lam), budget

    def update(self, k, train_C, policy, env, rng):
        return {"p_hat": float(np.mean(train_C)), "moved": 0, "eval_episodes": 0, "eval_collisions": 0}


class StandardLagrangian:
    """Baseline: lambda <- max(0, lambda + beta * (p_hat - d)).

    n_eval = 0: p_hat from the last K training episodes (the usual practice).
    n_eval > 0: p_hat from n_eval evaluation episodes of the frozen greedy policy. This variant gets
                exactly the same data as the gated rule, so comparing the two isolates the effect of gating.
    """

    def __init__(self, budget, beta=None, beta_up=None, beta_down=None, lam0=0.0, n_eval=0, **_):
        # beta sets both step sizes; beta_up / beta_down give asymmetric steps, as for the gated rule.
        self.beta_up = beta_up if beta_up is not None else beta
        self.beta_down = beta_down if beta_down is not None else beta
        self.lam, self.budget, self.n_eval = float(lam0), budget, n_eval

    def update(self, k, train_C, policy, env, rng):
        if self.n_eval > 0:
            C, _ = env.rollout(policy, self.n_eval, rng)
            p_hat, eval_collisions = float(C.mean()), int(C.sum())
        else:
            p_hat, eval_collisions = float(np.mean(train_C)), 0
        old = self.lam
        step = self.beta_up if p_hat > self.budget else self.beta_down
        self.lam = max(0.0, self.lam + step * (p_hat - self.budget))
        moved = int(np.sign(self.lam - old))
        return {"p_hat": p_hat, "moved": moved, "eval_episodes": self.n_eval,
                "eval_collisions": eval_collisions}


class GatedLagrangian:
    """Our method: freeze the greedy policy, run n evaluation episodes, and move lambda
    only when the confidence interval [L, U] lies fully above or fully below the budget d.
    """

    def __init__(self, beta_up, beta_down, budget, delta=0.05, n_eval=1000,
                 bound="hoeffding", delta_schedule=True, lam0=0.0, **_):
        self.lam = float(lam0)
        self.beta_up, self.beta_down = beta_up, beta_down
        self.budget, self.delta, self.n_eval = budget, delta, n_eval
        self.bound = BOUNDS[bound]
        self.delta_schedule = delta_schedule

    def update(self, k, train_C, policy, env, rng):
        C, _ = env.rollout(policy, self.n_eval, rng)
        p_hat = float(C.mean())
        delta_k = self.delta / (2 * k ** 2) if self.delta_schedule else self.delta
        L, U = self.bound(p_hat, self.n_eval, delta_k)
        d = self.budget
        if L > d:
            self.lam = self.lam + self.beta_up * (L - d)
            moved = 1
        elif U < d:
            self.lam = max(0.0, self.lam - self.beta_down * (d - U))
            moved = -1
        else:
            moved = 0      # evidence is not clear: hold lambda
        return {"p_hat": p_hat, "L": L, "U": U, "delta_k": delta_k,
                "moved": moved, "eval_episodes": self.n_eval, "eval_collisions": int(C.sum())}


RULES = {"fixed": FixedLambda, "standard": StandardLagrangian, "gated": GatedLagrangian}


def make_rule(cfg, budget):
    cfg = dict(cfg)
    kind = cfg.pop("type")
    cfg.pop("name", None)
    return RULES[kind](budget=budget, **cfg)
