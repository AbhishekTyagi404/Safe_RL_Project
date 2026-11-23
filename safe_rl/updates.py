"""Rules for updating the penalty weight lambda every K episodes.

Every rule has the same interface:
    rule.lam                                   current lambda
    rule.update(k, train_C, policy, env, rng)  returns an info dict and changes rule.lam

k        index of this check (1, 2, 3, ...)
train_C  collision indicators of the last K training episodes
policy   current greedy policy (used by rules that run evaluation episodes)
"""

import numpy as np


# ---------------- confidence bounds ----------------
def hoeffding_bounds(p_hat, n, delta):
    eps = np.sqrt(np.log(2.0 / delta) / (2.0 * n))
    return max(0.0, p_hat - eps), min(1.0, p_hat + eps)


BOUNDS = {"hoeffding": hoeffding_bounds}   # Wilson and Bernstein are added in step 4


# ---------------- rules ----------------
class FixedLambda:
    """Baseline: a hand picked penalty that never changes."""

    def __init__(self, lam, budget, **_):
        self.lam, self.budget = float(lam), budget

    def update(self, k, train_C, policy, env, rng):
        return {"p_hat": float(np.mean(train_C)), "moved": 0, "eval_episodes": 0}


class StandardLagrangian:
    """Baseline: lambda <- max(0, lambda + beta * (p_hat - d)), p_hat from the last K training episodes."""

    def __init__(self, beta, budget, lam0=0.0, **_):
        self.lam, self.beta, self.budget = float(lam0), beta, budget

    def update(self, k, train_C, policy, env, rng):
        p_hat = float(np.mean(train_C))
        old = self.lam
        self.lam = max(0.0, self.lam + self.beta * (p_hat - self.budget))
        moved = int(np.sign(self.lam - old))
        return {"p_hat": p_hat, "moved": moved, "eval_episodes": 0}


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
                "moved": moved, "eval_episodes": self.n_eval}


RULES = {"fixed": FixedLambda, "standard": StandardLagrangian, "gated": GatedLagrangian}


def make_rule(cfg, budget):
    cfg = dict(cfg)
    kind = cfg.pop("type")
    cfg.pop("name", None)
    return RULES[kind](budget=budget, **cfg)
