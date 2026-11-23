"""One training run: Q learning plus a lambda update rule, checked every K episodes."""

import numpy as np

from .agent import QLearning
from .gridworld import GridWorld
from .updates import make_rule


def run(env_cfg, agent_cfg, rule_cfg, n_episodes, check_every, budget, seed):
    rng = np.random.default_rng(seed)
    env = GridWorld(**env_cfg)
    env.seed(int(rng.integers(2**31)))
    agent = QLearning(env.n_states, env.n_actions, rng=rng, **agent_cfg)
    rule = make_rule(rule_cfg, budget)
    eval_rng = np.random.default_rng(int(rng.integers(2**31)))

    train_C = np.zeros(n_episodes)
    train_G = np.zeros(n_episodes)
    log = {k: [] for k in ["episode", "lam", "p_true", "return_true", "p_goal_true",
                           "p_hat", "L", "U", "moved", "eval_episodes", "eval_collisions"]}

    for ep in range(n_episodes):
        s = env.reset()
        done = False
        while not done:
            a = agent.act(s)
            s2, r, c, done = env.step(a)
            agent.update(s, a, r - rule.lam * c, s2, done)
            train_C[ep] = max(train_C[ep], c)
            train_G[ep] += r
            s = s2

        if (ep + 1) % check_every == 0:
            k = (ep + 1) // check_every
            policy = agent.greedy_policy()
            p_true, p_goal, ret_true = env.evaluate(policy)     # ground truth before the update
            info = rule.update(k, train_C[ep + 1 - check_every: ep + 1], policy, env, eval_rng)
            log["episode"].append(ep + 1)
            log["lam"].append(rule.lam)
            log["p_true"].append(p_true)
            log["return_true"].append(ret_true)
            log["p_goal_true"].append(p_goal)
            log["p_hat"].append(info["p_hat"])
            log["L"].append(info.get("L", np.nan))
            log["U"].append(info.get("U", np.nan))
            log["moved"].append(info["moved"])
            log["eval_episodes"].append(info["eval_episodes"])
            log["eval_collisions"].append(info["eval_collisions"])

    out = {k: np.array(v) for k, v in log.items()}
    out["train_C"] = train_C
    out["train_G"] = train_G
    out["final_policy"] = agent.greedy_policy()
    return out, env


def summarize(out, check_every, budget):
    """Scalar metrics for one run."""
    p = out["p_true"]
    moved = out["moved"]
    # A move is wrong if lambda went up while the true p was under budget, or down while over it.
    wrong = ((moved == 1) & (p <= budget)) | ((moved == -1) & (p > budget))
    tail = max(1, len(p) // 5)                       # last 20% of checks
    return {
        "final_p_true": float(p[-1]),
        "final_return_true": float(out["return_true"][-1]),
        "episodes_over_budget": int((p > budget).sum() * check_every),
        "training_collisions": int(out["train_C"].sum()),
        "lambda_final": float(out["lam"][-1]),
        "lambda_std_last20pct": float(out["lam"][-tail:].std()),
        "lambda_moves": int((moved != 0).sum()),
        "wrong_direction_moves": int(wrong.sum()),
        "frozen_fraction": float((moved == 0).mean()),
        "eval_episodes_used": int(out["eval_episodes"].sum()),
        "eval_collisions": int(out["eval_collisions"].sum()),
    }
