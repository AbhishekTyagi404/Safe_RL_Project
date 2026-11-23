"""Run every method in a config file over several seeds and save the results.

Usage:
    python run_experiment.py configs/baselines.yaml
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import yaml

from safe_rl.train import run, summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--seeds", type=int, default=None, help="override the number of seeds")
    parser.add_argument("--seed-offset", type=int, default=None,
                        help="first seed; use different seeds for tuning and for the final comparison")
    parser.add_argument("--budget", type=float, default=None,
                        help="override the budget d; results go to results/<name>_d<budget>")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.budget is not None:
        cfg["budget"] = args.budget
        cfg["name"] = f"{cfg['name']}_d{args.budget}"
    n_seeds = args.seeds or cfg["seeds"]
    offset = args.seed_offset if args.seed_offset is not None else cfg.get("seed_offset", 0)
    out_dir = Path("results") / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(cfg, open(out_dir / "config.json", "w"), indent=2)

    rows = []
    for method in cfg["methods"]:
        name = method["name"]
        t0 = time.time()
        for seed in range(offset, offset + n_seeds):
            out, env = run(cfg.get("env", {}), cfg.get("agent", {}), method,
                           cfg["n_episodes"], cfg["check_every"], cfg["budget"], seed)
            np.savez(out_dir / f"{name}_seed{seed}.npz", **out)
            summary = summarize(out, cfg["check_every"], cfg["budget"])
            rows.append({"method": name, "seed": seed, **summary})
        print(f"  {name:<28} done in {time.time() - t0:5.1f}s")

    with open(out_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Mean over seeds, one row per method.
    cols = ["final_p_true", "final_return_true", "episodes_over_budget", "over_budget_2nd_half",
            "training_collisions", "eval_collisions", "lambda_final", "lambda_max",
            "lambda_std_last20pct", "wrong_direction_moves",
            "frozen_fraction", "eval_episodes_used"]
    means = []
    for method in cfg["methods"]:
        rs = [r for r in rows if r["method"] == method["name"]]
        means.append({"method": method["name"], **{c: float(np.mean([r[c] for r in rs])) for c in cols}})
    with open(out_dir / "summary_mean.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method"] + cols)
        writer.writeheader()
        writer.writerows(means)

    print(f"\nMean over {n_seeds} seeds (budget d = {cfg['budget']}):")
    print(f"  {'method':<28}{'final p':>8}{'return':>8}{'ep over':>9}{'over 2nd':>9}{'λ max':>7}"
          f"{'λ final':>8}{'λ std':>7}{'wrong':>7}{'frozen':>8}")
    for m in means:
        print(f"  {m['method']:<28}{m['final_p_true']:8.3f}{m['final_return_true']:8.2f}"
              f"{m['episodes_over_budget']:9.0f}{m['over_budget_2nd_half']:9.0f}{m['lambda_max']:7.1f}"
              f"{m['lambda_final']:8.1f}{m['lambda_std_last20pct']:7.2f}"
              f"{m['wrong_direction_moves']:7.1f}{m['frozen_fraction']:8.2f}")
    print(f"\nSaved results to {out_dir}")


if __name__ == "__main__":
    main()
