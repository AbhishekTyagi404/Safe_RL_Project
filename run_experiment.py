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
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    n_seeds = args.seeds or cfg["seeds"]
    out_dir = Path("results") / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(cfg, open(out_dir / "config.json", "w"), indent=2)

    rows = []
    for method in cfg["methods"]:
        name = method["name"]
        t0 = time.time()
        for seed in range(n_seeds):
            out, env = run(cfg.get("env", {}), cfg.get("agent", {}), method,
                           cfg["n_episodes"], cfg["check_every"], cfg["budget"], seed)
            np.savez(out_dir / f"{name}_seed{seed}.npz", **out)
            summary = summarize(out, cfg["check_every"], cfg["budget"])
            rows.append({"method": name, "seed": seed, **summary})
        done = [r for r in rows if r["method"] == name]
        print(f"{name:<22} {time.time() - t0:6.1f}s | "
              f"final p {np.mean([r['final_p_true'] for r in done]):.3f} | "
              f"final return {np.mean([r['final_return_true'] for r in done]):6.2f} | "
              f"episodes over budget {np.mean([r['episodes_over_budget'] for r in done]):7.0f} | "
              f"wrong moves {np.mean([r['wrong_direction_moves'] for r in done]):.1f}")

    with open(out_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved results to {out_dir}")


if __name__ == "__main__":
    main()
