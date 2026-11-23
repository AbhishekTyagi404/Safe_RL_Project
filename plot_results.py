"""Plot lambda and the true collision probability over training for every method.

Usage:
    python plot_results.py results/baselines
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir")
    args = parser.parse_args()
    d = Path(args.results_dir)
    cfg = json.load(open(d / "config.json"))
    budget = cfg["budget"]

    runs = defaultdict(list)
    for f in sorted(d.glob("*_seed*.npz")):
        runs[f.stem.rsplit("_seed", 1)[0]].append(np.load(f))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for name in [m["name"] for m in cfg["methods"]]:
        if name not in runs:
            continue
        ep = runs[name][0]["episode"]
        for ax, key in zip(axes, ["lam", "p_true", "return_true"]):
            vals = np.stack([r[key] for r in runs[name]])
            mean = vals.mean(0)
            se = vals.std(0) / np.sqrt(len(vals))
            line, = ax.plot(ep, mean, label=name)
            ax.fill_between(ep, mean - se, mean + se, color=line.get_color(), alpha=0.2)

    axes[0].set_title("Penalty weight λ")
    axes[1].set_title("True collision probability of greedy policy")
    axes[1].axhline(budget, color="black", linestyle="--", linewidth=1, label=f"budget d = {budget}")
    axes[2].set_title("True expected return of greedy policy")
    for ax in axes:
        ax.set_xlabel("training episode")
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    out = d / "training_curves.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
