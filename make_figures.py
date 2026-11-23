"""Build the figures used in the README and the report.

Run the experiments first:
    python run_experiment.py configs/final.yaml
    python run_experiment.py configs/long_check.yaml
Then:
    python make_figures.py

Figures are saved to docs/figures/ (tracked by git, unlike results/).
"""

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from safe_rl.gridworld import GridWorld, MOVES

OUT = Path("docs/figures")
BUDGET = 0.1
COLORS = {
    "fixed_0": "#9e9e9e", "fixed_30": "#5f5f5f",
    "standard": "#d1495b", "standard_b10": "#d1495b",
    "standard_eval": "#edae49", "standard_eval_b20": "#edae49",
    "gated": "#00798c", "gated_b20": "#00798c", "gated_b50": "#30638e",
}
LABELS = {
    "fixed_0": "Fixed λ = 0", "fixed_30": "Fixed λ = 30",
    "standard": "Standard", "standard_b10": "Standard (β = 10)",
    "standard_eval": "Standard + eval data", "standard_eval_b20": "Standard + eval data (β = 20)",
    "gated": "Gated (ours)", "gated_b20": "Gated (β = 20)", "gated_b50": "Gated (β = 50)",
}

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def load(results_dir):
    runs = defaultdict(list)
    for f in sorted(Path(results_dir).glob("*_seed*.npz")):
        runs[f.stem.rsplit("_seed", 1)[0]].append(np.load(f))
    return runs


def mean_se(runs, key):
    v = np.stack([r[key] for r in runs])
    return v.mean(0), v.std(0) / np.sqrt(len(v))


def gridworld_figure():
    env = GridWorld()
    fig, ax = plt.subplots(figsize=(6, 4.8))
    for s in range(env.n_states):
        r, c = divmod(s, env.cols)
        color = "#2f2f2f" if env.is_obstacle[s] else "white"
        ax.add_patch(plt.Rectangle((c, -r - 1), 1, 1, facecolor=color, edgecolor="#cccccc"))

    for lam, color, label in [(0, COLORS["standard"], "best route with λ = 0 (shortcut)"),
                              (40, COLORS["gated"], "best route with λ = 40 (safe)")]:
        policy = env.solve(lam)
        p, _, ret = env.evaluate(policy)
        s, xs, ys = env.start, [], []
        for _ in range(40):
            r, c = divmod(s, env.cols)
            xs.append(c + 0.5)
            ys.append(-r - 0.5)
            if s == env.goal or env.is_obstacle[s]:
                break
            s = env.next_state[s, policy[s]]
        ax.plot(xs, ys, color=color, linewidth=3, alpha=0.85,
                label=f"{label}: collision p = {p:.3f}, return = {ret:.2f}")

    for s, text in [(env.start, "S"), (env.goal, "G")]:
        r, c = divmod(s, env.cols)
        ax.text(c + 0.5, -r - 0.5, text, ha="center", va="center", fontsize=13, fontweight="bold",
                zorder=5, bbox=dict(boxstyle="circle,pad=0.25", facecolor="white", edgecolor="black"))
    ax.set_xlim(0, env.cols)
    ax.set_ylim(-env.rows, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), fontsize=9, frameon=False)
    ax.set_title(f"Gridworld (slip = {env.slip}, budget d = {BUDGET})")
    fig.tight_layout()
    fig.savefig(OUT / "gridworld.png", dpi=150)
    plt.close(fig)


def curves_figure(results_dir, methods, filename, title):
    runs = load(results_dir)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name in methods:
        if name not in runs:
            continue
        ep = runs[name][0]["episode"]
        for ax, key in zip(axes, ["lam", "p_true"]):
            m, se = mean_se(runs[name], key)
            ax.plot(ep, m, color=COLORS[name], label=LABELS[name], linewidth=1.6)
            ax.fill_between(ep, m - se, m + se, color=COLORS[name], alpha=0.15)
    axes[0].set_title("Penalty weight λ")
    axes[1].set_title("True collision probability of the greedy policy")
    axes[1].axhline(BUDGET, color="black", linestyle="--", linewidth=1, label=f"budget d = {BUDGET}")
    for ax in axes:
        ax.set_xlabel("training episode")
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=150)
    plt.close(fig)


def metrics_figure(results_dir, methods, filename):
    runs = load(results_dir)
    panels = [("wrong", "Wrong direction λ moves\n(lower is better)"),
              ("over", "Episodes over budget\n(lower is better)"),
              ("ret", "Final true return\n(higher is better)")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (key, title) in zip(axes, panels):
        means, ses = [], []
        for name in methods:
            vals = []
            for r in runs[name]:
                p, moved = r["p_true"], r["moved"]
                if key == "wrong":
                    vals.append((((moved == 1) & (p <= BUDGET)) | ((moved == -1) & (p > BUDGET))).sum())
                elif key == "over":
                    vals.append((p > BUDGET).sum() * (r["episode"][1] - r["episode"][0]))
                else:
                    vals.append(r["return_true"][-1])
            means.append(np.mean(vals))
            ses.append(np.std(vals) / np.sqrt(len(vals)))
        x = np.arange(len(methods))
        ax.bar(x, means, yerr=ses, capsize=4, color=[COLORS[m] for m in methods])
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[m].replace(" + ", "\n+ ") for m in methods], fontsize=8)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"Final comparison, mean ± standard error over {len(runs[methods[0]])} seeds")
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=150)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gridworld_figure()
    curves_figure("results/final", ["standard", "standard_eval", "gated"],
                  "final_curves.png", "Final comparison (20 seeds, 6,000 episodes)")
    metrics_figure("results/final", ["standard", "standard_eval", "gated"], "final_metrics.png")
    curves_figure("results/long_check", ["standard_b10", "standard_eval_b20", "gated_b20", "gated_b50"],
                  "long_check_curves.png", "Long horizon check (3 seeds, 20,000 episodes)")
    print(f"Saved figures to {OUT}/")


if __name__ == "__main__":
    main()
