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
    "standard_sym": "#e8a0a8", "standard_asym": "#d1495b", "standard_eval_asym": "#edae49",
    "gated_asym_hoeff": "#00798c", "gated_asym_cp": "#66a182",
}
LABELS = {
    "fixed_0": "Fixed λ = 0", "fixed_30": "Fixed λ = 30",
    "standard": "Standard", "standard_b10": "Standard (β = 10)",
    "standard_eval": "Standard + eval data", "standard_eval_b20": "Standard + eval data (β = 20)",
    "gated": "Gated (ours)", "gated_b20": "Gated (β = 20)", "gated_b50": "Gated (β = 50)",
    "standard_sym": "Standard, symmetric", "standard_asym": "Standard",
    "standard_eval_asym": "Standard + eval data",
    "gated_asym_hoeff": "Gated, Hoeffding", "gated_asym_cp": "Gated, Clopper–Pearson",
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


PANELS = {
    "wrong": "Wrong direction λ moves\n(lower is better)",
    "over": "Episodes over budget\n(lower is better)",
    "swing": "λ change in last 20% of training\n(std of λ, lower is steadier)",
    "ret": "Final true return\n(higher is better)",
}


def metric_values(run, key):
    p, moved, lam = run["p_true"], run["moved"], run["lam"]
    if key == "wrong":
        return (((moved == 1) & (p <= BUDGET)) | ((moved == -1) & (p > BUDGET))).sum()
    if key == "over":
        return (p > BUDGET).sum() * (run["episode"][1] - run["episode"][0])
    if key == "swing":
        return lam[-max(1, len(lam) // 5):].std()
    return run["return_true"][-1]


def metrics_figure(results_dir, methods, filename, panels=("wrong", "over", "ret"), title=None):
    runs = load(results_dir)
    fig, axes = plt.subplots(1, len(panels), figsize=(4.8 * len(panels), 4.2))
    for ax, key in zip(axes, panels):
        vals = [[metric_values(r, key) for r in runs[m]] for m in methods]
        means = [np.mean(v) for v in vals]
        ses = [np.std(v, ddof=1) / np.sqrt(len(v)) for v in vals]
        x = np.arange(len(methods))
        ax.bar(x, means, yerr=ses, capsize=4, color=[COLORS[m] for m in methods])
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[m].replace(" + ", "\n+ ").replace(", ", "\n").replace("Clopper–Pearson", "Clopper–\nPearson")
                            for m in methods], fontsize=8)
        ax.set_title(PANELS[key])
        ax.grid(axis="y", alpha=0.3)
    n = len(runs[methods[0]])
    fig.suptitle(title or f"Mean ± standard error over {n} seeds")
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=150)
    plt.close(fig)


def bounds_figure():
    """Where each bound lets the gated rule move: U at a safe policy (p = 0.01) and L at the shortcut (p = 0.24)."""
    from safe_rl.updates import BOUNDS
    names = {"hoeffding": "Hoeffding", "bernstein": "Empirical Bernstein",
             "wilson": "Wilson (approximate)", "clopper_pearson": "Clopper–Pearson"}
    ns = np.array([50, 100, 200, 300, 500, 750, 1000, 1500, 2000])
    delta_k = 0.05 / (2 * 50 ** 2)                    # the level at check k = 50 (episode 2,500)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    for b, label in names.items():
        U = [BOUNDS[b](0.01, int(n), delta_k)[1] for n in ns]
        L = [BOUNDS[b](0.24, int(n), delta_k)[0] for n in ns]
        style = "--" if b == "wilson" else "-"
        axes[0].plot(ns, U, style, marker="o", markersize=3, label=label)
        axes[1].plot(ns, L, style, marker="o", markersize=3, label=label)
    for ax in axes:
        ax.axhline(BUDGET, color="black", linestyle=":", linewidth=1)
        ax.set_xlabel("evaluation episodes n")
        ax.grid(alpha=0.3)
    axes[0].set_title("Safe policy (p̂ = 0.01): upper bound U\nλ can move down only below the dotted line")
    axes[1].set_title("Shortcut (p̂ = 0.24): lower bound L\nλ can move up only above the dotted line")
    axes[0].legend(fontsize=8)
    fig.suptitle("Confidence bounds at check k = 50 (δ_k = δ / 2k², δ = 0.05)")
    fig.tight_layout()
    fig.savefig(OUT / "bounds.png", dpi=150)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gridworld_figure()
    curves_figure("results/final", ["standard", "standard_eval", "gated"],
                  "final_curves.png", "Final comparison (20 seeds, 6,000 episodes)")
    metrics_figure("results/final", ["standard", "standard_eval", "gated"], "final_metrics.png")
    curves_figure("results/long_check", ["standard_b10", "standard_eval_b20", "gated_b20", "gated_b50"],
                  "long_check_curves.png", "Long horizon check (3 seeds, 20,000 episodes)")
    matched = ["standard_asym", "standard_eval_asym", "gated_asym_hoeff", "gated_asym_cp"]
    if Path("results/final_matched").exists():
        metrics_figure("results/final_matched", ["standard_sym"] + matched, "matched_metrics.png",
                       panels=("wrong", "over", "swing", "ret"),
                       title="Matched step sizes (β_up = 20, β_down = 2), 20 held-out seeds, 10,000 episodes")
        curves_figure("results/final_matched", matched, "matched_curves.png",
                      "Matched step sizes (20 held-out seeds, 10,000 episodes)")
    bounds_figure()
    print(f"Saved figures to {OUT}/")


if __name__ == "__main__":
    main()
