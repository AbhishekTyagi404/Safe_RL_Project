"""Paired comparison of two methods run on the same seeds.

For every metric it reports, over seeds i = 1..n:
    D_i      = metric(method A, seed i) - metric(method B, seed i)
    mean     = average of D_i
    SE       = sample standard deviation of D_i (n - 1 in the denominator) / sqrt(n)
    95% CI   = mean +/- t(0.975, n - 1) * SE
    p (t)    = two sided paired t test
    p (perm) = exact two sided sign flip permutation test (no normality assumption)

For return metrics it adds two tests against a margin M = (margin_pct / 100) * mean of method B:
    non inferiority: H0 "A is worse than B by at least M", one sided t test on D_i + M
    TOST equivalence: H0 "|A - B| >= M", two one sided t tests; p is the larger of the two.
                      Equivalence at level 0.05 is the same as the 90% CI lying inside [-M, +M].
The margin should be fixed before looking at results (ours, 5%, comes from the project proposal).

Usage:
    python paired_stats.py results/final_matched --a gated_asym_hoeff \
        --b standard_asym standard_eval_asym gated_asym_cp
"""

import argparse
import csv
import itertools
import math
from pathlib import Path

import numpy as np

METRICS = ["wrong_direction_moves", "episodes_over_budget", "over_budget_2nd_half",
           "lambda_std_last20pct", "final_return_true", "return_last20pct", "final_p_true"]

# t(0.975, df) for small df, used only if scipy is not installed.
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
        10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110,
        18: 2.101, 19: 2.093, 20: 2.086, 25: 2.060, 30: 2.042}


def t_quantile(df):
    try:
        from scipy.stats import t
        return float(t.ppf(0.975, df))
    except ImportError:
        return T975.get(df, 1.96)


def _t_sf(t, df, steps=20000):
    """P(T > t) for a t distribution, by numerical integration (fallback when scipy is missing)."""
    c = math.gamma((df + 1) / 2) / (math.sqrt(df * math.pi) * math.gamma(df / 2))
    x = np.linspace(0.0, t, steps + 1)
    pdf = c * (1 + x * x / df) ** (-(df + 1) / 2)
    return 0.5 - np.trapz(pdf, x) if hasattr(np, "trapz") else 0.5 - np.trapezoid(pdf, x)


def t_test_p(d):
    if not np.any(d != d[0]):                        # all differences equal: no variance
        return 1.0 if d[0] == 0 else 0.0
    n = len(d)
    t = abs(d.mean()) / (d.std(ddof=1) / np.sqrt(n))
    try:
        from scipy.stats import t as tdist
        return float(2 * tdist.sf(t, n - 1))
    except ImportError:
        return float(2 * _t_sf(t, n - 1))


def one_sided_p(t_stat, df):
    """P(T > t_stat)."""
    try:
        from scipy.stats import t as tdist
        return float(tdist.sf(t_stat, df))
    except ImportError:
        return float(_t_sf(t_stat, df)) if t_stat >= 0 else 1.0 - float(_t_sf(-t_stat, df))


def t90(df):
    try:
        from scipy.stats import t
        return float(t.ppf(0.95, df))
    except ImportError:
        return {19: 1.729}.get(df, 1.645)


def sign_flip_p(d):
    """Exact permutation test: under H0 each paired difference is equally likely to have either sign."""
    n = len(d)
    observed = abs(d.mean())
    if n <= 20:
        signs = np.array(list(itertools.product([1, -1], repeat=n)))
    else:
        signs = np.random.default_rng(0).choice([1, -1], size=(200000, n))
    means = np.abs((signs * d).mean(axis=1))
    return float(np.mean(means >= observed - 1e-12))


def load(results_dir):
    """Recompute every metric from the saved run logs, so new metrics need no rerun."""
    import json
    from safe_rl.train import summarize
    d = Path(results_dir)
    cfg = json.load(open(d / "config.json"))
    data = {}
    for f in d.glob("*_seed*.npz"):
        method, seed = f.stem.rsplit("_seed", 1)
        run = dict(np.load(f))
        data.setdefault(method, {})[int(seed)] = summarize(run, cfg["check_every"], cfg["budget"])
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--a", required=True, help="method A")
    ap.add_argument("--b", nargs="+", required=True, help="one or more methods B")
    ap.add_argument("--margin-pct", type=float, default=5.0,
                    help="equivalence and non inferiority margin for return, in percent of B's mean")
    args = ap.parse_args()
    data = load(args.results_dir)

    out_rows = []
    for b in args.b:
        seeds = sorted(set(data[args.a]) & set(data[b]))
        n = len(seeds)
        tq = t_quantile(n - 1)
        print(f"\n{args.a}  minus  {b}   ({n} paired seeds)")
        print(f"  {'metric':<24}{'mean diff':>11}{'SE':>8}{'95% CI':>22}{'p (t)':>9}{'p (perm)':>10}")
        for m in METRICS:
            d = np.array([data[args.a][s][m] - data[b][s][m] for s in seeds])
            mean, se = d.mean(), d.std(ddof=1) / np.sqrt(n)
            lo, hi = mean - tq * se, mean + tq * se
            pt, pp = t_test_p(d), sign_flip_p(d)
            print(f"  {m:<24}{mean:11.3f}{se:8.3f}   [{lo:8.3f}, {hi:8.3f}]{pt:9.4f}{pp:10.4f}")
            out_rows.append({"a": args.a, "b": b, "metric": m, "n": n, "mean_diff": mean, "se": se,
                             "ci95_low": lo, "ci95_high": hi, "p_t": pt, "p_perm": pp})

        print(f"\n  Return against a margin of {args.margin_pct:g}% of {b}'s mean:")
        print(f"  {'metric':<24}{'margin':>8}{'90% CI':>22}{'p non inf.':>12}{'p TOST':>9}  verdict")
        for m in ["final_return_true", "return_last20pct"]:
            d = np.array([data[args.a][s][m] - data[b][s][m] for s in seeds])
            margin = args.margin_pct / 100 * np.mean([data[b][s][m] for s in seeds])
            mean, se = d.mean(), d.std(ddof=1) / np.sqrt(n)
            lo90, hi90 = mean - t90(n - 1) * se, mean + t90(n - 1) * se
            p_noninf = one_sided_p((mean + margin) / se, n - 1)          # H0: mean <= -margin
            p_upper = one_sided_p((margin - mean) / se, n - 1)           # H0: mean >= +margin
            p_tost = max(p_noninf, p_upper)
            verdict = ("equivalent within margin" if p_tost < 0.05 else
                       "not worse by more than margin" if p_noninf < 0.05 else "inconclusive")
            print(f"  {m:<24}{margin:8.3f}   [{lo90:8.3f}, {hi90:8.3f}]{p_noninf:12.4f}{p_tost:9.4f}  {verdict}")
            for r in out_rows:
                if r["b"] == b and r["metric"] == m:
                    r.update({"margin": margin, "ci90_low": lo90, "ci90_high": hi90,
                              "p_noninferiority": p_noninf, "p_tost": p_tost, "verdict": verdict})

    out = Path(args.results_dir) / f"paired_{args.a}.csv"
    with open(out, "w", newline="") as f:
        fields = list(dict.fromkeys(k for r in out_rows for k in r))
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
