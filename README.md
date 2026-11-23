# Confidence Gated Lagrangian Updates for Safe Reinforcement Learning

ECE 59500 Reinforcement Learning course project.
Team: Abhishek Tyagi, Ashish Kale, Ayush Kumar.

A delivery robot in a slippery gridworld must reach its goal while keeping the chance of a
collision per episode at or below d = 0.1. We compare ways of tuning the collision penalty λ:

| Method | Rule |
|---|---|
| Fixed λ | hand picked, never changes |
| Standard Lagrangian | λ ← max(0, λ + β(p̂ − d)), p̂ from the last K training episodes |
| Gated (ours) | freeze the greedy policy, run n evaluation episodes, build a confidence interval [L, U] on the collision rate, and move λ only when the interval is fully above or fully below d |

## Environment

```
. . . . . . . . .      safe route: long way around
. . . . . . . . .
. . X X X X X . .
S . . . . . . . G      shortcut: narrow corridor, about 24% collision risk
. . X X X X X . .
. . . . . . . . .
. . . . . . . . .
```

With probability 0.1 the robot slips and moves in a random direction. Reward is −1 per step and
+20 at the goal; entering an obstacle ends the episode with cost 1. The model is known, so the
code computes the **exact** collision probability of any policy (`GridWorld.evaluate`). We use it
as ground truth to count episodes over budget and to check whether each λ move went the right way.

## Setup

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate      macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python run_experiment.py configs/baselines.yaml            # all methods, seeds from the config
python run_experiment.py configs/baselines.yaml --seeds 2  # quick test
python plot_results.py results/baselines
```

Results go to `results/<config name>/`:
- `summary.csv`: one row per method and seed
- `training_curves.png`: λ, true collision probability and true return over training
- `<method>_seed<k>.npz`: full logs of each run

## Metrics (in `summary.csv`)

| Column | Meaning |
|---|---|
| `final_p_true` | exact collision probability of the final greedy policy |
| `final_return_true` | exact expected return of the final greedy policy |
| `episodes_over_budget` | training episodes during which the greedy policy's true p was above d |
| `training_collisions` | crashes that happened while learning |
| `lambda_std_last20pct` | how much λ still swings late in training |
| `wrong_direction_moves` | λ went up while true p ≤ d, or down while true p > d |
| `frozen_fraction` | share of checks where λ did not move |
| `eval_episodes_used` | extra evaluation episodes the method needed |

## Project structure

```
safe_rl/
  gridworld.py   environment, batch rollouts, exact evaluation, value iteration reference
  agent.py       tabular Q learning on r − λc
  updates.py     λ update rules and confidence bounds
  train.py       training loop and summary metrics
configs/         experiment settings (one YAML file per experiment)
run_experiment.py
plot_results.py
```

## Roadmap

- [x] Step 2: environment, Q learning, fixed and standard Lagrangian baselines, gated update with Hoeffding
- [ ] Step 3: tune β, n and K; first full comparison with 20 seeds
- [ ] Step 4: Wilson and Bernstein bounds, sweeps over d, n and β, proof of the wrong direction bound
- [ ] Step 5: report, recorded presentation
