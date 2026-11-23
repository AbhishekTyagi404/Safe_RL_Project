# Wrong direction bound for the confidence gated update

## Setting

Training is split into checks k = 1, 2, 3, ... . At check k the current greedy policy π_k is frozen and
run for n evaluation episodes. Episode i gives the collision indicator C_{k,i} ∈ {0, 1}, and

  p_k = p(π_k) = P(an episode of π_k has a collision)

is the quantity the constraint p(π) ≤ d is about. Let F_k be everything that happened before the
evaluation episodes of check k (all training, all earlier checks). π_k is fixed given F_k.

The rule computes p̂_k = (1/n) Σ_i C_{k,i} and an interval [L_k, U_k] at level δ_k = δ / (2k²), then

  λ ← λ + β_up (L_k − d)          if L_k > d     (move up)
  λ ← max(0, λ − β_down (d − U_k)) if U_k < d     (move down)
  λ unchanged                       otherwise

**A move is in the wrong direction** if λ moves up while p_k ≤ d, or moves down while p_k > d.
This is exactly the `wrong_direction_moves` metric in the code, where p_k is computed exactly by
`GridWorld.evaluate` before each update.

## Assumption

(A) Given F_k, the evaluation episodes C_{k,1}, ..., C_{k,n} are independent Bernoulli(p_k) variables.

This holds in our implementation because the evaluation episodes use a frozen policy, do not update
the Q table, and draw fresh randomness from a separate generator. It would **not** hold if p̂_k were
computed from training episodes, because the policy changes during training.

## Lemma (coverage at one check)

Under (A), with ε_k = sqrt( ln(2/δ_k) / (2n) ), L_k = p̂_k − ε_k and U_k = p̂_k + ε_k (clipped to [0, 1]),

  P( p_k ∉ [L_k, U_k] | F_k ) ≤ δ_k.

*Proof.* Given F_k, p̂_k is the mean of n independent variables in [0, 1] with mean p_k. Hoeffding's
inequality gives P(|p̂_k − p_k| ≥ ε | F_k) ≤ 2 exp(−2nε²). Setting ε = ε_k makes the right side
equal to δ_k. Clipping to [0, 1] cannot exclude p_k, since p_k ∈ [0, 1]. ∎

Taking the expectation over F_k removes the conditioning: P(p_k ∉ [L_k, U_k]) ≤ δ_k.

## Theorem

Under (A), for any δ ∈ (0, 1), any step sizes β_up, β_down ≥ 0, and any number of checks,

  P( λ ever moves in the wrong direction during training ) ≤ (π²/12) δ < δ.

*Proof.* Let E_k = { p_k ∈ [L_k, U_k] } be the event that check k's interval covers the true value.

On E_k, a move up requires L_k > d, so p_k ≥ L_k > d: the move is in the right direction. A move down
requires U_k < d, so p_k ≤ U_k < d: also the right direction. So a wrong direction move at check k
can only happen on the complement of E_k.

By the union bound and the lemma,

  P( some check moves λ the wrong way ) ≤ Σ_{k≥1} P(E_kᶜ) ≤ Σ_{k≥1} δ / (2k²) = (δ/2)(π²/6) = (π²/12) δ ≈ 0.82 δ.

The sum runs over all k ≥ 1, so the bound holds no matter how long training runs. ∎

## Other bounds

The proof uses only the coverage property of the lemma, so it holds unchanged for any interval with
P(p_k ∉ [L_k, U_k] | F_k) ≤ δ_k for every n:

| Bound | Valid for the theorem? | Why |
|---|---|---|
| Hoeffding | yes | exact finite sample bound (used above) |
| Empirical Bernstein (Maurer and Pontil, 2009) | yes | finite sample bound for variables in [0, 1] |
| Clopper–Pearson | yes | exact binomial interval, coverage ≥ 1 − δ_k for every n and p |
| Wilson score | no | coverage is only approximate (normal approximation) |

## What the theorem does not say

* It says nothing about **how often** λ moves. If p_k stays close to d, the interval can contain d at
  every check and λ never moves at all (see the d = 0.05 run, where λ was held 96% of the time).
* It does not guarantee that the constraint is satisfied, or that training converges.
* It is about the frozen greedy policy π_k at each check, not about the exploring policy used during
  training.

## Contrast with the standard update

The standard rule moves λ up whenever p̂_k > d. Even with perfectly i.i.d. data, when the true p_k is
just below d, P(p̂_k > d) can be close to 1/2, so wrong moves are expected at almost every check near
the boundary. With training data it is worse: those episodes include exploratory actions, so p̂ is
biased upward relative to the greedy policy. In our runs this produced 8.9 wrong moves per run at
d = 0.1 and 55 per run at d = 0.05.

## Empirical check

Across every gated run in this repository (259 runs: tuning, held-out, long horizon and budget sweeps,
with all four bounds and δ = 0.05), the gated rule made **0** wrong direction moves. The theorem allows
a failure probability of up to 0.82 × 0.05 ≈ 0.041 per run, so a handful of failures would also have
been consistent with it; seeing none reflects that Hoeffding and Clopper–Pearson are conservative.
Wilson runs also had none, but the theorem does not cover them.
