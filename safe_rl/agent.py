"""Tabular Q learning on the Lagrangian reward r_t - lambda * c_t."""

import numpy as np


class QLearning:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1, q_init=0.0, rng=None):
        # Optimistic q_init pushes early exploration, so the agent finds the goal
        # instead of settling on crashing early (which only costs one step).
        self.Q = np.full((n_states, n_actions), float(q_init))
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.n_actions = n_actions
        self.rng = rng if rng is not None else np.random.default_rng()

    def act(self, s):
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        q = self.Q[s]
        best = np.flatnonzero(q == q.max())          # random tie break
        return int(self.rng.choice(best))

    def update(self, s, a, r, s2, done):
        target = r if done else r + self.gamma * self.Q[s2].max()
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])

    def greedy_policy(self):
        return self.Q.argmax(axis=1)
