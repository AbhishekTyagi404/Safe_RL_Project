"""Slippery gridworld for the delivery robot.

Layout (default 7 x 9, S = start, G = goal, X = obstacle):

    . . . . . . . . .      <- safe route: long way around, far from obstacles
    . . . . . . . . .      <- middle route: next to the obstacles on one side
    . . X X X X X . .
    S . . . . . . . G      <- shortcut: a narrow corridor with obstacles on both sides
    . . X X X X X . .
    . . . . . . . . .
    . . . . . . . . .

With probability `slip` the robot ignores its action and moves in a uniformly
random direction. The corridor is the shortest route but the riskiest, the
route along row 1 is in between, and the route along row 0 is long but safe.

Entering an obstacle ends the episode with collision cost c = 1, so the episode
cost C = sum_t c_t is 1 if the episode has any collision and 0 otherwise.
"""

import numpy as np

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
MOVES = {UP: (-1, 0), RIGHT: (0, 1), DOWN: (1, 0), LEFT: (0, -1)}


class GridWorld:
    def __init__(self, rows=7, cols=9, start=(3, 0), goal=(3, 8), obstacles=None,
                 slip=0.1, step_reward=-1.0, goal_reward=20.0, max_steps=100):
        self.rows, self.cols = rows, cols
        self.n_states = rows * cols
        self.n_actions = 4
        self.slip = slip
        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.max_steps = max_steps

        if obstacles is None:
            obstacles = [(r, c) for r in (2, 4) for c in range(2, 7)]
        self.start = self._idx(*start)
        self.goal = self._idx(*goal)
        self.is_obstacle = np.zeros(self.n_states, dtype=bool)
        for r, c in obstacles:
            self.is_obstacle[self._idx(r, c)] = True
        self.is_terminal = self.is_obstacle.copy()
        self.is_terminal[self.goal] = True

        # Deterministic move table: next_state[s, a]. Walking into the outer wall keeps you in place.
        self.next_state = np.zeros((self.n_states, self.n_actions), dtype=int)
        for s in range(self.n_states):
            r, c = divmod(s, cols)
            for a, (dr, dc) in MOVES.items():
                nr, nc = r + dr, c + dc
                if not (0 <= nr < rows and 0 <= nc < cols):
                    nr, nc = r, c
                self.next_state[s, a] = self._idx(nr, nc)

        # Transition matrix P[s, a, s'] including slip.
        self.P = np.zeros((self.n_states, self.n_actions, self.n_states))
        for s in range(self.n_states):
            for a in range(self.n_actions):
                self.P[s, a, self.next_state[s, a]] += 1.0 - slip
                for b in range(self.n_actions):
                    self.P[s, a, self.next_state[s, b]] += slip / self.n_actions

        self.rng = np.random.default_rng()
        self.state = self.start
        self.t = 0

    def _idx(self, r, c):
        return r * self.cols + c

    def seed(self, seed):
        self.rng = np.random.default_rng(seed)

    # ---------- single episode interface (used for Q learning) ----------
    def reset(self):
        self.state, self.t = self.start, 0
        return self.state

    def step(self, action):
        if self.rng.random() < self.slip:
            action = self.rng.integers(self.n_actions)
        s2 = self.next_state[self.state, action]
        self.t += 1
        reward = self.step_reward
        cost = 0.0
        if self.is_obstacle[s2]:
            cost = 1.0
        if s2 == self.goal:
            reward += self.goal_reward
        done = bool(self.is_terminal[s2]) or self.t >= self.max_steps
        self.state = s2
        return s2, reward, cost, done

    # ---------- batch rollouts of a frozen policy (used for evaluation episodes) ----------
    def rollout(self, policy, n, rng=None):
        """Run n independent episodes of a fixed deterministic policy.

        Returns (C, G): per episode collision indicator and undiscounted return.
        """
        rng = self.rng if rng is None else rng
        s = np.full(n, self.start)
        alive = np.ones(n, dtype=bool)
        C = np.zeros(n, dtype=bool)
        G = np.zeros(n)
        for _ in range(self.max_steps):
            a = policy[s]
            slipped = rng.random(n) < self.slip
            a = np.where(slipped, rng.integers(0, self.n_actions, n), a)
            s2 = self.next_state[s, a]
            G += np.where(alive, self.step_reward, 0.0)
            hit = self.is_obstacle[s2] & alive
            reached = (s2 == self.goal) & alive
            C |= hit
            G += np.where(reached, self.goal_reward, 0.0)
            alive &= ~(hit | reached)
            s = s2
            if not alive.any():
                break
        return C.astype(float), G

    # ---------- exact evaluation (ground truth, possible because the model is known) ----------
    def evaluate(self, policy):
        """Exact collision probability, goal probability and expected return of a deterministic policy."""
        M = self.P[np.arange(self.n_states), policy]          # M[s, s'] under the policy
        d = np.zeros(self.n_states)
        d[self.start] = 1.0
        p_collision = p_goal = expected_return = 0.0
        for _ in range(self.max_steps):
            expected_return += d.sum() * self.step_reward
            d = d @ M
            p_collision += d[self.is_obstacle].sum()
            p_goal += d[self.goal]
            d[self.is_terminal] = 0.0
        expected_return += p_goal * self.goal_reward
        return p_collision, p_goal, expected_return

    def solve(self, lam, gamma=0.99, iters=2000):
        """Optimal policy for the reward r - lam * c by value iteration (a reference for the report)."""
        R = np.full(self.n_states, self.step_reward)
        R[self.goal] += self.goal_reward
        R = R - lam * self.is_obstacle            # reward and cost are paid on entering s'
        V = np.zeros(self.n_states)
        for _ in range(iters):
            cont = np.where(self.is_terminal, 0.0, V)
            Q = self.P @ (R + gamma * cont)
            V = Q.max(axis=1)
        return Q.argmax(axis=1)

    def render_policy(self, policy):
        arrows = {UP: "^", RIGHT: ">", DOWN: "v", LEFT: "<"}
        lines = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                s = self._idx(r, c)
                if s == self.goal:
                    row.append("G")
                elif self.is_obstacle[s]:
                    row.append("X")
                elif s == self.start:
                    row.append("S")
                else:
                    row.append(arrows[int(policy[s])])
            lines.append(" ".join(row))
        return "\n".join(lines)
