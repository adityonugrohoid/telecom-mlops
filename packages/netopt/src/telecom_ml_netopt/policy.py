"""Netopt policies.

`QPolicy` is tabular Q-learning on the observed (SINR, interference, latency) bins, from the
network-optimization repo's QLearningAgent (learning to a table of state-action values,
discount 0.95, epsilon-greedy exploration). The step size and exploration schedule are set
once, with their reasons:
- Per-entry step size 1/(1 + visits since this training run), floored at 0.02: it decays
  within a run so the values settle, starts fresh at each retrain because the dynamics may
  have changed, and the floor keeps rarely visited entries moving.
- First training: 2,000 episodes, epsilon 1.0 decaying to 0.01. Retraining warm-starts from
  the live table for 500 episodes, epsilon 0.2 decaying to 0.01: after a change the agent has
  to try other actions to find out whether the best one moved.

`RulePolicy` is the static rule-based baseline: decrease power when interference is in the
top two bins; otherwise balance load when latency is in the top two bins; otherwise increase
power when SINR is in the bottom two bins; otherwise adjust tilt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from telecom_ml_netopt.generator import (
    ACTIONS,
    BINS,
    STEPS,
    Environment,
    initial_state,
    observe,
    step,
)

GAMMA = 0.95
STEP_FLOOR = 0.02
FIRST_EPISODES = 2_000
RETRAIN_EPISODES = 500
FIRST_EPSILON = (1.0, 0.01)
RETRAIN_EPSILON = (0.2, 0.01)

Observation = tuple[int, int, int]


class Policy(Protocol):
    def act(self, observation: Observation) -> int: ...


@dataclass(frozen=True)
class QPolicy:
    table: np.ndarray

    def act(self, observation: Observation) -> int:
        """The greedy action for an observation."""
        return int(np.argmax(self.table[observation]))


class RulePolicy:
    def act(self, observation: Observation) -> int:
        sinr, interference, latency = observation
        if interference >= BINS - 2:
            return ACTIONS.index("decrease_power")
        if latency >= BINS - 2:
            return ACTIONS.index("load_balance")
        if sinr <= 1:
            return ACTIONS.index("increase_power")
        return ACTIONS.index("adjust_tilt")


def run_episode(policy: Policy, env: Environment, seed: int) -> float:
    """Total reward of one 50-step episode on a fresh cell, seeded by `seed`."""
    rng = np.random.default_rng(seed)
    state, outage = initial_state(rng, env)
    total = 0.0
    for _ in range(STEPS):
        action = policy.act(observe(rng, state, env))
        state, reward = step(rng, state, ACTIONS[action], env, outage)
        total += reward
    return total


def learn(
    env: Environment, start: np.ndarray, episodes: int, epsilon: tuple[float, float], seed: int
) -> QPolicy:
    """Q-learning from `start` for `episodes`, epsilon decaying geometrically over the run."""
    rng = np.random.default_rng(seed)
    table = start.copy()
    visits = np.zeros_like(table)
    first, last = epsilon
    decay = (last / first) ** (1.0 / episodes)
    eps = first
    for _ in range(episodes):
        state, outage = initial_state(rng, env)
        obs = observe(rng, state, env)
        for _ in range(STEPS):
            if rng.random() < eps:
                action = int(rng.integers(len(ACTIONS)))
            else:
                action = int(np.argmax(table[obs]))
            state, reward = step(rng, state, ACTIONS[action], env, outage)
            nxt = observe(rng, state, env)
            key = (*obs, action)
            visits[key] += 1
            rate = max(1.0 / (1.0 + visits[key]), STEP_FLOOR)
            table[key] += rate * (reward + GAMMA * table[nxt].max() - table[key])
            obs = nxt
        eps *= decay
    return QPolicy(table)


def fit_policy(env: Environment, live: QPolicy | None, seed: int) -> QPolicy:
    """First training from an empty table, or a warm-started retrain from the live policy."""
    if live is None:
        empty = np.zeros((BINS, BINS, BINS, len(ACTIONS)))
        return learn(env, empty, FIRST_EPISODES, FIRST_EPSILON, seed)
    return learn(env, live.table, RETRAIN_EPISODES, RETRAIN_EPSILON, seed)
