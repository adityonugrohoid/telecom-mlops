# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Netopt: a stylised cell-optimisation simulator and a tabular Q-learning policy
#
# **All environments here are simulated.** The simulator in `telecom_ml_netopt.generator` is
# stylised, not a network model: five actions move a cell's SINR, interference, throughput,
# latency and load by simple rules, and the reward is a weighted KPI improvement.
#
# The notebook shows:
#
# 1. an episode, and what the agent can see
# 2. the drift calendar
# 3. a learned policy against a rule-based policy and a random one, on the same episodes
# 4. what each event does to a policy that is not retrained

# %%
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from telecom_ml_netopt import generator
from telecom_ml_netopt.generator import ACTIONS, Environment
from telecom_ml_netopt.policy import QPolicy, RulePolicy, fit_policy, run_episode
from telecom_ml_netopt.usecase import NetOptUseCase

usecase = NetOptUseCase()
scenario = usecase.scenario
SEEDS = range(1000, 1050)


def env_on(index: int) -> Environment:
    day = scenario.start + timedelta(days=index)
    return Environment.from_row(generator.generate(day, scenario).data.iloc[0])


class RandomPolicy:
    def __init__(self) -> None:
        self.rng = np.random.default_rng(0)

    def act(self, observation) -> int:
        return int(self.rng.integers(len(ACTIONS)))


def rewards(policy, env: Environment) -> np.ndarray:
    return np.array([run_episode(policy, env, seed) for seed in SEEDS])


# %% [markdown]
# ## 1. An episode, and what the agent sees
#
# An episode is 50 steps on one cell from a random starting state. The agent observes SINR
# (with measurement noise), interference and latency in five bins each; the cell's load moves
# the dynamics underneath but is not observed. The earlier work's agent learned from a replay
# of recorded rows in which its actions changed nothing; every step here is simulated.

# %%
base = env_on(0)
print(base)
print("actions:", ACTIONS)

# %% [markdown]
# ## 2. The drift calendar
#
# Day 60: mean load rises 20% over 21 days. Day 120: a neighbour outage puts 10% of cells
# under extra interference and load. Day 160, the benign event: SINR measurement noise
# doubles, which blurs what the agent sees without changing the cells.

# %%
calendar = pd.DataFrame(
    [
        generator.params_on(scenario.start + timedelta(days=i), scenario)
        for i in range(scenario.days)
    ]
)
fig, axes = plt.subplots(1, 3, figsize=(13, 3), sharex=True)
for ax, (param, title) in zip(
    axes,
    [
        ("load_mult", "mean load multiplier"),
        ("outage_share", "share of cells in an outage"),
        ("sinr_noise_sd", "SINR measurement noise, dB (benign)"),
    ],
    strict=True,
):
    ax.plot(calendar.index, calendar[param])
    ax.set_title(title)
    ax.set_xlabel("simulated day")
fig.tight_layout()

# %% [markdown]
# ## 3. Learned, rule-based and random, on the same 50 episodes

# %%
learned = fit_policy(base, None, seed=1)
table = pd.DataFrame(
    {
        name: {"mean reward": r.mean(), "sd": r.std()}
        for name, r in (
            ("Q-learning", rewards(learned, base)),
            ("rule-based", rewards(RulePolicy(), base)),
            ("random", rewards(RandomPolicy(), base)),
        )
    }
)
table.round(3)

# %% [markdown]
# ## 4. A policy that is not retrained, against one that is
#
# The day-0 policy and a warm-started retrain on each event's environment, scored on the same
# 50 episodes. Retraining here does not reliably help, and sometimes hurts: the promotion gate
# is what keeps a worse retrain from going live.

# %%
rows = []
for name, index in (("load +20%", 90), ("outage 10%", 130), ("SINR noise x2 (benign)", 170)):
    env = env_on(index)
    retrained = fit_policy(env, learned, seed=2)
    rows.append(
        {
            "event": name,
            "not retrained": rewards(learned, env).mean(),
            "retrained": rewards(retrained, env).mean(),
        }
    )
pd.DataFrame(rows).set_index("event").round(3)


# %%
def best_action_share(policy: QPolicy) -> pd.Series:
    return pd.Series(np.argmax(policy.table, axis=-1).ravel()).map(dict(enumerate(ACTIONS)))


best_action_share(learned).value_counts(normalize=True).round(2)
