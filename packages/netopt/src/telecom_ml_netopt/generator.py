"""The netopt environment: a stylised cell-optimisation simulator, one version per simulated
day. Not a network model.

Each day's batch is one row: the parameters of that day's environment version. Episodes
(50 steps on one cell) run against it: an agent observes the cell's radio measurements, picks
one of five actions, and the cell's state moves by the action's physics.

Ported from the network-optimization repo (data_generator.py, NetworkOptDataGenerator): the
random initial state, the five actions' effects on SINR, interference, throughput, latency and
load, and the reward (weighted KPI improvement). The source trained its agent on a replay of
pre-recorded rows, where the next state never depended on the action; here every step is
simulated.

Changes from the source, on purpose, each with its physical reason (set once, not tuned):
- Raising power in a loaded network raises interference at neighbours more: a power
  increase adds interference x (0.5 + load)^2, since both the cell's and its neighbours'
  activity grow with load.
- Load balancing matters more under congestion: its latency cut scales x (load / 0.55)^2
  (0.55 is the mean initial load), since queueing delay grows faster than load.
- Congestion builds latency: every step adds 20 x (load - 0.5) ms.
- After a neighbour outage the cell carries extra interference and load (+0.3 each) and
  moving load off it pays more (load balancing's cut x1.5).
- The agent observes SINR, interference and latency, not load: the controller acts on the
  radio measurements it has, and load moves the dynamics underneath them.
- Draws that exist only for scenario events (whether an episode's cell is in an outage) come
  from a child stream of the episode's seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from telecom_ml_core.contract import LABEL_RELEASE, Batch, Scenario

USECASE_ID = 6
LABEL_DELAY_DAYS = 0
STEPS = 50
ACTIONS = ("increase_power", "decrease_power", "adjust_tilt", "load_balance", "no_action")
MEAN_INITIAL_LOAD = 0.55

# Base value of every parameter a scenario event can move (names match scenario.yaml).
BASE_PARAMS: dict[str, float] = {
    "load_mult": 1.0,
    "outage_share": 0.0,
    "sinr_noise_sd": 1.0,
}


@dataclass(frozen=True)
class Coupling:
    """The load couplings. `LOAD_COUPLING` is the environment's physics; `NO_COUPLING` puts
    every term at its neutral value and reproduces the source generator's one-step dynamics."""

    interference_exponent: float
    balance_exponent: float
    congestion_ms: float
    outage_boost: float
    outage_balance_mult: float


LOAD_COUPLING = Coupling(2.0, 2.0, 20.0, 0.3, 1.5)
NO_COUPLING = Coupling(0.0, 0.0, 0.0, 0.0, 1.0)


@dataclass(frozen=True)
class Environment:
    """One day's environment version."""

    load_mult: float
    outage_share: float
    sinr_noise_sd: float
    coupling: Coupling

    @classmethod
    def from_row(cls, row: pd.Series) -> Environment:
        return cls(
            load_mult=float(row["load_mult"]),
            outage_share=float(row["outage_share"]),
            sinr_noise_sd=float(row["sinr_noise_sd"]),
            coupling=LOAD_COUPLING,
        )


State = dict[str, float]


def initial_state(rng: np.random.Generator, env: Environment) -> tuple[State, bool]:
    """The source's random initial state, with the day's load level and any outage."""
    state = {
        "load": rng.uniform(0.1, 1.0) * env.load_mult,
        "sinr": rng.uniform(-5, 25),
        "interference": rng.uniform(0, 1),
        "throughput": rng.uniform(1, 200),
        "latency": rng.uniform(10, 200),
    }
    event_rng = rng.spawn(1)[0]
    outage = bool(event_rng.random() < env.outage_share)
    if outage:
        state["interference"] += env.coupling.outage_boost
        state["load"] += env.coupling.outage_boost
    state["load"] = float(np.clip(state["load"], 0.1, 1.0))
    state["interference"] = float(np.clip(state["interference"], 0, 1))
    return state, outage


def step(
    rng: np.random.Generator, state: State, action: str, env: Environment, outage: bool
) -> tuple[State, float]:
    """Apply one action: the source's physics plus the load couplings. Returns the next state
    and the source's reward."""
    c = env.coupling
    load = state["load"]
    nxt = dict(state)
    if action == "increase_power":
        gain = rng.uniform(1, 3)
        nxt["sinr"] += gain
        nxt["interference"] += rng.uniform(0.05, 0.1) * (0.5 + load) ** c.interference_exponent
        nxt["throughput"] *= 1 + gain / 25
        nxt["latency"] *= 1 - gain / 100
    elif action == "decrease_power":
        loss = rng.uniform(1, 2)
        nxt["sinr"] -= loss
        nxt["interference"] -= rng.uniform(0.05, 0.1)
        nxt["throughput"] *= 1 - loss / 50
        nxt["latency"] *= 1 + loss / 100
    elif action == "adjust_tilt":
        nxt["sinr"] += rng.uniform(-1, 1)
        factor = 1 + rng.uniform(0.05, 0.10)
        nxt["throughput"] *= factor
        nxt["latency"] *= 1 - 0.02 * factor
    elif action == "load_balance":
        nxt["load"] += rng.uniform(-0.15, 0.10)
        cut = rng.uniform(0.05, 0.15) * (load / MEAN_INITIAL_LOAD) ** c.balance_exponent
        if outage:
            cut *= c.outage_balance_mult
        nxt["latency"] *= 1 - min(cut, 0.6)
    else:  # no_action
        nxt["sinr"] += rng.normal(0, 0.3)
        nxt["throughput"] += rng.normal(0, 1.0)
        nxt["latency"] += rng.normal(0, 1.0)
        nxt["load"] += rng.normal(0, 0.02)
    nxt["latency"] += c.congestion_ms * (load - 0.5)
    nxt["load"] = float(np.clip(nxt["load"], 0.1, 1.0))
    nxt["sinr"] = float(np.clip(nxt["sinr"], -5, 25))
    nxt["interference"] = float(np.clip(nxt["interference"], 0, 1))
    nxt["throughput"] = float(np.clip(nxt["throughput"], 1, 200))
    nxt["latency"] = float(np.clip(nxt["latency"], 10, 200))
    reward = (
        0.3 * (nxt["sinr"] - state["sinr"]) / 25
        + 0.3 * (nxt["throughput"] - state["throughput"]) / 200
        - 0.2 * (nxt["latency"] - state["latency"]) / 200
        - 0.2 * (nxt["interference"] - state["interference"])
    )
    return nxt, float(reward)


OBSERVED = ("sinr", "interference", "latency")
EDGES = {
    "sinr": np.linspace(-5, 25, 6),
    "interference": np.linspace(0, 1, 6),
    "latency": np.linspace(10, 200, 6),
}
BINS = 5


def observe(rng: np.random.Generator, state: State, env: Environment) -> tuple[int, int, int]:
    """What the agent sees: SINR (with measurement noise), interference and latency, in five
    bins each. Load is not observed."""
    sinr = state["sinr"] + rng.normal(0, env.sinr_noise_sd)
    values = {"sinr": sinr, "interference": state["interference"], "latency": state["latency"]}
    a, b, c = (int(np.clip(np.digitize(values[k], EDGES[k]) - 1, 0, BINS - 1)) for k in OBSERVED)
    return a, b, c


def params_on(day: date, scenario: Scenario) -> dict[str, float]:
    """Every environment parameter on `day` under `scenario`."""
    return {name: scenario.value(name, base, day) for name, base in BASE_PARAMS.items()}


def generate(day: date, scenario: Scenario) -> Batch:
    """The environment version for `day`: one row of parameters. Policies are judged by
    rollouts on it, so its 'label' is released the same day."""
    data = pd.DataFrame([{"day": pd.Timestamp(day), **params_on(day, scenario)}])
    data[LABEL_RELEASE] = pd.Timestamp(day)
    return Batch(day=day, data=data, manifest=scenario.manifest(day))
