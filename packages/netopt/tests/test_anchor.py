"""Day-0 anchor (docs/generators.md rule 3).

The earlier network-optimization repo's result (Q-agent 60.2% over random in a fresh run of
its notebook at 360a9fab) comes from a replay environment in which the next state never
depends on the action, so it is recorded, never anchored to. The anchor is the dynamics: per
action, the mean one-step reward and state changes from a fresh run of the earlier generator
(seed 42, 2,000 episodes of 50 random actions, 100,000 steps).

Here: the same process on this environment with every load coupling at its neutral value.
Tolerance: reward within 0.002; each state change within 10% or 0.05, whichever is larger.
"""

import numpy as np
import pandas as pd
import pytest
from telecom_ml_netopt.generator import (
    ACTIONS,
    LOAD_COUPLING,
    NO_COUPLING,
    Environment,
    initial_state,
    step,
)

# Per action: reward, then the mean change in sinr, throughput, latency, interference, load.
SOURCE = {
    "adjust_tilt": (0.0090, -0.0166, 5.2410, -1.3024, 0.0, 0.0),
    "decrease_power": (-0.0105, -1.4648, -4.0301, 0.9171, -0.0702, 0.0),
    "increase_power": (0.0174, 1.8291, 5.5023, -1.2078, 0.0700, 0.0),
    "load_balance": (0.0060, 0.0, 0.0, -6.0365, 0.0, -0.0185),
    "no_action": (-0.0002, -0.0049, -0.0952, 0.0238, 0.0, 0.0008),
}
KPIS = ("sinr", "throughput", "latency", "interference", "load")


def one_step_means(env: Environment) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for _ in range(2_000):
        state, outage = initial_state(rng, env)
        for _ in range(50):
            action = ACTIONS[int(rng.integers(len(ACTIONS)))]
            nxt, reward = step(rng, state, action, env, outage)
            rows.append(
                {"action": action, "reward": reward, **{k: nxt[k] - state[k] for k in KPIS}}
            )
            state = nxt
    return pd.DataFrame(rows).groupby("action").mean()


@pytest.fixture(scope="module")
def uncoupled() -> pd.DataFrame:
    return one_step_means(Environment(1.0, 0.0, 1.0, NO_COUPLING))


@pytest.mark.parametrize("action", ACTIONS)
def test_one_step_dynamics_match_the_source(uncoupled: pd.DataFrame, action: str) -> None:
    reward, *changes = SOURCE[action]
    assert uncoupled.loc[action, "reward"] == pytest.approx(reward, abs=0.002)
    for kpi, change in zip(KPIS, changes, strict=True):
        tolerance = max(0.1 * abs(change), 0.05)
        assert uncoupled.loc[action, kpi] == pytest.approx(change, abs=tolerance), kpi


def mean_step(action: str, load: float, coupling: object, kpi: str) -> float:
    """Mean one-step change of `kpi` from a fixed state far from every clip limit."""
    env = Environment(1.0, 0.0, 1.0, coupling)  # type: ignore[arg-type]
    state = {"load": load, "sinr": 10.0, "interference": 0.2, "throughput": 100.0, "latency": 100.0}
    rng = np.random.default_rng(3)
    return float(
        np.mean([step(rng, state, action, env, False)[0][kpi] - state[kpi] for _ in range(4_000)])
    )


@pytest.mark.parametrize("load", [0.5, 0.9])
def test_power_increase_interference_scales_with_load_squared(load: float) -> None:
    ratio = mean_step("increase_power", load, LOAD_COUPLING, "interference") / mean_step(
        "increase_power", load, NO_COUPLING, "interference"
    )
    assert ratio == pytest.approx((0.5 + load) ** 2, rel=0.05)


def test_load_balancing_cuts_latency_more_under_congestion() -> None:
    # Congestion adds 20 x (load - 0.5) ms a step in the coupled physics; take it out.
    def cut(load: float) -> float:
        return -(mean_step("load_balance", load, LOAD_COUPLING, "latency") - 20 * (load - 0.5))

    assert cut(0.9) / cut(0.55) == pytest.approx((0.9 / 0.55) ** 2, rel=0.05)
