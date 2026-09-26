"""The netopt use case bound to the core contract."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pandera.pandas as pa
from telecom_ml_core.contract import (
    Batch,
    DriftResult,
    Metrics,
    PromotionDecision,
    Scenario,
    UseCase,
)
from telecom_ml_core.evaluate import EPISODE_SEED

from telecom_ml_netopt import generator
from telecom_ml_netopt.generator import ACTIONS, STEPS, Environment, initial_state, observe, step
from telecom_ml_netopt.policy import QPolicy, RulePolicy, fit_policy, run_episode

SCENARIO_PATH = Path(__file__).with_name("scenario.yaml")
DRIFT_EPISODES = 50

# Retrain trigger beside dataset drift: mean episode reward falling below its value at
# promotion.
REWARD_DROP = 0.03
# Promotion margin (docs/generators.md rule 8): twice the largest reward gain of a
# scenario-off run that retrained every day for 180 days (2026-09-26): 0.0412.
REWARD_MARGIN = 0.0824

SCHEMA = pa.DataFrameSchema(
    {
        "day": pa.Column("datetime64[ns]"),
        "load_mult": pa.Column(float, pa.Check.in_range(0.5, 2.0)),
        "outage_share": pa.Column(float, pa.Check.in_range(0.0, 1.0)),
        "sinr_noise_sd": pa.Column(float, pa.Check.in_range(0.0, 10.0)),
        "label_released_on": pa.Column("datetime64[ns]"),
    },
    checks=[pa.Check(lambda df: len(df) == 1, error="one environment version per day")],
    strict=True,
)


def _seed(base_seed: int, day: pd.Timestamp, stream: int) -> int:
    """A training or drift seed derived from the environment's day; nothing reads the clock."""
    sequence = np.random.SeedSequence([base_seed, generator.USECASE_ID, day.toordinal(), stream])
    return int(sequence.generate_state(1)[0])


class NetOptUseCase(UseCase):
    name = "netopt"
    usecase_id = generator.USECASE_ID
    evaluation = "rollout"
    drift_test = "ks"
    max_label_delay_days = generator.LABEL_DELAY_DAYS
    train_window_days = 1
    eval_window_days = 1
    scenario = Scenario.load(SCENARIO_PATH)

    def generate(self, day: date, scenario: Scenario) -> Batch:
        return generator.generate(day, scenario)

    def schema(self) -> pa.DataFrameSchema:
        return SCHEMA

    def drift_frame(self, data: pd.DataFrame) -> pd.DataFrame:
        """One row per episode of a fixed rule-policy rollout on the day's environment: the
        mean observed SINR, interference and latency bins it visited."""
        row = data.iloc[0]
        env = Environment.from_row(row)
        rng = np.random.default_rng(_seed(self.scenario.base_seed, row["day"], 1))
        policy = RulePolicy()
        rows = []
        for _ in range(DRIFT_EPISODES):
            state, outage = initial_state(rng, env)
            seen = []
            for _ in range(STEPS):
                obs = observe(rng, state, env)
                seen.append(obs)
                state, _ = step(rng, state, ACTIONS[policy.act(obs)], env, outage)
            mean = np.mean(seen, axis=0)
            rows.append({"sinr_bin": mean[0], "interference_bin": mean[1], "latency_bin": mean[2]})
        return pd.DataFrame(rows)

    def fit(self, train: pd.DataFrame, warm_start: Any | None) -> Any:
        row = train.iloc[0]
        live = warm_start if isinstance(warm_start, QPolicy) else None
        return fit_policy(
            Environment.from_row(row), live, _seed(self.scenario.base_seed, row["day"], 0)
        )

    def fit_baseline(self, train: pd.DataFrame) -> Any:
        return RulePolicy()

    def score(self, model: Any, frame: pd.DataFrame) -> Metrics:
        """Mean total reward over the fixed episodes, one per row of the frame."""
        rewards = [
            run_episode(model, Environment.from_row(row), int(row[EPISODE_SEED]))
            for _, row in frame.iterrows()
        ]
        return {"reward": float(np.mean(rewards)), "reward_sd": float(np.std(rewards))}

    def should_retrain(self, drift: DriftResult, live: Metrics, at_promotion: Metrics) -> bool:
        """Dataset drift of the visited states, or reward falling below its value at promotion."""
        return drift.detected or live["reward"] < at_promotion["reward"] - REWARD_DROP

    def promote(self, candidate: Metrics, live: Metrics, baseline: Metrics) -> PromotionDecision:
        """Mean episode reward beats live by the margin on the same fixed episodes."""
        gain = candidate["reward"] - live["reward"]
        summary = f"reward {gain:+.4f} against live, rule-based {baseline['reward']:.4f}"
        if gain < REWARD_MARGIN:
            return PromotionDecision(False, f"gain below the margin {REWARD_MARGIN}: {summary}")
        return PromotionDecision(True, f"candidate wins by the margin: {summary}")
