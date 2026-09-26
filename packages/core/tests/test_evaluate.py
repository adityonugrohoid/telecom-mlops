from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pandera.pandas as pa
import pytest
from telecom_ml_core.cli import build_components, run
from telecom_ml_core.contract import (
    LABEL_RELEASE,
    Batch,
    DriftResult,
    Event,
    Metrics,
    PromotionDecision,
    Scenario,
    UseCase,
)
from telecom_ml_core.evaluate import EPISODE_SEED, HoldoutEvaluator, RolloutEvaluator
from telecom_ml_core.pipeline import DataSource, run_loop
from toy import START, TOY_SCENARIO, ToyUseCase

ARMS = 3
PULLS = 20


@dataclass
class Arm:
    arm: int
    seen: list[int] = field(default_factory=list)


class Bandit(UseCase):
    """Each day is an environment version: success probability per arm. Arm 0 wins at first;
    from day 3 arm 2 does."""

    name = "bandit"
    usecase_id = 98
    evaluation = "rollout"
    drift_test = "auto"
    max_label_delay_days = 0
    train_window_days = 1
    eval_window_days = 1
    scenario = Scenario(
        START, 10, 3, (Event("arm_swap", 3, 0, "dynamics", {"p0": 0.2, "p2": 0.9}, False, None),)
    )

    def generate(self, day: date, scenario: Scenario) -> Batch:
        row = {
            "p0": scenario.value("p0", 0.8, day),
            "p1": scenario.value("p1", 0.5, day),
            "p2": scenario.value("p2", 0.3, day),
            LABEL_RELEASE: pd.Timestamp(day),
        }
        return Batch(day, pd.DataFrame([row]), scenario.manifest(day))

    def schema(self) -> pa.DataFrameSchema:
        return pa.DataFrameSchema(
            {f"p{i}": pa.Column(float, pa.Check.in_range(0, 1)) for i in range(ARMS)}
        )

    def drift_frame(self, data: pd.DataFrame) -> pd.DataFrame:
        """Sampled pulls per arm, as netopt samples visited states: one environment row is too
        little for a drift test."""
        rng = np.random.default_rng(0)
        row = data.iloc[0]
        return pd.DataFrame({f"r{i}": rng.binomial(1, row[f"p{i}"], 200) for i in range(ARMS)})

    def fit(self, train: pd.DataFrame, warm_start: Any | None) -> Any:
        rng = np.random.default_rng(0)
        means = [rng.binomial(1, train[f"p{i}"].iloc[0], 400).mean() for i in range(ARMS)]
        return Arm(int(np.argmax(means)))

    def fit_baseline(self, train: pd.DataFrame) -> Any:
        return Arm(1)

    def score(self, model: Any, frame: pd.DataFrame) -> Metrics:
        rewards = []
        for _, row in frame.iterrows():
            seed = int(row[EPISODE_SEED])
            model.seen.append(seed)
            rng = np.random.default_rng(seed)
            rewards.append(rng.binomial(1, row[f"p{model.arm}"], PULLS).mean())
        return {"reward": float(np.mean(rewards))}

    def should_retrain(self, drift: DriftResult, live: Metrics, at_promotion: Metrics) -> bool:
        return drift.detected or live["reward"] < at_promotion["reward"] - 0.1

    def promote(self, candidate: Metrics, live: Metrics, baseline: Metrics) -> PromotionDecision:
        if candidate["reward"] > live["reward"] and candidate["reward"] > baseline["reward"]:
            return PromotionDecision(True, "candidate reward beats live and baseline")
        return PromotionDecision(False, "candidate reward does not beat live and baseline")


def released_on(frame: pd.DataFrame) -> set[date]:
    return {ts.date() for ts in pd.to_datetime(frame[LABEL_RELEASE])}


def test_holdout_trains_strictly_before_the_evaluation_window() -> None:
    usecase = ToyUseCase()
    source = DataSource(usecase, TOY_SCENARIO)
    day = START + timedelta(days=20)
    train = HoldoutEvaluator().training_data(usecase, day, source)
    eval_days = {day - timedelta(days=i) for i in range(usecase.eval_window_days)}
    train_days = {
        min(eval_days) - timedelta(days=i) for i in range(1, usecase.train_window_days + 1)
    }
    assert released_on(train) == train_days
    assert released_on(source.released(min(eval_days), day)) == eval_days


def test_holdout_scores_every_model_on_the_same_rows() -> None:
    usecase = ToyUseCase()
    source = DataSource(usecase, TOY_SCENARIO)
    day = START + timedelta(days=4)
    scores = HoldoutEvaluator().evaluate(
        usecase,
        {
            "a": usecase.fit_baseline(source.batch(day).data),
            "b": usecase.fit_baseline(source.batch(day).data),
        },
        day,
        source,
    )
    assert scores["a"] == scores["b"]


def test_holdout_refuses_an_empty_window() -> None:
    class Nothing(DataSource):
        def released(self, first: date, last: date) -> pd.DataFrame:
            return pd.DataFrame()

    with pytest.raises(ValueError, match="no labels released"):
        HoldoutEvaluator().evaluate(ToyUseCase(), {}, START, Nothing(ToyUseCase(), TOY_SCENARIO))


def test_rollout_runs_every_policy_on_the_same_fixed_episodes() -> None:
    usecase = Bandit()
    source = DataSource(usecase, usecase.scenario)
    evaluator = RolloutEvaluator(episodes=5, seed=100)
    live, candidate = Arm(0), Arm(2)
    evaluator.evaluate(usecase, {"live": live, "candidate": candidate}, START, source)
    evaluator.evaluate(usecase, {"live": live}, START + timedelta(days=1), source)
    assert candidate.seen == [100, 101, 102, 103, 104]
    assert live.seen == candidate.seen * 2


def test_rollout_trains_on_the_day_environment() -> None:
    usecase = Bandit()
    source = DataSource(usecase, usecase.scenario)
    day = START + timedelta(days=5)
    train = RolloutEvaluator(episodes=1, seed=0).training_data(usecase, day, source)
    pd.testing.assert_frame_equal(train, source.batch(day).data)


def test_rollout_refuses_a_clashing_column() -> None:
    usecase = Bandit()

    class Clash(DataSource):
        def batch(self, day: date) -> Batch:
            batch = super().batch(day)
            return Batch(day, batch.data.assign(**{EPISODE_SEED: 1}), batch.manifest)

    with pytest.raises(ValueError, match="already has"):
        RolloutEvaluator(episodes=1, seed=0).evaluate(
            usecase, {}, START, Clash(usecase, usecase.scenario)
        )


def test_policy_loop_promotes_after_the_dynamics_change(tmp_path: Path) -> None:
    decisions = run_loop(Bandit(), START, 6, build_components(tmp_path))
    assert not any(d.promoted for d in decisions[:3])
    swap = decisions[3]
    assert (swap.drift.detected, swap.promoted, swap.live_version) == (True, True, 2)


def test_tml_loop_runs_end_to_end_and_continues(tmp_path: Path) -> None:
    def usecases() -> dict[str, type[UseCase]]:
        return {"bandit": Bandit, "toy": ToyUseCase}

    state = str(tmp_path)
    assert (
        run(
            ["loop", "all", "--from", "2026-01-01", "--days", "5", "--state", state],
            usecases,
            build_components,
        )
        == 0
    )
    assert run(["loop", "toy", "--days", "2", "--state", state], usecases, build_components) == 0
    registry = build_components(tmp_path).registry
    assert registry.last_day("toy") == START + timedelta(days=6)
    assert registry.last_day("bandit") == START + timedelta(days=4)
