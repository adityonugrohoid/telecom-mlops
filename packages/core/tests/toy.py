"""A toy use case and in-memory stage fakes, used to exercise the core without a real use case.

Label: y = 1 when x crosses a threshold `t`, with 5% label noise. `z` is an input the label
ignores. The calendar moves `t` on day 5 (concept, hurts the live model) and shifts `z` on
day 20 (benign: detectable, harmless).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
import pandera.pandas as pa
from telecom_ml_core.contract import (
    LABEL_RELEASE,
    Batch,
    DayDecision,
    DriftResult,
    Event,
    LiveRecord,
    Metrics,
    PromotionDecision,
    Scenario,
    UseCase,
    rng_for_day,
)

START = date(2026, 1, 1)
ROWS_PER_DAY = 400
LABEL_DELAY_DAYS = 2

TOY_SCENARIO = Scenario(
    start=START,
    days=30,
    base_seed=7,
    events=(
        Event("threshold_moves", 5, 0, "concept", {"t": 1.5}, False, None),
        Event("z_shift", 20, 0, "covariate", {"z_mean": 3.0}, True, None),
    ),
)


@dataclass(frozen=True)
class Threshold:
    cut: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (x > self.cut).astype(int)


class ToyUseCase(UseCase):
    name = "toy"
    usecase_id = 99
    evaluation = "holdout"
    drift_test = "auto"
    max_label_delay_days = LABEL_DELAY_DAYS
    train_window_days = 5
    eval_window_days = 3
    scenario = TOY_SCENARIO

    def generate(self, day: date, scenario: Scenario) -> Batch:
        rng = rng_for_day(scenario.base_seed, self.usecase_id, day)
        t = scenario.value("t", 0.0, day)
        z_mean = scenario.value("z_mean", 0.0, day)
        x = rng.normal(0.0, 1.0, ROWS_PER_DAY)
        z = rng.normal(z_mean, 1.0, ROWS_PER_DAY)
        flip = rng.random(ROWS_PER_DAY) < 0.05
        y = (x > t).astype(int) ^ flip.astype(int)
        data = pd.DataFrame(
            {
                "x": x,
                "z": z,
                "y": y,
                LABEL_RELEASE: pd.Timestamp(day + timedelta(days=LABEL_DELAY_DAYS)),
            }
        )
        return Batch(day=day, data=data, manifest=scenario.manifest(day))

    def schema(self) -> pa.DataFrameSchema:
        return pa.DataFrameSchema(
            {
                "x": pa.Column(float),
                "z": pa.Column(float),
                "y": pa.Column(int, pa.Check.isin([0, 1])),
                LABEL_RELEASE: pa.Column("datetime64[ns]"),
            }
        )

    def drift_frame(self, data: pd.DataFrame) -> pd.DataFrame:
        return data[["x", "z"]]

    def fit(self, train: pd.DataFrame, warm_start: Any | None) -> Any:
        cuts = np.linspace(-3.0, 3.0, 121)
        x, y = train["x"].to_numpy(), train["y"].to_numpy()
        accuracy = [((x > c).astype(int) == y).mean() for c in cuts]
        return Threshold(float(cuts[int(np.argmax(accuracy))]))

    def fit_baseline(self, train: pd.DataFrame) -> Any:
        return Threshold(float("inf") if train["y"].mean() < 0.5 else float("-inf"))

    def score(self, model: Any, frame: pd.DataFrame) -> Metrics:
        predicted = model.predict(frame["x"].to_numpy())
        return {"accuracy": float((predicted == frame["y"].to_numpy()).mean())}

    def should_retrain(self, drift: DriftResult, live: Metrics, at_promotion: Metrics) -> bool:
        return drift.detected or live["accuracy"] < at_promotion["accuracy"] - 0.05

    def promote(self, candidate: Metrics, live: Metrics, baseline: Metrics) -> PromotionDecision:
        if candidate["accuracy"] <= live["accuracy"] + 0.01:
            return PromotionDecision(False, "candidate does not beat live by 0.01 accuracy")
        if candidate["accuracy"] <= baseline["accuracy"]:
            return PromotionDecision(False, "candidate does not beat the baseline")
        return PromotionDecision(True, "candidate beats live and baseline")


class SchemaValidator:
    def validate(self, usecase: UseCase, batch: Batch) -> None:
        usecase.schema().validate(batch.data)


class MeanShiftDrift:
    """Drift when any feature's mean moves more than half a reference standard deviation."""

    def check(
        self, reference: pd.DataFrame, current: pd.DataFrame, numeric_test: str
    ) -> DriftResult:
        shift = ((current.mean() - reference.mean()).abs() / reference.std()).to_dict()
        drifted = [name for name, value in shift.items() if value > 0.5]
        return DriftResult(bool(drifted), len(drifted) / len(shift), shift, tuple(drifted))


class MemoryRegistry:
    def __init__(self) -> None:
        self.records: dict[str, LiveRecord] = {}
        self.decisions: list[DayDecision] = []

    def live(self, usecase: str) -> LiveRecord | None:
        return self.records.get(usecase)

    def promote(self, usecase: str, record: LiveRecord) -> None:
        self.records[usecase] = record

    def log_decision(self, decision: DayDecision) -> None:
        self.decisions.append(decision)

    def last_day(self, usecase: str) -> date | None:
        days = [d.day for d in self.decisions if d.usecase == usecase]
        return max(days) if days else None
