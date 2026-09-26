"""The capacity use case bound to the core contract."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

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

from telecom_ml_capacity import generator
from telecom_ml_capacity.model import TARGET, SeasonalNaive, fit_forecaster, mape_pct
from telecom_ml_capacity.schema import SCHEMA

SCENARIO_PATH = Path(__file__).with_name("scenario.yaml")

# Retrain trigger beside dataset drift: MAPE rising above its value at promotion (points).
MAPE_RISE = 2.0
# Promotion margin (docs/generators.md rule 8): twice the largest MAPE improvement of a
# scenario-off run that retrained every day for 180 days (2026-09-26): 0.157 points.
MAPE_MARGIN = 0.315


class CapacityUseCase(UseCase):
    name = "capacity"
    usecase_id = generator.USECASE_ID
    evaluation = "holdout"
    # One row per cell per day is 60 rows a day; see anomaly for why KS suits small frames.
    drift_test = "ks"
    rule8_exception = (
        "A forecaster on same-hour lags tracks growth through its lags, so the growth events "
        "are detected and may retrain but a new model is not expected to win. The use case "
        "shows the other side: models retrained on the holiday week forecast worse after it, "
        "and the gate refuses them."
    )
    max_label_delay_days = generator.LABEL_DELAY_DAYS
    train_window_days = 28
    eval_window_days = 14
    scenario = Scenario.load(SCENARIO_PATH)

    def generate(self, day: date, scenario: Scenario) -> Batch:
        return generator.generate(day, scenario)

    def schema(self) -> pa.DataFrameSchema:
        return SCHEMA

    def drift_frame(self, data: pd.DataFrame) -> pd.DataFrame:
        """One row per cell per day: the day's traffic against the same weekday 7 and 14 days
        earlier. The ratios do not move with the weekly pattern, and every measured KPI here
        follows traffic, so raw daily means would drift together on any weekend."""
        day = pd.to_datetime(data["timestamp"]).dt.date
        columns = [TARGET, "lag_7d", "lag_14d"]
        means = data.assign(day=day).groupby(["cell_id", "day"])[columns].mean()
        return pd.DataFrame(
            {
                "ratio_7d": means[TARGET] / means["lag_7d"],
                "ratio_14d": means[TARGET] / means["lag_14d"],
            }
        ).reset_index(drop=True)

    def fit(self, train: pd.DataFrame, warm_start: Any | None) -> Any:
        return fit_forecaster(train, seed=self.scenario.base_seed)

    def fit_baseline(self, train: pd.DataFrame) -> Any:
        return SeasonalNaive()

    def score(self, model: Any, frame: pd.DataFrame) -> Metrics:
        """MAPE of the 7-day-ahead forecast, in percent."""
        return {"mape": mape_pct(frame[TARGET].to_numpy(), model.predict(frame))}

    def should_retrain(self, drift: DriftResult, live: Metrics, at_promotion: Metrics) -> bool:
        """Dataset drift, or MAPE rising above its value at promotion."""
        return drift.detected or live["mape"] > at_promotion["mape"] + MAPE_RISE

    def promote(self, candidate: Metrics, live: Metrics, baseline: Metrics) -> PromotionDecision:
        """MAPE beats live by the margin and beats the seasonal naive forecast."""
        gain = live["mape"] - candidate["mape"]
        summary = (
            f"mape {candidate['mape']:.2f} vs live {live['mape']:.2f}, naive {baseline['mape']:.2f}"
        )
        if candidate["mape"] >= baseline["mape"]:
            return PromotionDecision(False, f"does not beat the seasonal naive: {summary}")
        if gain < MAPE_MARGIN:
            return PromotionDecision(False, f"gain below the margin {MAPE_MARGIN}: {summary}")
        return PromotionDecision(True, f"candidate wins by the margin: {summary}")
