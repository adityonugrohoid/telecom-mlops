"""The QoE use case bound to the core contract."""

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

from telecom_ml_qoe import generator
from telecom_ml_qoe.features import CATEGORIES, NUMERIC, TARGET
from telecom_ml_qoe.model import fit_baseline, fit_model
from telecom_ml_qoe.schema import SCHEMA

SCENARIO_PATH = Path(__file__).with_name("scenario.yaml")
MONITORED = [*NUMERIC, *CATEGORIES]

# Retrain trigger beside dataset drift: MAE rising above its value at promotion.
MAE_RISE = 0.02
# Promotion margin (docs/generators.md rule 8): twice the largest MAE improvement of a
# scenario-off run that retrained every day for 180 days (2026-09-26): 0.0037.
MAE_MARGIN = 0.0074


class QoEUseCase(UseCase):
    name = "qoe"
    usecase_id = generator.USECASE_ID
    evaluation = "holdout"
    drift_test = "auto"
    max_label_delay_days = generator.LABEL_DELAY_DAYS
    train_window_days = 14
    eval_window_days = 7
    scenario = Scenario.load(SCENARIO_PATH)

    def generate(self, day: date, scenario: Scenario) -> Batch:
        return generator.generate(day, scenario)

    def schema(self) -> pa.DataFrameSchema:
        return SCHEMA

    def drift_frame(self, data: pd.DataFrame) -> pd.DataFrame:
        """One row per session: sessions are independent draws."""
        return data[MONITORED]

    def fit(self, train: pd.DataFrame, warm_start: Any | None) -> Any:
        return fit_model(train, seed=self.scenario.base_seed)

    def fit_baseline(self, train: pd.DataFrame) -> Any:
        return fit_baseline(train)

    def score(self, model: Any, frame: pd.DataFrame) -> Metrics:
        """MAE and RMSE of predicted MOS."""
        error = model.predict(frame) - frame[TARGET].to_numpy()
        return {
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error**2))),
        }

    def should_retrain(self, drift: DriftResult, live: Metrics, at_promotion: Metrics) -> bool:
        """Dataset drift, or MAE rising above its value at promotion."""
        return drift.detected or live["mae"] > at_promotion["mae"] + MAE_RISE

    def promote(self, candidate: Metrics, live: Metrics, baseline: Metrics) -> PromotionDecision:
        """MAE beats live by the margin."""
        gain = live["mae"] - candidate["mae"]
        if gain >= MAE_MARGIN:
            return PromotionDecision(True, f"mae {-gain:+.4f} against live")
        return PromotionDecision(False, f"mae {-gain:+.4f} against live, margin {MAE_MARGIN}")
