"""The churn use case bound to the core contract."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pandera.pandas as pa
from sklearn.metrics import brier_score_loss, roc_auc_score
from telecom_ml_core.contract import (
    Batch,
    DriftResult,
    Metrics,
    PromotionDecision,
    Scenario,
    UseCase,
)

from telecom_ml_churn import generator
from telecom_ml_churn.baseline import fit_baseline
from telecom_ml_churn.features import CATEGORIES, NUMERIC, TARGET
from telecom_ml_churn.model import fit_model
from telecom_ml_churn.schema import SCHEMA

SCENARIO_PATH = Path(__file__).with_name("scenario.yaml")
MONITORED = NUMERIC + list(CATEGORIES)

# Retrain triggers, beside any drifted input column.
AUROC_DROP = 0.02
CHURN_RATE_SHIFT = 0.03
# Promotion margins (docs/generators.md rule 8): at least twice the largest noise gain of a
# scenario-off run that retrained every day for 180 days (2026-09-26): AUROC +0.0057,
# Brier -0.0012.
AUROC_MARGIN = 0.012
BRIER_MARGIN = 0.0025


class ChurnUseCase(UseCase):
    name = "churn"
    usecase_id = generator.USECASE_ID
    evaluation = "holdout"
    max_label_delay_days = generator.LABEL_DELAY_DAYS
    train_window_days = 30
    eval_window_days = 14
    scenario = Scenario.load(SCENARIO_PATH)

    def generate(self, day: date, scenario: Scenario) -> Batch:
        return generator.generate(day, scenario)

    def schema(self) -> pa.DataFrameSchema:
        return SCHEMA

    def drift_frame(self, data: pd.DataFrame) -> pd.DataFrame:
        return data[MONITORED]

    def fit(self, train: pd.DataFrame, warm_start: Any | None) -> Any:
        return fit_model(train, seed=self.scenario.base_seed)

    def fit_baseline(self, train: pd.DataFrame) -> Any:
        return fit_baseline(train)

    def score(self, model: Any, frame: pd.DataFrame) -> Metrics:
        """AUROC and Brier score, plus the observed and predicted churn rate."""
        proba = model.predict_proba(frame)
        labels = frame[TARGET].to_numpy()
        return {
            "auroc": float(roc_auc_score(labels, proba)),
            "brier": float(brier_score_loss(labels, proba)),
            "churn_rate": float(labels.mean()),
            "predicted_rate": float(np.mean(proba)),
        }

    def should_retrain(self, drift: DriftResult, live: Metrics, at_promotion: Metrics) -> bool:
        """Any monitored input drifted, AUROC fell, or the churn rate moved."""
        return (
            bool(drift.drifted)
            or live["auroc"] < at_promotion["auroc"] - AUROC_DROP
            or abs(live["churn_rate"] - at_promotion["churn_rate"]) > CHURN_RATE_SHIFT
        )

    def promote(self, candidate: Metrics, live: Metrics, baseline: Metrics) -> PromotionDecision:
        """AUROC and Brier both no worse than live, and one better by its margin."""
        auroc_gain = candidate["auroc"] - live["auroc"]
        brier_gain = live["brier"] - candidate["brier"]
        summary = f"auroc {auroc_gain:+.4f}, brier {-brier_gain:+.4f} against live"
        if auroc_gain < 0 or brier_gain < 0:
            return PromotionDecision(False, f"candidate worse on one metric: {summary}")
        if auroc_gain >= AUROC_MARGIN or brier_gain >= BRIER_MARGIN:
            return PromotionDecision(True, f"candidate wins by a margin: {summary}")
        return PromotionDecision(False, f"candidate gain below the margins: {summary}")
