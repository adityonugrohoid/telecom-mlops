"""The anomaly use case bound to the core contract."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pandera.pandas as pa
from sklearn.metrics import f1_score, precision_score, recall_score
from telecom_ml_core.contract import (
    Batch,
    DriftResult,
    Metrics,
    PromotionDecision,
    Scenario,
    UseCase,
)

from telecom_ml_anomaly import generator
from telecom_ml_anomaly.features import KPIS, TARGET
from telecom_ml_anomaly.model import TriageDetector, fit_detector, fit_forest
from telecom_ml_anomaly.schema import SCHEMA

SCENARIO_PATH = Path(__file__).with_name("scenario.yaml")

# Retrain triggers beside dataset drift of the cell-day frame.
F1_DROP = 0.05
# Alerts on audited normal hours, absolute rise over the value at promotion.
ALERT_RISE = 0.005
# Promotion margins (docs/generators.md rule 8), from a scenario-off run at a 20% audit that
# retrained every day for 180 days (2026-09-26): twice the largest F1 gain (+0.0215), and
# twice the largest rise in alerts on normal hours from noise alone (+0.0019).
F1_MARGIN = 0.043
ALERT_GUARD = 0.0038


class AnomalyUseCase(UseCase):
    name = "anomaly"
    usecase_id = generator.USECASE_ID
    evaluation = "holdout"
    # One row per cell per day is 50 rows a day; Wasserstein at a fixed 0.1 flags that much
    # sampling noise, the KS test accounts for it.
    drift_test = "ks"
    rule8_exception = None
    max_label_delay_days = generator.LABEL_DELAY_DAYS
    train_window_days = 30
    eval_window_days = 14
    scenario = Scenario.load(SCENARIO_PATH)

    def generate(self, day: date, scenario: Scenario) -> Batch:
        return generator.generate(day, scenario)

    def schema(self) -> pa.DataFrameSchema:
        return SCHEMA

    def drift_frame(self, data: pd.DataFrame) -> pd.DataFrame:
        """One row per cell per day, the mean of each KPI: hours of one cell and day share
        its load and conditions, so they are not independent samples."""
        day = pd.to_datetime(data["timestamp"]).dt.date
        return data.assign(day=day).groupby(["cell_id", "day"])[KPIS].mean().reset_index(drop=True)

    def fit(self, train: pd.DataFrame, warm_start: Any | None) -> Any:
        live = warm_start if isinstance(warm_start, TriageDetector) else None
        return fit_detector(train, live=live, seed=self.scenario.base_seed)

    def fit_baseline(self, train: pd.DataFrame) -> Any:
        return fit_forest(train, seed=self.scenario.base_seed)

    def score(self, model: Any, frame: pd.DataFrame) -> Metrics:
        """F1, precision and recall on the random audit sample, the one set of hours labelled
        regardless of what any model alerted on; alerts on audited normal hours; and recall on
        audited outages of the new type."""
        audited = frame[frame["audited"].to_numpy(bool)]
        labels = audited[TARGET].to_numpy()
        alerts = model.alerts(audited)
        new_type = (audited["anomaly_type"] == generator.NEW_TYPE).fillna(False).to_numpy(bool)
        return {
            "f1": float(f1_score(labels, alerts, zero_division=0)),
            "precision": float(precision_score(labels, alerts, zero_division=0)),
            "recall": float(recall_score(labels, alerts, zero_division=0)),
            "normal_alert_rate": float(alerts[labels == 0].mean()),
            "new_type_recall": float(alerts[new_type].mean()) if new_type.any() else 0.0,
            "audited_hours": float(len(audited)),
        }

    def should_retrain(self, drift: DriftResult, live: Metrics, at_promotion: Metrics) -> bool:
        """Dataset drift, F1 falling, or alerts on normal hours rising."""
        return (
            drift.detected
            or live["f1"] < at_promotion["f1"] - F1_DROP
            or live["normal_alert_rate"] > at_promotion["normal_alert_rate"] + ALERT_RISE
        )

    def promote(self, candidate: Metrics, live: Metrics, baseline: Metrics) -> PromotionDecision:
        """F1 beats live by the margin, and alerts on normal hours rise by no more than noise."""
        gain = candidate["f1"] - live["f1"]
        summary = (
            f"f1 {gain:+.4f}, normal alerts {live['normal_alert_rate']:.4f} to "
            f"{candidate['normal_alert_rate']:.4f}"
        )
        if candidate["normal_alert_rate"] > live["normal_alert_rate"] + ALERT_GUARD:
            return PromotionDecision(False, f"alerts on normal hours rise: {summary}")
        if gain < F1_MARGIN:
            return PromotionDecision(False, f"gain below the margin {F1_MARGIN}: {summary}")
        return PromotionDecision(True, f"candidate wins by the margin: {summary}")
