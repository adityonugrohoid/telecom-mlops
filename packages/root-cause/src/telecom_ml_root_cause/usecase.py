"""The root-cause use case bound to the core contract."""

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

from telecom_ml_root_cause import generator
from telecom_ml_root_cause.baseline import FirstAlarm
from telecom_ml_root_cause.features import INCIDENT, TARGET
from telecom_ml_root_cause.model import RootCauseModel, fit_model
from telecom_ml_root_cause.schema import SCHEMA

SCENARIO_PATH = Path(__file__).with_name("scenario.yaml")
# Drift is checked per incident, not per event: the 20 events of an incident share one root
# impact, so they are not independent samples. One row per incident summarises it.
# Retrain triggers beside dataset drift and any event type the live model never saw.
TOP1_DROP = 0.05
# Incident volume is a daily count, not a per-row distribution: it is watched as incidents
# per day over the evaluation window.
VOLUME_SHIFT = 0.2
# Promotion margin (docs/generators.md rule 8): twice the largest top-1 gain of a
# scenario-off run that retrained every day for 180 days (2026-09-26): +0.0213.
TOP1_MARGIN = 0.043


def top_k(scores: np.ndarray, frame: pd.DataFrame, k: int) -> float:
    """Share of incidents whose root cause is among their k highest-scored events."""
    ranked = pd.DataFrame(
        {"incident": frame[INCIDENT].to_numpy(), "root": frame[TARGET].to_numpy(), "s": scores}
    )
    ranked["rank"] = ranked.groupby("incident")["s"].rank(ascending=False, method="first")
    return float(ranked.loc[ranked["root"] == 1, "rank"].le(k).mean())


def mean_reciprocal_rank(scores: np.ndarray, frame: pd.DataFrame) -> float:
    ranked = pd.DataFrame(
        {"incident": frame[INCIDENT].to_numpy(), "root": frame[TARGET].to_numpy(), "s": scores}
    )
    ranked["rank"] = ranked.groupby("incident")["s"].rank(ascending=False, method="first")
    return float((1.0 / ranked.loc[ranked["root"] == 1, "rank"]).mean())


class RootCauseUseCase(UseCase):
    name = "root-cause"
    usecase_id = generator.USECASE_ID
    evaluation = "holdout"
    drift_test = "auto"
    max_label_delay_days = generator.MAX_LABEL_DELAY_DAYS
    train_window_days = 30
    eval_window_days = 21
    scenario = Scenario.load(SCENARIO_PATH)

    def generate(self, day: date, scenario: Scenario) -> Batch:
        return generator.generate(day, scenario)

    def schema(self) -> pa.DataFrameSchema:
        return SCHEMA

    def drift_frame(self, data: pd.DataFrame) -> pd.DataFrame:
        """One row per incident: its strongest KPI impact, spread, timing and first alarm."""
        ordered = data.sort_values([INCIDENT, "event_sequence_position"])
        by_incident = ordered.groupby(INCIDENT, sort=False)
        return pd.DataFrame(
            {
                "max_sinr_drop": by_incident["sinr_delta"].min().abs(),
                "max_throughput_drop": by_incident["throughput_delta"].min().abs(),
                "max_latency_rise": by_incident["latency_delta"].max(),
                "max_affected_cells": by_incident["affected_cells"].max(),
                "mean_time_lag_seconds": by_incident["time_lag_seconds"].mean(),
                "first_alarm_type": by_incident["event_type"].first(),
                "first_alarm_severity": by_incident["alarm_severity"].first(),
            }
        ).reset_index(drop=True)

    def fit(self, train: pd.DataFrame, warm_start: Any | None) -> Any:
        return fit_model(train, seed=self.scenario.base_seed)

    def fit_baseline(self, train: pd.DataFrame) -> Any:
        return FirstAlarm()

    def score(self, model: Any, frame: pd.DataFrame) -> Metrics:
        """Top-1 and top-3 hit rate, MRR, the share of events of a type the model never saw,
        and the mean incidents per day of the scored incidents."""
        scores = model.root_score(frame)
        if isinstance(model, RootCauseModel):
            unknown = float((~frame["event_type"].isin(model.seen_types)).mean())
        else:
            unknown = 0.0
        return {
            "top1": top_k(scores, frame, 1),
            "top3": top_k(scores, frame, 3),
            "mrr": mean_reciprocal_rank(scores, frame),
            "unknown_signature_share": unknown,
            "incidents_per_day": float(frame.groupby(INCIDENT)["incidents_in_day"].first().mean()),
        }

    def should_retrain(self, drift: DriftResult, live: Metrics, at_promotion: Metrics) -> bool:
        """Dataset drift, an unseen event type, top-1 falling, or incident volume moving."""
        volume_ratio = live["incidents_per_day"] / at_promotion["incidents_per_day"]
        return (
            drift.detected
            or live["unknown_signature_share"] > 0.0
            or live["top1"] < at_promotion["top1"] - TOP1_DROP
            or abs(volume_ratio - 1.0) > VOLUME_SHIFT
        )

    def promote(self, candidate: Metrics, live: Metrics, baseline: Metrics) -> PromotionDecision:
        """Top-1 hit rate beats live by the margin."""
        gain = candidate["top1"] - live["top1"]
        if gain >= TOP1_MARGIN:
            return PromotionDecision(True, f"top1 {gain:+.4f} against live")
        return PromotionDecision(False, f"top1 {gain:+.4f} against live, margin {TOP1_MARGIN}")
