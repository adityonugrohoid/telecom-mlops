"""Stage 3: compare the day's monitored inputs with the live model's reference, using Evidently.

Evidently picks a test per column from its type and sample size. For p-value tests a column
drifts when the value falls below the threshold; for distance tests, when it reaches it.
"""

from __future__ import annotations

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from telecom_ml_core.contract import DriftResult


class EvidentlyDrift:
    """Dataset drift: detected when the share of drifted columns reaches `drift_share`."""

    def __init__(self, drift_share: float) -> None:
        if not 0.0 < drift_share <= 1.0:
            raise ValueError(f"drift_share must be in (0, 1], got {drift_share}")
        self.drift_share = drift_share

    def check(self, reference: pd.DataFrame, current: pd.DataFrame) -> DriftResult:
        """Test every column of `current` against `reference`.

        Args:
            reference: The monitored inputs the live model trained on.
            current: The same columns for the day.

        Returns:
            Whether the dataset drifted, the drifted share and each column's test value.
        """
        if list(reference.columns) != list(current.columns):
            raise ValueError(
                f"drift columns differ: reference {list(reference.columns)}, "
                f"current {list(current.columns)}"
            )
        report = Report([DataDriftPreset(drift_share=self.drift_share)]).run(current, reference)
        scores: dict[str, float] = {}
        drifted: list[str] = []
        for metric in report.dict()["metrics"]:
            config = metric["config"]
            if not config["type"].endswith(":ValueDrift"):
                continue
            column, value = config["column"], float(metric["value"])
            scores[column] = value
            is_p_value = "p_value" in config["method"]
            if (value < config["threshold"]) if is_p_value else (value >= config["threshold"]):
                drifted.append(column)
        missing = set(reference.columns) - set(scores)
        if missing:
            raise RuntimeError(f"Evidently returned no drift test for columns {sorted(missing)}")
        share = len(drifted) / len(scores)
        return DriftResult(
            detected=share >= self.drift_share,
            share_drifted=share,
            per_feature=scores,
            drifted=tuple(drifted),
        )
