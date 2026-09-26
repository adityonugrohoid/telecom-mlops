"""Root-cause features, ported from the root-cause-analysis repo (features.py): the raw event
columns, one-hot event type and severity, the source's temporal features and its six
interaction features. Categories are fixed, so `power_supply` has a column from day 0."""

from __future__ import annotations

import numpy as np
import pandas as pd

from telecom_ml_root_cause.generator import EVENT_TYPES, SEVERITIES

TARGET = "is_root_cause"
INCIDENT = "incident_id"
NUMERIC = [
    "event_sequence_position",
    "time_lag_seconds",
    "affected_cells",
    "sinr_delta",
    "throughput_delta",
    "latency_delta",
]
CATEGORIES: dict[str, tuple[str, ...]] = {
    "event_type": EVENT_TYPES,
    "alarm_severity": SEVERITIES,
}
SEVERITY_SCORE = {"critical": 4, "major": 3, "minor": 2, "warning": 1}


def _one_hot(data: pd.DataFrame) -> pd.DataFrame:
    """One-hot with the first category dropped, as the source's get_dummies(drop_first)."""
    columns = {}
    for column, categories in CATEGORIES.items():
        for category in sorted(categories)[1:]:
            columns[f"{column}_{category}"] = (data[column] == category).astype(int)
    return pd.DataFrame(columns, index=data.index)


def engineered_features(data: pd.DataFrame) -> pd.DataFrame:
    """Every model input for each alarm event."""
    timestamp = pd.to_datetime(data["timestamp"])
    hour = timestamp.dt.hour
    weekday = timestamp.dt.dayofweek
    by_incident = data.groupby(INCIDENT)
    max_lag = by_incident["time_lag_seconds"].transform("max").replace(0, np.nan)
    max_position = by_incident["event_sequence_position"].transform("max").replace(0, np.nan)
    extra = pd.DataFrame(
        {
            "hour": hour,
            "day_of_week": weekday,
            "is_weekend": (weekday >= 5).astype(int),
            "is_peak_hour": (((hour >= 9) & (hour <= 11)) | ((hour >= 18) & (hour <= 21))).astype(
                int
            ),
            "severity_encoded": data["alarm_severity"].map(SEVERITY_SCORE),
            "is_first_event": (data["event_sequence_position"] == 0).astype(int),
            "normalized_time_lag": (data["time_lag_seconds"] / max_lag).fillna(0.0),
            "kpi_impact_score": data["sinr_delta"].abs()
            + data["throughput_delta"].abs() / 10
            + data["latency_delta"].abs() / 50,
            "cascade_depth_ratio": (data["event_sequence_position"] / max_position).fillna(0.0),
            "multi_cell_indicator": (data["affected_cells"] > 1).astype(int),
        },
        index=data.index,
    )
    return pd.concat([data[NUMERIC].astype(float), _one_hot(data), extra.astype(float)], axis=1)
