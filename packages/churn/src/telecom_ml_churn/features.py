"""Churn features, ported from the churn-prediction repo (features.py).

`raw_features` is what the logistic baseline sees (17 columns in the source evidence);
`engineered_features` adds the source's temporal and interaction features (27 columns).
Categories are fixed here, so one-hot columns never depend on which values a day holds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from telecom_ml_churn.generator import CONTRACTS, DEVICES, NETWORKS, PAYMENTS

TARGET = "is_churned"
NUMERIC = [
    "tenure_months",
    "monthly_charges",
    "avg_sinr_db",
    "avg_throughput_mbps",
    "avg_latency_ms",
    "avg_packet_loss_pct",
    "avg_qoe_mos",
    "total_tickets",
    "total_sessions",
]
CATEGORIES: dict[str, tuple[str, ...]] = {
    "network_type": NETWORKS,
    "device_class": DEVICES,
    "contract_type": CONTRACTS,
    "payment_method": PAYMENTS,
}


def _one_hot(data: pd.DataFrame) -> pd.DataFrame:
    """One-hot with the first category dropped, as the source's get_dummies(drop_first)."""
    columns = {}
    for column, categories in CATEGORIES.items():
        for category in sorted(categories)[1:]:
            columns[f"{column}_{category}"] = (data[column] == category).astype(int)
    return pd.DataFrame(columns, index=data.index)


def raw_features(data: pd.DataFrame) -> pd.DataFrame:
    """Generator columns only: numeric inputs plus one-hot categories."""
    return pd.concat([data[NUMERIC].astype(float), _one_hot(data)], axis=1)


def engineered_features(data: pd.DataFrame) -> pd.DataFrame:
    """Raw features plus the source's temporal and interaction features."""
    timestamp = pd.to_datetime(data["timestamp"])
    hour = timestamp.dt.hour
    weekday = timestamp.dt.dayofweek
    tenure = data["tenure_months"]
    sessions = data["total_sessions"]
    extra = pd.DataFrame(
        {
            "hour": hour,
            "day_of_week": weekday,
            "is_weekend": (weekday >= 5).astype(int),
            "is_peak_hour": (((hour >= 9) & (hour <= 11)) | ((hour >= 18) & (hour <= 21))).astype(
                int
            ),
            "qoe_trend": data["avg_qoe_mos"] * np.where(tenure > 12, 1, -1),
            "network_quality_index": 0.5 * (data["avg_sinr_db"] / 25)
            + 0.5 * (data["avg_throughput_mbps"] / 100),
            "service_degradation": data["avg_latency_ms"] / 100 + data["avg_packet_loss_pct"] * 2,
            "ticket_rate": data["total_tickets"] / np.maximum(tenure, 1),
            "session_frequency": sessions / 30,
            "charges_per_session": data["monthly_charges"] / np.maximum(sessions, 1),
        },
        index=data.index,
    )
    return pd.concat([raw_features(data), extra.astype(float)], axis=1)
