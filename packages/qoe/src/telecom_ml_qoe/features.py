"""QoE features, ported from the qoe-prediction repo (features.py): the seven session
measurements, four temporal features, the source's five interaction features and one-hot
network, device and app. Categories are fixed, so columns never depend on a day's mix."""

from __future__ import annotations

import pandas as pd

from telecom_ml_qoe.generator import APPS, DEVICES, NETWORKS

TARGET = "mos_score"
NUMERIC = [
    "sinr_db",
    "throughput_mbps",
    "latency_ms",
    "packet_loss_pct",
    "congestion_level",
    "session_duration_min",
    "data_volume_mb",
]
CATEGORIES: dict[str, tuple[str, ...]] = {
    "network_type": NETWORKS,
    "device_class": DEVICES,
    "app_type": APPS,
}
APP_SENSITIVITY = {
    "gaming": 0.9,
    "voip": 0.85,
    "video_streaming": 0.8,
    "browsing": 0.3,
    "social": 0.4,
    "cloud_gaming": 0.95,
}


def features(data: pd.DataFrame) -> pd.DataFrame:
    """Every model input for each session."""
    timestamp = pd.to_datetime(data["timestamp"])
    hour = timestamp.dt.hour
    weekday = timestamp.dt.dayofweek
    one_hot = {
        f"{column}_{category}": (data[column] == category).astype(int)
        for column, categories in CATEGORIES.items()
        for category in sorted(categories)[1:]
    }
    extra = pd.DataFrame(
        {
            "hour": hour,
            "day_of_week": weekday,
            "is_weekend": (weekday >= 5).astype(int),
            "is_peak_hour": (((hour >= 9) & (hour <= 11)) | ((hour >= 18) & (hour <= 21))).astype(
                int
            ),
            "network_quality_index": 0.5 * (data["sinr_db"] / 25)
            + 0.5 * (data["throughput_mbps"] / 100),
            "service_degradation": data["latency_ms"] / 100 + data["packet_loss_pct"] * 2,
            "throughput_per_user_mbps": data["throughput_mbps"],
            "bandwidth_utilization": data["data_volume_mb"]
            / (data["throughput_mbps"] * data["session_duration_min"] * 60 / 8 + 0.01),
            "app_sensitivity_score": data["app_type"].map(APP_SENSITIVITY),
            **one_hot,
        },
        index=data.index,
    )
    return pd.concat([data[NUMERIC].astype(float), extra.astype(float)], axis=1)
