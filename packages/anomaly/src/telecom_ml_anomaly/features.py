"""Anomaly features, ported from the anomaly-detection repo (features.py and the notebook's
feature matrix): the seven hourly KPIs, four temporal features and five interaction
features, 16 in all. Cell and area type are not model inputs, as in the source."""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "label_anomaly"
KPIS = [
    "traffic_load_gb",
    "avg_sinr_db",
    "avg_throughput_mbps",
    "avg_latency_ms",
    "packet_loss_pct",
    "connected_users",
    "prb_utilization",
]


def features(data: pd.DataFrame) -> pd.DataFrame:
    """The 16 model inputs for each cell-hour."""
    timestamp = pd.to_datetime(data["timestamp"])
    hour = timestamp.dt.hour
    weekday = timestamp.dt.dayofweek
    traffic, users = data["traffic_load_gb"], data["connected_users"]
    sinr, throughput = data["avg_sinr_db"], data["avg_throughput_mbps"]
    prb, latency = data["prb_utilization"], data["avg_latency_ms"]
    extra = pd.DataFrame(
        {
            "hour": hour,
            "day_of_week": weekday,
            "is_weekend": (weekday >= 5).astype(int),
            "is_peak_hour": (((hour >= 9) & (hour <= 11)) | ((hour >= 18) & (hour <= 21))).astype(
                int
            ),
            "load_per_user": traffic / np.maximum(users, 1),
            "spectral_efficiency": throughput / np.maximum(sinr + 5, 0.1),
            "congestion_index": prb * (latency / 50),
            "sinr_throughput_ratio": sinr / np.maximum(throughput, 0.1),
            "utilization_load_gap": prb - traffic / 50,
        },
        index=data.index,
    )
    return pd.concat([data[KPIS].astype(float), extra.astype(float)], axis=1)
