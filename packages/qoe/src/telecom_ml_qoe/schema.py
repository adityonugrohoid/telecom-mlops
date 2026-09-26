"""The contract every QoE batch must meet."""

from __future__ import annotations

import pandera.pandas as pa
from telecom_ml_core.contract import LABEL_RELEASE

from telecom_ml_qoe.generator import APPS, DEVICES, NETWORKS

SCHEMA = pa.DataFrameSchema(
    {
        "session_id": pa.Column(str, unique=True),
        "timestamp": pa.Column("datetime64[ns]"),
        "network_type": pa.Column(str, pa.Check.isin(NETWORKS)),
        "device_class": pa.Column(str, pa.Check.isin(DEVICES)),
        "app_type": pa.Column(str, pa.Check.isin(APPS)),
        "sinr_db": pa.Column(float, pa.Check.in_range(-5, 25)),
        "throughput_mbps": pa.Column(float, pa.Check.in_range(0.1, 300)),
        "latency_ms": pa.Column(float, pa.Check.in_range(10, 300)),
        "packet_loss_pct": pa.Column(float, pa.Check.in_range(0, 5)),
        "congestion_level": pa.Column(float, pa.Check.in_range(0, 1)),
        "session_duration_min": pa.Column(float, pa.Check.in_range(1, 120)),
        "data_volume_mb": pa.Column(float, pa.Check.ge(0)),
        "mos_score": pa.Column(float, pa.Check.in_range(1, 5)),
        LABEL_RELEASE: pa.Column("datetime64[ns]"),
    },
    strict=True,
)
