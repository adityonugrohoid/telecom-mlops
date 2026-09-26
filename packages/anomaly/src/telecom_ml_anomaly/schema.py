"""The contract every anomaly batch must meet."""

from __future__ import annotations

import pandera.pandas as pa
from telecom_ml_core.contract import LABEL_RELEASE

from telecom_ml_anomaly.generator import ANOMALY_TYPES, AREA_TYPES, CELL_TYPES

SCHEMA = pa.DataFrameSchema(
    {
        "cell_id": pa.Column(str),
        "cell_type": pa.Column(str, pa.Check.isin(CELL_TYPES)),
        "area_type": pa.Column(str, pa.Check.isin(AREA_TYPES)),
        "timestamp": pa.Column("datetime64[ns]"),
        "traffic_load_gb": pa.Column(float, pa.Check.in_range(0.5, 50.0)),
        "avg_sinr_db": pa.Column(float, pa.Check.in_range(-5, 25)),
        "avg_throughput_mbps": pa.Column(float, pa.Check.gt(0)),
        "avg_latency_ms": pa.Column(float, pa.Check.in_range(10, 300)),
        "packet_loss_pct": pa.Column(float, pa.Check.in_range(0, 5)),
        "connected_users": pa.Column(int, pa.Check.in_range(50, 500)),
        "prb_utilization": pa.Column(float, pa.Check.in_range(0.1, 0.95)),
        "label_anomaly": pa.Column(int, pa.Check.isin([0, 1])),
        "anomaly_type": pa.Column("string", pa.Check.isin(ANOMALY_TYPES), nullable=True),
        "audited": pa.Column(bool),
        LABEL_RELEASE: pa.Column("datetime64[ns]"),
    },
    checks=[
        pa.Check(
            lambda df: (df["label_anomaly"] == 1).eq(df["anomaly_type"].notna()).all(),
            error="anomaly_type is set exactly on labelled anomalies",
        ),
        pa.Check(
            lambda df: (df.groupby("cell_id")["timestamp"].count() == 24).all(),
            error="every cell has 24 hourly rows",
        ),
    ],
    strict=True,
)
