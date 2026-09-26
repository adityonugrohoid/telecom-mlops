"""The contract every capacity batch must meet."""

from __future__ import annotations

import pandera.pandas as pa
from telecom_ml_core.contract import LABEL_RELEASE

from telecom_ml_capacity.generator import AREA_TYPES, CELL_TYPES

SCHEMA = pa.DataFrameSchema(
    {
        "cell_id": pa.Column(str),
        "cell_type": pa.Column(str, pa.Check.isin(CELL_TYPES)),
        "area_type": pa.Column(str, pa.Check.isin(AREA_TYPES)),
        "timestamp": pa.Column("datetime64[ns]"),
        "traffic_load_gb": pa.Column(float, pa.Check.ge(0)),
        "connected_users": pa.Column(int, pa.Check.ge(10)),
        "prb_utilization": pa.Column(float, pa.Check.in_range(0.1, 0.95)),
        "avg_throughput_mbps": pa.Column(float, pa.Check.in_range(1.0, 100.0)),
        "avg_latency_ms": pa.Column(float, pa.Check.in_range(10, 300)),
        "avg_sinr_db": pa.Column(float, pa.Check.in_range(-5, 25)),
        "lag_7d": pa.Column(float, pa.Check.ge(0)),
        "lag_8d": pa.Column(float, pa.Check.ge(0)),
        "lag_14d": pa.Column(float, pa.Check.ge(0)),
        "mean_same_hour_7_13d": pa.Column(float, pa.Check.ge(0)),
        LABEL_RELEASE: pa.Column("datetime64[ns]"),
    },
    checks=[
        pa.Check(
            lambda df: (df.groupby("cell_id")["timestamp"].count() == 24).all(),
            error="every cell has 24 hourly rows",
        ),
    ],
    strict=True,
)
