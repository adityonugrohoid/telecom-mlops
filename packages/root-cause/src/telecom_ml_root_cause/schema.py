"""The contract every root-cause batch must meet."""

from __future__ import annotations

import pandera.pandas as pa
from telecom_ml_core.contract import LABEL_RELEASE

from telecom_ml_root_cause.generator import EVENT_TYPES, EVENTS_PER_INCIDENT, N_CELLS, SEVERITIES

SCHEMA = pa.DataFrameSchema(
    {
        "event_id": pa.Column(str, unique=True),
        "incident_id": pa.Column(str),
        "event_sequence_position": pa.Column(int, pa.Check.in_range(0, EVENTS_PER_INCIDENT - 1)),
        "timestamp": pa.Column("datetime64[ns]"),
        "cell_id": pa.Column(int, pa.Check.in_range(1, N_CELLS)),
        "event_type": pa.Column(str, pa.Check.isin(EVENT_TYPES)),
        "alarm_severity": pa.Column(str, pa.Check.isin(SEVERITIES)),
        "time_lag_seconds": pa.Column(int, pa.Check.in_range(0, 300)),
        "is_root_cause": pa.Column(int, pa.Check.isin([0, 1])),
        "affected_cells": pa.Column(int, pa.Check.in_range(1, N_CELLS)),
        "sinr_delta": pa.Column(float),
        "throughput_delta": pa.Column(float),
        "latency_delta": pa.Column(float),
        "incidents_in_day": pa.Column(int, pa.Check.ge(1)),
        LABEL_RELEASE: pa.Column("datetime64[ns]"),
    },
    checks=[
        pa.Check(
            lambda df: (df.groupby("incident_id")["is_root_cause"].sum() == 1).all(),
            error="every incident has exactly one root cause",
        ),
        pa.Check(
            lambda df: (df.groupby("incident_id")["event_id"].count() == EVENTS_PER_INCIDENT).all(),
            error=f"every incident has {EVENTS_PER_INCIDENT} events",
        ),
    ],
    strict=True,
)
