"""The contract every churn batch must meet."""

from __future__ import annotations

import pandera.pandas as pa
from telecom_ml_core.contract import LABEL_RELEASE

from telecom_ml_churn.generator import CONTRACTS, DEVICES, NETWORKS, PAYMENTS

SCHEMA = pa.DataFrameSchema(
    {
        "customer_id": pa.Column(str, unique=True),
        "timestamp": pa.Column("datetime64[ns]"),
        "tenure_months": pa.Column(int, pa.Check.in_range(1, 72)),
        "contract_type": pa.Column(str, pa.Check.isin(CONTRACTS)),
        "payment_method": pa.Column(str, pa.Check.isin(PAYMENTS)),
        "monthly_charges": pa.Column(float, pa.Check.in_range(0, 200)),
        "network_type": pa.Column(str, pa.Check.isin(NETWORKS)),
        "device_class": pa.Column(str, pa.Check.isin(DEVICES)),
        "avg_sinr_db": pa.Column(float, pa.Check.in_range(-5, 25)),
        "avg_throughput_mbps": pa.Column(float, pa.Check.in_range(0.1, 300)),
        "avg_latency_ms": pa.Column(float, pa.Check.in_range(10, 300)),
        "avg_packet_loss_pct": pa.Column(float, pa.Check.in_range(0, 10)),
        "avg_qoe_mos": pa.Column(float, pa.Check.in_range(1, 5)),
        "total_tickets": pa.Column(int, pa.Check.ge(0)),
        "total_sessions": pa.Column(int, pa.Check.in_range(50, 500)),
        "is_churned": pa.Column(int, pa.Check.isin([0, 1])),
        LABEL_RELEASE: pa.Column("datetime64[ns]"),
    },
    strict=True,
)
