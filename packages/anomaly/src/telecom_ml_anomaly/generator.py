"""The anomaly generator: one simulated day of hourly KPIs for 50 cells, with labelled
anomalies injected into 5% of cell-hours.

Ported from the anomaly-detection repo (data_generator.py, AnomalyDataGenerator). Per-cell
diurnal profiles, the KPI physics and the four anomaly signatures are kept.

Changes from the source, on purpose:
- Seeded by date (base seed, use case id, day); nothing reads the clock. Cell profiles are
  static: drawn once from the base seed, so the same 50 cells appear every day.
- Scenario parameters for the drift calendar: `grown_share` (the share of cells where new
  demand raises traffic, users and latency: the normal baseline moving toward congestion),
  `outage_rate` (a fifth anomaly type, intermittent_outage, the source did not have) and
  `weekend_factor` (the weekend traffic dip, smoothed by the benign event).
- Labels exist only where someone looked. Each hour carries `audited`, a random 10% audit
  sample; the use case labels an hour only if it was audited or a detector alerted on it.
- The radio helpers are copied here, not shared, so another use case's physics cannot move
  these results.
- Each day's triage labels are released one day later.
"""

from __future__ import annotations

from datetime import date, timedelta
from functools import cache
from typing import Any

import numpy as np
import pandas as pd
from telecom_ml_core.contract import LABEL_RELEASE, Batch, Scenario, rng_for_day

USECASE_ID = 3
N_CELLS = 50
HOURS = 24
LABEL_DELAY_DAYS = 1
ANOMALY_RATE = 0.05
AUDIT_SHARE = 0.20

SOURCE_TYPES = ("traffic_spike", "sinr_drop", "latency_surge", "throughput_collapse")
NEW_TYPE = "intermittent_outage"
ANOMALY_TYPES = (*SOURCE_TYPES, NEW_TYPE)
CELL_TYPES = ("macro", "micro", "small")
AREA_TYPES = ("urban", "suburban", "rural")

BASE_TRAFFIC = {"macro": 25.0, "micro": 12.0, "small": 5.0}
AREA_FACTOR = {"urban": 1.3, "suburban": 1.0, "rural": 0.7}
BASE_SINR = {"urban": 8.0, "suburban": 12.0, "rural": 14.0}
BASE_USERS = {"macro": 350, "micro": 200, "small": 80}

# Base value of every parameter a scenario event can move (names match scenario.yaml).
BASE_PARAMS: dict[str, float] = {
    "grown_share": 0.0,
    "growth_demand_mult": 1.8,
    "growth_latency_mult": 1.8,
    "outage_rate": 0.0,
    "weekend_factor": 0.8,
}


def params_on(day: date, scenario: Scenario) -> dict[str, float]:
    """Every generator parameter on `day` under `scenario`."""
    return {name: scenario.value(name, base, day) for name, base in BASE_PARAMS.items()}


@cache
def cell_profiles(base_seed: int) -> pd.DataFrame:
    """The 50 cells: type, area, and a fixed rank deciding which cells new demand reaches."""
    rng = np.random.default_rng([base_seed, USECASE_ID, 0])
    return pd.DataFrame(
        {
            "cell_id": [f"CELL_{i:04d}" for i in range(N_CELLS)],
            "cell_type": rng.choice(CELL_TYPES, N_CELLS, p=[0.4, 0.35, 0.25]),
            "area_type": rng.choice(AREA_TYPES, N_CELLS, p=[0.5, 0.3, 0.2]),
            "growth_rank": rng.permutation(N_CELLS) / N_CELLS,
        }
    )


# -- radio helpers, copied from the source's TelecomDataGenerator ------------------------


def _congestion(
    rng: np.random.Generator, timestamps: pd.DatetimeIndex, weekend_factor: float
) -> np.ndarray:
    hour = timestamps.hour.to_numpy()
    weekday = timestamps.dayofweek.to_numpy()
    congestion = 0.5 + 0.3 * np.sin((hour - 6) * np.pi / 12)
    peak = ((hour >= 9) & (hour <= 11)) | ((hour >= 18) & (hour <= 21))
    congestion = np.where(peak, congestion * 1.3, congestion)
    congestion = np.where(weekday >= 5, congestion * weekend_factor, congestion)
    return np.asarray(np.clip(congestion + rng.normal(0, 0.1, len(congestion)), 0, 1))


def _sinr_to_throughput(
    rng: np.random.Generator, sinr_db: np.ndarray, network: np.ndarray
) -> np.ndarray:
    capacity = np.log2(1 + 10 ** (sinr_db / 10))
    max_throughput = np.where(network == "5G", 300, 50)
    throughput = capacity * max_throughput / 5 * rng.normal(1, 0.2, len(sinr_db))
    return np.asarray(np.clip(throughput, 0.1, max_throughput))


def _latency(rng: np.random.Generator, congestion: np.ndarray) -> np.ndarray:
    latency = 20 * (1 + 5 * congestion**2) + rng.normal(0, 5, len(congestion))
    return np.asarray(np.clip(latency, 10, 300))


# -- one day -----------------------------------------------------------------------------


def _cell_day(
    rng: np.random.Generator,
    cell: Any,
    timestamps: pd.DatetimeIndex,
    p: dict[str, float],
) -> pd.DataFrame:
    n = len(timestamps)
    congestion = _congestion(rng, timestamps, p["weekend_factor"])
    traffic = (
        BASE_TRAFFIC[cell.cell_type]
        * AREA_FACTOR[cell.area_type]
        * (0.3 + 0.7 * congestion)
        * rng.normal(1, 0.15, n)
    )
    traffic = np.clip(traffic, 0.5, 50.0)
    sinr = np.clip(rng.normal(BASE_SINR[cell.area_type], 4.0, n), -5, 25)
    network = np.where(rng.random(n) < 0.4, "5G", "4G")
    throughput = _sinr_to_throughput(rng, sinr, network)
    latency = _latency(rng, congestion)
    loss = np.clip(rng.exponential(0.3, n), 0, 5)
    users = BASE_USERS[cell.cell_type] * (0.3 + 0.7 * congestion) * rng.normal(1, 0.1, n)
    if cell.growth_rank < p["grown_share"]:
        traffic = np.clip(traffic * p["growth_demand_mult"], 0.5, 50.0)
        users = users * p["growth_demand_mult"]
        latency = np.clip(latency * p["growth_latency_mult"], 10, 300)
    users = np.clip(users, 50, 500).astype(int)
    prb = np.clip(0.2 + 0.6 * (traffic / 50.0) + rng.normal(0, 0.05, n), 0.1, 0.95)
    return pd.DataFrame(
        {
            "cell_id": cell.cell_id,
            "cell_type": cell.cell_type,
            "area_type": cell.area_type,
            "timestamp": timestamps,
            "traffic_load_gb": traffic,
            "avg_sinr_db": sinr,
            "avg_throughput_mbps": throughput,
            "avg_latency_ms": latency,
            "packet_loss_pct": loss,
            "connected_users": users,
            "prb_utilization": prb,
        }
    )


def _inject(rng: np.random.Generator, data: pd.DataFrame, p: dict[str, float]) -> pd.DataFrame:
    """Corrupt 5% of cell-hours with the source's four signatures, plus intermittent outages
    at `outage_rate`. Each signature moves several KPIs at once, as real faults do."""
    n = len(data)
    label = np.zeros(n, dtype=int)
    kind = np.full(n, "", dtype=object)
    traffic = data["traffic_load_gb"].to_numpy(copy=True)
    sinr = data["avg_sinr_db"].to_numpy(copy=True)
    throughput = data["avg_throughput_mbps"].to_numpy(copy=True)
    latency = data["avg_latency_ms"].to_numpy(copy=True)
    loss = data["packet_loss_pct"].to_numpy(copy=True)
    users = data["connected_users"].to_numpy(copy=True)
    prb = data["prb_utilization"].to_numpy(copy=True)

    n_source = int(n * ANOMALY_RATE)
    chosen = rng.choice(n, size=n_source, replace=False)
    types = rng.choice(SOURCE_TYPES, size=n_source)
    for idx, atype in zip(chosen, types, strict=True):
        label[idx], kind[idx] = 1, atype
        if atype == "traffic_spike":
            traffic[idx] = min(traffic[idx] * rng.uniform(4, 8), 50.0)
            latency[idx] = min(latency[idx] * rng.uniform(1.5, 2.0), 300.0)
            sinr[idx] = max(sinr[idx] - rng.uniform(3, 5), -5.0)
            users[idx] = min(int(users[idx] * rng.uniform(1.5, 2.0)), 500)
        elif atype == "sinr_drop":
            sinr[idx] = max(sinr[idx] - rng.uniform(12, 18), -5.0)
            throughput[idx] = max(throughput[idx] * rng.uniform(0.3, 0.5), 0.1)
            latency[idx] = min(latency[idx] * rng.uniform(1.3, 1.6), 300.0)
            loss[idx] = min(loss[idx] + rng.uniform(1, 3), 5.0)
        elif atype == "latency_surge":
            latency[idx] = min(latency[idx] * rng.uniform(5, 8), 300.0)
            throughput[idx] = max(throughput[idx] * rng.uniform(0.5, 0.7), 0.1)
            loss[idx] = min(loss[idx] + rng.uniform(1, 2), 5.0)
            prb[idx] = min(prb[idx] + rng.uniform(0.20, 0.40), 0.95)
        else:  # throughput_collapse
            throughput[idx] = max(throughput[idx] / rng.uniform(8, 15), 0.1)
            latency[idx] = min(latency[idx] * rng.uniform(1.5, 2.0), 300.0)
            sinr[idx] = max(sinr[idx] - rng.uniform(5, 8), -5.0)
            loss[idx] = min(loss[idx] + rng.uniform(2, 4), 5.0)

    # Intermittent outages: short drops in hours the source signatures left clean. Users and
    # traffic fall away while radio conditions look normal, so no single KPI is extreme.
    clean = np.flatnonzero(label == 0)
    n_outage = round(n * p["outage_rate"])
    for idx in rng.choice(clean, size=min(n_outage, len(clean)), replace=False):
        label[idx], kind[idx] = 1, NEW_TYPE
        drop = rng.uniform(0.35, 0.6)
        traffic[idx] = max(traffic[idx] * drop, 0.5)
        users[idx] = max(int(users[idx] * drop), 50)
        throughput[idx] = max(throughput[idx] * rng.uniform(0.5, 0.8), 0.1)
        loss[idx] = min(loss[idx] + rng.uniform(0.5, 1.5), 5.0)

    return data.assign(
        traffic_load_gb=traffic,
        avg_sinr_db=sinr,
        avg_throughput_mbps=throughput,
        avg_latency_ms=latency,
        packet_loss_pct=loss,
        connected_users=users,
        prb_utilization=prb,
        label_anomaly=label,
        anomaly_type=pd.Series(kind, dtype="string").replace("", pd.NA),
    )


def generate(day: date, scenario: Scenario) -> Batch:
    """The 50 cells' 24 hourly KPI rows for `day`; labels are released the next day."""
    rng = rng_for_day(scenario.base_seed, USECASE_ID, day)
    p = params_on(day, scenario)
    timestamps = pd.date_range(start=pd.Timestamp(day), periods=HOURS, freq="h")
    cells = cell_profiles(scenario.base_seed)
    frames = [_cell_day(rng, cell, timestamps, p) for cell in cells.itertuples(index=False)]
    data = _inject(rng, pd.concat(frames, ignore_index=True), p)
    data["audited"] = rng.random(len(data)) < AUDIT_SHARE
    data[LABEL_RELEASE] = pd.Timestamp(day + timedelta(days=LABEL_DELAY_DAYS))
    return Batch(day=day, data=data, manifest=scenario.manifest(day))
