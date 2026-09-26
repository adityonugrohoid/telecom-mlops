"""The root-cause generator: one simulated day of network incidents, each a cascade of alarms
from a single root event.

Ported from the root-cause-analysis repo (data_generator.py, RCADataGenerator). The cascade
model is kept event by event: a root event with large KPI impact, usually but not always
first, followed by cascading alarms whose severity and KPI impact decay with distance.

Changes from the source, on purpose:
- Seeded by date (base seed, use case id, day); nothing reads the clock. Incidents start
  inside the batch day, where the source spread them over the 90 days before `now()`.
- Incidents per day are drawn from Poisson(`incident_rate`), so volume is a scenario input.
- Scenario parameters for the drift calendar: `config_error_late_root_share` (the share of
  config_error incidents whose root is no longer the first alarm) and `power_supply_share`
  (a root cause class the source did not have, with its own signature).
- Each incident's label (the ticket closing on its root cause) is released 1 to 7 days later.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
from telecom_ml_core.contract import LABEL_RELEASE, Batch, Scenario, rng_for_day

USECASE_ID = 2
EVENTS_PER_INCIDENT = 20
N_CELLS = 50
MIN_LABEL_DELAY_DAYS = 1
MAX_LABEL_DELAY_DAYS = 7

SOURCE_TYPES = ("hardware_failure", "software_bug", "config_error", "overload", "external")
NEW_TYPE = "power_supply"
EVENT_TYPES = (*SOURCE_TYPES, NEW_TYPE)
SEVERITIES = ("critical", "major", "minor", "warning")
ROOT_SEVERITY = {
    "hardware_failure": "critical",
    "software_bug": "critical",
    "config_error": "major",
    "overload": "critical",
    "external": "major",
    NEW_TYPE: "major",
}

# Base value of every parameter a scenario event can move (names match scenario.yaml).
BASE_PARAMS: dict[str, float] = {
    "incident_rate": 30.0,
    "config_error_late_root_share": 0.0,
    "power_supply_share": 0.0,
}

# The power_supply signature: the root alarm arrives after the first symptoms, with small
# KPI impact (a power dip) spread over more cells.
POWER_ROOT_POSITIONS = (2, 3, 4)
POWER_KPI_SCALE = 0.3
POWER_AFFECTED_CELLS = (3, 8)


def params_on(day: date, scenario: Scenario) -> dict[str, float]:
    """Every generator parameter on `day` under `scenario`."""
    return {name: scenario.value(name, base, day) for name, base in BASE_PARAMS.items()}


def _cascade_severity(rng: np.random.Generator, root_severity: str, position: int) -> str:
    idx = SEVERITIES.index(root_severity)
    drop = min(position // 4, len(SEVERITIES) - 1 - idx)
    jitter = rng.choice([-1, 0, 0, 1], p=[0.1, 0.4, 0.3, 0.2])
    return SEVERITIES[int(np.clip(idx + drop + jitter, 0, len(SEVERITIES) - 1))]


def _incident(
    rng: np.random.Generator, incident_id: str, start: pd.Timestamp, p: dict[str, float]
) -> list[dict[str, Any]]:
    """All alarm events of one incident, root and cascade."""
    if rng.random() < p["power_supply_share"]:
        root_type = NEW_TYPE
    else:
        root_type = str(rng.choice(SOURCE_TYPES))
    root_severity = ROOT_SEVERITY[root_type]
    primary_cell = int(rng.integers(1, N_CELLS + 1))

    # ~20% of the time the monitoring system files the root one severity lower.
    if rng.random() < 0.20:
        idx = SEVERITIES.index(root_severity)
        root_severity = SEVERITIES[min(idx + 1, len(SEVERITIES) - 1)]

    scale = POWER_KPI_SCALE if root_type == NEW_TYPE else 1.0
    root_sinr = float(rng.uniform(-15, -5)) * scale
    root_throughput = float(rng.uniform(-80, -30)) * scale
    root_latency = float(rng.uniform(30, 150)) * scale

    root_lag = min(int(rng.exponential(scale=3)), 15)
    root_position = int(rng.choice([0, 0, 0, 0, 0, 0, 0, 1, 1, 2]))
    late_config = root_type == "config_error" and rng.random() < p["config_error_late_root_share"]
    if root_type == NEW_TYPE or late_config:
        root_position = int(rng.choice(POWER_ROOT_POSITIONS))
        root_lag = int(rng.integers(5, 16))
    if root_type == NEW_TYPE:
        root_cells = int(rng.integers(POWER_AFFECTED_CELLS[0], POWER_AFFECTED_CELLS[1] + 1))
    else:
        root_cells = 1

    events: list[dict[str, Any]] = []
    for position in range(EVENTS_PER_INCIDENT):
        if position == root_position:
            events.append(
                {
                    "incident_id": incident_id,
                    "event_sequence_position": position,
                    "timestamp": start + pd.Timedelta(seconds=root_lag),
                    "cell_id": primary_cell,
                    "event_type": root_type,
                    "alarm_severity": root_severity,
                    "time_lag_seconds": root_lag,
                    "is_root_cause": 1,
                    "affected_cells": root_cells,
                    "sinr_delta": round(root_sinr + float(rng.normal(0, 1.5)), 2),
                    "throughput_delta": round(root_throughput + float(rng.normal(0, 5)), 2),
                    "latency_delta": round(root_latency + float(rng.normal(0, 8)), 2),
                }
            )
            continue

        distance = abs(position - root_position)
        if position < root_position:
            # a symptom seen before the root is identified
            lag = int(np.clip(rng.exponential(scale=5), 0, root_lag))
        else:
            lag = int(np.clip(rng.exponential(scale=30) + distance * 5, root_lag + 1, 300))
        cascade_type = str(rng.choice(SOURCE_TYPES))
        severity = _cascade_severity(rng, root_severity, distance)
        if distance <= 2:
            overlap = 0.7 + 0.3 * rng.random()
            noise = 1.5
        else:
            overlap = float(np.exp(-0.15 * distance))
            noise = 1.0
        # Cascades take their impact from the incident, not from the power dip.
        impact = 1.0 / scale
        sinr = round(root_sinr * impact * overlap + float(rng.normal(0, 2 * noise)), 2)
        throughput = round(root_throughput * impact * overlap + float(rng.normal(0, 6 * noise)), 2)
        latency = round(root_latency * impact * overlap + float(rng.normal(0, 10 * noise)), 2)
        affected = int(np.clip(1 + rng.poisson(distance * 0.5), 1, N_CELLS))
        cell = int(rng.integers(1, N_CELLS + 1)) if rng.random() < 0.3 else primary_cell
        events.append(
            {
                "incident_id": incident_id,
                "event_sequence_position": position,
                "timestamp": start + pd.Timedelta(seconds=lag),
                "cell_id": cell,
                "event_type": cascade_type,
                "alarm_severity": severity,
                "time_lag_seconds": lag,
                "is_root_cause": 0,
                "affected_cells": affected,
                "sinr_delta": sinr,
                "throughput_delta": throughput,
                "latency_delta": latency,
            }
        )
    return events


def generate(day: date, scenario: Scenario) -> Batch:
    """The incidents that start on `day`; each one's label is released 1 to 7 days later."""
    rng = rng_for_day(scenario.base_seed, USECASE_ID, day)
    p = params_on(day, scenario)
    n_incidents = max(1, int(rng.poisson(p["incident_rate"])))
    midnight = pd.Timestamp(day)
    rows: list[dict[str, Any]] = []
    for k in range(n_incidents):
        incident_id = f"I{day:%Y%m%d}-{k:03d}"
        start = midnight + pd.Timedelta(seconds=int(rng.integers(0, 86_400 - 300)))
        delay = int(rng.integers(MIN_LABEL_DELAY_DAYS, MAX_LABEL_DELAY_DAYS + 1))
        for event in _incident(rng, incident_id, start, p):
            event[LABEL_RELEASE] = pd.Timestamp(day + timedelta(days=delay))
            rows.append(event)
    data = pd.DataFrame(rows)
    data.insert(0, "event_id", [f"E{day:%Y%m%d}-{i:05d}" for i in range(len(data))])
    data["incidents_in_day"] = n_incidents
    return Batch(day=day, data=data, manifest=scenario.manifest(day))
