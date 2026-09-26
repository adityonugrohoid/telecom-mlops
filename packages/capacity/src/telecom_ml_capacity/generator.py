"""The capacity generator: one simulated day of hourly traffic for 60 cells, with the history a
7-day-ahead forecast may use.

Ported from the capacity-forecasting repo (data_generator.py, CapacityDataGenerator). The
diurnal x weekly traffic shape, per-hour noise, short AR noise, special-event bursts, rare
outages and the KPIs derived from congestion are kept.

Changes from the source, on purpose:
- Seeded by date (base seed, use case id, day); nothing reads the clock. Cell profiles are
  static, drawn once from the base seed.
- The source's hourly random-walk drift has no bound in time and would sit at its clip limits
  over a 180-day calendar. Each cell instead carries a slow daily level (AR(1) with 0.97 per
  day, stationary standard deviation 0.04), close to the source's drift over its 30 days.
  Its shocks come from a child stream of each day's seed.
- Growth is scenario data: the 2% per month trend and the faster growth of some cells are
  events on the calendar, so a scenario-off calendar is stationary.
- Each row carries the traffic of the same cell and hour 7, 8 and 14 days earlier and the
  mean over days 7 to 13 earlier: the history a 7-day-ahead forecast may use. It comes from
  the same deterministic days.
- The radio helpers are copied here, not shared, so another use case's physics cannot move
  these results.
"""

from __future__ import annotations

from datetime import date, timedelta
from functools import cache

import numpy as np
import pandas as pd
from telecom_ml_core.contract import LABEL_RELEASE, Batch, Scenario, rng_for_day

USECASE_ID = 5
N_CELLS = 60
HOURS = 24
LABEL_DELAY_DAYS = 0
HORIZON_DAYS = 7
FAST_GROWTH_CELLS = 15
HOLIDAY_CELL_SHARE = 0.3
SPECIAL_EVENT_PROBABILITY = 0.02
SPECIAL_EVENT_MULTIPLIER = 2.5
LEVEL_PERSISTENCE = 0.97
LEVEL_SD = 0.04
LEVEL_MEMORY_DAYS = 150

CELL_TYPES = ("macro", "micro", "small")
AREA_TYPES = ("urban", "suburban", "rural")
BASE_LOAD = {"urban": 15.0, "suburban": 8.0, "rural": 3.0}
BASE_USERS = {"urban": 500, "suburban": 300, "rural": 100}
MAX_THROUGHPUT = {"urban": 100.0, "suburban": 60.0, "rural": 30.0}
BASE_SINR = {"urban": 8.0, "suburban": 12.0, "rural": 15.0}

# Base value of every parameter a scenario event can move (names match scenario.yaml).
BASE_PARAMS: dict[str, float] = {
    "growth_level": 1.0,
    "fast_growth_extra": 1.0,
    "holiday_mult": 1.0,
}


def params_on(day: date, scenario: Scenario) -> dict[str, float]:
    """Every generator parameter on `day` under `scenario`."""
    return {name: scenario.value(name, base, day) for name, base in BASE_PARAMS.items()}


@cache
def cell_profiles(base_seed: int) -> pd.DataFrame:
    """The 60 cells, with fixed ranks deciding which grow fast and which see the holiday."""
    rng = np.random.default_rng([base_seed, USECASE_ID, 0])
    area = rng.choice(AREA_TYPES, N_CELLS, p=[0.5, 0.3, 0.2])
    return pd.DataFrame(
        {
            "cell_id": [f"CELL_{i:04d}" for i in range(N_CELLS)],
            "cell_type": rng.choice(CELL_TYPES, N_CELLS, p=[0.4, 0.35, 0.25]),
            "area_type": area,
            "base_load_gb": [BASE_LOAD[a] for a in area],
            "base_users": [BASE_USERS[a] for a in area],
            "fast_growth": rng.permutation(N_CELLS) < FAST_GROWTH_CELLS,
            "holiday": rng.permutation(N_CELLS) < round(N_CELLS * HOLIDAY_CELL_SHARE),
        }
    )


@cache
def _level_shock(base_seed: int, ordinal: int) -> np.ndarray:
    """One day's shocks to each cell's slow level, from a child stream of that day's seed."""
    child = rng_for_day(base_seed, USECASE_ID, date.fromordinal(ordinal)).spawn(1)[0]
    step_sd = LEVEL_SD * np.sqrt(1 - LEVEL_PERSISTENCE**2)
    return np.asarray(child.normal(0.0, step_sd, N_CELLS))


def _level(base_seed: int, day: date) -> np.ndarray:
    """Each cell's slow level on `day`: AR(1) by day, summed over its last 150 shocks."""
    ordinal = day.toordinal()
    weights = LEVEL_PERSISTENCE ** np.arange(LEVEL_MEMORY_DAYS)
    shocks = np.stack([_level_shock(base_seed, ordinal - j) for j in range(LEVEL_MEMORY_DAYS)])
    return np.asarray(weights @ shocks)


def _diurnal(hour: np.ndarray) -> np.ndarray:
    """24-hour pattern peaking at hour 20, trough at hour 4 (source)."""
    return np.asarray(1.0 + 0.5 * np.sin((hour - 4) * np.pi / 8 - np.pi / 2))


def _latency(rng: np.random.Generator, congestion: np.ndarray) -> np.ndarray:
    latency = 20 * (1 + 5 * congestion**2) + rng.normal(0, 5, len(congestion))
    return np.asarray(np.clip(latency, 10, 300))


@cache
def _day(base_seed: int, ordinal: int, levels: tuple[float, float, float]) -> pd.DataFrame:
    """Every cell's 24 hours on one day, for the given (growth, fast growth, holiday) levels."""
    day = date.fromordinal(ordinal)
    growth, fast_extra, holiday = levels
    rng = rng_for_day(base_seed, USECASE_ID, day)
    timestamps = pd.date_range(start=pd.Timestamp(day), periods=HOURS, freq="h")
    hour = timestamps.hour.to_numpy().astype(float)
    weekly = np.where(timestamps.dayofweek.to_numpy() >= 5, 0.85, 1.0)
    level = _level(base_seed, day)
    frames = []
    profiles = cell_profiles(base_seed)
    cell_ids = profiles["cell_id"].tolist()
    cell_types = profiles["cell_type"].tolist()
    areas = profiles["area_type"].tolist()
    base_loads = profiles["base_load_gb"].to_numpy(dtype=float)
    base_users_all = profiles["base_users"].to_numpy(dtype=float)
    fast_cells = profiles["fast_growth"].to_numpy(dtype=bool)
    holiday_cells = profiles["holiday"].to_numpy(dtype=bool)
    for i in range(N_CELLS):
        base = float(base_loads[i])
        base_users = float(base_users_all[i])
        area = str(areas[i])
        noise = rng.normal(1.0, 0.12, HOURS)
        ar = np.zeros(HOURS)
        ar[0] = rng.normal(0, 0.04)
        for t in range(1, HOURS):
            ar[t] = 0.6 * ar[t - 1] + rng.normal(0, 0.04)
        traffic = base * _diurnal(hour) * weekly * noise * (1.0 + level[i]) + ar * base

        burst = np.zeros(HOURS, dtype=bool)
        t = 0
        while t < HOURS:
            if rng.random() < SPECIAL_EVENT_PROBABILITY * 0.5:
                duration = int(rng.integers(3, 7))
                burst[t : t + duration] = True
                t += duration
            else:
                t += 1
        multiplier = rng.uniform(1.3, SPECIAL_EVENT_MULTIPLIER * 0.8, HOURS)
        traffic = np.where(burst, traffic * multiplier, traffic)
        outage = rng.random(HOURS) < 0.001
        traffic = np.where(outage, traffic * rng.uniform(0.05, 0.2), traffic)

        scale = growth * (fast_extra if fast_cells[i] else 1.0)
        scale *= holiday if holiday_cells[i] else 1.0
        traffic = np.clip(traffic * scale, 0.0, None)

        congestion = np.clip(traffic / (base * 3.0), 0.0, 1.0)
        users = base_users * congestion + rng.normal(0, base_users * 0.05, HOURS)
        prb = np.clip(0.1 + 0.85 * congestion + rng.normal(0, 0.03, HOURS), 0.1, 0.95)
        max_tp = MAX_THROUGHPUT[area]
        throughput = np.clip(
            max_tp * (1.0 - 0.7 * congestion) + rng.normal(0, 3, HOURS), 1.0, max_tp
        )
        latency = _latency(rng, congestion)
        sinr = np.clip(rng.normal(BASE_SINR[area], 4.0, HOURS), -5, 25)
        frames.append(
            pd.DataFrame(
                {
                    "cell_id": cell_ids[i],
                    "cell_type": cell_types[i],
                    "area_type": area,
                    "timestamp": timestamps,
                    "traffic_load_gb": np.round(traffic, 4),
                    "connected_users": np.clip(users, 10, base_users * 1.5).astype(int),
                    "prb_utilization": np.round(prb, 4),
                    "avg_throughput_mbps": np.round(throughput, 2),
                    "avg_latency_ms": np.round(latency, 2),
                    "avg_sinr_db": np.round(sinr, 2),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _traffic_on(day: date, scenario: Scenario) -> pd.DataFrame:
    p = params_on(day, scenario)
    levels = (p["growth_level"], p["fast_growth_extra"], p["holiday_mult"])
    return _day(scenario.base_seed, day.toordinal(), levels)


def generate(day: date, scenario: Scenario) -> Batch:
    """The 60 cells' hourly traffic on `day`, with the history a 7-day-ahead forecast may use.
    Actual traffic arrives each hour, so labels are released the same day."""
    data = _traffic_on(day, scenario).copy()

    def past(days_back: int) -> np.ndarray:
        frame = _traffic_on(day - timedelta(days=days_back), scenario)
        return frame["traffic_load_gb"].to_numpy()

    data["lag_7d"] = past(7)
    data["lag_8d"] = past(8)
    data["lag_14d"] = past(14)
    data["mean_same_hour_7_13d"] = np.mean([past(k) for k in range(7, 14)], axis=0)
    data[LABEL_RELEASE] = pd.Timestamp(day)
    return Batch(day=day, data=data, manifest=scenario.manifest(day))
