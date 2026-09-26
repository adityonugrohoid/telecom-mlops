"""The churn generator: one simulated day of customers, their network experience and churn.

Ported from the churn-prediction repo (data_generator.py). The structural model is kept:
a logit on QoE MOS, tenure, charges, tickets and contract, with an intercept that gives a
15% churn rate. Every coefficient and input mix is a named parameter the scenario can move.

Changes from the source, on purpose:
- Seeded by date (base seed, use case id, day); nothing reads the clock. Observation
  timestamps fall in the 30 days before the batch day, where the source used `now()`.
- The intercept is solved once, on the scenario-off distribution, and then held. The source
  re-solved it for every sample, which would cancel any drift in the churn rate.
- The radio helpers are copied here, not shared, so another use case's physics cannot move
  these results.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from functools import cache

import numpy as np
import pandas as pd
from telecom_ml_core.contract import LABEL_RELEASE, Batch, Scenario, rng_for_day

USECASE_ID = 1
CUSTOMERS_PER_DAY = 1_000
LABEL_DELAY_DAYS = 30
OBSERVATION_WINDOW_DAYS = 30
TARGET_CHURN_RATE = 0.15
INTERCEPT_SAMPLE = 200_000

CONTRACTS = ("month-to-month", "one-year", "two-year")
PAYMENTS = ("electronic", "bank_transfer", "credit_card", "mail_check")
NETWORKS = ("4G", "5G")
DEVICES = ("low", "mid", "high")
APPS = ("web_browsing", "video_streaming", "gaming", "voip")

# Base value of every parameter a scenario event can move (names match scenario.yaml).
BASE_PARAMS: dict[str, float] = {
    # label mechanism
    "qoe_coef": -1.8,
    "tenure_coef": -0.8,
    "charge_coef": 0.4,
    "ticket_coef": 0.35,
    "contract_mtm": 0.5,
    "contract_one_year": -0.2,
    "contract_two_year": -0.6,
    "logit_noise_sd": 0.6,
    # inputs
    "share_5g": 0.4,
    "share_mtm": 0.5,
    "share_one_year": 0.3,
    "mtm_charge_mult": 1.0,
    "ticket_rate": 2.0,
}


@dataclass(frozen=True)
class Draw:
    """One sample of customers before labelling, and its churn logit without intercept."""

    frame: pd.DataFrame
    logit: np.ndarray


def params_on(day: date, scenario: Scenario) -> dict[str, float]:
    """Every generator parameter on `day` under `scenario`."""
    return {name: scenario.value(name, base, day) for name, base in BASE_PARAMS.items()}


# -- radio helpers, copied from the source's TelecomDataGenerator ------------------------


def _sinr(rng: np.random.Generator, n: int) -> np.ndarray:
    return np.clip(rng.normal(10.0, 5.0, n), -5, 25)


def _sinr_to_throughput(
    rng: np.random.Generator, sinr_db: np.ndarray, network: np.ndarray
) -> np.ndarray:
    capacity = np.log2(1 + 10 ** (sinr_db / 10))
    max_throughput = np.where(network == "5G", 300, 50)
    throughput = capacity * max_throughput / 5 * rng.normal(1, 0.2, len(sinr_db))
    return np.clip(throughput, 0.1, max_throughput)


def _congestion(rng: np.random.Generator, timestamps: pd.DatetimeIndex) -> np.ndarray:
    hour = timestamps.hour.to_numpy()
    weekday = timestamps.dayofweek.to_numpy()
    congestion = 0.5 + 0.3 * np.sin((hour - 6) * np.pi / 12)
    peak = ((hour >= 9) & (hour <= 11)) | ((hour >= 18) & (hour <= 21))
    congestion = np.where(peak, congestion * 1.3, congestion)
    congestion = np.where(weekday >= 5, congestion * 0.8, congestion)
    return np.clip(congestion + rng.normal(0, 0.1, len(congestion)), 0, 1)


def _latency(rng: np.random.Generator, congestion: np.ndarray) -> np.ndarray:
    latency = 20 * (1 + 5 * congestion**2) + rng.normal(0, 5, len(congestion))
    return np.clip(latency, 10, 300)


def _qoe_mos(
    throughput: np.ndarray, latency: np.ndarray, loss: np.ndarray, app: np.ndarray
) -> np.ndarray:
    mos = 1 + 4 * (1 - np.exp(-throughput / 10))
    latency_penalty = np.clip(latency / 100, 0, 2)
    mos = mos - latency_penalty - loss / 2
    mos = np.where(app == "video_streaming", mos - loss * 0.5, mos)
    mos = np.where(app == "gaming", mos - latency_penalty * 0.5, mos)
    return np.asarray(np.clip(mos, 1, 5))


# -- the structural model ----------------------------------------------------------------


def _draw(rng: np.random.Generator, n: int, day: date, p: dict[str, float]) -> Draw:
    """Customers observed in the window ending on `day`, with their churn logit."""
    window_start = pd.Timestamp(day - timedelta(days=OBSERVATION_WINDOW_DAYS))
    timestamp = pd.DatetimeIndex(
        window_start
        + pd.to_timedelta(rng.integers(0, OBSERVATION_WINDOW_DAYS, n), unit="D")
        + pd.to_timedelta(rng.integers(0, 86_400, n), unit="s")
    )
    tenure = rng.integers(1, 73, n)
    share_two_year = 1.0 - p["share_mtm"] - p["share_one_year"]
    contract = rng.choice(CONTRACTS, n, p=[p["share_mtm"], p["share_one_year"], share_two_year])
    payment = rng.choice(PAYMENTS, n, p=[0.4, 0.3, 0.2, 0.1])
    charge = np.where(
        contract == "month-to-month",
        rng.uniform(30, 90, n) * p["mtm_charge_mult"],
        np.where(contract == "one-year", rng.uniform(50, 110, n), rng.uniform(70, 120, n)),
    )
    charges = np.round(charge, 2)
    network = rng.choice(NETWORKS, n, p=[1.0 - p["share_5g"], p["share_5g"]])
    device = rng.choice(DEVICES, n, p=[0.2, 0.5, 0.3])

    sinr = _sinr(rng, n)
    throughput = _sinr_to_throughput(rng, sinr, network)
    congestion = _congestion(rng, timestamp)
    latency = _latency(rng, congestion)
    loss = np.clip(congestion * 3 + rng.normal(0, 0.5, n), 0, 10)
    app = rng.choice(APPS, n, p=[0.4, 0.3, 0.15, 0.15])
    mos = _qoe_mos(throughput, latency, loss, app)

    tickets = rng.poisson(p["ticket_rate"], n)
    sessions = np.clip(rng.integers(50, 501, n) + (tenure * 3).astype(int), 50, 500)

    contract_score = np.where(
        contract == "month-to-month",
        p["contract_mtm"],
        np.where(contract == "one-year", p["contract_one_year"], p["contract_two_year"]),
    )
    logit = (
        p["qoe_coef"] * (mos - 3.0)
        + p["tenure_coef"] * (np.log1p(tenure) - np.log1p(12))
        + p["charge_coef"] * ((charges - 70) / 30)
        + p["ticket_coef"] * (tickets - 2)
        + contract_score
        + rng.normal(0, p["logit_noise_sd"], n)
    )
    frame = pd.DataFrame(
        {
            "timestamp": timestamp,
            "tenure_months": tenure,
            "contract_type": contract,
            "payment_method": payment,
            "monthly_charges": charges,
            "network_type": network,
            "device_class": device,
            "avg_sinr_db": np.round(sinr, 2),
            "avg_throughput_mbps": np.round(throughput, 2),
            "avg_latency_ms": np.round(latency, 2),
            "avg_packet_loss_pct": np.round(loss, 2),
            "avg_qoe_mos": np.round(mos, 2),
            "total_tickets": tickets,
            "total_sessions": sessions,
        }
    )
    return Draw(frame=frame, logit=logit)


@cache
def intercept(base_seed: int) -> float:
    """The intercept that gives TARGET_CHURN_RATE on the scenario-off distribution.

    Solved once by bisection on a large fixed sample, as the source did per sample.
    """
    rng = np.random.default_rng([base_seed, USECASE_ID, 0])
    logit = _draw(rng, INTERCEPT_SAMPLE, date(2000, 1, 1), BASE_PARAMS).logit
    lo, hi = -10.0, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if np.mean(1.0 / (1.0 + np.exp(-(logit + mid)))) < TARGET_CHURN_RATE:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def generate(day: date, scenario: Scenario) -> Batch:
    """The customers whose observation window ends on `day`, labelled 30 days later."""
    rng = rng_for_day(scenario.base_seed, USECASE_ID, day)
    draw = _draw(rng, CUSTOMERS_PER_DAY, day, params_on(day, scenario))
    churn_prob = 1.0 / (1.0 + np.exp(-(draw.logit + intercept(scenario.base_seed))))
    data = draw.frame.assign(
        customer_id=[f"C{day:%Y%m%d}-{i:04d}" for i in range(CUSTOMERS_PER_DAY)],
        is_churned=(rng.random(CUSTOMERS_PER_DAY) < churn_prob).astype(int),
    )
    data[LABEL_RELEASE] = pd.Timestamp(day + timedelta(days=LABEL_DELAY_DAYS))
    return Batch(day=day, data=data, manifest=scenario.manifest(day))
