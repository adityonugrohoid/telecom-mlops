"""The QoE generator: one simulated day of user sessions, their radio conditions and the MOS
each user reports.

Ported from the qoe-prediction repo (data_generator.py, QoEDataGenerator). The session model
is kept: device class sets the SINR and throughput ceiling, the hour's congestion sets
latency, and MOS follows throughput, latency and loss per app, plus perception noise.

Changes from the source, on purpose:
- Seeded by date (base seed, use case id, day); nothing reads the clock. Sessions start
  inside the batch day, where the source spread them over 30 days.
- Scenario parameters for the drift calendar: `high_device_share` (the device mix),
  `video_codec_gain` (how much more MOS a video session gets from the same throughput) and
  `gaming_share` (the app mix).
- The radio and MOS helpers are copied here, not shared, so another use case's physics
  cannot move these results.
- MOS is measured with the session, so each label is released the same day.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from telecom_ml_core.contract import LABEL_RELEASE, Batch, Scenario, rng_for_day

USECASE_ID = 4
SESSIONS_PER_DAY = 1_500
LABEL_DELAY_DAYS = 0

SOURCE_APPS = ("video_streaming", "browsing", "gaming", "social", "voip")
APP_WEIGHTS = (0.25, 0.30, 0.15, 0.15, 0.15)
NEW_APP = "cloud_gaming"
APPS = (*SOURCE_APPS, NEW_APP)
# A cloud gaming session needs four times the throughput for the same picture and pays twice
# the latency penalty of local gaming.
CLOUD_THROUGHPUT_NEED = 4.0
CLOUD_LATENCY_WEIGHT = 2.0
DEVICES = ("low", "mid", "high")
DEVICE_WEIGHTS = (0.2, 0.5, 0.3)
NETWORKS = ("4G", "5G")
DEVICE_SINR = {"low": 7.0, "mid": 10.0, "high": 13.0}
DEVICE_MAX_THROUGHPUT = {"low": 20.0, "mid": 50.0, "high": 150.0}

# Base value of every parameter a scenario event can move (names match scenario.yaml).
BASE_PARAMS: dict[str, float] = {
    "high_device_share": DEVICE_WEIGHTS[2],
    "video_codec_gain": 1.0,
    "gaming_share": APP_WEIGHTS[2],
    "cloud_gaming_share": 0.0,
}


def params_on(day: date, scenario: Scenario) -> dict[str, float]:
    """Every generator parameter on `day` under `scenario`."""
    return {name: scenario.value(name, base, day) for name, base in BASE_PARAMS.items()}


def _rescaled(weights: tuple[float, ...], index: int, share: float) -> list[float]:
    """The weights with entry `index` set to `share` and the others scaled to fill the rest."""
    others = sum(w for i, w in enumerate(weights) if i != index)
    return [share if i == index else w * (1.0 - share) / others for i, w in enumerate(weights)]


# -- radio and MOS helpers, copied from the source's TelecomDataGenerator ----------------


def _sinr_to_throughput(
    rng: np.random.Generator, sinr_db: np.ndarray, network: np.ndarray
) -> np.ndarray:
    capacity = np.log2(1 + 10 ** (sinr_db / 10))
    max_throughput = np.where(network == "5G", 300, 50)
    throughput = capacity * max_throughput / 5 * rng.normal(1, 0.2, len(sinr_db))
    return np.asarray(np.clip(throughput, 0.1, max_throughput))


def _congestion(rng: np.random.Generator, timestamps: pd.DatetimeIndex) -> np.ndarray:
    hour = timestamps.hour.to_numpy()
    weekday = timestamps.dayofweek.to_numpy()
    congestion = 0.5 + 0.3 * np.sin((hour - 6) * np.pi / 12)
    peak = ((hour >= 9) & (hour <= 11)) | ((hour >= 18) & (hour <= 21))
    congestion = np.where(peak, congestion * 1.3, congestion)
    congestion = np.where(weekday >= 5, congestion * 0.8, congestion)
    return np.asarray(np.clip(congestion + rng.normal(0, 0.1, len(congestion)), 0, 1))


def _latency(rng: np.random.Generator, congestion: np.ndarray) -> np.ndarray:
    latency = 20 * (1 + 5 * congestion**2) + rng.normal(0, 5, len(congestion))
    return np.asarray(np.clip(latency, 10, 300))


def _qoe_mos(
    throughput: np.ndarray,
    latency: np.ndarray,
    loss: np.ndarray,
    app: np.ndarray,
    video_codec_gain: float,
) -> np.ndarray:
    """The source's MOS model. `video_codec_gain` multiplies the throughput a video session
    effectively gets: a better codec gives the same picture from fewer bits."""
    effective = np.where(app == "video_streaming", throughput * video_codec_gain, throughput)
    effective = np.where(app == NEW_APP, throughput / CLOUD_THROUGHPUT_NEED, effective)
    mos = 1 + 4 * (1 - np.exp(-effective / 10))
    latency_penalty = np.clip(latency / 100, 0, 2)
    mos = mos - latency_penalty - loss / 2
    mos = np.where(app == "video_streaming", mos - loss * 0.5, mos)
    mos = np.where(app == "gaming", mos - latency_penalty * 0.5, mos)
    mos = np.where(app == NEW_APP, mos - latency_penalty * CLOUD_LATENCY_WEIGHT, mos)
    return np.asarray(np.clip(mos, 1, 5))


def generate(day: date, scenario: Scenario) -> Batch:
    """The sessions that start on `day`, each with the MOS its user reported."""
    rng = rng_for_day(scenario.base_seed, USECASE_ID, day)
    # Draws that exist only for scenario events come from a child stream, so adding an event
    # never shifts the sessions the source model draws.
    event_rng = rng.spawn(1)[0]
    p = params_on(day, scenario)
    n = SESSIONS_PER_DAY
    timestamps = pd.DatetimeIndex(
        pd.Timestamp(day) + pd.to_timedelta(rng.integers(0, 24 * 3600, n), unit="s")
    )
    network = rng.choice(NETWORKS, n, p=[0.6, 0.4])
    device = rng.choice(DEVICES, n, p=_rescaled(DEVICE_WEIGHTS, 2, p["high_device_share"]))
    app = rng.choice(SOURCE_APPS, n, p=_rescaled(APP_WEIGHTS, 2, p["gaming_share"]))
    app = np.where(event_rng.random(n) < p["cloud_gaming_share"], NEW_APP, app)

    base_sinr = np.array([DEVICE_SINR[d] for d in device])
    sinr = np.clip(rng.normal(base_sinr, 4.0), -5, 25)
    ceiling = np.array([DEVICE_MAX_THROUGHPUT[d] for d in device])
    throughput = np.minimum(_sinr_to_throughput(rng, sinr, network), ceiling)
    congestion = _congestion(rng, timestamps)
    latency = _latency(rng, congestion)
    loss = np.clip(rng.exponential(0.5, n), 0, 5)
    duration = np.clip(rng.gamma(shape=3, scale=5, size=n), 1, 120)
    volume = throughput * duration * (60 / 8) * rng.uniform(0.5, 1.0, n)

    mos_base = _qoe_mos(throughput, latency, loss, app, p["video_codec_gain"])
    perception = rng.normal(0, 0.4, n)
    user_bias = rng.normal(0, 0.15, n)
    content = rng.uniform(-0.3, 0.3, n)
    mos = np.clip(mos_base + perception + user_bias + content, 1, 5)

    data = pd.DataFrame(
        {
            "session_id": [f"S{day:%Y%m%d}-{i:05d}" for i in range(n)],
            "timestamp": timestamps,
            "network_type": network,
            "device_class": device,
            "app_type": app,
            "sinr_db": sinr,
            "throughput_mbps": throughput,
            "latency_ms": latency,
            "packet_loss_pct": loss,
            "congestion_level": congestion,
            "session_duration_min": duration,
            "data_volume_mb": volume,
            "mos_score": mos,
        }
    )
    data[LABEL_RELEASE] = pd.Timestamp(day)
    return Batch(day=day, data=data, manifest=scenario.manifest(day))
