"""Capacity forecasters, 7 days ahead.

`Forecaster` is LightGBM with the capacity-forecasting repo's hyperparameters (models.py,
LightGBMForecaster). Its inputs hold only information at least 7 days old: the same hour 7, 8
and 14 days earlier, the mean over days 7 to 13, the hour and weekday, and the cell's type and
area. The source's own features included the previous hour, which forecasts one hour ahead.

`SeasonalNaive` forecasts the same cell and hour 7 days earlier.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from telecom_ml_capacity.generator import AREA_TYPES, CELL_TYPES

TARGET = "traffic_load_gb"
LAGS = ["lag_7d", "lag_8d", "lag_14d", "mean_same_hour_7_13d"]


def features(data: pd.DataFrame) -> pd.DataFrame:
    """Model inputs per cell-hour; nothing newer than 7 days before the forecast hour."""
    timestamp = pd.to_datetime(data["timestamp"])
    one_hot = {
        f"{column}_{category}": (data[column] == category).astype(float)
        for column, categories in (("cell_type", CELL_TYPES), ("area_type", AREA_TYPES))
        for category in sorted(categories)[1:]
    }
    return pd.DataFrame(
        {
            **{lag: data[lag].astype(float) for lag in LAGS},
            "hour": timestamp.dt.hour.astype(float),
            "day_of_week": timestamp.dt.dayofweek.astype(float),
            **one_hot,
        },
        index=data.index,
    )


@dataclass(frozen=True)
class Forecaster:
    regressor: LGBMRegressor

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Forecast traffic per cell-hour."""
        return np.asarray(self.regressor.predict(features(data)))


def fit_forecaster(train: pd.DataFrame, seed: int) -> Forecaster:
    """Train on the window's cell-hours, in time order. Retrains from scratch."""
    regressor = LGBMRegressor(
        num_leaves=63,
        learning_rate=0.05,
        n_estimators=300,
        max_depth=-1,
        min_child_samples=30,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=0.5,
        random_state=seed,
        verbose=-1,
    )
    regressor.fit(features(train), train[TARGET])
    return Forecaster(regressor)


@dataclass(frozen=True)
class SeasonalNaive:
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """The same cell and hour 7 days earlier."""
        return data["lag_7d"].to_numpy(dtype=float)


def mape_pct(truth: np.ndarray, forecast: np.ndarray) -> float:
    """Mean absolute percentage error over hours with nonzero traffic, as the source did."""
    mask = truth != 0
    if not mask.any():
        raise ValueError("cannot compute MAPE: every actual value is zero")
    return float(np.mean(np.abs((truth[mask] - forecast[mask]) / truth[mask])) * 100)
