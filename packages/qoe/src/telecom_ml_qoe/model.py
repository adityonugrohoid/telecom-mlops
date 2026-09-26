"""QoE models.

`QoEModel` is LightGBM on the session features, ported from the qoe-prediction repo
(models.py, LightGBMQoERegressor) with its hyperparameters. `MeanMos` is the source's
baseline: predict the training window's mean MOS for every session.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from telecom_ml_qoe.features import TARGET, features


@dataclass(frozen=True)
class QoEModel:
    regressor: LGBMRegressor

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Predicted MOS per session."""
        return np.asarray(self.regressor.predict(features(data)))


def fit_model(train: pd.DataFrame, seed: int) -> QoEModel:
    """Train on the window's sessions with the source's hyperparameters. Retrains from scratch;
    no warm start."""
    regressor = LGBMRegressor(
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=200,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=seed,
        verbose=-1,
    )
    regressor.fit(features(train), train[TARGET])
    return QoEModel(regressor)


@dataclass(frozen=True)
class MeanMos:
    mean: float

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        return np.full(len(data), self.mean)


def fit_baseline(train: pd.DataFrame) -> MeanMos:
    return MeanMos(float(train[TARGET].mean()))
