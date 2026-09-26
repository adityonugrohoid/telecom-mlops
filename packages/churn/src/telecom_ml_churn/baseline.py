"""The churn baseline: logistic regression on raw features, ported from the churn-prediction
repo (baseline.py). On the source evidence it scored AUROC 0.8727 against XGBoost's 0.8548."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from telecom_ml_churn.features import TARGET, raw_features


@dataclass(frozen=True)
class ChurnBaseline:
    pipeline: Pipeline

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """Churn probability per customer."""
        return np.asarray(self.pipeline.predict_proba(raw_features(data))[:, 1])


def fit_baseline(train: pd.DataFrame) -> ChurnBaseline:
    pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    pipeline.fit(raw_features(train), train[TARGET])
    return ChurnBaseline(pipeline)
