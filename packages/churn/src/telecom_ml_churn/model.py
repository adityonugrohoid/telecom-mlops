"""The churn model: XGBoost on engineered features, ported from the churn-prediction repo
(models.py, XGBoostChurnClassifier) with the hyperparameters its `train` passed."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from telecom_ml_churn.features import TARGET, engineered_features

HYPERPARAMETERS = {
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 200,
    "objective": "binary:logistic",
    "eval_metric": "auc",
}


@dataclass(frozen=True)
class ChurnModel:
    classifier: XGBClassifier

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """Churn probability per customer."""
        return np.asarray(self.classifier.predict_proba(engineered_features(data))[:, 1])


def fit_model(train: pd.DataFrame, seed: int) -> ChurnModel:
    """Train on released rows. Churn retrains from scratch on the window; no warm start."""
    classifier = XGBClassifier(**HYPERPARAMETERS, random_state=seed)
    classifier.fit(engineered_features(train), train[TARGET])
    return ChurnModel(classifier)
