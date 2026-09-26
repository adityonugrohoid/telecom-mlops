"""The root-cause ranker: XGBoost with the source's `multi:softprob` objective over the two
classes of `is_root_cause`, ported from the root-cause-analysis repo (models.py,
XGBoostRCAClassifier). Events are ranked within each incident by their root probability."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from telecom_ml_root_cause.features import TARGET, engineered_features

HYPERPARAMETERS = {
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 200,
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "num_class": 2,
}


@dataclass(frozen=True)
class RootCauseModel:
    classifier: XGBClassifier
    seen_types: frozenset[str]

    def root_score(self, data: pd.DataFrame) -> np.ndarray:
        """Probability that each event is its incident's root cause."""
        return np.asarray(self.classifier.predict_proba(engineered_features(data))[:, 1])


def fit_model(train: pd.DataFrame, seed: int) -> RootCauseModel:
    """Train on released incidents. Retrains from scratch on the window; no warm start."""
    classifier = XGBClassifier(**HYPERPARAMETERS, random_state=seed)
    classifier.fit(engineered_features(train), train[TARGET])
    return RootCauseModel(classifier, frozenset(train["event_type"].unique()))
