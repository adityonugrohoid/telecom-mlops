"""Anomaly models.

`IsolationForestModel` is the source's unsupervised detector, ported from the
anomaly-detection repo (models.py) with its hyperparameters. It is the baseline and holds the
day-0 anchor.

`TriageDetector` is the loop's model: a gradient-boosted classifier on the 16 features plus
the Isolation Forest score, trained on triage labels. Only hours someone looked at carry a
label: hours the Isolation Forest or the live detector alerted on, and the random audit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest

from telecom_ml_anomaly.features import TARGET, features

FOREST_HYPERPARAMETERS = {"n_estimators": 200, "contamination": 0.05, "max_features": 1.0}
ALERT_PROBABILITY = 0.5


@dataclass(frozen=True)
class IsolationForestModel:
    forest: IsolationForest

    def score(self, data: pd.DataFrame) -> np.ndarray:
        """Anomaly score per cell-hour; lower is more anomalous (sklearn's decision function)."""
        return np.asarray(self.forest.decision_function(features(data)))

    def alerts(self, data: pd.DataFrame) -> np.ndarray:
        """1 where the forest raises an alert, else 0."""
        return np.asarray(self.forest.predict(features(data)) == -1).astype(int)


def fit_forest(train: pd.DataFrame, seed: int) -> IsolationForestModel:
    """Train on every cell-hour of the window; labels are not used."""
    forest = IsolationForest(**FOREST_HYPERPARAMETERS, random_state=seed)
    forest.fit(features(train))
    return IsolationForestModel(forest)


@dataclass(frozen=True)
class TriageDetector:
    forest: IsolationForestModel
    classifier: HistGradientBoostingClassifier

    def _inputs(self, data: pd.DataFrame) -> pd.DataFrame:
        return features(data).assign(forest_score=self.forest.score(data))

    def probability(self, data: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.classifier.predict_proba(self._inputs(data))[:, 1])

    def alerts(self, data: pd.DataFrame) -> np.ndarray:
        """1 where the detector raises an alert, else 0."""
        return (self.probability(data) >= ALERT_PROBABILITY).astype(int)


def triaged(
    data: pd.DataFrame, forest: IsolationForestModel, live: TriageDetector | None
) -> np.ndarray:
    """Which hours carry a triage label: audited, or alerted on by the forest or the live
    detector. The live detector at retraining time stands in for the one live at the hour."""
    looked_at = data["audited"].to_numpy(bool) | forest.alerts(data).astype(bool)
    if live is not None:
        looked_at |= live.alerts(data).astype(bool)
    return np.asarray(looked_at)


def fit_detector(train: pd.DataFrame, live: TriageDetector | None, seed: int) -> TriageDetector:
    """Refit the forest on every hour of the window, then the classifier on triaged hours."""
    forest = fit_forest(train, seed)
    labelled = train[triaged(train, forest, live)]
    inputs = features(labelled).assign(forest_score=forest.score(labelled))
    classifier = HistGradientBoostingClassifier(random_state=seed)
    classifier.fit(inputs, labelled[TARGET])
    return TriageDetector(forest, classifier)
