"""Day-0 anchor (docs/generators.md rule 3).

Source: churn-prediction at 8f3735cb, evidence/baseline_metrics.json (2026-09-19): 10,000
customers, stratified 80/20 split with seed 42, churn rate 0.151 on the test split,
logistic regression on raw features AUROC 0.8727, XGBoost on engineered features 0.8548.

Here: 10 scenario-off days from the calendar start (10,000 customers), the same split.
Tolerance: churn rate within 0.015, each AUROC within 0.02.
"""

from datetime import timedelta

import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from telecom_ml_churn import generator
from telecom_ml_churn.baseline import fit_baseline
from telecom_ml_churn.features import TARGET, engineered_features, raw_features
from telecom_ml_churn.model import fit_model
from telecom_ml_churn.usecase import ChurnUseCase

EVIDENCE = {"churn_rate_test": 0.151, "baseline_auroc": 0.8727, "model_auroc": 0.8548}
RATE_TOLERANCE = 0.015
AUROC_TOLERANCE = 0.02
SEED = 42


@pytest.fixture(scope="module")
def anchor() -> dict[str, float]:
    scenario = ChurnUseCase.scenario.without_events()
    data = pd.concat(
        [generator.generate(scenario.start + timedelta(days=i), scenario).data for i in range(10)],
        ignore_index=True,
    )
    train, test = train_test_split(
        data.index, test_size=0.2, random_state=SEED, stratify=data[TARGET]
    )
    labels = data.loc[test, TARGET]
    baseline = fit_baseline(data.loc[train])
    model = fit_model(data.loc[train], seed=SEED)
    return {
        "churn_rate_test": float(labels.mean()),
        "baseline_auroc": float(roc_auc_score(labels, baseline.predict_proba(data.loc[test]))),
        "model_auroc": float(roc_auc_score(labels, model.predict_proba(data.loc[test]))),
        "raw_columns": raw_features(data).shape[1],
        "engineered_columns": engineered_features(data).shape[1],
    }


def test_feature_counts_match_the_source(anchor: dict[str, float]) -> None:
    assert (anchor["raw_columns"], anchor["engineered_columns"]) == (17, 27)


def test_churn_rate_matches_the_evidence(anchor: dict[str, float]) -> None:
    assert anchor["churn_rate_test"] == pytest.approx(
        EVIDENCE["churn_rate_test"], abs=RATE_TOLERANCE
    )


@pytest.mark.parametrize("metric", ["baseline_auroc", "model_auroc"])
def test_auroc_matches_the_evidence(anchor: dict[str, float], metric: str) -> None:
    assert anchor[metric] == pytest.approx(EVIDENCE[metric], abs=AUROC_TOLERANCE)
