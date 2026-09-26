"""Day-0 anchor (docs/generators.md rule 3).

The earlier anomaly-detection repo has no evidence file, so the anchor is a fresh run of its
code: e2c311fd, its own notebook at seed 42, in a separate environment on its own pinned
versions (python 3.13.12, numpy 1.26.4, scikit-learn 1.9.1, pandas 3.0.6). Isolation Forest
(contamination 0.05) trained and scored on all 36,000 cell-hours: F1 0.6983, ROC AUC 0.9691.

The earlier README's precision 0.95 and recall 0.38 are the 2nd-percentile row of the
notebook's threshold table (F1 0.545), not its default threshold.

Here: the same forest on 30 scenario-off days (36,000 cell-hours), trained and scored on
all of them. Tolerance: F1 within 0.05, AUC within 0.01.
"""

from datetime import timedelta

import pandas as pd
import pytest
from sklearn.metrics import f1_score, roc_auc_score
from telecom_ml_anomaly import generator
from telecom_ml_anomaly.features import TARGET
from telecom_ml_anomaly.model import fit_forest
from telecom_ml_anomaly.usecase import AnomalyUseCase

SOURCE_RUN = {"f1": 0.6983, "auc": 0.9691}
F1_TOLERANCE = 0.05
AUC_TOLERANCE = 0.01
SEED = 42


@pytest.fixture(scope="module")
def anchor() -> dict[str, float]:
    scenario = AnomalyUseCase.scenario.without_events()
    data = pd.concat(
        [generator.generate(scenario.start + timedelta(days=i), scenario).data for i in range(30)],
        ignore_index=True,
    )
    forest = fit_forest(data, seed=SEED)
    labels = data[TARGET]
    return {
        "rows": float(len(data)),
        "anomaly_rate": float(labels.mean()),
        "f1": float(f1_score(labels, forest.alerts(data))),
        "auc": float(roc_auc_score(labels, -forest.score(data))),
    }


def test_same_scale_as_the_source(anchor: dict[str, float]) -> None:
    assert (anchor["rows"], anchor["anomaly_rate"]) == (36_000, pytest.approx(0.05))


def test_f1_matches_the_source_run(anchor: dict[str, float]) -> None:
    assert anchor["f1"] == pytest.approx(SOURCE_RUN["f1"], abs=F1_TOLERANCE)


def test_auc_matches_the_source_run(anchor: dict[str, float]) -> None:
    assert anchor["auc"] == pytest.approx(SOURCE_RUN["auc"], abs=AUC_TOLERANCE)
