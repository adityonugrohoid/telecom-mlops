from datetime import timedelta

import numpy as np
import pandas as pd
from telecom_ml_anomaly import generator
from telecom_ml_anomaly.model import fit_forest, triaged
from telecom_ml_anomaly.usecase import AnomalyUseCase

USECASE = AnomalyUseCase()
SCENARIO = USECASE.scenario


def days(first: int, count: int) -> pd.DataFrame:
    return pd.concat(
        [
            generator.generate(SCENARIO.start + timedelta(days=first + i), SCENARIO).data
            for i in range(count)
        ],
        ignore_index=True,
    )


def test_audit_is_a_random_tenth_of_hours() -> None:
    data = days(0, 10)
    assert data["audited"].mean() == np.float64(data["audited"].mean())
    assert abs(data["audited"].mean() - generator.AUDIT_SHARE) < 0.01


def test_drift_frame_has_one_row_per_cell_per_day() -> None:
    frame = USECASE.drift_frame(days(0, 3))
    assert len(frame) == generator.N_CELLS * 3


def test_triage_labels_only_audited_or_alerted_hours() -> None:
    data = days(0, 10)
    forest = fit_forest(data, seed=1)
    looked_at = triaged(data, forest, None)
    expected = data["audited"].to_numpy(bool) | forest.alerts(data).astype(bool)
    assert (looked_at == expected).all()
    assert 0.10 < looked_at.mean() < 0.25


def test_scores_use_the_audit_sample_only() -> None:
    data = days(0, 5)
    model = USECASE.fit(days(-30, 30), None)
    unaudited_flipped = data.copy()
    hidden = ~unaudited_flipped["audited"]
    unaudited_flipped.loc[hidden, "label_anomaly"] = (
        1 - unaudited_flipped.loc[hidden, "label_anomaly"]
    )
    assert USECASE.score(model, data) == USECASE.score(model, unaudited_flipped)


def test_promotion_refuses_rising_normal_alerts() -> None:
    live = {"f1": 0.90, "normal_alert_rate": 0.001}
    within_noise = {"f1": 0.99, "normal_alert_rate": 0.004}
    worse_alerts = {"f1": 0.99, "normal_alert_rate": 0.006}
    assert USECASE.promote(within_noise, live, live).promote
    assert not USECASE.promote(worse_alerts, live, live).promote
    assert not USECASE.promote({"f1": 0.94, "normal_alert_rate": 0.001}, live, live).promote
