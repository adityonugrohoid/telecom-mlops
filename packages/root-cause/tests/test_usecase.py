from datetime import timedelta

import pandas as pd
from telecom_ml_root_cause import generator
from telecom_ml_root_cause.baseline import FirstAlarm
from telecom_ml_root_cause.features import INCIDENT
from telecom_ml_root_cause.model import fit_model
from telecom_ml_root_cause.usecase import RootCauseUseCase

USECASE = RootCauseUseCase()
SCENARIO = USECASE.scenario


def days(first: int, count: int) -> pd.DataFrame:
    return pd.concat(
        [
            generator.generate(SCENARIO.start + timedelta(days=first + i), SCENARIO).data
            for i in range(count)
        ],
        ignore_index=True,
    )


def test_drift_frame_has_one_row_per_incident() -> None:
    data = days(0, 2)
    frame = USECASE.drift_frame(data)
    assert len(frame) == data[INCIDENT].nunique()
    assert "incidents_in_day" not in frame.columns
    assert frame["max_throughput_drop"].ge(0).all()


def test_score_reports_volume_and_unseen_types() -> None:
    model = fit_model(days(60, 10), seed=1)
    before, after = days(80, 5), days(95, 5)
    assert USECASE.score(model, before)["unknown_signature_share"] == 0.0
    assert USECASE.score(model, after)["unknown_signature_share"] > 0.0
    assert USECASE.score(FirstAlarm(), after)["unknown_signature_share"] == 0.0
    volume = USECASE.score(model, before)["incidents_per_day"]
    assert volume == before.groupby(INCIDENT)["incidents_in_day"].first().mean()


def test_promotion_needs_the_margin() -> None:
    live = {"top1": 0.90}
    assert not USECASE.promote({"top1": 0.94}, live, {"top1": 0.7}).promote
    assert USECASE.promote({"top1": 0.95}, live, {"top1": 0.7}).promote
