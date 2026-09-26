"""Day-0 anchor (docs/generators.md rule 3).

Source: root-cause-analysis at 6bcf53bd, evidence/baseline_metrics.json (2026-09-19): 500
incidents, incident-grouped 80/20 split with seed 42, XGBoost accuracy@1 0.89, first-alarm
baseline 0.58.

The baseline's 0.58 is one 100-incident test split: the source generator puts 70% of roots
at position 0 (a fresh run of the source generator, seed 42, 5,000 incidents: 0.7024), and
that split happened to hold 58. So the anchor is:
(a) distribution at scale: 5,000 scenario-off incidents against the source generator's
    fresh run, root at position 0 within 0.02 and the KPI column means within 3%;
(b) model: the evidence setup at seed 42, accuracy@1 within 0.05 of 0.89;
(c) baseline: first-alarm accuracy@1 at scale within 0.02 of the source's 0.7024.
"""

from datetime import timedelta

import pandas as pd
import pytest
from sklearn.model_selection import GroupShuffleSplit
from telecom_ml_root_cause import generator
from telecom_ml_root_cause.baseline import FirstAlarm
from telecom_ml_root_cause.features import INCIDENT, TARGET
from telecom_ml_root_cause.model import fit_model
from telecom_ml_root_cause.usecase import RootCauseUseCase, top_k

SOURCE_AT_SCALE = {
    "root_at_first_alarm": 0.7024,
    "time_lag_seconds": 73.32,
    "affected_cells": 5.57,
    "sinr_delta": -3.60,
    "throughput_delta": -19.97,
    "latency_delta": 32.50,
}
EVIDENCE_MODEL_TOP1 = 0.89
SHARE_TOLERANCE = 0.02
MEAN_TOLERANCE = 0.03
MODEL_TOLERANCE = 0.05
SEED = 42


def scenario_off_incidents(count: int) -> pd.DataFrame:
    scenario = RootCauseUseCase.scenario.without_events()
    frames, seen, index = [], 0, 0
    while seen < count:
        data = generator.generate(scenario.start + timedelta(days=index), scenario).data
        frames.append(data)
        seen += data[INCIDENT].nunique()
        index += 1
    data = pd.concat(frames, ignore_index=True)
    keep = data[INCIDENT].drop_duplicates().iloc[:count]
    return data[data[INCIDENT].isin(keep)].reset_index(drop=True)


@pytest.fixture(scope="module")
def at_scale() -> pd.DataFrame:
    return scenario_off_incidents(5_000)


def test_root_position_matches_the_source_at_scale(at_scale: pd.DataFrame) -> None:
    roots = at_scale[at_scale[TARGET] == 1]
    share = float((roots["event_sequence_position"] == 0).mean())
    assert share == pytest.approx(SOURCE_AT_SCALE["root_at_first_alarm"], abs=SHARE_TOLERANCE)


@pytest.mark.parametrize(
    "column",
    ["time_lag_seconds", "affected_cells", "sinr_delta", "throughput_delta", "latency_delta"],
)
def test_kpi_means_match_the_source_at_scale(at_scale: pd.DataFrame, column: str) -> None:
    assert at_scale[column].mean() == pytest.approx(SOURCE_AT_SCALE[column], rel=MEAN_TOLERANCE)


def test_first_alarm_baseline_matches_the_source_at_scale(at_scale: pd.DataFrame) -> None:
    scores = FirstAlarm().root_score(at_scale)
    assert top_k(scores, at_scale, 1) == pytest.approx(
        SOURCE_AT_SCALE["root_at_first_alarm"], abs=SHARE_TOLERANCE
    )


def test_model_matches_the_evidence_setup() -> None:
    data = scenario_off_incidents(500)
    train, test = next(
        GroupShuffleSplit(1, test_size=0.2, random_state=SEED).split(data, groups=data[INCIDENT])
    )
    model = fit_model(data.iloc[train], seed=SEED)
    holdout = data.iloc[test]
    assert top_k(model.root_score(holdout), holdout, 1) == pytest.approx(
        EVIDENCE_MODEL_TOP1, abs=MODEL_TOLERANCE
    )
