"""Day-0 anchor (docs/generators.md rule 3).

Source: qoe-prediction at 266b8524, evidence/baseline_metrics.json (2026-09-19): 10,000
sessions, random 80/20 split with seed 42, LightGBM MAE 0.3577, mean-MOS baseline MAE 0.5461.

At scale the source generator (seed 42, 10,000 sessions) gives MOS mean 3.874 and standard
deviation 0.704, and its mean baseline's MAE over all sessions is 0.559: the evidence split's
0.546 sits below that by split noise.

Here: the first 10,000 sessions of 7 scenario-off days, the same split. Tolerance: model MAE
within 0.02, baseline MAE within 0.03, MOS mean and standard deviation within 0.02.
"""

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from telecom_ml_qoe import generator
from telecom_ml_qoe.features import TARGET
from telecom_ml_qoe.model import fit_baseline, fit_model
from telecom_ml_qoe.usecase import QoEUseCase

EVIDENCE = {"model_mae": 0.3577, "baseline_mae": 0.5461}
SOURCE_AT_SCALE = {"mos_mean": 3.874, "mos_std": 0.704}
SEED = 42


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    scenario = QoEUseCase.scenario.without_events()
    sessions = pd.concat(
        [generator.generate(scenario.start + timedelta(days=i), scenario).data for i in range(7)],
        ignore_index=True,
    )
    return sessions.iloc[:10_000]


def mae(model: object, frame: pd.DataFrame) -> float:
    return float(np.mean(np.abs(model.predict(frame) - frame[TARGET].to_numpy())))  # type: ignore[attr-defined]


def test_mos_distribution_matches_the_source(data: pd.DataFrame) -> None:
    assert data[TARGET].mean() == pytest.approx(SOURCE_AT_SCALE["mos_mean"], abs=0.02)
    assert data[TARGET].std() == pytest.approx(SOURCE_AT_SCALE["mos_std"], abs=0.02)


def test_model_and_baseline_match_the_evidence(data: pd.DataFrame) -> None:
    train, test = train_test_split(data.index, test_size=0.2, random_state=SEED)
    model = fit_model(data.loc[train], seed=SEED)
    baseline = fit_baseline(data.loc[train])
    assert mae(model, data.loc[test]) == pytest.approx(EVIDENCE["model_mae"], abs=0.02)
    assert mae(baseline, data.loc[test]) == pytest.approx(EVIDENCE["baseline_mae"], abs=0.03)
