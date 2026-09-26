"""Day-0 anchor (docs/generators.md rule 3).

Source: capacity-forecasting at 82492521. Its evidence (MAPE 14.51) is reproduced exactly on
current libraries, but its features include the previous hour's traffic: a 1-hour nowcast.
The loop forecasts 7 days ahead, so the anchor is the same data with inputs at least 7 days
old, measured in a fresh run of the source generator (seed 42, 60 cells, 30 days, the last
20% of time as test): LightGBM MAPE 16.05, seasonal naive 20.98. Traffic mean by area:
rural 2.645, suburban 7.148, urban 13.324.

Here: 30 scenario-off days, the same chronological split. Tolerance: each MAPE within 2
points, each area's mean traffic within 5%.
"""

from datetime import timedelta

import pandas as pd
import pytest
from telecom_ml_capacity import generator
from telecom_ml_capacity.model import TARGET, SeasonalNaive, fit_forecaster, mape_pct
from telecom_ml_capacity.usecase import CapacityUseCase

SOURCE_SEVEN_DAY = {"lightgbm": 16.05, "naive": 20.98}
SOURCE_AREA_MEAN = {"rural": 2.645, "suburban": 7.148, "urban": 13.324}
SEED = 42


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    scenario = CapacityUseCase.scenario.without_events()
    return pd.concat(
        [generator.generate(scenario.start + timedelta(days=i), scenario).data for i in range(30)],
        ignore_index=True,
    )


@pytest.mark.parametrize("area", sorted(SOURCE_AREA_MEAN))
def test_area_traffic_matches_the_source(data: pd.DataFrame, area: str) -> None:
    mean = data.loc[data["area_type"] == area, TARGET].mean()
    assert mean == pytest.approx(SOURCE_AREA_MEAN[area], rel=0.05)


def test_seven_day_forecasts_match_the_source(data: pd.DataFrame) -> None:
    cut = data["timestamp"].sort_values().iloc[int(len(data) * 0.8)]
    train, test = data[data["timestamp"] < cut], data[data["timestamp"] >= cut]
    truth = test[TARGET].to_numpy()
    lightgbm = mape_pct(truth, fit_forecaster(train, seed=SEED).predict(test))
    naive = mape_pct(truth, SeasonalNaive().predict(test))
    assert lightgbm == pytest.approx(SOURCE_SEVEN_DAY["lightgbm"], abs=2.0)
    assert naive == pytest.approx(SOURCE_SEVEN_DAY["naive"], abs=2.0)
