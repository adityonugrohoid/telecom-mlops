"""The tests every generator carries (docs/generators.md), for capacity."""

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
import time_machine
from scipy.stats import ks_2samp
from telecom_ml_capacity import generator
from telecom_ml_capacity.model import TARGET
from telecom_ml_capacity.usecase import CapacityUseCase
from telecom_ml_core.contract import LABEL_RELEASE, Scenario

USECASE = CapacityUseCase()
SCENARIO = USECASE.scenario
OFF = SCENARIO.without_events()


def day(index: int) -> date:
    return SCENARIO.start + timedelta(days=index)


def days(first: int, count: int, scenario: Scenario) -> pd.DataFrame:
    return pd.concat(
        [generator.generate(day(first + i), scenario).data for i in range(count)],
        ignore_index=True,
    )


def cells(flag: str) -> set[str]:
    profiles = generator.cell_profiles(SCENARIO.base_seed)
    return set(profiles.loc[profiles[flag], "cell_id"])


def test_same_seed_and_day_give_identical_data_and_days_differ() -> None:
    first = generator.generate(day(5), SCENARIO).data
    pd.testing.assert_frame_equal(first, generator.generate(day(5), SCENARIO).data)
    assert not first[TARGET].equals(generator.generate(day(6), SCENARIO).data[TARGET])


def test_output_does_not_depend_on_the_clock() -> None:
    generator.cell_profiles.cache_clear()
    generator._day.cache_clear()
    with time_machine.travel(datetime(2031, 7, 4, 3, 0), tick=False):
        then = generator.generate(day(5), SCENARIO).data
    generator.cell_profiles.cache_clear()
    generator._day.cache_clear()
    with time_machine.travel(datetime(2024, 2, 29, 22, 0), tick=False):
        now = generator.generate(day(5), SCENARIO).data
    pd.testing.assert_frame_equal(then, now)


def test_lags_are_the_same_cell_and_hour_days_earlier() -> None:
    today = generator.generate(day(20), SCENARIO).data
    week_ago = generator.generate(day(13), SCENARIO).data
    fortnight_ago = generator.generate(day(6), SCENARIO).data
    np.testing.assert_array_equal(today["lag_7d"], week_ago[TARGET])
    np.testing.assert_array_equal(today["lag_14d"], fortnight_ago[TARGET])
    earlier = np.mean(
        [generator.generate(day(20 - k), SCENARIO).data[TARGET] for k in range(7, 14)], axis=0
    )
    np.testing.assert_allclose(today["mean_same_hour_7_13d"], earlier)


def test_scenario_off_day_0_and_day_179_share_a_distribution() -> None:
    early, late = days(0, 14, OFF), days(166, 14, OFF)
    assert ks_2samp(early[TARGET], late[TARGET]).pvalue > 0.001
    assert late[TARGET].mean() == pytest.approx(early[TARGET].mean(), rel=0.05)


def test_events_move_their_parameters_by_the_stated_amount_and_nothing_else() -> None:
    assert generator.params_on(day(-1), SCENARIO) == generator.BASE_PARAMS
    assert generator.params_on(day(179), SCENARIO)["growth_level"] == pytest.approx(1.12)
    assert generator.params_on(day(89), SCENARIO)["growth_level"] == pytest.approx(1.06)
    assert generator.params_on(day(69), SCENARIO)["fast_growth_extra"] == 1.0
    assert generator.params_on(day(179), SCENARIO)["fast_growth_extra"] == pytest.approx(1.11)
    assert generator.params_on(day(129), SCENARIO)["holiday_mult"] == 1.0
    assert generator.params_on(day(130), SCENARIO)["holiday_mult"] == 1.8
    assert generator.params_on(day(136), SCENARIO)["holiday_mult"] == 1.8
    assert generator.params_on(day(137), SCENARIO)["holiday_mult"] == 1.0


def test_growth_raises_traffic_and_fast_cells_grow_faster() -> None:
    def mean(data: pd.DataFrame, fast: bool) -> float:
        return float(data.loc[data["cell_id"].isin(cells("fast_growth")) == fast, TARGET].mean())

    early, late = days(0, 28, SCENARIO), days(152, 28, SCENARIO)
    slow_growth = mean(late, fast=False) / mean(early, fast=False)
    fast_growth = mean(late, fast=True) / mean(early, fast=True)
    assert slow_growth == pytest.approx(1.10, abs=0.04)
    assert fast_growth > slow_growth + 0.05


def test_holiday_week_lifts_its_cells_for_seven_days_only() -> None:
    holiday_cells = cells("holiday")

    def ratio(first: int) -> float:
        data = days(first, 7, SCENARIO)
        on = data["cell_id"].isin(holiday_cells)
        return float(data.loc[on, TARGET].mean() / data.loc[~on, TARGET].mean())

    before, during, after = ratio(123), ratio(130), ratio(137)
    assert during / before == pytest.approx(1.8, abs=0.1)
    assert after / before == pytest.approx(1.0, abs=0.08)


def test_manifest_lists_exactly_the_active_events() -> None:
    def names(index: int) -> list[str]:
        return [e.name for e in generator.generate(day(index), SCENARIO).manifest.events]

    assert names(0) == ["steady_growth"]
    assert names(70) == ["steady_growth", "fast_growth_cells"]
    assert names(130) == ["steady_growth", "fast_growth_cells", "holiday_week"]
    assert names(137) == ["steady_growth", "fast_growth_cells"]


def test_labels_release_the_same_day_and_batches_pass_the_schema() -> None:
    batch = generator.generate(day(3), SCENARIO)
    assert (batch.data[LABEL_RELEASE] == pd.Timestamp(day(3))).all()
    assert len(batch.data) == generator.N_CELLS * generator.HOURS
    USECASE.schema().validate(batch.data)
