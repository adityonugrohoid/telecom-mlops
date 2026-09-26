"""The tests every generator carries (docs/generators.md), for anomaly."""

from datetime import date, datetime, timedelta

import pandas as pd
import pytest
import time_machine
from scipy.stats import ks_2samp
from telecom_ml_anomaly import generator
from telecom_ml_anomaly.features import KPIS
from telecom_ml_anomaly.usecase import AnomalyUseCase
from telecom_ml_core.contract import LABEL_RELEASE, Scenario

USECASE = AnomalyUseCase()
SCENARIO = USECASE.scenario
OFF = SCENARIO.without_events()


def day(index: int) -> date:
    return SCENARIO.start + timedelta(days=index)


def days(first: int, count: int, scenario: Scenario) -> pd.DataFrame:
    return pd.concat(
        [generator.generate(day(first + i), scenario).data for i in range(count)],
        ignore_index=True,
    )


def normal(data: pd.DataFrame) -> pd.DataFrame:
    return data[data["label_anomaly"] == 0]


def moved_cells() -> set[str]:
    cells = generator.cell_profiles(SCENARIO.base_seed)
    return set(cells.loc[cells["growth_rank"] < 0.2, "cell_id"])


def test_same_seed_and_day_give_identical_data_and_days_differ() -> None:
    first = generator.generate(day(5), SCENARIO).data
    pd.testing.assert_frame_equal(first, generator.generate(day(5), SCENARIO).data)
    assert not first[KPIS].equals(generator.generate(day(6), SCENARIO).data[KPIS])


def test_output_does_not_depend_on_the_clock() -> None:
    generator.cell_profiles.cache_clear()
    with time_machine.travel(datetime(2031, 7, 4, 3, 0), tick=False):
        then = generator.generate(day(5), SCENARIO).data
    generator.cell_profiles.cache_clear()
    with time_machine.travel(datetime(2024, 2, 29, 22, 0), tick=False):
        now = generator.generate(day(5), SCENARIO).data
    pd.testing.assert_frame_equal(then, now)


def test_cells_are_the_same_every_day() -> None:
    a = generator.generate(day(0), SCENARIO).data[["cell_id", "cell_type", "area_type"]]
    b = generator.generate(day(90), SCENARIO).data[["cell_id", "cell_type", "area_type"]]
    pd.testing.assert_frame_equal(a.drop_duplicates(), b.drop_duplicates())


def test_scenario_off_day_0_and_day_179_share_a_distribution() -> None:
    early, late = normal(days(0, 7, OFF)), normal(days(173, 7, OFF))
    for column in KPIS:
        assert ks_2samp(early[column], late[column]).pvalue > 0.001, column
    assert days(173, 7, OFF)["label_anomaly"].mean() == pytest.approx(0.05)


def test_manifest_lists_exactly_the_active_events() -> None:
    def names(index: int) -> list[str]:
        return [e.name for e in generator.generate(day(index), SCENARIO).manifest.events]

    events = [e.name for e in SCENARIO.events]
    assert names(49) == []
    assert names(50) == events[:1]
    assert names(110) == events[:2]
    assert names(155) == events


def test_outages_appear_at_day_110_at_one_percent() -> None:
    before, after = days(100, 10, SCENARIO), days(110, 10, SCENARIO)
    assert not (before["anomaly_type"] == generator.NEW_TYPE).any()
    share = (after["anomaly_type"] == generator.NEW_TYPE).fillna(False).mean()
    assert share == pytest.approx(0.01, abs=0.001)
    assert after["label_anomaly"].mean() == pytest.approx(0.06, abs=0.001)


def test_benign_event_lifts_weekend_traffic_only() -> None:
    def traffic(data: pd.DataFrame, weekend: bool) -> float:
        rows = normal(data)
        on_weekend = pd.to_datetime(rows["timestamp"]).dt.dayofweek >= 5
        others = ~rows["cell_id"].isin(moved_cells())
        return float(rows.loc[(on_weekend == weekend) & others, "traffic_load_gb"].mean())

    before, after = days(141, 14, SCENARIO), days(162, 14, SCENARIO)
    # Traffic follows 0.3 + 0.7 x congestion, so the 0.8 to 1.0 weekend factor lifts it ~12%.
    assert traffic(after, weekend=True) / traffic(before, weekend=True) > 1.08
    assert traffic(after, weekend=False) / traffic(before, weekend=False) == pytest.approx(
        1.0, abs=0.03
    )
    assert generator.params_on(day(155), SCENARIO)["weekend_factor"] == 1.0


def test_labels_release_the_next_day_and_batches_pass_the_schema() -> None:
    batch = generator.generate(day(3), SCENARIO)
    assert (batch.data[LABEL_RELEASE] == pd.Timestamp(day(4))).all()
    assert pd.to_datetime(batch.data["timestamp"]).dt.date.eq(day(3)).all()
    USECASE.schema().validate(batch.data)
