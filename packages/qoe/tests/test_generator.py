"""The tests every generator carries (docs/generators.md), for QoE."""

from datetime import date, datetime, timedelta

import pandas as pd
import pytest
import time_machine
from scipy.stats import ks_2samp
from telecom_ml_core.contract import LABEL_RELEASE, Scenario
from telecom_ml_qoe import generator
from telecom_ml_qoe.features import NUMERIC
from telecom_ml_qoe.usecase import QoEUseCase

USECASE = QoEUseCase()
SCENARIO = USECASE.scenario
OFF = SCENARIO.without_events()


def day(index: int) -> date:
    return SCENARIO.start + timedelta(days=index)


def days(first: int, count: int, scenario: Scenario) -> pd.DataFrame:
    return pd.concat(
        [generator.generate(day(first + i), scenario).data for i in range(count)],
        ignore_index=True,
    )


def test_same_seed_and_day_give_identical_data_and_days_differ() -> None:
    first = generator.generate(day(5), SCENARIO).data
    pd.testing.assert_frame_equal(first, generator.generate(day(5), SCENARIO).data)
    assert not first[NUMERIC].equals(generator.generate(day(6), SCENARIO).data[NUMERIC])


def test_output_does_not_depend_on_the_clock() -> None:
    with time_machine.travel(datetime(2031, 7, 4, 3, 0), tick=False):
        then = generator.generate(day(5), SCENARIO).data
    with time_machine.travel(datetime(2024, 2, 29, 22, 0), tick=False):
        now = generator.generate(day(5), SCENARIO).data
    pd.testing.assert_frame_equal(then, now)


def test_scenario_off_day_0_and_day_179_share_a_distribution() -> None:
    early, late = days(0, 7, OFF), days(173, 7, OFF)
    for column in [*NUMERIC, "mos_score"]:
        assert ks_2samp(early[column], late[column]).pvalue > 0.001, column
    assert "cloud_gaming" not in set(late["app_type"])


def test_events_move_their_parameters_by_the_stated_amount_and_nothing_else() -> None:
    base = generator.BASE_PARAMS
    assert generator.params_on(day(39), SCENARIO) == base
    mid = generator.params_on(day(54), SCENARIO)
    assert mid["high_device_share"] == pytest.approx(0.3 + 0.2 * 15 / 30)
    assert generator.params_on(day(69), SCENARIO)["high_device_share"] == pytest.approx(0.5)
    assert generator.params_on(day(90), SCENARIO)["video_codec_gain"] == 3.0
    assert generator.params_on(day(150), SCENARIO)["cloud_gaming_share"] == 0.1
    final = generator.params_on(day(179), SCENARIO)
    moved = {"high_device_share", "video_codec_gain", "cloud_gaming_share"}
    assert {k for k in base if final[k] != base[k]} == moved


def test_device_mix_reaches_half_high_end() -> None:
    before, after = days(30, 5, SCENARIO), days(75, 5, SCENARIO)
    assert (before["device_class"] == "high").mean() == pytest.approx(0.3, abs=0.02)
    assert (after["device_class"] == "high").mean() == pytest.approx(0.5, abs=0.02)


def test_codec_change_lifts_video_mos_only() -> None:
    before, after = days(80, 7, SCENARIO), days(95, 7, SCENARIO)

    def mean_mos(data: pd.DataFrame, video: bool) -> float:
        rows = data[(data["app_type"] == "video_streaming") == video]
        return float(rows["mos_score"].mean())

    assert mean_mos(after, video=True) > mean_mos(before, video=True) + 0.1
    assert mean_mos(after, video=False) == pytest.approx(mean_mos(before, video=False), abs=0.03)


def test_cloud_gaming_arrives_at_day_150_with_its_signature() -> None:
    assert "cloud_gaming" not in set(days(140, 10, SCENARIO)["app_type"])
    after = days(150, 7, SCENARIO)
    cloud = after["app_type"] == "cloud_gaming"
    assert cloud.mean() == pytest.approx(0.1, abs=0.01)
    assert after.loc[cloud, "mos_score"].mean() < after.loc[~cloud, "mos_score"].mean() - 0.5


def test_manifest_lists_exactly_the_active_events() -> None:
    def names(index: int) -> list[str]:
        return [e.name for e in generator.generate(day(index), SCENARIO).manifest.events]

    events = [e.name for e in SCENARIO.events]
    assert names(39) == []
    assert names(40) == events[:1]
    assert names(90) == events[:2]
    assert names(150) == events


def test_labels_release_the_same_day_and_batches_pass_the_schema() -> None:
    batch = generator.generate(day(3), SCENARIO)
    assert (batch.data[LABEL_RELEASE] == pd.Timestamp(day(3))).all()
    assert pd.to_datetime(batch.data["timestamp"]).dt.date.eq(day(3)).all()
    assert len(batch.data) == generator.SESSIONS_PER_DAY
    USECASE.schema().validate(batch.data)
