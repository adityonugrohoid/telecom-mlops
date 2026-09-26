"""The tests every generator carries (docs/generators.md), for root-cause."""

from datetime import date, datetime, timedelta

import pandas as pd
import pytest
import time_machine
from scipy.stats import chisquare, ks_2samp
from telecom_ml_core.contract import LABEL_RELEASE, Scenario
from telecom_ml_root_cause import generator
from telecom_ml_root_cause.features import NUMERIC
from telecom_ml_root_cause.usecase import RootCauseUseCase

USECASE = RootCauseUseCase()
SCENARIO = USECASE.scenario
OFF = SCENARIO.without_events()


def day(index: int) -> date:
    return SCENARIO.start + timedelta(days=index)


def days(first: int, count: int, scenario: Scenario) -> pd.DataFrame:
    return pd.concat(
        [generator.generate(day(first + i), scenario).data for i in range(count)],
        ignore_index=True,
    )


def roots(data: pd.DataFrame) -> pd.DataFrame:
    return data[data["is_root_cause"] == 1]


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
    early, late = days(0, 10, OFF), days(170, 10, OFF)
    for column in NUMERIC:
        assert ks_2samp(early[column], late[column]).pvalue > 0.001, column
    for column in ("event_type", "alarm_severity"):
        categories = sorted(set(early[column]) | set(late[column]))
        observed = late[column].value_counts().reindex(categories, fill_value=0)
        expected = early[column].value_counts(normalize=True).reindex(categories) * len(late)
        assert chisquare(observed, expected).pvalue > 0.001, column
    assert "power_supply" not in set(early["event_type"]) | set(late["event_type"])


def test_events_move_their_parameters_by_the_stated_amount_and_nothing_else() -> None:
    base = generator.BASE_PARAMS
    assert generator.params_on(day(29), SCENARIO) == base
    assert generator.params_on(day(30), SCENARIO)["config_error_late_root_share"] == 0.5
    assert generator.params_on(day(89), SCENARIO)["power_supply_share"] == 0.0
    assert generator.params_on(day(90), SCENARIO)["power_supply_share"] == 0.1
    assert generator.params_on(day(149), SCENARIO)["incident_rate"] == base["incident_rate"]
    final = generator.params_on(day(179), SCENARIO)
    assert final["incident_rate"] == pytest.approx(base["incident_rate"] * 1.4)
    assert {k for k in base if final[k] != base[k]} == set(base)


def test_firmware_rollout_moves_config_error_roots_off_the_first_alarm() -> None:
    def first_alarm_share(data: pd.DataFrame) -> float:
        config = roots(data)[roots(data)["event_type"] == "config_error"]
        return float((config["event_sequence_position"] == 0).mean())

    # config_error roots only; power_supply from day 90 does not touch them.
    before, after = days(0, 30, SCENARIO), days(30, 150, SCENARIO)
    assert first_alarm_share(before) == pytest.approx(0.7, abs=0.07)
    assert first_alarm_share(after) == pytest.approx(0.7 * 0.5, abs=0.05)
    other = roots(after)[~roots(after)["event_type"].isin(["config_error", "power_supply"])]
    assert (other["event_sequence_position"] == 0).mean() == pytest.approx(0.7, abs=0.05)


def test_power_supply_appears_at_day_90_with_its_signature() -> None:
    assert "power_supply" not in set(days(60, 30, SCENARIO)["event_type"])
    after = days(90, 30, SCENARIO)
    power = roots(after)[roots(after)["event_type"] == "power_supply"]
    share = len(power) / after["incident_id"].nunique()
    assert share == pytest.approx(0.1, abs=0.03)
    assert power["event_sequence_position"].between(2, 4).all()
    assert power["affected_cells"].between(3, 8).all()
    assert (power["alarm_severity"].isin(["major", "minor"])).all()
    normal = roots(after)[roots(after)["event_type"] != "power_supply"]
    assert power["throughput_delta"].abs().mean() < 0.4 * normal["throughput_delta"].abs().mean()
    cascade = after[after["is_root_cause"] == 0]
    assert "power_supply" not in set(cascade["event_type"])


def test_benign_event_raises_volume_by_40_percent() -> None:
    def per_day(first: int) -> float:
        return float(
            pd.Series(
                [
                    generator.generate(day(first + i), SCENARIO).data["incidents_in_day"].iloc[0]
                    for i in range(20)
                ]
            ).mean()
        )

    assert per_day(160) / per_day(120) == pytest.approx(1.4, abs=0.15)


def test_manifest_lists_exactly_the_active_events() -> None:
    def names(index: int) -> list[str]:
        return [e.name for e in generator.generate(day(index), SCENARIO).manifest.events]

    assert names(29) == []
    assert names(30) == ["firmware_rollout"]
    assert names(90) == ["firmware_rollout", "power_supply_faults"]
    assert names(150) == ["firmware_rollout", "power_supply_faults", "incident_volume_growth"]


def test_labels_release_per_incident_1_to_7_days_later_and_batches_pass_the_schema() -> None:
    batch = generator.generate(day(3), SCENARIO)
    delay = (pd.to_datetime(batch.data[LABEL_RELEASE]) - pd.Timestamp(day(3))).dt.days
    assert delay.between(1, 7).all()
    assert (batch.data.groupby("incident_id")[LABEL_RELEASE].nunique() == 1).all()
    assert delay.nunique() > 1
    stamps = pd.to_datetime(batch.data["timestamp"])
    assert stamps.min() >= pd.Timestamp(day(3))
    assert stamps.max() < pd.Timestamp(day(4))
    USECASE.schema().validate(batch.data)
