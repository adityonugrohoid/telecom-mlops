from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest
from telecom_ml_core.contract import LABEL_RELEASE, Batch, Event, Scenario, rng_for_day
from toy import START, TOY_SCENARIO, ToyUseCase


def test_event_ramps_linearly_then_holds() -> None:
    event = Event("e", 10, 4, "covariate", {"p": 1.0}, False, None)
    assert [event.strength(d) for d in (9, 10, 11, 13, 50)] == [0.0, 0.25, 0.5, 1.0, 1.0]


def test_event_with_duration_switches_off() -> None:
    event = Event("holiday", 10, 0, "trend", {"p": 1.8}, True, 7)
    assert [event.strength(d) for d in (9, 10, 16, 17)] == [0.0, 1.0, 1.0, 0.0]


def test_event_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="unknown kind"):
        Event("e", 0, 0, "seasonal", {}, False, None)  # type: ignore[arg-type]


def test_scenario_value_moves_by_stated_amount_on_the_ramp() -> None:
    scenario = Scenario(
        START, 180, 1, (Event("rise", 60, 14, "covariate", {"charge": 1.15}, False, None),)
    )
    before = scenario.value("charge", 1.0, START + timedelta(days=59))
    mid = scenario.value("charge", 1.0, START + timedelta(days=66))
    end = scenario.value("charge", 1.0, START + timedelta(days=73))
    assert (before, end) == (1.0, pytest.approx(1.15))
    assert mid == pytest.approx(1.0 + 0.15 * 7 / 14)
    assert scenario.value("other", 2.0, START + timedelta(days=100)) == 2.0


def test_manifest_lists_exactly_the_active_events() -> None:
    names = [e.name for e in TOY_SCENARIO.manifest(START + timedelta(days=4)).events]
    assert names == []
    names = [e.name for e in TOY_SCENARIO.manifest(START + timedelta(days=5)).events]
    assert names == ["threshold_moves"]
    names = [e.name for e in TOY_SCENARIO.manifest(START + timedelta(days=20)).events]
    assert names == ["threshold_moves", "z_shift"]


def test_same_seed_and_day_give_identical_draws_different_days_differ() -> None:
    day = date(2026, 3, 1)
    a = rng_for_day(1, 2, day).random(5)
    b = rng_for_day(1, 2, day).random(5)
    c = rng_for_day(1, 2, day + timedelta(days=1)).random(5)
    assert (a == b).all()
    assert not (a == c).all()


def test_toy_generation_is_deterministic_by_date() -> None:
    usecase = ToyUseCase()
    day = START + timedelta(days=3)
    pd.testing.assert_frame_equal(
        usecase.generate(day, TOY_SCENARIO).data, usecase.generate(day, TOY_SCENARIO).data
    )


def test_scenario_loads_from_yaml(tmp_path: Path) -> None:
    path = tmp_path / "scenario.yaml"
    path.write_text(
        "start: 2026-01-01\ndays: 180\nbase_seed: 42\nevents:\n"
        "  - {name: holiday, day: 130, ramp_days: 0, kind: trend, params: {traffic: 1.8},"
        " benign: true, duration_days: 7}\n"
    )
    scenario = Scenario.load(path)
    assert scenario.start == date(2026, 1, 1)
    assert scenario.events[0] == Event("holiday", 130, 0, "trend", {"traffic": 1.8}, True, 7)
    assert scenario.without_events().events == ()


def test_scenario_load_names_the_missing_key(tmp_path: Path) -> None:
    path = tmp_path / "scenario.yaml"
    path.write_text("start: 2026-01-01\ndays: 180\nevents: []\n")
    with pytest.raises(ValueError, match="base_seed"):
        Scenario.load(path)


def test_batch_requires_label_release_column() -> None:
    manifest = TOY_SCENARIO.manifest(START)
    with pytest.raises(ValueError, match=LABEL_RELEASE):
        Batch(day=START, data=pd.DataFrame({"x": [1.0]}), manifest=manifest)
