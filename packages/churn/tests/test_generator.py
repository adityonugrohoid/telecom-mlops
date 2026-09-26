"""The tests every generator carries (docs/generators.md), for churn."""

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
import time_machine
from scipy.stats import chisquare, ks_2samp
from telecom_ml_churn import generator
from telecom_ml_churn.features import CATEGORIES, NUMERIC
from telecom_ml_churn.usecase import ChurnUseCase
from telecom_ml_core.contract import LABEL_RELEASE

USECASE = ChurnUseCase()
SCENARIO = USECASE.scenario
OFF = SCENARIO.without_events()
DAY0 = SCENARIO.start


def day(index: int) -> date:
    return SCENARIO.start + timedelta(days=index)


def days(first: int, count: int, scenario=OFF) -> pd.DataFrame:  # type: ignore[no-untyped-def]
    return pd.concat(
        [generator.generate(day(first + i), scenario).data for i in range(count)],
        ignore_index=True,
    )


def test_same_seed_and_day_give_identical_data_and_days_differ() -> None:
    first = generator.generate(day(5), SCENARIO).data
    pd.testing.assert_frame_equal(first, generator.generate(day(5), SCENARIO).data)
    other = generator.generate(day(6), SCENARIO).data
    assert not first[NUMERIC].equals(other[NUMERIC])


def test_output_does_not_depend_on_the_clock() -> None:
    generator.intercept.cache_clear()
    with time_machine.travel(datetime(2031, 7, 4, 3, 0), tick=False):
        then = generator.generate(day(5), SCENARIO).data
    generator.intercept.cache_clear()
    with time_machine.travel(datetime(2024, 2, 29, 22, 0), tick=False):
        now = generator.generate(day(5), SCENARIO).data
    pd.testing.assert_frame_equal(then, now)


def test_scenario_off_day_0_and_day_179_share_a_distribution() -> None:
    early, late = days(0, 5), days(175, 5)
    for column in NUMERIC:
        assert ks_2samp(early[column], late[column]).pvalue > 0.001, column
    for column, categories in CATEGORIES.items():
        observed = late[column].value_counts().reindex(categories, fill_value=0)
        expected = early[column].value_counts(normalize=True).reindex(categories) * len(late)
        assert chisquare(observed, expected).pvalue > 0.001, column
    assert abs(early["is_churned"].mean() - late["is_churned"].mean()) < 0.03


def test_events_move_their_parameters_by_the_stated_amount_and_nothing_else() -> None:
    base = generator.BASE_PARAMS
    assert generator.params_on(day(59), SCENARIO) == base
    mid = generator.params_on(day(66), SCENARIO)
    assert mid["mtm_charge_mult"] == pytest.approx(1.0 + 0.15 * 7 / 14)
    assert mid["charge_coef"] == pytest.approx(0.4 + 1.6 * 7 / 14)
    after = generator.params_on(day(73), SCENARIO)
    assert (after["mtm_charge_mult"], after["charge_coef"]) == (pytest.approx(1.15), 2.0)
    assert generator.params_on(day(169), SCENARIO)["share_5g"] == pytest.approx(0.7)
    moved = {"mtm_charge_mult", "charge_coef", "share_5g"}
    final = generator.params_on(day(179), SCENARIO)
    assert {k for k in base if final[k] != base[k]} == moved


def test_price_rise_raises_month_to_month_charges_by_15_percent() -> None:
    def mtm_mean(data: pd.DataFrame) -> float:
        return float(data.loc[data.contract_type == "month-to-month", "monthly_charges"].mean())

    before, after = days(40, 10, SCENARIO), days(80, 10, SCENARIO)
    assert mtm_mean(after) / mtm_mean(before) == pytest.approx(1.15, abs=0.02)
    other = before.contract_type != "month-to-month"
    assert before.loc[other, "monthly_charges"].mean() == pytest.approx(
        after.loc[after.contract_type != "month-to-month", "monthly_charges"].mean(), abs=1.5
    )
    assert after["is_churned"].mean() > before["is_churned"].mean() + 0.03


def test_benign_event_shifts_the_5g_share() -> None:
    before, after = days(140, 5, SCENARIO), days(172, 5, SCENARIO)
    assert (before.network_type == "5G").mean() == pytest.approx(0.4, abs=0.03)
    assert (after.network_type == "5G").mean() == pytest.approx(0.7, abs=0.03)


def test_manifest_lists_exactly_the_active_events() -> None:
    def names(index: int) -> list[str]:
        return [e.name for e in generator.generate(day(index), SCENARIO).manifest.events]

    assert names(59) == []
    assert names(60) == ["price_rise_charges", "price_rise_sensitivity"]
    assert names(150) == ["price_rise_charges", "price_rise_sensitivity", "fiveg_share_growth"]
    benign = [e for e in generator.generate(day(160), SCENARIO).manifest.events if e.benign]
    assert [(e.name, e.strength) for e in benign] == [("fiveg_share_growth", pytest.approx(0.55))]


def test_labels_release_after_the_30_day_window_and_batches_pass_the_schema() -> None:
    batch = generator.generate(day(3), SCENARIO)
    assert (batch.data[LABEL_RELEASE] == pd.Timestamp(day(33))).all()
    stamps = pd.to_datetime(batch.data["timestamp"])
    assert stamps.min() >= pd.Timestamp(day(3) - timedelta(days=30))
    assert stamps.max() < pd.Timestamp(day(3))
    USECASE.schema().validate(batch.data)
    assert len(batch.data) == generator.CUSTOMERS_PER_DAY
    assert np.isfinite(generator.intercept(SCENARIO.base_seed))
