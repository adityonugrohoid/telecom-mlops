"""The tests every generator carries (docs/generators.md), for netopt, plus the rollout rules."""

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
import time_machine
from telecom_ml_core.contract import LABEL_RELEASE
from telecom_ml_netopt import generator
from telecom_ml_netopt.generator import Environment, initial_state
from telecom_ml_netopt.policy import RulePolicy, run_episode
from telecom_ml_netopt.usecase import NetOptUseCase

USECASE = NetOptUseCase()
SCENARIO = USECASE.scenario
OFF = SCENARIO.without_events()


def day(index: int) -> date:
    return SCENARIO.start + timedelta(days=index)


def env_on(index: int) -> Environment:
    return Environment.from_row(generator.generate(day(index), SCENARIO).data.iloc[0])


def test_same_seed_gives_the_same_episode_and_seeds_differ() -> None:
    env = env_on(0)
    assert run_episode(RulePolicy(), env, 7) == run_episode(RulePolicy(), env, 7)
    assert run_episode(RulePolicy(), env, 7) != run_episode(RulePolicy(), env, 8)


def test_output_does_not_depend_on_the_clock() -> None:
    with time_machine.travel(datetime(2031, 7, 4, 3, 0), tick=False):
        then = (
            generator.generate(day(90), SCENARIO).data,
            run_episode(RulePolicy(), env_on(90), 3),
        )
    with time_machine.travel(datetime(2024, 2, 29, 22, 0), tick=False):
        now = (generator.generate(day(90), SCENARIO).data, run_episode(RulePolicy(), env_on(90), 3))
    pd.testing.assert_frame_equal(then[0], now[0])
    assert then[1] == now[1]


def test_scenario_off_day_0_and_day_179_are_the_same_environment() -> None:
    first = generator.generate(day(0), OFF).data.drop(columns=["day", LABEL_RELEASE])
    last = generator.generate(day(179), OFF).data.drop(columns=["day", LABEL_RELEASE])
    pd.testing.assert_frame_equal(first, last)


def test_events_move_their_parameters_by_the_stated_amount_and_nothing_else() -> None:
    base = generator.BASE_PARAMS
    assert generator.params_on(day(59), SCENARIO) == base
    assert generator.params_on(day(70), SCENARIO)["load_mult"] == pytest.approx(1 + 0.2 * 11 / 21)
    assert generator.params_on(day(80), SCENARIO)["load_mult"] == pytest.approx(1.2)
    assert generator.params_on(day(120), SCENARIO)["outage_share"] == 0.1
    assert generator.params_on(day(160), SCENARIO)["sinr_noise_sd"] == 2.0
    assert set(generator.params_on(day(179), SCENARIO)) == set(base)


def test_outage_hits_its_share_of_episodes_with_its_stated_boost() -> None:
    env = env_on(130)
    hits = []
    for seed in range(4_000):
        _, outage = initial_state(np.random.default_rng(seed), env)
        hits.append(outage)
    assert np.mean(hits) == pytest.approx(0.1, abs=0.015)


def test_outage_draws_leave_the_base_states_unchanged() -> None:
    before, after = env_on(100), env_on(130)
    for seed in range(20):
        a, _ = initial_state(np.random.default_rng(seed), before)
        b, outage = initial_state(np.random.default_rng(seed), after)
        if not outage:
            assert a["sinr"] == b["sinr"]
            assert a["throughput"] == b["throughput"]


def test_manifest_lists_exactly_the_active_events() -> None:
    def names(index: int) -> list[str]:
        return [e.name for e in generator.generate(day(index), SCENARIO).manifest.events]

    events = [e.name for e in SCENARIO.events]
    assert names(59) == []
    assert names(60) == events[:1]
    assert names(120) == events[:2]
    assert names(160) == events


def test_one_environment_row_a_day_released_the_same_day() -> None:
    batch = generator.generate(day(3), SCENARIO)
    assert len(batch.data) == 1
    assert batch.data[LABEL_RELEASE].iloc[0] == pd.Timestamp(day(3))
    USECASE.schema().validate(batch.data)
