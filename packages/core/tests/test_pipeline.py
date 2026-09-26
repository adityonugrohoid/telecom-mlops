from datetime import timedelta

import pandas as pd
import pandera.errors
import pytest
from telecom_ml_core.contract import LABEL_RELEASE, Batch, Scenario
from telecom_ml_core.pipeline import Components, DataSource, run_loop
from toy import (
    LABEL_DELAY_DAYS,
    START,
    TOY_SCENARIO,
    MeanShiftDrift,
    MemoryRegistry,
    SchemaValidator,
    ToyUseCase,
    WindowEvaluator,
)


def components(registry: MemoryRegistry) -> Components:
    return Components(
        validator=SchemaValidator(),
        drift=MeanShiftDrift(),
        evaluators={"holdout": WindowEvaluator()},
        registry=registry,
    )


def test_released_rows_respect_the_label_delay() -> None:
    source = DataSource(ToyUseCase(), TOY_SCENARIO)
    day = START + timedelta(days=10)
    released = source.released(day, day)
    assert (pd.to_datetime(released[LABEL_RELEASE]) == pd.Timestamp(day)).all()
    assert len(released) == len(source.batch(day - timedelta(days=LABEL_DELAY_DAYS)).data)


def test_loop_logs_one_decision_per_day_and_bootstraps_version_one() -> None:
    registry = MemoryRegistry()
    decisions = run_loop(ToyUseCase(), START, 4, components(registry))
    assert [d.day for d in decisions] == [START + timedelta(days=i) for i in range(4)]
    assert registry.decisions == decisions
    assert decisions[0].live_version == 1
    assert registry.last_day("toy") == START + timedelta(days=3)


def test_concept_event_leads_to_a_promotion() -> None:
    registry = MemoryRegistry()
    decisions = run_loop(ToyUseCase(), START, 20, components(registry))
    before = [d for d in decisions if d.day < START + timedelta(days=5)]
    after = [d for d in decisions if d.day >= START + timedelta(days=5)]
    assert not any(d.promoted for d in before)
    assert any(d.promoted for d in after)
    assert registry.records["toy"].version > 1


def test_benign_shift_is_detected_retrained_and_not_promoted() -> None:
    registry = MemoryRegistry()
    decisions = run_loop(ToyUseCase(), START, 28, components(registry))
    benign = [d for d in decisions if any(e.benign for e in d.manifest.events)]
    assert benign
    assert all(d.drift.detected for d in benign)
    assert any(d.retrained and not d.promoted for d in benign)
    assert not any(d.promoted for d in benign)


def test_a_later_run_continues_where_the_last_stopped() -> None:
    registry = MemoryRegistry()
    run_loop(ToyUseCase(), START, 3, components(registry))
    decisions = run_loop(ToyUseCase(), None, 2, components(registry))
    assert decisions[0].day == START + timedelta(days=3)


def test_loop_refuses_to_skip_or_repeat_days() -> None:
    registry = MemoryRegistry()
    run_loop(ToyUseCase(), START, 3, components(registry))
    with pytest.raises(ValueError, match="state continues at 2026-01-04"):
        run_loop(ToyUseCase(), START, 1, components(registry))


def test_loop_needs_a_start_without_state() -> None:
    with pytest.raises(ValueError, match="pass --from"):
        run_loop(ToyUseCase(), None, 1, components(MemoryRegistry()))


def test_invalid_batch_stops_the_loop() -> None:
    class Broken(ToyUseCase):
        def generate(self, day, scenario: Scenario) -> Batch:  # type: ignore[no-untyped-def]
            batch = super().generate(day, scenario)
            return Batch(day, batch.data.assign(y=5), batch.manifest)

    with pytest.raises(pandera.errors.SchemaError):
        run_loop(Broken(), START, 1, components(MemoryRegistry()))
