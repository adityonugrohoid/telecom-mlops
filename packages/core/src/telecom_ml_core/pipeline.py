"""The daily loop: the same six stages for every use case.

1. generate the day's batch   2. validate it   3. check drift against the reference
4. score the live model        5. retrain on the trigger
6. promote only if the candidate wins; log every decision

Stage implementations arrive as a `Components` bundle so the runner holds no
storage, drift or evaluation logic of its own.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Protocol

import pandas as pd

from telecom_ml_core.contract import (
    LABEL_RELEASE,
    Batch,
    DayDecision,
    DriftResult,
    DriftTest,
    EvaluationKind,
    LiveRecord,
    Metrics,
    Scenario,
    UseCase,
)

log = logging.getLogger(__name__)


class DataSource:
    """Generated batches for one use case, cached per day. Generation is deterministic by date,
    so any past day can be rebuilt instead of stored."""

    def __init__(self, usecase: UseCase, scenario: Scenario) -> None:
        self.usecase = usecase
        self.scenario = scenario
        self._cache: dict[date, Batch] = {}

    def batch(self, day: date) -> Batch:
        if day not in self._cache:
            self._cache[day] = self.usecase.generate(day, self.scenario)
        return self._cache[day]

    def released(self, first: date, last: date) -> pd.DataFrame:
        """Rows whose label was released on a day in [first, last].

        Looks back `max_label_delay_days` before `first` so rows generated earlier and released
        inside the window are included.
        """
        if last < first:
            raise ValueError(f"empty release window {first}..{last}")
        start = first - timedelta(days=self.usecase.max_label_delay_days)
        span = (last - start).days + 1
        frames = [self.batch(start + timedelta(days=i)).data for i in range(span)]
        data = pd.concat(frames, ignore_index=True)
        released_on = pd.to_datetime(data[LABEL_RELEASE])
        mask = (released_on >= pd.Timestamp(first)) & (released_on <= pd.Timestamp(last))
        return data.loc[mask].reset_index(drop=True)


class Validator(Protocol):
    def validate(self, usecase: UseCase, batch: Batch) -> None:
        """Raise with context when the batch breaks the use case's schema."""
        ...


class DriftDetector(Protocol):
    def check(
        self, reference: pd.DataFrame, current: pd.DataFrame, numeric_test: DriftTest
    ) -> DriftResult: ...


class Evaluator(Protocol):
    def training_data(self, usecase: UseCase, day: date, source: DataSource) -> pd.DataFrame:
        """The rows a candidate trains on when retrained on `day`."""
        ...

    def evaluate(
        self, usecase: UseCase, models: Mapping[str, Any], day: date, source: DataSource
    ) -> dict[str, Metrics]:
        """Score every model on the same evaluation data for `day`."""
        ...


class Registry(Protocol):
    def live(self, usecase: str) -> LiveRecord | None: ...

    def promote(self, usecase: str, record: LiveRecord) -> None: ...

    def log_decision(self, decision: DayDecision) -> None: ...

    def last_day(self, usecase: str) -> date | None:
        """The last simulated day the loop completed for this use case."""
        ...


@dataclass(frozen=True)
class Components:
    validator: Validator
    drift: DriftDetector
    evaluators: Mapping[EvaluationKind, Evaluator]
    registry: Registry


def _bootstrap(usecase: UseCase, components: Components, source: DataSource) -> LiveRecord:
    """Train the first live model on the history before the calendar starts."""
    evaluator = components.evaluators[usecase.evaluation]
    day = usecase.scenario.start - timedelta(days=1)
    train = evaluator.training_data(usecase, day, source)
    model = usecase.fit(train, None)
    metrics = evaluator.evaluate(usecase, {"live": model}, day, source)["live"]
    record = LiveRecord(
        model=model,
        reference=usecase.drift_frame(train),
        metrics=metrics,
        promoted_on=day,
        version=1,
    )
    components.registry.promote(usecase.name, record)
    log.info(f"{usecase.name}: bootstrapped v1 on history to {day}: {metrics}")
    return record


def run_day(usecase: UseCase, day: date, components: Components, source: DataSource) -> DayDecision:
    evaluator = components.evaluators[usecase.evaluation]
    live = components.registry.live(usecase.name)
    if live is None:
        live = _bootstrap(usecase, components, source)

    # 1-2. generate and validate
    batch = source.batch(day)
    components.validator.validate(usecase, batch)

    # 3. drift against the live model's reference
    drift = components.drift.check(
        live.reference, usecase.drift_frame(batch.data), usecase.drift_test
    )

    # 4. score the live model
    live_metrics = evaluator.evaluate(usecase, {"live": live.model}, day, source)["live"]

    # 5. retrain on the trigger
    if not usecase.should_retrain(drift, live_metrics, live.metrics):
        decision = DayDecision(
            usecase=usecase.name,
            day=day,
            drift=drift,
            live_metrics=live_metrics,
            retrained=False,
            candidate_metrics=None,
            baseline_metrics=None,
            promoted=False,
            reason="no retrain trigger",
            live_version=live.version,
            manifest=batch.manifest,
            notes=(),
        )
        components.registry.log_decision(decision)
        return decision

    train = evaluator.training_data(usecase, day, source)
    candidate = usecase.fit(train, live.model)
    baseline = usecase.fit_baseline(train)
    scores = evaluator.evaluate(
        usecase, {"live": live.model, "candidate": candidate, "baseline": baseline}, day, source
    )

    # 6. promote only if the candidate wins
    verdict = usecase.promote(scores["candidate"], scores["live"], scores["baseline"])
    version = live.version
    if verdict.promote:
        version = live.version + 1
        components.registry.promote(
            usecase.name,
            LiveRecord(
                model=candidate,
                reference=usecase.drift_frame(train),
                metrics=scores["candidate"],
                promoted_on=day,
                version=version,
            ),
        )
    decision = DayDecision(
        usecase=usecase.name,
        day=day,
        drift=drift,
        live_metrics=scores["live"],
        retrained=True,
        candidate_metrics=scores["candidate"],
        baseline_metrics=scores["baseline"],
        promoted=verdict.promote,
        reason=verdict.reason,
        live_version=version,
        manifest=batch.manifest,
        notes=(),
    )
    components.registry.log_decision(decision)
    return decision


def run_loop(
    usecase: UseCase, first: date | None, days: int, components: Components
) -> list[DayDecision]:
    """Replay `days` simulated days. Continues from the registry's last day when `first` is None;
    when both exist they must agree, so a run never skips or repeats a day."""
    if days <= 0:
        raise ValueError(f"days must be > 0, got {days}")
    last = components.registry.last_day(usecase.name)
    resume = None if last is None else last + timedelta(days=1)
    if first is None and resume is None:
        raise ValueError(f"{usecase.name}: no state yet, pass --from to start the loop")
    if first is not None and resume is not None and first != resume:
        raise ValueError(
            f"{usecase.name}: state continues at {resume}, but --from is {first}; "
            "drop --from or start from a fresh state folder"
        )
    start = first if first is not None else resume
    assert start is not None
    source = DataSource(usecase, usecase.scenario)
    return [run_day(usecase, start + timedelta(days=i), components, source) for i in range(days)]
