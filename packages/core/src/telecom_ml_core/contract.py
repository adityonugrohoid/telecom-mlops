"""The contract every use case implements, and the records passed between pipeline stages.

A use case owns its generator, schema, features, model and baseline. The core
only calls the methods below; it never imports a use case package.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Literal, get_args

import numpy as np
import pandas as pd
import pandera.pandas as pa
import yaml

EventKind = Literal["covariate", "concept", "prior", "new_class", "dynamics", "trend"]
EvaluationKind = Literal["holdout", "rollout"]

# Column every batch carries: the simulated date on which that row's label becomes known.
LABEL_RELEASE = "label_released_on"

Metrics = dict[str, float]


@dataclass(frozen=True)
class Event:
    """One scenario event: from `day`, ramps its parameters to their end values over `ramp_days`.

    `duration_days` bounds a temporary event (a holiday week); None means the change stays.
    `benign` marks the calendar's one input shift that should not hurt the model.
    """

    name: str
    day: int
    ramp_days: int
    kind: EventKind
    params: dict[str, float]
    benign: bool
    duration_days: int | None

    def __post_init__(self) -> None:
        if self.kind not in get_args(EventKind):
            raise ValueError(f"event {self.name!r}: unknown kind {self.kind!r}")
        if self.ramp_days < 0:
            raise ValueError(f"event {self.name!r}: ramp_days must be >= 0, got {self.ramp_days}")
        if self.duration_days is not None and self.duration_days <= 0:
            raise ValueError(f"event {self.name!r}: duration_days must be > 0 when set")

    def strength(self, day_index: int) -> float:
        """0 before the event, rising linearly to 1 over the ramp, 0 again after its duration."""
        if day_index < self.day:
            return 0.0
        if self.duration_days is not None and day_index >= self.day + self.duration_days:
            return 0.0
        if self.ramp_days == 0:
            return 1.0
        return min(1.0, (day_index - self.day + 1) / self.ramp_days)


@dataclass(frozen=True)
class ActiveEvent:
    name: str
    kind: EventKind
    strength: float
    benign: bool


@dataclass(frozen=True)
class Manifest:
    """Ground truth for one day: the events active and their strength. Never reaches a model."""

    day: date
    events: tuple[ActiveEvent, ...]


@dataclass(frozen=True)
class Scenario:
    """A drift calendar: `days` simulated days from `start`, and the events on it."""

    start: date
    days: int
    base_seed: int
    events: tuple[Event, ...]

    @classmethod
    def load(cls, path: Path) -> Scenario:
        raw = yaml.safe_load(path.read_text())
        try:
            events = tuple(
                Event(
                    name=e["name"],
                    day=int(e["day"]),
                    ramp_days=int(e["ramp_days"]),
                    kind=e["kind"],
                    params={k: float(v) for k, v in e["params"].items()},
                    benign=bool(e["benign"]),
                    duration_days=e.get("duration_days"),
                )
                for e in raw["events"]
            )
            return cls(
                start=date.fromisoformat(str(raw["start"])),
                days=int(raw["days"]),
                base_seed=int(raw["base_seed"]),
                events=events,
            )
        except KeyError as exc:
            raise ValueError(f"scenario {path}: missing key {exc}") from exc

    def without_events(self) -> Scenario:
        """The same calendar with the scenario off."""
        return Scenario(start=self.start, days=self.days, base_seed=self.base_seed, events=())

    def day_index(self, day: date) -> int:
        """Days since the calendar start; negative for the history before it."""
        return (day - self.start).days

    def manifest(self, day: date) -> Manifest:
        index = self.day_index(day)
        active = tuple(
            ActiveEvent(e.name, e.kind, e.strength(index), e.benign)
            for e in self.events
            if e.strength(index) > 0.0
        )
        return Manifest(day=day, events=active)

    def value(self, param: str, base: float, day: date) -> float:
        """A parameter's value on `day`: its base, moved toward each event's end value."""
        index = self.day_index(day)
        current = base
        for event in self.events:
            if param in event.params:
                current += (event.params[param] - base) * event.strength(index)
        return current


def rng_for_day(base_seed: int, usecase_id: int, day: date) -> np.random.Generator:
    """The seed for one simulated day. Derived from the date only; nothing reads the clock."""
    return np.random.default_rng([base_seed, usecase_id, day.toordinal()])


@dataclass(frozen=True)
class Batch:
    """One simulated day of data. `data` carries LABEL_RELEASE per row."""

    day: date
    data: pd.DataFrame
    manifest: Manifest

    def __post_init__(self) -> None:
        if LABEL_RELEASE not in self.data.columns:
            raise ValueError(f"batch for {self.day}: missing column {LABEL_RELEASE!r}")


@dataclass(frozen=True)
class DriftResult:
    detected: bool
    share_drifted: float
    per_feature: dict[str, float]


@dataclass(frozen=True)
class PromotionDecision:
    promote: bool
    reason: str


@dataclass(frozen=True)
class LiveRecord:
    """The model in production for a use case, with what it was judged on."""

    model: Any
    reference: pd.DataFrame
    metrics: Metrics
    promoted_on: date
    version: int


@dataclass(frozen=True)
class DayDecision:
    """Everything the loop decided on one day. Every day gets one, whatever happened."""

    usecase: str
    day: date
    drift: DriftResult
    live_metrics: Metrics
    retrained: bool
    candidate_metrics: Metrics | None
    baseline_metrics: Metrics | None
    promoted: bool
    reason: str
    live_version: int
    manifest: Manifest
    notes: tuple[str, ...]


class UseCase(ABC):
    """What a use case supplies to the pipeline.

    Subclasses set the class attributes and implement the abstract methods.
    """

    name: str
    usecase_id: int
    evaluation: EvaluationKind
    # Longest label delay in days; the core looks this far back to collect released labels.
    max_label_delay_days: int
    train_window_days: int
    eval_window_days: int
    scenario: Scenario

    @abstractmethod
    def generate(self, day: date, scenario: Scenario) -> Batch:
        """The simulated batch for `day` under `scenario`, seeded by date."""

    @abstractmethod
    def schema(self) -> pa.DataFrameSchema:
        """The schema every batch must pass."""

    @abstractmethod
    def drift_frame(self, data: pd.DataFrame) -> pd.DataFrame:
        """The monitored inputs, compared against the live model's reference."""

    @abstractmethod
    def fit(self, train: pd.DataFrame, warm_start: Any | None) -> Any:
        """Train a model on released rows. `warm_start` is the live model, or None."""

    @abstractmethod
    def fit_baseline(self, train: pd.DataFrame) -> Any:
        """The simple reference model the candidate is also compared against."""

    @abstractmethod
    def score(self, model: Any, frame: pd.DataFrame) -> Metrics:
        """Metrics of `model` on an evaluation frame."""

    @abstractmethod
    def should_retrain(self, drift: DriftResult, live: Metrics, at_promotion: Metrics) -> bool:
        """The retrain trigger: drift, or the live score falling from what it was promoted on."""

    @abstractmethod
    def promote(self, candidate: Metrics, live: Metrics, baseline: Metrics) -> PromotionDecision:
        """The promotion test for this use case."""

    def history_start(self) -> date:
        """First day of the pre-calendar history the initial model trains on."""
        span = self.train_window_days + self.eval_window_days + self.max_label_delay_days
        return self.scenario.start - timedelta(days=span)
