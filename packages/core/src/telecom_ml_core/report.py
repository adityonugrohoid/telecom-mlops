"""The results summary: what the loop did on each use case's drift calendar, read from the
promotion logs of a run.

For each event it reports the day it began, the first day of dataset drift and of a retrain
on or after it, and each promotion that followed it. A promotion belongs to the latest event
that began on or before its day, and it counts as learned when the candidate's training data
held at least one row generated on or after that event's day; otherwise it was a refresh on
recent data. That check runs through the same data source and evaluator the loop used.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from telecom_ml_core.contract import LABEL_RELEASE, Event, UseCase
from telecom_ml_core.evaluate import HoldoutEvaluator
from telecom_ml_core.pipeline import DataSource

SIMULATED = (
    "**All data and environments in this summary are simulated.** Every use case runs on a "
    "synthetic generator and a scripted drift calendar; the numbers say nothing about any real "
    "network, customer or operator."
)


@dataclass(frozen=True)
class Promotion:
    day: int
    learned: bool
    reason: str


@dataclass(frozen=True)
class EventOutcome:
    event: Event
    first_dataset_drift: int | None
    first_retrain: int | None
    promotions: tuple[Promotion, ...]


def _index(usecase: UseCase, day: str) -> int:
    return (date.fromisoformat(day) - usecase.scenario.start).days


def learned(usecase: UseCase, promoted_on: date, event_day: date, source: DataSource) -> bool:
    """Whether a candidate retrained on `promoted_on` trained on any row generated on or after
    `event_day`."""
    if usecase.evaluation == "rollout":
        # A policy trains on the day's environment version.
        return promoted_on >= event_day
    first, last = HoldoutEvaluator().train_window(usecase, promoted_on)
    generated = event_day
    while generated <= last:
        released = pd.to_datetime(source.batch(generated).data[LABEL_RELEASE])
        if ((released >= pd.Timestamp(first)) & (released <= pd.Timestamp(last))).any():
            return True
        generated += timedelta(days=1)
    return False


def outcomes(usecase: UseCase, decisions: list[dict[str, Any]]) -> list[EventOutcome]:
    """Each calendar event with what the loop did after it."""
    if not decisions:
        raise ValueError(f"{usecase.name}: no decisions in the promotion log")
    events = sorted(usecase.scenario.events, key=lambda e: e.day)
    source = DataSource(usecase, usecase.scenario)
    start = usecase.scenario.start

    def owner(index: int) -> Event | None:
        begun = [e for e in events if e.day <= index]
        return begun[-1] if begun else None

    result = []
    for event in events:
        after = [d for d in decisions if _index(usecase, d["day"]) >= event.day]
        drift = next((d for d in after if d["drift"]["detected"]), None)
        retrain = next((d for d in after if d["retrained"]), None)
        promotions = tuple(
            Promotion(
                day=_index(usecase, d["day"]),
                learned=learned(
                    usecase,
                    date.fromisoformat(d["day"]),
                    start + timedelta(days=event.day),
                    source,
                ),
                reason=d["reason"],
            )
            for d in decisions
            if d["promoted"] and owner(_index(usecase, d["day"])) is event
        )
        result.append(
            EventOutcome(
                event=event,
                first_dataset_drift=None if drift is None else _index(usecase, drift["day"]),
                first_retrain=None if retrain is None else _index(usecase, retrain["day"]),
                promotions=promotions,
            )
        )
    return result


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _day(value: int | None) -> str:
    return "none" if value is None else str(value)


def _promotions(promotions: tuple[Promotion, ...]) -> str:
    if not promotions:
        return "none"
    return ", ".join(f"{p.day} ({'learned' if p.learned else 'refresh'})" for p in promotions)


def section(usecase: UseCase, decisions: list[dict[str, Any]], model_card: str) -> str:
    """The Markdown section for one use case."""
    found = outcomes(usecase, decisions)
    last = decisions[-1]
    lines = [
        f"## {usecase.name}",
        "",
        f"{_count(len(decisions), 'simulated day')} from {usecase.scenario.start}; "
        f"{_count(sum(d['retrained'] for d in decisions), 'retrain')}, "
        f"{_count(sum(d['promoted'] for d in decisions), 'promotion')}, "
        f"dataset drift on {_count(sum(d['drift']['detected'] for d in decisions), 'day')}.",
        "",
        "| Event | Day | Kind | First dataset drift | First retrain | Promotions |",
        "|---|---|---|---|---|---|",
    ]
    for o in found:
        label = f"{o.event.name} (benign)" if o.event.benign else o.event.name
        lines.append(
            f"| {label} | {o.event.day} | {o.event.kind} | {_day(o.first_dataset_drift)} | "
            f"{_day(o.first_retrain)} | {_promotions(o.promotions)} |"
        )
    benign = [o for o in found if o.event.benign]
    lines.append("")
    for o in benign:
        detected = o.first_retrain is not None or o.first_dataset_drift is not None
        lines.append(
            f"Benign event: {'detected' if detected else 'not detected'}, "
            f"{_count(len(o.promotions), 'promotion')}."
        )
    if usecase.rule8_exception is not None:
        lines += [
            "",
            f"No promotion is expected for the real events here. {usecase.rule8_exception}",
        ]
    metrics = ", ".join(f"{k} {v:.4f}" for k, v in last["live_metrics"].items())
    lines += [
        "",
        f"Day {_index(usecase, last['day'])}: live model v{last['live_version']}, {metrics}.",
        f"Details: {model_card}.",
        "",
    ]
    return "\n".join(lines)


def summary(entries: list[tuple[UseCase, list[dict[str, Any]], str]]) -> str:
    """The whole summary: the simulated-data statement first, then one section per use case."""
    if not entries:
        raise ValueError("no use case has a promotion log to report")
    parts = [
        SIMULATED,
        "",
        "# Results",
        "",
        "Learned: the candidate's training data held rows from after the event began. Refresh: "
        "it did not, and won on recent data alone.",
        "",
    ]
    parts += [section(usecase, decisions, card) for usecase, decisions, card in entries]
    return "\n".join(parts)


def model_card_path(usecase: UseCase, root: Path) -> str:
    """Where the use case's model card sits, relative to `root`: next to its `src` folder."""
    package_root = Path(inspect.getfile(type(usecase))).resolve().parents[2]
    return str((package_root / "MODEL_CARD.md").relative_to(root.resolve()))
