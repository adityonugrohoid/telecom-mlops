"""The moving split, computed from the code, and the figures in docs/moving-split.md.

Every number here comes from the use cases' own settings and the loop's own functions:
window sizes from each use case, the train and test windows from `HoldoutEvaluator`, which
rows each window holds from the use case's generator, and whether a candidate could have
learned an event from `report.learned`. Promotion days come from the committed results
summary.

Run `uv run python docs/figures/moving_split.py` to redraw the figures.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from telecom_ml_core.cli import discover_usecases
from telecom_ml_core.contract import LABEL_RELEASE, UseCase
from telecom_ml_core.evaluate import HoldoutEvaluator
from telecom_ml_core.pipeline import DataSource
from telecom_ml_core.report import learned

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "calendar-2026-01-01-180-days.md"
HOLDOUT_ORDER = ("churn", "root-cause", "anomaly", "qoe", "capacity")
SNAPSHOT_DAY = 110


@dataclass(frozen=True)
class Span:
    """An inclusive range of calendar day indexes (day 0 is the calendar start)."""

    first: int
    last: int


@dataclass(frozen=True)
class Windows:
    """The moving split of one holdout use case on one day."""

    usecase: str
    day: int
    train_released: Span
    test_released: Span
    train_generated: Span
    test_generated: Span
    delay_days: Span


def _index(usecase: UseCase, day: date) -> int:
    return (day - usecase.scenario.start).days


def _generated(usecase: UseCase, source: DataSource, released: Span) -> Span:
    """The days whose rows have their answers released inside `released`."""
    start = usecase.scenario.start
    first_release = start + timedelta(days=released.first)
    last_release = start + timedelta(days=released.last)
    days = []
    generated = first_release - timedelta(days=usecase.max_label_delay_days)
    while generated <= last_release:
        released_on = pd.to_datetime(source.batch(generated).data[LABEL_RELEASE])
        inside = (released_on >= pd.Timestamp(first_release)) & (
            released_on <= pd.Timestamp(last_release)
        )
        if inside.any():
            days.append(_index(usecase, generated))
        generated += timedelta(days=1)
    return Span(min(days), max(days))


def delay_days(usecase: UseCase, source: DataSource, day: int) -> Span:
    """The shortest and longest wait for an answer among one day's rows."""
    generated = usecase.scenario.start + timedelta(days=day)
    released = pd.to_datetime(source.batch(generated).data[LABEL_RELEASE])
    waits = (released - pd.Timestamp(generated)).dt.days
    return Span(int(waits.min()), int(waits.max()))


def windows(usecase: UseCase, day: int) -> Windows:
    """The train and test windows of a holdout use case on `day`, by answer release day and
    by the days the rows were generated."""
    if usecase.evaluation != "holdout":
        raise ValueError(f"{usecase.name} is judged by rollouts, not a holdout split")
    source = DataSource(usecase, usecase.scenario)
    on = usecase.scenario.start + timedelta(days=day)
    train_first, train_last = HoldoutEvaluator().train_window(usecase, on)
    train = Span(_index(usecase, train_first), _index(usecase, train_last))
    test = Span(day - usecase.eval_window_days + 1, day)
    return Windows(
        usecase=usecase.name,
        day=day,
        train_released=train,
        test_released=test,
        train_generated=_generated(usecase, source, train),
        test_generated=_generated(usecase, source, test),
        delay_days=delay_days(usecase, source, day),
    )


def earliest_learnable(usecase: UseCase, event_day: int) -> int:
    """The first day a candidate's training rows include a row from on or after the event."""
    source = DataSource(usecase, usecase.scenario)
    start = usecase.scenario.start
    event = start + timedelta(days=event_day)
    day = event_day
    while not learned(usecase, start + timedelta(days=day), event, source):
        day += 1
    return day


@dataclass(frozen=True)
class Promotion:
    day: int
    learned: bool


@dataclass(frozen=True)
class EventRow:
    usecase: str
    event: str
    day: int
    promotions: tuple[Promotion, ...]


_ROW = re.compile(
    r"^\| (?P<event>[^|]+?) \| (?P<day>\d+) \| [^|]+ \| [^|]+ \| [^|]+ \| (?P<promos>[^|]+) \|$"
)
_PROMO = re.compile(r"(\d+) \((learned|refresh)\)")


def promotions_from_results(path: Path) -> list[EventRow]:
    """Every event row of the results summary, with its promotions."""
    rows, usecase = [], None
    for line in path.read_text().splitlines():
        if line.startswith("## "):
            usecase = line[3:].strip()
            continue
        match = _ROW.match(line)
        if usecase is None or match is None:
            continue
        promotions = tuple(
            Promotion(int(day), kind == "learned") for day, kind in _PROMO.findall(match["promos"])
        )
        rows.append(EventRow(usecase, match["event"], int(match["day"]), promotions))
    if not rows:
        raise ValueError(f"no event rows found in {path}")
    return rows


def holdout_usecases() -> list[UseCase]:
    found = {name: factory() for name, factory in discover_usecases().items()}
    return [found[name] for name in HOLDOUT_ORDER]


# -- drawing -----------------------------------------------------------------------------

FIGURES = Path(__file__).resolve().parent
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "wait": "#c3c2b7",
        "train": "#2a78d6",
        "test": "#eb6834",
        "train_tint": "#b7d3f6",
        "test_tint": "#f7c7b2",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "wait": "#52514e",
        "train": "#3987e5",
        "test": "#d95926",
        "train_tint": "#1c4a80",
        "test_tint": "#6e2f16",
    },
}


def _delay_text(delay: Span) -> str:
    """How long this use case waits for an answer, as measured from its generator."""
    if delay.last == 0:
        return "answers at once"
    if delay.first == delay.last:
        return f"answers after {delay.first} day{'s' if delay.first > 1 else ''}"
    return f"answers after {delay.first} to {delay.last} days"


def _style(plt: object, theme: dict[str, str]) -> None:
    plt.rcParams.update(  # type: ignore[attr-defined]
        {
            "svg.fonttype": "none",
            "svg.hashsalt": "telecom-mlops",
            "font.family": "sans-serif",
            "font.sans-serif": ["system-ui", "Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "text.color": theme["ink"],
            "axes.edgecolor": theme["axis"],
            "axes.labelcolor": theme["secondary"],
            "xtick.color": theme["muted"],
            "ytick.color": theme["secondary"],
            "figure.facecolor": theme["surface"],
            "axes.facecolor": theme["surface"],
        }
    )


def _bar(ax: object, y: float, span: Span, height: float, color: str, surface: str) -> None:
    from matplotlib.patches import FancyBboxPatch

    ax.add_patch(  # type: ignore[attr-defined]
        FancyBboxPatch(
            (span.first - 0.5, y - height / 2),
            span.last - span.first + 1,
            height,
            boxstyle="round,pad=0,rounding_size=0.6",
            linewidth=1.5,
            edgecolor=surface,
            facecolor=color,
            mutation_aspect=0.05,
        )
    )


def _axes_common(ax: object, theme: dict[str, str], x0: int, x1: int) -> None:
    ax.set_xlim(x0, x1)  # type: ignore[attr-defined]
    ax.grid(axis="x", color=theme["grid"], linewidth=0.8)  # type: ignore[attr-defined]
    ax.set_axisbelow(True)  # type: ignore[attr-defined]
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)  # type: ignore[attr-defined]
    ax.tick_params(axis="y", length=0)  # type: ignore[attr-defined]
    ax.set_xlabel("simulated day")  # type: ignore[attr-defined]


def _legend(ax: object, theme: dict[str, str], entries: list[tuple[str, str]]) -> None:
    from matplotlib.patches import Patch

    handles = [Patch(facecolor=color, edgecolor="none", label=label) for label, color in entries]
    ax.legend(  # type: ignore[attr-defined]
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        frameon=False,
        fontsize=9,
        labelcolor=theme["secondary"],
        handlelength=1.4,
    )


def _split_rows(ax: object, theme: dict[str, str], rows: list[tuple[str, Windows]]) -> None:
    for i, (_label, w) in enumerate(rows):
        y = -i
        _bar(ax, y + 0.17, w.train_released, 0.26, theme["train"], theme["surface"])
        _bar(ax, y + 0.17, w.test_released, 0.26, theme["test"], theme["surface"])
        _bar(ax, y - 0.17, w.train_generated, 0.2, theme["train_tint"], theme["surface"])
        _bar(ax, y - 0.17, w.test_generated, 0.2, theme["test_tint"], theme["surface"])
        ax.plot([w.day + 0.5] * 2, [y - 0.36, y + 0.36], color=theme["ink"], linewidth=1.2)  # type: ignore[attr-defined]
        ax.text(  # type: ignore[attr-defined]
            w.day + 1.3,
            y + 0.17,
            "today",
            fontsize=8.5,
            color=theme["secondary"],
            va="center",
        )
    ax.set_yticks([-i for i in range(len(rows))], [label for label, _ in rows])  # type: ignore[attr-defined]
    ax.set_ylim(-len(rows) + 0.4, 0.6)  # type: ignore[attr-defined]


def draw_split(mode: str, rows: list[Windows]) -> None:
    """Figure A: every holdout use case's train and test windows on one day."""
    import matplotlib.pyplot as plt

    theme = THEMES[mode]
    _style(plt, theme)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    labelled = [(f"{w.usecase}\n{_delay_text(w.delay_days)}", w) for w in rows]
    _split_rows(ax, theme, labelled)
    _axes_common(ax, theme, 30, 118)
    ax.set_title(
        f"The moving split on day {SNAPSHOT_DAY}",
        loc="left",
        fontsize=11,
        color=theme["ink"],
        pad=48,
    )
    _legend(
        ax,
        theme,
        [
            ("train: answers released", theme["train"]),
            ("test: answers released", theme["test"]),
            ("train: when the rows happened", theme["train_tint"]),
            ("test: when the rows happened", theme["test_tint"]),
        ],
    )
    fig.tight_layout()
    fig.savefig(FIGURES / f"split-day-{SNAPSHOT_DAY}-{mode}.svg", metadata={"Date": None})
    plt.close(fig)


def draw_slide(mode: str, rows: list[Windows]) -> None:
    """Figure B: one use case's windows on three days, moving together."""
    import matplotlib.pyplot as plt

    theme = THEMES[mode]
    _style(plt, theme)
    fig, ax = plt.subplots(figsize=(9, 3.2))
    _split_rows(ax, theme, [(f"day {w.day}", w) for w in rows])
    _axes_common(ax, theme, 20, 128)
    ax.set_title(
        f"{rows[0].usecase}: both windows move forward with each day",
        loc="left",
        fontsize=11,
        color=theme["ink"],
        pad=48,
    )
    _legend(
        ax,
        theme,
        [
            ("train: answers released", theme["train"]),
            ("test: answers released", theme["test"]),
            ("train: when the rows happened", theme["train_tint"]),
            ("test: when the rows happened", theme["test_tint"]),
        ],
    )
    fig.tight_layout()
    fig.savefig(FIGURES / f"split-slide-{rows[0].usecase}-{mode}.svg", metadata={"Date": None})
    plt.close(fig)


@dataclass(frozen=True)
class EventTiming:
    label: str
    event_day: int
    learnable_day: int
    promotions: tuple[Promotion, ...]


def draw_events(mode: str, timings: list[EventTiming]) -> None:
    """Figure C: each event, the first day a model could learn it, and what was promoted."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    theme = THEMES[mode]
    _style(plt, theme)
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for i, t in enumerate(timings):
        y = -i
        ax.plot(
            [t.event_day, t.learnable_day],
            [y, y],
            color=theme["wait"],
            linewidth=6,
            solid_capstyle="round",
            zorder=1,
        )
        ax.plot(
            [t.event_day],
            [y],
            marker="|",
            markersize=14,
            markeredgewidth=2,
            color=theme["ink"],
            zorder=2,
        )
        for p in t.promotions:
            ax.plot(
                [p.day],
                [y],
                marker="o",
                markersize=9,
                zorder=3,
                markerfacecolor=theme["train"] if p.learned else theme["surface"],
                markeredgecolor=theme["train"] if p.learned else theme["ink"],
                markeredgewidth=1.8,
            )
    ax.set_yticks([-i for i in range(len(timings))], [t.label for t in timings])
    ax.set_ylim(-len(timings) + 0.4, 0.6)
    _axes_common(ax, theme, 0, 180)
    ax.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="|",
                linestyle="none",
                markersize=12,
                markeredgewidth=2,
                color=theme["ink"],
                label="event begins",
            ),
            Line2D(
                [], [], color=theme["wait"], linewidth=6, label="waiting until a model can learn it"
            ),
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markersize=8,
                markerfacecolor=theme["train"],
                markeredgecolor=theme["train"],
                label="promoted, learned",
            ),
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markersize=8,
                markerfacecolor=theme["surface"],
                markeredgecolor=theme["ink"],
                label="promoted, refresh",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        frameon=False,
        fontsize=9,
        labelcolor=theme["secondary"],
    )
    ax.set_title("From event to promotion", loc="left", fontsize=11, color=theme["ink"], pad=48)
    fig.tight_layout()
    fig.savefig(FIGURES / f"events-to-promotions-{mode}.svg", metadata={"Date": None})
    plt.close(fig)


def draw_netopt(mode: str, days: list[int]) -> None:
    """Figure D: netopt's split is practice episodes against a fixed exam, not time."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    theme = THEMES[mode]
    _style(plt, theme)
    fig, ax = plt.subplots(figsize=(9, 2.8))
    for j, day in enumerate(days):
        x = j * 3.1
        for y, label, color in (
            (1.0, f"practice: 500 episodes\nseeded from day {day}", theme["train"]),
            (0.0, "exam: 50 fixed episodes\nseeds 1000 to 1049", theme["test"]),
        ):
            ax.add_patch(
                FancyBboxPatch(
                    (x, y),
                    2.8,
                    0.8,
                    boxstyle="round,pad=0,rounding_size=0.08",
                    facecolor=color,
                    edgecolor=theme["surface"],
                    linewidth=1.5,
                )
            )
            ax.text(x + 1.4, y + 0.4, label, ha="center", va="center", fontsize=9, color="#ffffff")
        ax.text(
            x + 1.4,
            1.95,
            f"if a retrain runs on day {day}",
            ha="center",
            fontsize=9.5,
            color=theme["secondary"],
        )
    ax.set_xlim(-0.2, len(days) * 3.1)
    ax.set_ylim(-0.15, 2.55)
    ax.axis("off")
    ax.text(
        -0.2,
        2.45,
        "netopt: practice episodes against a fixed exam (the first training "
        "practises 2,000 episodes)",
        fontsize=11,
        color=theme["ink"],
    )
    fig.tight_layout()
    fig.savefig(FIGURES / f"netopt-practice-and-exam-{mode}.svg", metadata={"Date": None})
    plt.close(fig)


def _event_label(event: str) -> str:
    """Events that begin together share one row; name the row by what their names share."""
    parts = event.split(" + ")
    shared = parts[0]
    for part in parts[1:]:
        while not part.startswith(shared):
            shared = shared[:-1]
    return shared.strip("_").replace("_", " ")


def event_timings() -> list[EventTiming]:
    """The real events with a promotion, in calendar order within each use case."""
    found = {u.name: u for u in holdout_usecases()}
    rows = [r for r in promotions_from_results(RESULTS) if r.promotions and r.usecase in found]
    return [
        EventTiming(
            label=f"{r.usecase}: {_event_label(r.event)}",
            event_day=r.day,
            learnable_day=earliest_learnable(found[r.usecase], r.day),
            promotions=r.promotions,
        )
        for usecase in HOLDOUT_ORDER
        for r in rows
        if r.usecase == usecase
    ]


def main() -> None:
    usecases = holdout_usecases()
    split = [windows(u, SNAPSHOT_DAY) for u in usecases]
    churn = next(u for u in usecases if u.name == "churn")
    slide = [windows(churn, d) for d in (100, 110, 120)]
    timings = event_timings()
    for mode in THEMES:
        draw_split(mode, split)
        draw_slide(mode, slide)
        draw_events(mode, timings)
        draw_netopt(mode, [59, 60, 61])


if __name__ == "__main__":
    main()
