"""The `tml` command line.

Use cases register through the `telecom_ml.usecases` entry point group, so the core
finds them without importing any use case package by name.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable, Sequence
from datetime import date
from importlib.metadata import entry_points
from pathlib import Path

from telecom_ml_core.contract import UseCase
from telecom_ml_core.drift import EvidentlyDrift
from telecom_ml_core.evaluate import HoldoutEvaluator, RolloutEvaluator
from telecom_ml_core.pipeline import Components, run_loop
from telecom_ml_core.registry import MlflowRegistry, sqlite_uri
from telecom_ml_core.report import model_card_path, summary
from telecom_ml_core.validate import SchemaValidator

ENTRY_POINT_GROUP = "telecom_ml.usecases"
DEFAULT_STATE_DIR = Path("state")
# Evidently's default: the dataset drifts when half the monitored columns do.
DRIFT_SHARE = 0.5
# Fixed episodes every policy runs on each environment version (generator spec, netopt).
ROLLOUT_EPISODES = 50
ROLLOUT_SEED = 1000

log = logging.getLogger(__name__)

UseCaseLoader = Callable[[], dict[str, Callable[[], UseCase]]]
ComponentsFactory = Callable[[Path], Components]


def discover_usecases() -> dict[str, Callable[[], UseCase]]:
    """Find installed use cases.

    Returns:
        Use case name to a factory that builds it, sorted by name.
    """
    found = {ep.name: ep.load() for ep in entry_points(group=ENTRY_POINT_GROUP)}
    return dict(sorted(found.items()))


def build_components(state_dir: Path) -> Components:
    """The production stage implementations, stored under `state_dir`.

    Args:
        state_dir: Folder holding run state (never committed).

    Returns:
        Validator, drift detector, both evaluators and the MLflow registry.
    """
    state_dir.mkdir(parents=True, exist_ok=True)
    return Components(
        validator=SchemaValidator(),
        drift=EvidentlyDrift(drift_share=DRIFT_SHARE),
        evaluators={
            "holdout": HoldoutEvaluator(),
            "rollout": RolloutEvaluator(episodes=ROLLOUT_EPISODES, seed=ROLLOUT_SEED),
        },
        registry=MlflowRegistry(
            tracking_uri=sqlite_uri(state_dir),
            artifact_root=state_dir / "mlartifacts",
            log_dir=state_dir / "promotion_log",
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tml", description="telecom-mlops daily drift loop")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="list the installed use cases")
    loop = sub.add_parser("loop", help="replay simulated days for one use case, or all")
    loop.add_argument("usecase", help="a use case name, or 'all'")
    loop.add_argument(
        "--from",
        dest="first",
        type=date.fromisoformat,
        help="first simulated day (YYYY-MM-DD); omit to continue from state",
    )
    loop.add_argument("--days", type=int, required=True, help="number of simulated days")
    loop.add_argument("--state", type=Path, default=DEFAULT_STATE_DIR, help="run state folder")
    report = sub.add_parser("report", help="write the results summary of a run's promotion logs")
    report.add_argument("--state", type=Path, required=True, help="run state folder")
    report.add_argument("--out", type=Path, required=True, help="Markdown file to write")
    return parser


def run(
    argv: Sequence[str], load_usecases: UseCaseLoader, make_components: ComponentsFactory
) -> int:
    """Run one `tml` command.

    Args:
        argv: Command line arguments without the program name.
        load_usecases: Returns the available use cases.
        make_components: Builds the stage implementations for a state folder.

    Returns:
        The process exit code.
    """
    args = _parser().parse_args(argv)
    usecases = load_usecases()
    if args.command == "list":
        for name in usecases:
            sys.stdout.write(f"{name}\n")
        return 0

    if args.command == "report":
        return write_report(args.state, args.out, usecases)

    if args.usecase == "all":
        names = list(usecases)
    elif args.usecase in usecases:
        names = [args.usecase]
    else:
        known = ", ".join(usecases) or "none installed"
        raise SystemExit(f"unknown use case {args.usecase!r}; known: {known}")

    components = make_components(args.state)
    for name in names:
        decisions = run_loop(usecases[name](), args.first, args.days, components)
        promoted = sum(d.promoted for d in decisions)
        retrained = sum(d.retrained for d in decisions)
        drifted = sum(d.drift.detected for d in decisions)
        log.info(
            f"{name}: {len(decisions)} days to {decisions[-1].day}, drift on {drifted}, "
            f"retrained {retrained}, promoted {promoted}"
        )
    return 0


def write_report(state_dir: Path, out: Path, usecases: dict[str, Callable[[], UseCase]]) -> int:
    """Write the results summary for every use case with a promotion log under `state_dir`.

    Raises:
        FileNotFoundError: when the state folder holds no promotion logs.
    """
    log_dir = state_dir / "promotion_log"
    logged = sorted(p.stem for p in log_dir.glob("*.jsonl")) if log_dir.is_dir() else []
    if not logged:
        raise FileNotFoundError(f"no promotion logs under {log_dir}")
    unknown = [name for name in logged if name not in usecases]
    if unknown:
        raise SystemExit(f"promotion logs for use cases that are not installed: {unknown}")
    registry = MlflowRegistry(sqlite_uri(state_dir), state_dir / "mlartifacts", log_dir)
    root = Path.cwd()
    entries = []
    for name in logged:
        usecase = usecases[name]()
        entries.append((usecase, registry.decisions(name), model_card_path(usecase, root)))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(summary(entries))
    log.info(f"wrote {out} for {', '.join(logged)}")
    return 0


def main() -> int:
    """Entry point for the `tml` script."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    return run(sys.argv[1:], discover_usecases, build_components)


if __name__ == "__main__":
    raise SystemExit(main())
