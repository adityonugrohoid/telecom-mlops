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
from telecom_ml_core.pipeline import Components, run_loop

ENTRY_POINT_GROUP = "telecom_ml.usecases"
DEFAULT_STATE_DIR = Path("state")

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

    Raises:
        NotImplementedError: until the validate, drift, registry and evaluator stages land.
    """
    raise NotImplementedError(
        f"no production components yet for {state_dir}: validate, drift, registry and the "
        "evaluators are not wired in this version"
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


def main() -> int:
    """Entry point for the `tml` script."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    return run(sys.argv[1:], discover_usecases, build_components)


if __name__ == "__main__":
    raise SystemExit(main())
