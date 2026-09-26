from pathlib import Path

import pytest
from telecom_ml_core.cli import build_components, run
from telecom_ml_core.contract import UseCase
from telecom_ml_core.pipeline import Components
from toy import MeanShiftDrift, MemoryRegistry, SchemaValidator, ToyUseCase, WindowEvaluator


class OtherToy(ToyUseCase):
    name = "toy2"


def usecases() -> dict[str, type[UseCase]]:
    return {"toy": ToyUseCase, "toy2": OtherToy}


def test_list_prints_the_use_cases(capsys: pytest.CaptureFixture[str]) -> None:
    assert run(["list"], usecases, build_components) == 0
    assert capsys.readouterr().out.split() == ["toy", "toy2"]


def test_loop_all_runs_every_use_case_on_the_given_state(tmp_path: Path) -> None:
    registry = MemoryRegistry()
    seen: list[Path] = []

    def make(state: Path) -> Components:
        seen.append(state)
        return Components(
            SchemaValidator(), MeanShiftDrift(), {"holdout": WindowEvaluator()}, registry
        )

    argv = ["loop", "all", "--from", "2026-01-01", "--days", "3", "--state", str(tmp_path)]
    assert run(argv, usecases, make) == 0
    assert seen == [tmp_path]
    assert {d.usecase for d in registry.decisions} == {"toy", "toy2"}
    assert len(registry.decisions) == 6


def test_unknown_use_case_names_the_known_ones() -> None:
    with pytest.raises(SystemExit, match="known: toy, toy2"):
        run(["loop", "nope", "--days", "1"], usecases, build_components)


def test_production_components_are_not_wired_yet(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="not wired"):
        build_components(tmp_path)
