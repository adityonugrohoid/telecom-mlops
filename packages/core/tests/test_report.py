from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest
from telecom_ml_core.cli import build_components, run
from telecom_ml_core.contract import UseCase
from telecom_ml_core.pipeline import DataSource, run_loop
from telecom_ml_core.registry import MlflowRegistry, sqlite_uri
from telecom_ml_core.report import SIMULATED, learned, outcomes, summary
from toy import START, TOY_SCENARIO, ToyUseCase


@pytest.fixture(scope="module")
def decisions(tmp_path_factory: pytest.TempPathFactory) -> list[dict[str, object]]:
    state = tmp_path_factory.mktemp("state")
    run_loop(ToyUseCase(), START, 28, build_components(state))
    registry = MlflowRegistry(sqlite_uri(state), state / "mlartifacts", state / "promotion_log")
    return registry.decisions("toy")


def test_the_concept_event_promotes_on_learned_data_and_the_benign_one_never(
    decisions: list[dict[str, object]],
) -> None:
    found = {o.events[0].name: o for o in outcomes(ToyUseCase(), decisions)}
    concept, benign = found["threshold_moves"], found["z_shift"]
    assert concept.promotions
    assert all(p.learned for p in concept.promotions)
    assert benign.first_dataset_drift == 20
    assert benign.promotions == ()


def test_learned_needs_post_event_rows_in_the_training_window() -> None:
    usecase = ToyUseCase()
    source = DataSource(usecase, TOY_SCENARIO)
    event_day = START + timedelta(days=5)
    # Label delay 2, eval window 3: training on day d uses releases up to d - 3, so the first
    # post-event row (generated day 5, released day 7) enters on day 10.
    assert not learned(usecase, START + timedelta(days=9), event_day, source)
    assert learned(usecase, START + timedelta(days=10), event_day, source)


def test_summary_opens_with_the_simulated_data_statement(
    decisions: list[dict[str, object]],
) -> None:
    text = summary([(ToyUseCase(), decisions, "packages/toy/MODEL_CARD.md")])
    assert text.splitlines()[0] == SIMULATED
    assert "| z_shift (benign) | 20 |" in text
    assert "Benign event: detected, 0 promotions." in text


def test_events_starting_the_same_day_share_one_row(decisions: list[dict[str, object]]) -> None:
    class SameDay(ToyUseCase):
        scenario = replace(
            TOY_SCENARIO,
            events=(*TOY_SCENARIO.events, replace(TOY_SCENARIO.events[0], name="twin")),
        )

    text = summary([(SameDay(), decisions, "card")])
    assert "| threshold_moves + twin | 5 | concept |" in text


def test_events_after_the_last_day_are_marked_not_reached(
    decisions: list[dict[str, object]],
) -> None:
    text = summary([(ToyUseCase(), decisions[:15], "card")])
    assert "| z_shift (benign) | 20 | covariate | not reached in this run |" in text
    assert "Benign event: not reached in this run." in text
    assert "| threshold_moves | 5 |" in text


def test_rule8_exception_is_read_from_the_use_case(decisions: list[dict[str, object]]) -> None:
    class Excepted(ToyUseCase):
        rule8_exception = "Lags absorb the change."

    text = summary([(Excepted(), decisions, "card")])
    assert "No promotion is expected for the real events here. Lags absorb the change." in text


def test_tml_report_writes_the_file(tmp_path: Path) -> None:
    def usecases() -> dict[str, type[UseCase]]:
        return {"toy": ToyUseCase}

    state = tmp_path / "state"
    loop = ["loop", "toy", "--from", "2026-01-01", "--days", "6", "--state", str(state)]
    assert run(loop, usecases, build_components) == 0
    out = tmp_path / "results" / "summary.md"
    assert (
        run(["report", "--state", str(state), "--out", str(out)], usecases, build_components) == 0
    )
    assert out.read_text().splitlines()[0] == SIMULATED


def test_report_refuses_an_empty_state(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no promotion logs"):
        run(
            ["report", "--state", str(tmp_path), "--out", str(tmp_path / "x.md")],
            dict,
            build_components,
        )
