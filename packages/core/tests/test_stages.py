from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from telecom_ml_core.contract import LABEL_RELEASE, Batch, LiveRecord
from telecom_ml_core.drift import EvidentlyDrift
from telecom_ml_core.evaluate import HoldoutEvaluator
from telecom_ml_core.pipeline import Components, run_loop
from telecom_ml_core.registry import MlflowRegistry, sqlite_uri
from telecom_ml_core.validate import BatchValidationError, SchemaValidator
from toy import START, TOY_SCENARIO, Threshold, ToyUseCase


def registry_at(state: Path) -> MlflowRegistry:
    return MlflowRegistry(sqlite_uri(state), state / "mlartifacts", state / "promotion_log")


def toy_batch() -> Batch:
    return ToyUseCase().generate(START, TOY_SCENARIO)


def test_valid_batch_passes() -> None:
    SchemaValidator().validate(ToyUseCase(), toy_batch())


def test_schema_failure_names_the_use_case_day_and_column() -> None:
    batch = toy_batch()
    broken = Batch(batch.day, batch.data.assign(y=5), batch.manifest)
    with pytest.raises(BatchValidationError, match=r"toy 2026-01-01: schema failed(.|\n)*\by\b"):
        SchemaValidator().validate(ToyUseCase(), broken)


def test_label_released_before_the_day_is_rejected() -> None:
    batch = toy_batch()
    early = batch.data.assign(**{LABEL_RELEASE: pd.Timestamp(START - timedelta(days=1))})
    with pytest.raises(BatchValidationError, match="400 rows release their label before"):
        SchemaValidator().validate(ToyUseCase(), Batch(batch.day, early, batch.manifest))


def test_manifest_for_another_day_is_rejected() -> None:
    batch = toy_batch()
    other = TOY_SCENARIO.manifest(START + timedelta(days=1))
    with pytest.raises(BatchValidationError, match="manifest is for 2026-01-02"):
        SchemaValidator().validate(ToyUseCase(), Batch(batch.day, batch.data, other))


def frames(shift: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    reference = pd.DataFrame({"a": rng.normal(0, 1, 800), "b": rng.normal(0, 1, 800)})
    current = pd.DataFrame({"a": rng.normal(shift, 1, 800), "b": rng.normal(0, 1, 800)})
    return reference, current


def test_same_distribution_does_not_drift() -> None:
    result = EvidentlyDrift(drift_share=0.5).check(*frames(0.0), "auto")
    assert not result.detected
    assert result.drifted == ()
    assert set(result.per_feature) == {"a", "b"}


def test_shifted_column_is_named_and_share_decides_detection() -> None:
    reference, current = frames(1.0)
    result = EvidentlyDrift(drift_share=0.5).check(reference, current, "auto")
    assert (result.detected, result.share_drifted, result.drifted) == (True, 0.5, ("a",))
    assert not EvidentlyDrift(drift_share=0.9).check(reference, current, "auto").detected


def test_ks_keeps_small_same_distribution_days_quiet() -> None:
    rng = np.random.default_rng(3)
    reference = pd.DataFrame({"a": rng.normal(0, 1, 1500), "b": rng.normal(0, 1, 1500)})
    days = [pd.DataFrame({"a": rng.normal(0, 1, 50), "b": rng.normal(0, 1, 50)}) for _ in range(20)]
    detector = EvidentlyDrift(drift_share=0.5)
    flagged = {
        test: sum(detector.check(reference, day, test).detected for day in days)
        for test in ("auto", "ks")
    }
    assert flagged["ks"] <= 2
    assert flagged["auto"] > flagged["ks"]


def test_drift_rejects_mismatched_columns() -> None:
    reference, current = frames(0.0)
    with pytest.raises(ValueError, match="drift columns differ"):
        EvidentlyDrift(0.5).check(reference, current[["b"]], "auto")


def record(version: int) -> LiveRecord:
    return LiveRecord(
        model=Threshold(0.25),
        reference=pd.DataFrame({"x": [0.1, 0.2], "z": [1.0, 2.0]}),
        metrics={"accuracy": 0.9},
        promoted_on=date(2026, 1, 3),
        version=version,
    )


def test_registry_is_empty_until_the_first_promotion(tmp_path: Path) -> None:
    registry = registry_at(tmp_path)
    assert registry.live("toy") is None
    assert registry.last_day("toy") is None


def test_promoted_model_survives_a_new_process(tmp_path: Path) -> None:
    registry_at(tmp_path).promote("toy", record(1))
    live = registry_at(tmp_path).live("toy")
    assert live is not None
    assert (live.model, live.metrics, live.promoted_on, live.version) == (
        Threshold(0.25),
        {"accuracy": 0.9},
        date(2026, 1, 3),
        1,
    )
    pd.testing.assert_frame_equal(live.reference, record(1).reference)


def test_registry_refuses_a_version_out_of_step(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="registry created version 1, record says 2"):
        registry_at(tmp_path).promote("toy", record(2))


def real_components(state: Path) -> Components:
    return Components(
        SchemaValidator(), EvidentlyDrift(0.5), {"holdout": HoldoutEvaluator()}, registry_at(state)
    )


def test_loop_on_real_stages_persists_and_resumes(tmp_path: Path) -> None:
    first = run_loop(ToyUseCase(), START, 14, real_components(tmp_path))
    assert any(d.promoted for d in first)

    later = run_loop(ToyUseCase(), None, 10, real_components(tmp_path))
    assert later[0].day == START + timedelta(days=14)
    assert later[0].live_version == first[-1].live_version

    logged = registry_at(tmp_path).decisions("toy")
    assert [d["day"] for d in logged] == [
        (START + timedelta(days=i)).isoformat() for i in range(24)
    ]
    benign = [d for d in later if any(e.benign for e in d.manifest.events)]
    assert all(d.drift.detected and "z" in d.drift.drifted for d in benign)
    assert not any(d.promoted for d in benign)
