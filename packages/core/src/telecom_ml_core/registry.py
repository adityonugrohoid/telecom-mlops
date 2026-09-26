"""Run state: live models in the MLflow model registry, one MLflow run per loop day, and the
promotion log as JSON lines. Everything lives under the gitignored state folder.

Registered model names are the use case names; the live version carries the alias `live`.
"""

from __future__ import annotations

import json
import pickle
import tempfile
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST, ErrorCode

from telecom_ml_core.contract import DayDecision, LiveRecord

LIVE_ALIAS = "live"
MODEL_FILE = "model.pkl"
REFERENCE_FILE = "reference.parquet"


def sqlite_uri(state_dir: Path) -> str:
    """The local tracking and registry URI for a state folder."""
    return f"sqlite:///{(state_dir / 'mlflow.db').resolve()}"


def _is_missing(exc: MlflowException) -> bool:
    return bool(exc.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST))


def _jsonable(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(f"cannot serialise {type(value).__name__}")


class MlflowRegistry:
    """Live models, per-day runs and the promotion log for every use case.

    Args:
        tracking_uri: MLflow tracking and registry URI.
        artifact_root: Folder for run artifacts (models and drift references).
        log_dir: Folder for the promotion logs, one JSON-lines file per use case.
    """

    def __init__(self, tracking_uri: str, artifact_root: Path, log_dir: Path) -> None:
        self.client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
        self.artifact_root = artifact_root
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._live: dict[str, LiveRecord] = {}

    def _experiment(self, usecase: str) -> str:
        found = self.client.get_experiment_by_name(usecase)
        if found is not None:
            return str(found.experiment_id)
        location = (self.artifact_root / usecase).resolve().as_uri()
        return str(self.client.create_experiment(usecase, artifact_location=location))

    def _log_path(self, usecase: str) -> Path:
        return self.log_dir / f"{usecase}.jsonl"

    def live(self, usecase: str) -> LiveRecord | None:
        """The live model for a use case, or None before its first promotion."""
        if usecase in self._live:
            return self._live[usecase]
        try:
            version = self.client.get_model_version_by_alias(usecase, LIVE_ALIAS)
        except MlflowException as exc:
            if _is_missing(exc):
                return None
            raise
        run_id = version.run_id
        if run_id is None:
            raise RuntimeError(f"{usecase}: live version {version.version} has no source run")
        run = self.client.get_run(run_id)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(self.client.download_artifacts(run_id, "live", tmp))
            model = pickle.loads((folder / MODEL_FILE).read_bytes())
            reference = pd.read_parquet(folder / REFERENCE_FILE)
        record = LiveRecord(
            model=model,
            reference=reference,
            metrics=dict(run.data.metrics),
            promoted_on=date.fromisoformat(run.data.params["promoted_on"]),
            version=int(version.version),
        )
        self._live[usecase] = record
        return record

    def promote(self, usecase: str, record: LiveRecord) -> None:
        """Store a model as a new registered version and point the live alias at it.

        Raises:
            RuntimeError: when the registry's version number disagrees with the record's.
        """
        run = self.client.create_run(
            self._experiment(usecase),
            run_name=f"promote-v{record.version}-{record.promoted_on}",
            tags={"kind": "promotion"},
        )
        run_id = run.info.run_id
        self.client.log_param(run_id, "promoted_on", record.promoted_on.isoformat())
        self.client.log_param(run_id, "version", record.version)
        for name, value in record.metrics.items():
            self.client.log_metric(run_id, name, value)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / MODEL_FILE).write_bytes(pickle.dumps(record.model))
            record.reference.to_parquet(folder / REFERENCE_FILE)
            self.client.log_artifacts(run_id, str(folder), "live")
        self.client.set_terminated(run_id)

        try:
            self.client.get_registered_model(usecase)
        except MlflowException as exc:
            if not _is_missing(exc):
                raise
            self.client.create_registered_model(usecase)
        version = self.client.create_model_version(usecase, f"{run.info.artifact_uri}/live", run_id)
        if int(version.version) != record.version:
            raise RuntimeError(
                f"{usecase}: registry created version {version.version}, record says "
                f"{record.version}; state folder and loop disagree"
            )
        self.client.set_registered_model_alias(usecase, LIVE_ALIAS, version.version)
        self._live[usecase] = record

    def log_decision(self, decision: DayDecision) -> None:
        """Append the day's decision to the promotion log and record it as an MLflow run."""
        with self._log_path(decision.usecase).open("a") as log:
            log.write(json.dumps(asdict(decision), default=_jsonable) + "\n")

        run = self.client.create_run(
            self._experiment(decision.usecase),
            run_name=f"day-{decision.day}",
            tags={
                "kind": "day",
                "day": decision.day.isoformat(),
                "retrained": str(decision.retrained),
                "promoted": str(decision.promoted),
                "reason": decision.reason,
            },
        )
        run_id = run.info.run_id
        self.client.log_metric(run_id, "drift_share", decision.drift.share_drifted)
        self.client.log_metric(run_id, "live_version", decision.live_version)
        for prefix, metrics in (
            ("live", decision.live_metrics),
            ("candidate", decision.candidate_metrics),
            ("baseline", decision.baseline_metrics),
        ):
            for name, value in (metrics or {}).items():
                self.client.log_metric(run_id, f"{prefix}_{name}", value)
        self.client.set_terminated(run_id)

    def decisions(self, usecase: str) -> list[dict[str, Any]]:
        """The promotion log for a use case, oldest first."""
        path = self._log_path(usecase)
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines()]

    def last_day(self, usecase: str) -> date | None:
        """The last simulated day in the promotion log."""
        logged = self.decisions(usecase)
        return date.fromisoformat(logged[-1]["day"]) if logged else None
