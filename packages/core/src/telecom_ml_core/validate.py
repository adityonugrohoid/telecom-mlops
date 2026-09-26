"""Stage 2: check a day's batch against its use case's schema before anything uses it."""

from __future__ import annotations

import pandas as pd
import pandera.errors

from telecom_ml_core.contract import LABEL_RELEASE, Batch, UseCase


class BatchValidationError(ValueError):
    """A batch broke its use case's contract. Generators are deterministic, so this is a bug."""


class SchemaValidator:
    """Runs the use case's pandera schema lazily, so one error lists every failing check."""

    def validate(self, usecase: UseCase, batch: Batch) -> None:
        """Validate one batch.

        Args:
            usecase: The use case that generated the batch.
            batch: The day's batch.

        Raises:
            BatchValidationError: naming the use case, the day and every failed check.
        """
        where = f"{usecase.name} {batch.day}"
        if batch.manifest.day != batch.day:
            raise BatchValidationError(
                f"{where}: manifest is for {batch.manifest.day}, not the batch day"
            )
        if batch.data.empty:
            raise BatchValidationError(f"{where}: empty batch")
        released = pd.to_datetime(batch.data[LABEL_RELEASE])
        early = int((released < pd.Timestamp(batch.day)).sum())
        if early:
            raise BatchValidationError(f"{where}: {early} rows release their label before the day")
        try:
            usecase.schema().validate(batch.data, lazy=True)
        except pandera.errors.SchemaErrors as exc:
            failures = exc.failure_cases[["column", "check", "failure_case"]].head(20)
            raise BatchValidationError(
                f"{where}: schema failed\n{failures.to_string(index=False)}"
            ) from exc
