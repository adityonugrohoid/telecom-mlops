"""Chronological holdout for the supervised use cases.

Windows are keyed on the day a label was released, never on a random split:
the candidate trains on labels released before the evaluation window, and every model
is scored on the labels released inside it.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, timedelta
from typing import Any

import pandas as pd

from telecom_ml_core.contract import Metrics, UseCase
from telecom_ml_core.pipeline import DataSource


class HoldoutEvaluator:
    def train_window(self, usecase: UseCase, day: date) -> tuple[date, date]:
        """The release days a candidate retrained on `day` learns from: the
        `train_window_days` before the evaluation window."""
        last = day - timedelta(days=usecase.eval_window_days)
        return last - timedelta(days=usecase.train_window_days - 1), last

    def training_data(self, usecase: UseCase, day: date, source: DataSource) -> pd.DataFrame:
        """Rows released in the `train_window_days` before the evaluation window.

        Raises:
            ValueError: when no labelled rows were released in the window.
        """
        first, last = self.train_window(usecase, day)
        train = source.released(first, last)
        if train.empty:
            raise ValueError(f"{usecase.name} {day}: no labels released in {first}..{last}")
        return train

    def evaluate(
        self, usecase: UseCase, models: Mapping[str, Any], day: date, source: DataSource
    ) -> dict[str, Metrics]:
        """Score every model on the rows released in the last `eval_window_days` up to `day`.

        Raises:
            ValueError: when no labelled rows were released in the window.
        """
        first = day - timedelta(days=usecase.eval_window_days - 1)
        frame = source.released(first, day)
        if frame.empty:
            raise ValueError(f"{usecase.name} {day}: no labels released in {first}..{day}")
        return {name: usecase.score(model, frame) for name, model in models.items()}
