"""Simulation rollouts for the policy use case.

Each day is an environment version. The evaluator owns the episode seeds, so the live
policy, the candidate and the baseline all run exactly the same episodes.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import Any

import pandas as pd

from telecom_ml_core.contract import Metrics, UseCase
from telecom_ml_core.pipeline import DataSource

EPISODE_SEED = "episode_seed"


class RolloutEvaluator:
    """Fixed-seed episodes on each day's environment version.

    Args:
        episodes: Episodes per environment version.
        seed: First episode seed; episode i uses `seed + i` on every day and for every model.
    """

    def __init__(self, episodes: int, seed: int) -> None:
        if episodes <= 0:
            raise ValueError(f"episodes must be > 0, got {episodes}")
        self.seeds = [seed + i for i in range(episodes)]

    def training_data(self, usecase: UseCase, day: date, source: DataSource) -> pd.DataFrame:
        """The day's environment version; the policy trains by interacting with it."""
        return source.batch(day).data

    def evaluate(
        self, usecase: UseCase, models: Mapping[str, Any], day: date, source: DataSource
    ) -> dict[str, Metrics]:
        """Score every policy on the day's environment crossed with the fixed episode seeds.

        The use case's `score` runs one episode per row of the frame it receives.
        """
        env = source.batch(day).data
        if EPISODE_SEED in env.columns:
            raise ValueError(f"{usecase.name}: environment already has a {EPISODE_SEED!r} column")
        frame = env.merge(pd.DataFrame({EPISODE_SEED: self.seeds}), how="cross")
        return {name: usecase.score(model, frame) for name, model in models.items()}
