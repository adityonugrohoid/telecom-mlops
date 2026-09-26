"""The root-cause baseline: always pick the first alarm, ported from the root-cause-analysis
repo (baseline.py). On the source evidence it found the root in 58% of incidents against the
model's 89%."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FirstAlarm:
    def root_score(self, data: pd.DataFrame) -> np.ndarray:
        """Earlier in the cascade scores higher; position 0 ranks first."""
        return -data["event_sequence_position"].to_numpy(dtype=float)
