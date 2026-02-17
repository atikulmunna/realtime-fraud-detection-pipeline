"""Adaptive threshold helpers for streaming anomaly routing."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def compute_adaptive_threshold(
    scores: Iterable[float],
    *,
    current_threshold: float = 0.5,
    quantile: float = 0.995,
    min_samples: int = 50,
    floor: float = 0.05,
    ceiling: float = 0.99,
) -> float:
    values = [float(s) for s in scores]
    if len(values) < min_samples:
        return float(current_threshold)

    q = float(np.quantile(np.array(values, dtype=float), quantile))
    return float(min(max(q, floor), ceiling))
