"""Chronological dataset splitting helpers shared by training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DatasetSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def chronological_split(
    df: pd.DataFrame,
    *,
    step_column: str = "step",
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> DatasetSplits:
    """Split rows chronologically without placing the same PaySim step in multiple splits."""
    if len(df) < 3:
        raise ValueError("At least three rows are required for train/validation/test splits.")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be within (0, 1).")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be within (0, 1).")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train_fraction + validation_fraction must be < 1.")

    if step_column not in df.columns:
        ordered = df.reset_index(drop=True)
        train_end = max(1, int(len(ordered) * train_fraction))
        validation_end = max(train_end + 1, int(len(ordered) * (train_fraction + validation_fraction)))
        validation_end = min(validation_end, len(ordered) - 1)
        return DatasetSplits(
            train=ordered.iloc[:train_end].copy(),
            validation=ordered.iloc[train_end:validation_end].copy(),
            test=ordered.iloc[validation_end:].copy(),
        )

    ordered_steps = sorted(df[step_column].dropna().unique().tolist())
    if len(ordered_steps) < 3:
        raise ValueError("At least three distinct chronological steps are required.")

    train_end = max(1, int(len(ordered_steps) * train_fraction))
    validation_end = max(train_end + 1, int(len(ordered_steps) * (train_fraction + validation_fraction)))
    validation_end = min(validation_end, len(ordered_steps) - 1)
    train_steps = set(ordered_steps[:train_end])
    validation_steps = set(ordered_steps[train_end:validation_end])
    test_steps = set(ordered_steps[validation_end:])

    return DatasetSplits(
        train=df[df[step_column].isin(train_steps)].sort_values(step_column).reset_index(drop=True),
        validation=df[df[step_column].isin(validation_steps)].sort_values(step_column).reset_index(drop=True),
        test=df[df[step_column].isin(test_steps)].sort_values(step_column).reset_index(drop=True),
    )
