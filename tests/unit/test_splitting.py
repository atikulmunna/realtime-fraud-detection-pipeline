import pandas as pd
import pytest

from src.data.splitting import chronological_split


def test_chronological_split_keeps_steps_disjoint_and_ordered():
    df = pd.DataFrame({"step": [4, 1, 2, 1, 3, 5, 6, 7, 8, 9], "value": range(10)})

    splits = chronological_split(df)

    train_steps = set(splits.train["step"])
    validation_steps = set(splits.validation["step"])
    test_steps = set(splits.test["step"])
    assert train_steps.isdisjoint(validation_steps | test_steps)
    assert validation_steps.isdisjoint(test_steps)
    assert max(train_steps) < min(validation_steps) < min(test_steps)


def test_chronological_split_falls_back_to_row_order_without_step():
    df = pd.DataFrame({"value": range(20)})

    splits = chronological_split(df)

    assert list(splits.train["value"]) == list(range(14))
    assert list(splits.validation["value"]) == [14, 15, 16]
    assert list(splits.test["value"]) == [17, 18, 19]


def test_chronological_split_rejects_invalid_fractions():
    df = pd.DataFrame({"step": range(10)})
    with pytest.raises(ValueError, match="must be < 1"):
        chronological_split(df, train_fraction=0.8, validation_fraction=0.2)
