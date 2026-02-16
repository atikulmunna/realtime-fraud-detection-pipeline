"""Feature engineering for PaySim training data."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

BASE_DATETIME = datetime(2024, 1, 1, 0, 0, 0)


def add_timestamp_from_step(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = out["step"].apply(lambda s: BASE_DATETIME + timedelta(hours=int(s)))
    return out


def add_training_features(df: pd.DataFrame) -> pd.DataFrame:
    out = add_timestamp_from_step(df)
    txn_type = out["type"].astype(str)

    out["amount_ratio"] = out["amount"] / (out["oldbalanceOrg"] + 1.0)
    out["balance_diff_orig"] = out["oldbalanceOrg"] - out["newbalanceOrig"] - out["amount"]
    out["is_transfer"] = (txn_type == "TRANSFER").astype(int)
    out["is_cashout"] = txn_type.isin(["CASH-OUT", "CASH_OUT"]).astype(int)
    out["hour_of_day"] = pd.to_datetime(out["timestamp"]).dt.hour

    out = out.sort_values(["nameOrig", "timestamp"]).reset_index(drop=True)
    out["txn_velocity_1h"] = out.groupby(["nameOrig", "step"]).cumcount() + 1

    return out


def build_features(input_csv: str | Path, output_parquet: str | Path) -> None:
    df = pd.read_csv(input_csv)
    feat_df = add_training_features(df)
    Path(output_parquet).parent.mkdir(parents=True, exist_ok=True)
    feat_df.to_parquet(output_parquet, index=False)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build PaySim engineered feature parquet.")
    parser.add_argument("--input", required=True, help="Path to PaySim CSV.")
    parser.add_argument("--output", required=True, help="Path to output parquet.")
    args = parser.parse_args()
    build_features(args.input, args.output)
    print(f"Saved features to {args.output}")


if __name__ == "__main__":
    main()
