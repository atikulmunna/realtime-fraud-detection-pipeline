"""EDA summary utilities for PaySim."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def build_summary(df: pd.DataFrame) -> dict[str, Any]:
    fraud_by_type = (
        df.groupby("type", dropna=False)["isFraud"]
        .agg(["count", "sum", "mean"])
        .rename(columns={"sum": "fraud_count", "mean": "fraud_rate"})
        .reset_index()
    )

    fraud_types = set(df.loc[df["isFraud"] == 1, "type"].astype(str).unique())
    allowed_fraud_types = {"TRANSFER", "CASH-OUT", "CASH_OUT"}
    invalid_fraud_types = sorted(fraud_types - allowed_fraud_types)

    amount_series = df["amount"]
    step_series = df["step"]
    step_min = int(step_series.min())
    step_max = int(step_series.max())

    summary: dict[str, Any] = {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "fraud_count": int(df["isFraud"].sum()),
        "fraud_rate": float(df["isFraud"].mean()),
        "class_imbalance_ratio": float((df["isFraud"] == 0).sum() / max((df["isFraud"] == 1).sum(), 1)),
        "step_range": {"min": step_min, "max": step_max},
        "timestamp_hours_span_from_step": int(step_max - step_min),
        "invalid_fraud_types": invalid_fraud_types,
        "fraud_by_type": fraud_by_type.to_dict(orient="records"),
        "amount_stats": {
            "min": float(amount_series.min()),
            "p50": float(amount_series.quantile(0.5)),
            "p95": float(amount_series.quantile(0.95)),
            "p99": float(amount_series.quantile(0.99)),
            "max": float(amount_series.max()),
        },
    }
    return summary


def run_eda(input_csv: str | Path, output_json: str | Path | None = None) -> dict[str, Any]:
    df = pd.read_csv(input_csv)
    summary = build_summary(df)
    if output_json:
        out = Path(output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run PaySim EDA checks and emit JSON summary.")
    parser.add_argument("--input", required=True, help="Path to PaySim CSV.")
    parser.add_argument("--out", default="docs/eda_summary_v1.json", help="Path to output summary JSON.")
    args = parser.parse_args()

    summary = run_eda(args.input, args.out)
    print(f"rows={summary['rows']} fraud_count={summary['fraud_count']} fraud_rate={summary['fraud_rate']:.6f}")
    if summary["invalid_fraud_types"]:
        print(f"invalid_fraud_types={summary['invalid_fraud_types']}")
    print(f"saved={args.out}")


if __name__ == "__main__":
    main()
