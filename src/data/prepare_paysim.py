"""PaySim ingestion and validation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

EXPECTED_COLUMNS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
]

ALLOWED_TYPES = {"CASH-IN", "CASH-OUT", "CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"}


def _normalize_txn_type(value: str) -> str:
    return value.replace("_", "-")


@dataclass(frozen=True)
class ValidationResult:
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    rows: int
    cols: int

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0


def _check_required_columns(columns: Iterable[str]) -> list[str]:
    colset = set(columns)
    missing = [c for c in EXPECTED_COLUMNS if c not in colset]
    if not missing:
        return []
    return [f"Missing required columns: {', '.join(missing)}"]


def validate_paysim_dataframe(df: pd.DataFrame) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    errors.extend(_check_required_columns(df.columns))

    if "type" in df.columns:
        unknown_types = set(df["type"].astype(str).unique()) - ALLOWED_TYPES
        if unknown_types:
            errors.append(f"Unknown transaction types: {sorted(unknown_types)}")

    if "isFraud" in df.columns and "type" in df.columns:
        normalized = df["type"].astype(str).map(_normalize_txn_type)
        fraud_non_allowed = df[(df["isFraud"] == 1) & (~normalized.isin(["TRANSFER", "CASH-OUT"]))]
        if len(fraud_non_allowed) > 0:
            errors.append("Found fraudulent rows outside TRANSFER/CASH-OUT.")

    if "isFlaggedFraud" in df.columns:
        flagged = int((df["isFlaggedFraud"] == 1).sum())
        if flagged <= 20:
            warnings.append("isFlaggedFraud is sparse and should not be used as target.")

    if "step" in df.columns:
        max_step = int(df["step"].max())
        min_step = int(df["step"].min())
        if min_step < 1:
            errors.append("step must start at 1 or greater.")
        if max_step < 100:
            warnings.append("Maximum step is unexpectedly small; verify dataset completeness.")

    return ValidationResult(
        errors=tuple(errors),
        warnings=tuple(warnings),
        rows=int(df.shape[0]),
        cols=int(df.shape[1]),
    )


def load_and_validate(csv_path: str | Path) -> ValidationResult:
    df = pd.read_csv(csv_path)
    return validate_paysim_dataframe(df)


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Validate PaySim CSV format and constraints.")
    parser.add_argument("--input", required=True, help="Path to PaySim CSV.")
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    args = parser.parse_args()

    result = load_and_validate(args.input)
    payload: dict[str, Any] = {
        "ok": result.ok,
        "rows": result.rows,
        "cols": result.cols,
        "errors": list(result.errors),
        "warnings": list(result.warnings),
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"ok={payload['ok']} rows={payload['rows']} cols={payload['cols']}")
        if payload["errors"]:
            print("errors:")
            for item in payload["errors"]:
                print(f"- {item}")
        if payload["warnings"]:
            print("warnings:")
            for item in payload["warnings"]:
                print(f"- {item}")

    if not result.ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
