# Feature Contract v1

Date: 2026-02-16

This document defines the production feature contract used by both offline training and streaming inference.

## Canonical feature list (ordered)

`FEATURES_V1`:

1. `amount`
2. `amount_ratio`
3. `balance_diff_orig`
4. `is_transfer`
5. `is_cashout`
6. `hour_of_day`
7. `txn_velocity_1h`

## Definitions

- `amount`: raw transaction amount.
  Formula: `amount`
- `amount_ratio`: relative size of transfer against origin balance.
  Formula: `amount / (oldbalanceOrg + 1.0)`
- `balance_diff_orig`: accounting consistency check on origin side.
  Formula: `oldbalanceOrg - newbalanceOrig - amount`
- `is_transfer`: type indicator.
  Formula: `1 if type == "TRANSFER" else 0`
- `is_cashout`: type indicator.
  Formula: `1 if type in {"CASH-OUT", "CASH_OUT"} else 0`
- `hour_of_day`: event hour.
  Formula: `timestamp.hour` (from converted `step` in offline)
- `txn_velocity_1h`: within-user transaction counter.
  Offline formula (v1): cumulative count grouped by `nameOrig` and `step`.
  Streaming formula (v1): keyed per `user_id` state counter in 1-hour scope.

## Constraints

- Ordering is strict. Models must consume features in exactly this order.
- Leakage fields must never appear in model input:
  - `is_fraud`
  - `isFraud`
  - `label`
- Defaults are not allowed silently. Missing required upstream fields should route event to DLQ.

## Versioning policy

- Any feature addition, removal, rename, order change, or formula change requires a new contract version.
- Code reference: `src/common/feature_contract.py`.
