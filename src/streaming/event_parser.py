"""Event parsing and validation helpers for streaming ingestion."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import ValidationError

from src.common.feature_contract import LEAKAGE_FIELDS

# Resolved from the module location rather than the working directory. Flink's
# Python harness executes UDFs from its own temp directory, where a CWD-relative
# path raises FileNotFoundError and fails every record. `src` and `schemas` are
# siblings both in the repository and in the image at /opt/fraud.
DEFAULT_EVENT_SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schemas" / "event_v1.json"
TRANSACTION_TYPE_ALIASES = {"CASH-OUT": "CASH_OUT", "CASH-IN": "CASH_IN"}


class EventValidationError(ValueError):
    def __init__(self, message: str, *, error_code: str) -> None:
        super().__init__(message)
        self.error_code = error_code


@dataclass(frozen=True)
class ParseResult:
    event: dict[str, Any] | None
    dlq: dict[str, Any] | None

    @property
    def ok(self) -> bool:
        return self.event is not None


def route_parse_result(
    result: ParseResult,
    *,
    valid_topic: str = "parsed-events",
    dlq_topic: str = "dead-letter",
) -> tuple[str, dict[str, Any]]:
    if result.ok:
        return valid_topic, result.event or {}
    return dlq_topic, result.dlq or {}


def _utc_now_iso() -> str:
    # The Flink 1.19 runtime uses Python 3.10, where datetime.UTC is unavailable.
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")  # noqa: UP017


def _load_schema(schema_path: str | Path = DEFAULT_EVENT_SCHEMA_PATH) -> dict[str, Any]:
    return json.loads(Path(schema_path).read_text(encoding="utf-8"))


def _coerce_to_dict(raw_payload: str | bytes | dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw_payload, dict):
        return dict(raw_payload)
    if isinstance(raw_payload, bytes):
        raw_payload = raw_payload.decode("utf-8")
    if isinstance(raw_payload, str):
        obj = json.loads(raw_payload)
        if not isinstance(obj, dict):
            raise EventValidationError("Payload JSON must decode to an object.", error_code="INVALID_PAYLOAD_SHAPE")
        return obj
    raise EventValidationError(
        f"Unsupported payload type: {type(raw_payload).__name__}",
        error_code="UNSUPPORTED_PAYLOAD_TYPE",
    )


def build_dlq_record(
    error: str,
    raw_event: Any,
    *,
    stage: str,
    error_code: str,
) -> dict[str, Any]:
    raw_value: Any = raw_event
    if isinstance(raw_value, bytes):
        raw_value = raw_value.decode("utf-8", errors="replace")
    try:
        json.dumps(raw_value)
    except TypeError:
        raw_value = str(raw_value)

    record = {
        "error": error,
        "error_code": error_code,
        "stage": stage,
        "raw_event": raw_value,
        "received_at": _utc_now_iso(),
    }
    if isinstance(raw_event, dict) and isinstance(raw_event.get("event_id"), str) and raw_event["event_id"]:
        record["event_id"] = raw_event["event_id"]
    return record


def _field_name(error: ValidationError) -> str:
    return str(error.absolute_path[-1]) if error.absolute_path else "payload"


def _format_schema_error(error: ValidationError, event: dict[str, Any], schema: dict[str, Any]) -> str:
    field = _field_name(error)
    if error.validator == "required":
        missing = [name for name in schema.get("required", []) if name not in event]
        return f"Missing required fields: {missing}"
    if error.validator == "enum":
        return f"Invalid value for '{field}': {error.instance}"
    if error.validator == "type":
        expected = error.validator_value
        article = "an" if str(expected)[:1].lower() in "aeiou" else "a"
        return f"Field '{field}' must be {article} {expected}."
    if error.validator == "minimum":
        return f"Field '{field}' must be >= {error.validator_value}."
    if error.validator == "format":
        return f"Field '{field}' must be a valid {error.validator_value}."
    return error.message


def _schema_error_code(error: ValidationError) -> str:
    return {
        "required": "MISSING_REQUIRED_FIELD",
        "enum": "INVALID_ENUM",
        "type": "INVALID_FIELD_TYPE",
        "minimum": "MINIMUM_VIOLATION",
        "format": "INVALID_FORMAT",
    }.get(str(error.validator), "SCHEMA_VALIDATION_ERROR")


def _validate_against_schema(event: dict[str, Any], schema: dict[str, Any]) -> None:
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    error = next(iter(validator.iter_errors(event)), None)
    if error is not None:
        raise EventValidationError(
            _format_schema_error(error, event, schema),
            error_code=_schema_error_code(error),
        )

    for field, spec in schema.get("properties", {}).items():
        if isinstance(spec, dict) and spec.get("type") == "number" and field in event:
            if not isfinite(float(event[field])):
                raise EventValidationError(
                    f"Field '{field}' must be finite.",
                    error_code="NON_FINITE_NUMBER",
                )

    timestamp = event.get("timestamp")
    if isinstance(timestamp, str):
        parsed_timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if parsed_timestamp.utcoffset() is None:
            raise EventValidationError(
                "Field 'timestamp' must include a timezone offset.",
                error_code="INVALID_FORMAT",
            )


def _sanitize_event(event: dict[str, Any]) -> dict[str, Any]:
    blocked = set(LEAKAGE_FIELDS) | {"is_fraud"}
    sanitized = {k: v for k, v in event.items() if k not in blocked}
    if "type" in sanitized:
        sanitized["type"] = TRANSACTION_TYPE_ALIASES.get(str(sanitized["type"]), sanitized["type"])
    return sanitized


def parse_and_validate_event(
    raw_payload: str | bytes | dict[str, Any],
    schema_path: str | Path = DEFAULT_EVENT_SCHEMA_PATH,
) -> ParseResult:
    schema = _load_schema(schema_path)
    try:
        event = _coerce_to_dict(raw_payload)
        _validate_against_schema(event, schema)
        sanitized = _sanitize_event(event)
        return ParseResult(event=sanitized, dlq=None)
    except json.JSONDecodeError as exc:
        return ParseResult(
            event=None,
            dlq=build_dlq_record(
                str(exc),
                raw_payload,
                stage="parse",
                error_code="INVALID_JSON",
            ),
        )
    except EventValidationError as exc:
        return ParseResult(
            event=None,
            dlq=build_dlq_record(
                str(exc),
                raw_payload,
                stage="parse",
                error_code=exc.error_code,
            ),
        )
    except Exception as exc:
        return ParseResult(
            event=None,
            dlq=build_dlq_record(
                str(exc),
                raw_payload,
                stage="parse",
                error_code="PARSER_ERROR",
            ),
        )
