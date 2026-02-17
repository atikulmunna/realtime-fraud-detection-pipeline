"""Event parsing and validation helpers for streaming ingestion."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.common.feature_contract import LEAKAGE_FIELDS


DEFAULT_EVENT_SCHEMA_PATH = Path("schemas/event_v1.json")


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
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_schema(schema_path: str | Path = DEFAULT_EVENT_SCHEMA_PATH) -> dict[str, Any]:
    return json.loads(Path(schema_path).read_text(encoding="utf-8"))


def _coerce_to_dict(raw_payload: str | bytes | dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw_payload, dict):
        return raw_payload
    if isinstance(raw_payload, bytes):
        raw_payload = raw_payload.decode("utf-8")
    if isinstance(raw_payload, str):
        obj = json.loads(raw_payload)
        if not isinstance(obj, dict):
            raise ValueError("Payload JSON must decode to an object.")
        return obj
    raise ValueError(f"Unsupported payload type: {type(raw_payload).__name__}")


def _build_dlq(error: str, raw_event: Any) -> dict[str, Any]:
    raw_value: Any = raw_event
    if isinstance(raw_value, bytes):
        raw_value = raw_value.decode("utf-8", errors="replace")
    try:
        json.dumps(raw_value)
    except TypeError:
        raw_value = str(raw_value)

    return {
        "error": error,
        "raw_event": raw_value,
        "received_at": _utc_now_iso(),
    }


def _validate_required(event: dict[str, Any], required: list[str]) -> None:
    missing = [k for k in required if k not in event]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")


def _validate_types_and_lengths(event: dict[str, Any], props: dict[str, Any]) -> None:
    for field, spec in props.items():
        if field not in event or not isinstance(spec, dict):
            continue

        value = event[field]
        expected = spec.get("type")
        if expected == "string":
            if not isinstance(value, str):
                raise ValueError(f"Field '{field}' must be a string.")
            min_len = spec.get("minLength")
            if isinstance(min_len, int) and len(value) < min_len:
                raise ValueError(f"Field '{field}' must have length >= {min_len}.")
        elif expected == "number":
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"Field '{field}' must be numeric.")
        elif expected == "integer":
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"Field '{field}' must be an integer.")
        elif expected == "boolean":
            if not isinstance(value, bool):
                raise ValueError(f"Field '{field}' must be a boolean.")


def _validate_enum(event: dict[str, Any], enum_map: dict[str, set[Any]]) -> None:
    for field, allowed in enum_map.items():
        if field in event and event[field] not in allowed:
            raise ValueError(f"Invalid value for '{field}': {event[field]}")


def _validate_number_min(event: dict[str, Any], minimums: dict[str, float]) -> None:
    for field, min_value in minimums.items():
        if field in event:
            value = event[field]
            if not isinstance(value, (int, float)):
                raise ValueError(f"Field '{field}' must be numeric.")
            if value < min_value:
                raise ValueError(f"Field '{field}' must be >= {min_value}.")


def _sanitize_event(event: dict[str, Any]) -> dict[str, Any]:
    blocked = set(LEAKAGE_FIELDS) | {"is_fraud"}
    return {k: v for k, v in event.items() if k not in blocked}


def parse_and_validate_event(
    raw_payload: str | bytes | dict[str, Any],
    schema_path: str | Path = DEFAULT_EVENT_SCHEMA_PATH,
) -> ParseResult:
    schema = _load_schema(schema_path)
    required = list(schema.get("required", []))
    props = schema.get("properties", {})

    enum_map: dict[str, set[Any]] = {}
    minimums: dict[str, float] = {}
    for field, spec in props.items():
        if isinstance(spec, dict) and "enum" in spec:
            enum_map[field] = set(spec["enum"])
        if isinstance(spec, dict) and "minimum" in spec:
            minimums[field] = float(spec["minimum"])

    try:
        event = _coerce_to_dict(raw_payload)
        _validate_required(event, required)
        _validate_types_and_lengths(event, props)
        _validate_enum(event, enum_map)
        _validate_number_min(event, minimums)
        sanitized = _sanitize_event(event)
        return ParseResult(event=sanitized, dlq=None)
    except Exception as exc:
        return ParseResult(event=None, dlq=_build_dlq(str(exc), raw_payload))
