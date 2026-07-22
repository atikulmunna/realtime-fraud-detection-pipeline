import json
import math
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

from src.streaming.event_parser import parse_and_validate_event, route_parse_result


def _valid_event() -> dict:
    return {
        "event_id": "evt-1",
        "timestamp": "2026-02-16T12:00:00Z",
        "user_id": "C123",
        "type": "TRANSFER",
        "amount": 100.0,
        "old_balance_orig": 500.0,
        "new_balance_orig": 400.0,
    }


def test_parse_and_validate_event_happy_path():
    result = parse_and_validate_event(_valid_event())
    assert result.ok
    assert result.event is not None
    assert result.event["event_id"] == "evt-1"
    assert result.dlq is None


def test_parse_and_validate_event_strips_leakage_fields():
    payload = _valid_event()
    payload["is_fraud"] = 1
    payload["isFraud"] = 1
    payload["label"] = "true_positive"
    result = parse_and_validate_event(payload)
    assert result.ok
    assert "is_fraud" not in result.event
    assert "isFraud" not in result.event
    assert "label" not in result.event


def test_parse_and_validate_event_missing_required_routes_dlq():
    payload = _valid_event()
    payload.pop("user_id")
    result = parse_and_validate_event(payload)
    assert not result.ok
    assert result.dlq is not None
    assert "Missing required fields" in result.dlq["error"]


def test_parse_and_validate_event_invalid_type_enum_routes_dlq():
    payload = _valid_event()
    payload["type"] = "WIRE"
    result = parse_and_validate_event(payload)
    assert not result.ok
    assert result.dlq is not None
    assert "Invalid value for 'type'" in result.dlq["error"]


def test_parse_and_validate_event_invalid_json_routes_dlq():
    result = parse_and_validate_event("{bad json}")
    assert not result.ok
    assert result.dlq is not None
    assert "Expecting property name enclosed in double quotes" in result.dlq["error"]


def test_route_parse_result_routes_valid_to_main_topic():
    result = parse_and_validate_event(_valid_event())
    topic, payload = route_parse_result(result, valid_topic="clean-events", dlq_topic="dead-letter")
    assert topic == "clean-events"
    assert payload["event_id"] == "evt-1"


def test_route_parse_result_routes_invalid_to_dlq_topic():
    payload = _valid_event()
    payload["type"] = "BAD_TYPE"
    result = parse_and_validate_event(payload)
    topic, dlq_payload = route_parse_result(result, valid_topic="clean-events", dlq_topic="dead-letter")
    assert topic == "dead-letter"
    assert "Invalid value for 'type'" in dlq_payload["error"]


def test_parse_and_validate_event_rejects_wrong_field_type():
    payload = _valid_event()
    payload["timestamp"] = 123456
    result = parse_and_validate_event(payload)
    assert not result.ok
    assert "must be a string" in result.dlq["error"]


def test_parse_and_validate_event_dlq_raw_event_bytes_are_serializable():
    bad_bytes = b'{"event_id":"x", bad_json}'
    result = parse_and_validate_event(bad_bytes)
    assert not result.ok
    assert isinstance(result.dlq["raw_event"], str)


def test_parse_and_validate_event_normalizes_legacy_transaction_types():
    for legacy, canonical in (("CASH-OUT", "CASH_OUT"), ("CASH-IN", "CASH_IN")):
        payload = _valid_event()
        payload["type"] = legacy

        result = parse_and_validate_event(payload)

        assert result.ok
        assert result.event["type"] == canonical


def test_parse_and_validate_event_accepts_canonical_paysim_transaction_types():
    for transaction_type in ("CASH_OUT", "CASH_IN"):
        payload = _valid_event()
        payload["type"] = transaction_type
        assert parse_and_validate_event(payload).ok


def test_parse_and_validate_event_rejects_timestamp_without_timezone():
    payload = _valid_event()
    payload["timestamp"] = "2026-02-16T12:00:00"

    result = parse_and_validate_event(payload)

    assert not result.ok
    assert result.dlq["error_code"] == "INVALID_FORMAT"
    assert result.dlq["stage"] == "parse"


def test_parse_and_validate_event_rejects_non_finite_numbers():
    payload = _valid_event()
    payload["amount"] = math.nan

    result = parse_and_validate_event(payload)

    assert not result.ok
    assert result.dlq["error_code"] == "NON_FINITE_NUMBER"


def test_parse_dlq_record_conforms_to_schema():
    result = parse_and_validate_event("{bad json}")
    schema = json.loads(Path("schemas/dlq_v1.json").read_text(encoding="utf-8"))

    Draft202012Validator(schema, format_checker=FormatChecker()).validate(result.dlq)
