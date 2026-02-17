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
