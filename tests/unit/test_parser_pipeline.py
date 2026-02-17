from src.streaming.parser_pipeline import process_payload_batch


def _valid_event(event_id: str) -> dict:
    return {
        "event_id": event_id,
        "timestamp": "2026-02-16T12:00:00Z",
        "user_id": "C123",
        "type": "TRANSFER",
        "amount": 100.0,
        "old_balance_orig": 500.0,
        "new_balance_orig": 400.0,
    }


def test_process_payload_batch_routes_valid_and_invalid():
    payloads = [
        _valid_event("evt-1"),
        {**_valid_event("evt-2"), "type": "BAD_TYPE"},
        "{bad json}",
        {**_valid_event("evt-3"), "is_fraud": 1},
    ]

    out = process_payload_batch(payloads, valid_topic="clean-events", dlq_topic="dead-letter")

    assert set(out.keys()) == {"clean-events", "dead-letter"}
    assert len(out["clean-events"]) == 2
    assert len(out["dead-letter"]) == 2
    assert {e["event_id"] for e in out["clean-events"]} == {"evt-1", "evt-3"}
    assert "is_fraud" not in out["clean-events"][1]
    assert "error" in out["dead-letter"][0]


def test_process_payload_batch_all_valid():
    payloads = [_valid_event("evt-1"), _valid_event("evt-2")]
    out = process_payload_batch(payloads)
    assert len(out["parsed-events"]) == 2
    assert len(out["dead-letter"]) == 0
