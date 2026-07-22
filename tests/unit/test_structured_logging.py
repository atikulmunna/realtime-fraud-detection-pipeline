import json
import logging

from src.common.structured_logging import JsonFormatter, log_event


def test_json_formatter_emits_correlation_and_event_fields():
    formatter = JsonFormatter()
    record = logging.LogRecord("fraud", logging.INFO, __file__, 1, "accepted", (), None)
    record.structured_fields = {"event": "feedback_accepted", "correlation_id": "feedback-1"}

    payload = json.loads(formatter.format(record))

    assert payload["level"] == "INFO"
    assert payload["event"] == "feedback_accepted"
    assert payload["correlation_id"] == "feedback-1"
    assert payload["timestamp"].endswith("Z")


def test_log_event_attaches_structured_fields(caplog):
    logger = logging.getLogger("fraud-test")
    with caplog.at_level(logging.INFO):
        log_event(logger, "candidate_promoted", correlation_id="update-3", version="7")

    record = caplog.records[-1]
    assert record.structured_fields == {
        "event": "candidate_promoted",
        "version": "7",
        "correlation_id": "update-3",
    }
