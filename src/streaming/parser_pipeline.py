"""Lightweight parser pipeline helpers for batch-style integration tests."""

from __future__ import annotations

from typing import Any

from src.streaming.event_parser import parse_and_validate_event, route_parse_result


def process_payload_batch(
    payloads: list[str | bytes | dict[str, Any]],
    *,
    valid_topic: str = "parsed-events",
    dlq_topic: str = "dead-letter",
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {
        valid_topic: [],
        dlq_topic: [],
    }

    for payload in payloads:
        result = parse_and_validate_event(payload)
        topic, routed_payload = route_parse_result(
            result,
            valid_topic=valid_topic,
            dlq_topic=dlq_topic,
        )
        out[topic].append(routed_payload)

    return out
