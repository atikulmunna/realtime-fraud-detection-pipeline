"""Publisher abstractions for feedback events."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from kafka import KafkaProducer


class FeedbackPublisher(Protocol):
    def publish(self, payload: dict[str, Any]) -> None:
        ...


@dataclass
class KafkaFeedbackPublisher:
    producer: KafkaProducer
    topic: str

    def publish(self, payload: dict[str, Any]) -> None:
        self.producer.send(self.topic, payload)
        self.producer.flush()


def build_kafka_feedback_publisher(
    *,
    bootstrap_servers: str = "localhost:9092",
    topic: str = "feedback",
) -> KafkaFeedbackPublisher:
    producer = KafkaProducer(
        bootstrap_servers=[bootstrap_servers],
        # Avoid eager broker version probe at startup so API can boot before Kafka is available.
        api_version=(2, 8, 0),
        request_timeout_ms=3000,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
    return KafkaFeedbackPublisher(producer=producer, topic=topic)
