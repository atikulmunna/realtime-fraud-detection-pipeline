"""Publisher abstractions for feedback events."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from confluent_kafka import Producer
from kafka import KafkaProducer


class FeedbackPublisher(Protocol):
    def publish(self, payload: dict[str, Any]) -> None: ...


@dataclass
class KafkaFeedbackPublisher:
    producer: KafkaProducer
    topic: str

    def publish(self, payload: dict[str, Any]) -> None:
        self.producer.send(self.topic, payload)
        self.producer.flush()


@dataclass
class ConfluentFeedbackPublisher:
    producer: Producer
    topic: str
    key_field: str = "feedback_id"

    def publish(self, payload: dict[str, Any]) -> None:
        self.producer.produce(
            self.topic,
            value=json.dumps(payload).encode("utf-8"),
            key=str(payload[self.key_field]),
        )
        remaining = self.producer.flush(timeout=10.0)
        if remaining:
            raise RuntimeError(f"Kafka did not acknowledge {remaining} message(s).")


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


def build_reliable_feedback_publisher(
    *,
    bootstrap_servers: str = "localhost:9092",
    topic: str = "feedback",
    key_field: str = "feedback_id",
) -> ConfluentFeedbackPublisher:
    producer = Producer(
        {
            "bootstrap.servers": bootstrap_servers,
            "enable.idempotence": True,
            "acks": "all",
            "retries": 10,
            "delivery.timeout.ms": 30000,
            "client.id": "fraud-feedback-outbox",
        }
    )
    return ConfluentFeedbackPublisher(producer=producer, topic=topic, key_field=key_field)
