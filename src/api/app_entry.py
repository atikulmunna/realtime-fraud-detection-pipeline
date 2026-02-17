"""Factory entrypoint for running the feedback API with uvicorn."""

from __future__ import annotations

import os

from src.api.feedback_publisher import build_kafka_feedback_publisher
from src.api.main import create_app


def app_factory():
    bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    topic = os.getenv("FEEDBACK_TOPIC", "feedback")
    publisher = build_kafka_feedback_publisher(bootstrap_servers=bootstrap, topic=topic)
    return create_app(publisher=publisher)
