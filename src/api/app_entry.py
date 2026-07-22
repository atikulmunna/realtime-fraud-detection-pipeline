"""Factory entrypoint for running the feedback API with uvicorn."""

from __future__ import annotations

import os
from pathlib import Path

from src.api.feedback_publisher import build_kafka_feedback_publisher
from src.api.feedback_store import SqlFeedbackStore
from src.api.main import create_app
from src.common.structured_logging import configure_json_logging


def app_factory():
    configure_json_logging()
    app_env = os.getenv("APP_ENV", "production").strip().lower()
    database_url = os.getenv("DATABASE_URL")
    api_key = os.getenv("FEEDBACK_API_KEY")
    api_key_file = os.getenv("FEEDBACK_API_KEY_FILE")
    if not api_key and api_key_file:
        api_key = Path(api_key_file).read_text(encoding="utf-8").strip()
    if app_env != "development" and not api_key:
        raise RuntimeError("FEEDBACK_API_KEY is required outside development mode.")
    if database_url:
        store = SqlFeedbackStore(database_url, create_schema=os.getenv("CREATE_DB_SCHEMA") == "true")
        return create_app(feedback_store=store, api_key=api_key)
    if app_env != "development":
        raise RuntimeError("DATABASE_URL is required outside development mode.")
    bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    topic = os.getenv("FEEDBACK_TOPIC", "feedback")
    publisher = build_kafka_feedback_publisher(bootstrap_servers=bootstrap, topic=topic)
    return create_app(publisher=publisher, api_key=api_key)
