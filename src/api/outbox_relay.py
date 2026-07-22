"""Relay transactionally stored feedback outbox records to Kafka."""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass

from src.api.feedback_publisher import FeedbackPublisher, build_reliable_feedback_publisher
from src.api.feedback_store import SqlFeedbackStore
from src.common.metrics_http import start_metrics_http_server
from src.common.metrics_stub import MetricsRegistry
from src.common.structured_logging import configure_json_logging


@dataclass(frozen=True)
class RelayResult:
    published: int
    failed: int


def relay_once(
    *,
    store: SqlFeedbackStore,
    publisher: FeedbackPublisher,
    batch_size: int = 100,
    metrics: MetricsRegistry | None = None,
) -> RelayResult:
    metric_registry = metrics or MetricsRegistry()
    published = 0
    failed = 0
    for record in store.pending(limit=batch_size):
        try:
            payload = dict(record["payload"])
            publisher.publish(payload)
            store.mark_published(str(record["feedback_id"]))
            published += 1
            metric_registry.inc("outbox_published_total")
        except Exception:  # noqa: BLE001
            failed += 1
            metric_registry.inc("outbox_publish_failures_total")
            break
    metric_registry.set_gauge("outbox_backlog", store.pending_count())
    return RelayResult(published=published, failed=failed)


def run_relay_loop(
    *,
    store: SqlFeedbackStore,
    publisher: FeedbackPublisher,
    batch_size: int = 100,
    poll_interval_s: float = 1.0,
    max_iterations: int | None = None,
    metrics: MetricsRegistry | None = None,
) -> RelayResult:
    total_published = 0
    total_failed = 0
    iterations = 0
    while max_iterations is None or iterations < max_iterations:
        result = relay_once(store=store, publisher=publisher, batch_size=batch_size, metrics=metrics)
        total_published += result.published
        total_failed += result.failed
        iterations += 1
        if result.published == 0:
            time.sleep(poll_interval_s)
    return RelayResult(published=total_published, failed=total_failed)


def main() -> None:
    configure_json_logging()
    parser = argparse.ArgumentParser(description="Relay the durable feedback outbox to Kafka.")
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL"))
    parser.add_argument("--bootstrap-servers", default=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"))
    parser.add_argument("--topic", default=os.getenv("FEEDBACK_TOPIC", "feedback"))
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--poll-interval-s", type=float, default=1.0)
    parser.add_argument("--metrics-port", type=int, default=8003)
    args = parser.parse_args()
    if not args.database_url:
        raise RuntimeError("DATABASE_URL is required.")

    metrics = MetricsRegistry()
    server = start_metrics_http_server(metrics=metrics, port=args.metrics_port)
    try:
        run_relay_loop(
            store=SqlFeedbackStore(args.database_url, create_schema=True),
            publisher=build_reliable_feedback_publisher(
                bootstrap_servers=args.bootstrap_servers,
                topic=args.topic,
            ),
            batch_size=args.batch_size,
            poll_interval_s=args.poll_interval_s,
            metrics=metrics,
        )
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
