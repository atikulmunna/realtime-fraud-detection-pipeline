"""Continuous Kafka feedback consumer that feeds online SGD updates."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Protocol

from kafka import KafkaConsumer

from src.common.metrics_stub import MetricsRegistry
from src.online.online_sgd_updater import OnlineSGDUpdater


class ConsumerMessage(Protocol):
    value: Any


class FeedbackConsumer(Protocol):
    def poll(self, timeout_ms: int = 0, max_records: int | None = None) -> dict[Any, list[ConsumerMessage]]:
        ...

    def close(self) -> None:
        ...


def build_kafka_feedback_consumer(
    *,
    bootstrap_servers: str = "localhost:9092",
    topic: str = "feedback",
    group_id: str = "fraud-online-updater",
    auto_offset_reset: str = "latest",
) -> KafkaConsumer:
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=[bootstrap_servers],
        group_id=group_id,
        enable_auto_commit=True,
        auto_offset_reset=auto_offset_reset,
        value_deserializer=lambda b: json.loads(b.decode("utf-8")),
    )
    return consumer


def run_feedback_consumer_loop(
    *,
    updater: OnlineSGDUpdater,
    consumer: FeedbackConsumer,
    metrics: MetricsRegistry | None = None,
    poll_timeout_ms: int = 1000,
    max_records_per_poll: int = 100,
    flush_interval_s: float = 15.0,
    max_messages: int | None = None,
    max_idle_polls: int | None = None,
    force_flush_on_exit: bool = True,
) -> dict[str, Any]:
    if poll_timeout_ms <= 0:
        raise ValueError("poll_timeout_ms must be > 0")
    if max_records_per_poll <= 0:
        raise ValueError("max_records_per_poll must be > 0")
    if flush_interval_s <= 0:
        raise ValueError("flush_interval_s must be > 0")

    m = metrics or updater.metrics
    started = time.monotonic()
    last_flush_at = started
    idle_polls = 0

    messages_seen = 0
    accepted = 0
    updates = 0
    skipped = 0

    try:
        while True:
            polled = consumer.poll(timeout_ms=poll_timeout_ms, max_records=max_records_per_poll)
            records = [rec for recs in polled.values() for rec in recs]

            if records:
                idle_polls = 0
                for rec in records:
                    messages_seen += 1
                    payload = rec.value
                    if isinstance(payload, (bytes, str)):
                        payload = json.loads(payload.decode("utf-8") if isinstance(payload, bytes) else payload)
                    if isinstance(payload, dict) and updater.add_feedback(payload):
                        accepted += 1
                        if updater.ready():
                            res = updater.flush(force=False)
                            if res.updated:
                                updates += 1
                    else:
                        skipped += 1

                    m.inc("online_consumer_messages_total")

                    if max_messages is not None and messages_seen >= max_messages:
                        break
            else:
                idle_polls += 1

            now = time.monotonic()
            if now - last_flush_at >= flush_interval_s:
                res = updater.flush(force=False)
                if res.updated:
                    updates += 1
                last_flush_at = now

            if max_messages is not None and messages_seen >= max_messages:
                break
            if max_idle_polls is not None and idle_polls >= max_idle_polls:
                break

        if force_flush_on_exit:
            final = updater.flush(force=True)
            if final.updated:
                updates += 1
    finally:
        consumer.close()

    return {
        "messages_seen": int(messages_seen),
        "accepted": int(accepted),
        "skipped": int(skipped),
        "updates": int(updates),
        "runtime_s": float(time.monotonic() - started),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run continuous feedback consumer for online updates.")
    parser.add_argument("--model-path", default="models/sgd_classifier_v1.joblib")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument("--topic", default="feedback")
    parser.add_argument("--group-id", default="fraud-online-updater")
    parser.add_argument("--auto-offset-reset", default="latest", choices=["latest", "earliest"])
    parser.add_argument("--poll-timeout-ms", type=int, default=1000)
    parser.add_argument("--max-records-per-poll", type=int, default=100)
    parser.add_argument("--flush-interval-s", type=float, default=15.0)
    parser.add_argument("--max-messages", type=int, default=None)
    parser.add_argument("--max-idle-polls", type=int, default=None)
    parser.add_argument("--no-force-flush-on-exit", action="store_true")
    args = parser.parse_args()

    metrics = MetricsRegistry()
    updater = OnlineSGDUpdater(model_path=Path(args.model_path), batch_size=args.batch_size, metrics=metrics)
    consumer = build_kafka_feedback_consumer(
        bootstrap_servers=args.bootstrap_servers,
        topic=args.topic,
        group_id=args.group_id,
        auto_offset_reset=args.auto_offset_reset,
    )
    summary = run_feedback_consumer_loop(
        updater=updater,
        consumer=consumer,
        metrics=metrics,
        poll_timeout_ms=args.poll_timeout_ms,
        max_records_per_poll=args.max_records_per_poll,
        flush_interval_s=args.flush_interval_s,
        max_messages=args.max_messages,
        max_idle_polls=args.max_idle_polls,
        force_flush_on_exit=not args.no_force_flush_on_exit,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
