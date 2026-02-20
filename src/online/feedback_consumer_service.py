"""Continuous Kafka feedback consumer that feeds online SGD updates."""

from __future__ import annotations

import argparse
import json
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any, Protocol

import joblib
from kafka import KafkaConsumer

from src.common.metrics_stub import MetricsRegistry
from src.online.model_promotion import PromotionThresholds, evaluate_and_maybe_rollback
from src.online.online_sgd_updater import OnlineSGDUpdater


class ConsumerMessage(Protocol):
    value: Any


class FeedbackConsumer(Protocol):
    def poll(self, timeout_ms: int = 0, max_records: int | None = None) -> dict[Any, list[ConsumerMessage]]:
        ...

    def close(self) -> None:
        ...


def start_metrics_http_server(
    *,
    metrics: MetricsRegistry,
    host: str = "0.0.0.0",
    port: int = 8002,
) -> ThreadingHTTPServer:
    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/metrics":
                body = metrics.render_prometheus().encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/plain; version=0.0.4")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path == "/health":
                body = b'{"status":"ok"}'
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            self.send_response(HTTPStatus.NOT_FOUND)
            self.end_headers()

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return

    server = ThreadingHTTPServer((host, int(port)), _Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


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
    promotion_holdout_parquet: str | Path | None = None,
    promotion_thresholds: PromotionThresholds | None = None,
) -> dict[str, Any]:
    if poll_timeout_ms <= 0:
        raise ValueError("poll_timeout_ms must be > 0")
    if max_records_per_poll <= 0:
        raise ValueError("max_records_per_poll must be > 0")
    if flush_interval_s <= 0:
        raise ValueError("flush_interval_s must be > 0")

    m = metrics or updater.metrics
    m.inc("online_consumer_messages_total", 0.0)
    m.inc("promotion_pass_total", 0.0)
    m.inc("promotion_fail_total", 0.0)
    m.inc("promotion_rollback_total", 0.0)
    started = time.monotonic()
    last_flush_at = started
    idle_polls = 0

    messages_seen = 0
    accepted = 0
    updates = 0
    skipped = 0
    promotion_passed = 0
    promotion_failed = 0
    rollbacks = 0

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
                            backup_payload = joblib.load(updater.model_path)
                            res = updater.flush(force=False)
                            if res.updated:
                                updates += 1
                                if promotion_holdout_parquet is not None:
                                    decision = evaluate_and_maybe_rollback(
                                        updater=updater,
                                        backup_payload=backup_payload,
                                        holdout_parquet=promotion_holdout_parquet,
                                        thresholds=promotion_thresholds or PromotionThresholds(),
                                    )
                                    if decision.passed:
                                        promotion_passed += 1
                                        m.inc("promotion_pass_total")
                                    else:
                                        promotion_failed += 1
                                        m.inc("promotion_fail_total")
                                        if decision.rolled_back:
                                            rollbacks += 1
                                            m.inc("promotion_rollback_total")
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
            backup_payload = joblib.load(updater.model_path)
            final = updater.flush(force=True)
            if final.updated:
                updates += 1
                if promotion_holdout_parquet is not None:
                    decision = evaluate_and_maybe_rollback(
                        updater=updater,
                        backup_payload=backup_payload,
                        holdout_parquet=promotion_holdout_parquet,
                        thresholds=promotion_thresholds or PromotionThresholds(),
                    )
                    if decision.passed:
                        promotion_passed += 1
                        m.inc("promotion_pass_total")
                    else:
                        promotion_failed += 1
                        m.inc("promotion_fail_total")
                        if decision.rolled_back:
                            rollbacks += 1
                            m.inc("promotion_rollback_total")
    finally:
        consumer.close()

    return {
        "messages_seen": int(messages_seen),
        "accepted": int(accepted),
        "skipped": int(skipped),
        "updates": int(updates),
        "promotion_passed": int(promotion_passed),
        "promotion_failed": int(promotion_failed),
        "rollbacks": int(rollbacks),
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
    parser.add_argument("--promotion-holdout", default=None, help="Parquet path with FEATURES_V1 + isFraud.")
    parser.add_argument("--min-precision", type=float, default=0.0)
    parser.add_argument("--min-recall", type=float, default=0.0)
    parser.add_argument("--min-pr-auc", type=float, default=0.0)
    parser.add_argument("--metrics-host", default="0.0.0.0")
    parser.add_argument("--metrics-port", type=int, default=8002)
    args = parser.parse_args()

    metrics = MetricsRegistry()
    updater = OnlineSGDUpdater(model_path=Path(args.model_path), batch_size=args.batch_size, metrics=metrics)
    consumer = build_kafka_feedback_consumer(
        bootstrap_servers=args.bootstrap_servers,
        topic=args.topic,
        group_id=args.group_id,
        auto_offset_reset=args.auto_offset_reset,
    )
    metrics_server = start_metrics_http_server(
        metrics=metrics,
        host=args.metrics_host,
        port=args.metrics_port,
    )
    try:
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
            promotion_holdout_parquet=args.promotion_holdout,
            promotion_thresholds=PromotionThresholds(
                min_precision=float(args.min_precision),
                min_recall=float(args.min_recall),
                min_pr_auc=float(args.min_pr_auc),
            ),
        ),
        print(json.dumps(summary, indent=2))
    finally:
        metrics_server.shutdown()
        metrics_server.server_close()


if __name__ == "__main__":
    main()
