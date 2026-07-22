"""Continuous Kafka feedback consumer that feeds online SGD updates."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Protocol

import joblib
from kafka import KafkaConsumer

from src.api.feedback_publisher import build_reliable_feedback_publisher
from src.common.metrics_http import start_metrics_http_server
from src.common.metrics_stub import MetricsRegistry
from src.common.structured_logging import configure_json_logging, log_event
from src.models.registry import MlflowOnlineCandidateRegistry
from src.online.model_promotion import PromotionThresholds, evaluate_and_maybe_rollback
from src.online.online_sgd_updater import OnlineSGDUpdater

logger = logging.getLogger(__name__)


class ConsumerMessage(Protocol):
    value: Any


class FeedbackConsumer(Protocol):
    def poll(self, timeout_ms: int = 0, max_records: int | None = None) -> dict[Any, list[ConsumerMessage]]: ...

    def close(self) -> None: ...

    def commit(self) -> None: ...


class CandidateRegistry(Protocol):
    def register_candidate(self, candidate_path: str | Path, metadata: dict[str, Any]) -> str: ...

    def promote_candidate(self, version: str) -> None: ...


def _consumer_lag(consumer: FeedbackConsumer) -> int | None:
    assignment = getattr(consumer, "assignment", None)
    end_offsets = getattr(consumer, "end_offsets", None)
    position = getattr(consumer, "position", None)
    if not callable(assignment) or not callable(end_offsets) or not callable(position):
        return None
    partitions = assignment()
    if not partitions:
        return 0
    ends = end_offsets(partitions)
    return sum(max(0, int(ends[partition]) - int(position(partition))) for partition in partitions)


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
        enable_auto_commit=False,
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
    candidate_registry: CandidateRegistry | None = None,
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
    commits = 0
    has_uncommitted_records = False

    def commit_processed() -> None:
        nonlocal commits
        commit = getattr(consumer, "commit", None)
        if callable(commit):
            commit()
        commits += 1
        m.inc("online_consumer_commits_total")
        log_event(logger, "feedback_offsets_committed", commit_count=commits)

    def flush_and_evaluate(*, force: bool) -> bool:
        nonlocal updates, promotion_passed, promotion_failed, rollbacks
        backup_payload = joblib.load(updater.model_path)
        result = updater.flush(force=force, stage_candidate=True)
        if not result.updated:
            return False

        updates += 1
        if result.signal is None:
            raise RuntimeError("Updated candidate did not include a model-update signal.")
        candidate_version: str | None = None
        if candidate_registry is not None:
            candidate_version = candidate_registry.register_candidate(
                updater.candidate_path,
                {
                    "online_update_count": updater.online_update_count,
                    "batch_size": result.batch_size,
                    "features_order": updater.features_order,
                },
            )

        passed = True
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
                log_event(
                    logger,
                    "online_candidate_approved",
                    correlation_id=str(updater.online_update_count),
                    metrics=decision.metrics,
                )
            else:
                passed = False
                promotion_failed += 1
                m.inc("promotion_fail_total")
                if decision.rolled_back:
                    rollbacks += 1
                    m.inc("promotion_rollback_total")
                log_event(
                    logger,
                    "online_candidate_rejected",
                    correlation_id=str(updater.online_update_count),
                    level=logging.WARNING,
                    reasons=decision.reasons,
                )
        if passed:
            if candidate_registry is not None and candidate_version is not None:
                candidate_registry.promote_candidate(candidate_version)
            updater.promote_candidate()
        return True

    try:
        while True:
            remaining = None if max_messages is None else max_messages - messages_seen
            poll_limit = max_records_per_poll if remaining is None else min(max_records_per_poll, max(remaining, 1))
            polled = consumer.poll(timeout_ms=poll_timeout_ms, max_records=poll_limit)
            lag = _consumer_lag(consumer)
            if lag is not None:
                m.set_gauge("online_consumer_lag", lag)
            m.set_gauge("online_model_age_seconds", updater.model_age_seconds)
            records = [rec for recs in polled.values() for rec in recs]

            if records:
                has_uncommitted_records = True
                idle_polls = 0
                for rec in records:
                    messages_seen += 1
                    payload = rec.value
                    if isinstance(payload, (bytes, str)):
                        payload = json.loads(payload.decode("utf-8") if isinstance(payload, bytes) else payload)
                    if isinstance(payload, dict) and updater.add_feedback(payload):
                        accepted += 1
                        if updater.ready():
                            flush_and_evaluate(force=False)
                    else:
                        skipped += 1

                    m.inc("online_consumer_messages_total")

                    if max_messages is not None and messages_seen >= max_messages:
                        break
            else:
                idle_polls += 1

            now = time.monotonic()
            if now - last_flush_at >= flush_interval_s:
                flush_and_evaluate(force=True)
                last_flush_at = now

            if has_uncommitted_records and updater.buffer_size == 0:
                commit_processed()
                has_uncommitted_records = False

            if max_messages is not None and messages_seen >= max_messages:
                break
            if max_idle_polls is not None and idle_polls >= max_idle_polls:
                break

        if force_flush_on_exit:
            flush_and_evaluate(force=True)
            if has_uncommitted_records and updater.buffer_size == 0:
                commit_processed()
                has_uncommitted_records = False
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
        "commits": int(commits),
        "runtime_s": float(time.monotonic() - started),
    }


def main() -> None:
    configure_json_logging()
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
    parser.add_argument("--mlflow-tracking-uri", default=None)
    parser.add_argument("--registered-model-name", default="fraud-online-sgd")
    parser.add_argument("--model-update-topic", default="model-updates")
    parser.add_argument("--min-precision", type=float, default=0.0)
    parser.add_argument("--min-recall", type=float, default=0.0)
    parser.add_argument("--min-pr-auc", type=float, default=0.0)
    parser.add_argument("--metrics-host", default="0.0.0.0")
    parser.add_argument("--metrics-port", type=int, default=8002)
    args = parser.parse_args()

    metrics = MetricsRegistry()
    update_publisher = build_reliable_feedback_publisher(
        bootstrap_servers=args.bootstrap_servers,
        topic=args.model_update_topic,
        key_field="model_type",
    )
    updater = OnlineSGDUpdater(
        model_path=Path(args.model_path),
        batch_size=args.batch_size,
        metrics=metrics,
        publisher=update_publisher,
    )
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
    candidate_registry = (
        MlflowOnlineCandidateRegistry(
            tracking_uri=args.mlflow_tracking_uri,
            registered_model_name=args.registered_model_name,
        )
        if args.mlflow_tracking_uri
        else None
    )
    try:
        summary = (
            run_feedback_consumer_loop(
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
                candidate_registry=candidate_registry,
            ),
        )
        print(json.dumps(summary, indent=2))
    finally:
        metrics_server.shutdown()
        metrics_server.server_close()


if __name__ == "__main__":
    main()
