"""Checkpointed Kafka-to-Kafka PyFlink fraud scoring job."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from src.common.feature_contract import FEATURES_V1
from src.streaming.ensemble_scoring import EnsembleModels, load_ensemble_models, score_event_features
from src.streaming.event_parser import build_dlq_record, parse_and_validate_event
from src.streaming.feature_extractor import enrich_event_with_features


@dataclass(frozen=True)
class FlinkJobSettings:
    bootstrap_servers: str = "kafka:29092"
    input_topic: str = "raw-events"
    model_update_topic: str = "model-updates"
    anomaly_topic: str = "anomalies"
    metrics_topic: str = "metrics"
    dlq_topic: str = "dead-letter"
    group_id: str = "fraud-flink-scoring"
    checkpoint_interval_ms: int = 30_000
    out_of_order_ms: int = 120_000
    model_dir: str = "/opt/fraud/models"
    if_model_path: str = "/opt/fraud/models/isolation_forest_v1.joblib"
    ae_model_path: str = "/opt/fraud/models/autoencoder_v1.joblib"
    sgd_model_path: str = "/opt/fraud/models/sgd_classifier_v1.joblib"


def event_timestamp_ms(event: dict[str, Any]) -> int:
    timestamp = datetime.fromisoformat(str(event["timestamp"]).replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        raise ValueError("Event timestamp must be timezone-aware.")
    return int(timestamp.timestamp() * 1000)


def utc_hour_key(timestamp_ms: int) -> str:
    timestamp = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc)  # noqa: UP017
    return timestamp.replace(minute=0, second=0, microsecond=0).isoformat()


def is_late_event(timestamp_ms: int, watermark_ms: int) -> bool:
    return watermark_ms >= 0 and timestamp_ms < watermark_ms


def event_user_key(raw: str) -> str:
    try:
        parsed = json.loads(raw)
        return str(parsed.get("user_id", "__invalid__")) if isinstance(parsed, dict) else "__invalid__"
    except (TypeError, ValueError, json.JSONDecodeError):
        return "__invalid__"


def envelope_user_key(raw: str) -> str:
    try:
        envelope = json.loads(raw)
        payload = envelope.get("payload", {})
        return str(payload.get("user_id", "__invalid__")) if isinstance(payload, dict) else "__invalid__"
    except (TypeError, ValueError, json.JSONDecodeError):
        return "__invalid__"


def validate_model_update_signal(signal: dict[str, Any], model_dir: str | Path) -> Path:
    if signal.get("model_type") != "sgd_classifier":
        raise ValueError("Only SGD champion update signals are supported.")
    source_name = Path(str(signal.get("model_path", ""))).name
    if not source_name or source_name.endswith(".candidate"):
        raise ValueError("Model update must reference a promoted artifact.")
    root = Path(model_dir).resolve()
    candidate = (root / source_name).resolve()
    if candidate.parent != root or not candidate.is_file():
        raise ValueError("Model update artifact is outside the model directory or missing.")
    payload = joblib.load(candidate)
    if list(payload.get("features_order", [])) != FEATURES_V1:
        raise ValueError("Model update feature order does not match FEATURES_V1.")
    if "model" not in payload:
        raise ValueError("Model update artifact has no model payload.")
    expected_checksum = str(signal.get("artifact_sha256", ""))
    actual_checksum = hashlib.sha256(candidate.read_bytes()).hexdigest()
    if len(expected_checksum) != 64 or expected_checksum != actual_checksum:
        raise ValueError("Model update artifact checksum is missing or invalid.")
    return candidate


def _settings_from_env() -> FlinkJobSettings:
    return FlinkJobSettings(
        bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092"),
        input_topic=os.getenv("RAW_EVENTS_TOPIC", "raw-events"),
        model_update_topic=os.getenv("MODEL_UPDATE_TOPIC", "model-updates"),
        anomaly_topic=os.getenv("ANOMALIES_TOPIC", "anomalies"),
        metrics_topic=os.getenv("METRICS_TOPIC", "metrics"),
        dlq_topic=os.getenv("DLQ_TOPIC", "dead-letter"),
        group_id=os.getenv("FLINK_CONSUMER_GROUP", "fraud-flink-scoring"),
        checkpoint_interval_ms=int(os.getenv("CHECKPOINT_INTERVAL_MS", "30000")),
        out_of_order_ms=int(os.getenv("OUT_OF_ORDER_MS", "120000")),
        model_dir=os.getenv("MODEL_DIR", "/opt/fraud/models"),
        if_model_path=os.getenv("IF_MODEL_PATH", "/opt/fraud/models/isolation_forest_v1.joblib"),
        ae_model_path=os.getenv("AE_MODEL_PATH", "/opt/fraud/models/autoencoder_v1.joblib"),
        sgd_model_path=os.getenv("SGD_MODEL_PATH", "/opt/fraud/models/sgd_classifier_v1.joblib"),
    )


def build_flink_job(settings: FlinkJobSettings) -> Any:  # pragma: no cover - exercised in the Flink image
    from pyflink.common import Duration, SimpleStringSchema, Types, WatermarkStrategy
    from pyflink.common.watermark_strategy import TimestampAssigner
    from pyflink.datastream import CheckpointingMode, StreamExecutionEnvironment
    from pyflink.datastream.checkpoint_config import ExternalizedCheckpointCleanup
    from pyflink.datastream.connectors.kafka import (
        DeliveryGuarantee,
        KafkaOffsetResetStrategy,
        KafkaOffsetsInitializer,
        KafkaRecordSerializationSchema,
        KafkaSink,
        KafkaSource,
    )
    from pyflink.datastream.functions import KeyedBroadcastProcessFunction, KeyedProcessFunction
    from pyflink.datastream.state import MapStateDescriptor

    class _TimestampAssigner(TimestampAssigner):
        def extract_timestamp(self, value: str, record_timestamp: int) -> int:
            try:
                raw = json.loads(value)
                return event_timestamp_ms(raw)
            except Exception:
                return record_timestamp if record_timestamp >= 0 else 0

    class _VelocityFunction(KeyedProcessFunction):
        def open(self, runtime_context: Any) -> None:
            self.hour_counts = runtime_context.get_map_state(
                MapStateDescriptor("user_hour_counts", Types.STRING(), Types.INT())
            )

        def process_element(self, value: str, ctx: Any):
            parsed = parse_and_validate_event(value)
            if not parsed.ok:
                yield json.dumps({"route": "dlq", "payload": parsed.dlq})
                return
            event = parsed.event or {}
            timestamp_ms = event_timestamp_ms(event)
            watermark = ctx.timer_service().current_watermark()
            if is_late_event(timestamp_ms, watermark):
                yield json.dumps(
                    {
                        "route": "dlq",
                        "payload": build_dlq_record(
                            "Event arrived behind the two-minute watermark.",
                            event,
                            stage="late_event",
                            error_code="LATE_EVENT",
                        ),
                    }
                )
                return
            hour = utc_hour_key(timestamp_ms)
            count = int(self.hour_counts.get(hour) or 0) + 1
            self.hour_counts.put(hour, count)
            hour_end_ms = ((timestamp_ms // 3_600_000) + 1) * 3_600_000
            ctx.timer_service().register_event_time_timer(hour_end_ms + settings.out_of_order_ms)
            enriched = enrich_event_with_features(event, txn_velocity_1h=count)
            yield json.dumps({"route": "score", "payload": enriched})

        def on_timer(self, timestamp: int, ctx: Any):
            expired_hour_start = timestamp - settings.out_of_order_ms - 3_600_000
            self.hour_counts.remove(utc_hour_key(expired_hour_start))

    update_descriptor = MapStateDescriptor("champion_model_update", Types.STRING(), Types.STRING())

    class _ScoringFunction(KeyedBroadcastProcessFunction):
        def open(self, runtime_context: Any) -> None:
            self.models: EnsembleModels = load_ensemble_models(
                if_model_path=settings.if_model_path,
                ae_model_path=settings.ae_model_path,
                sgd_model_path=settings.sgd_model_path,
            )
            self.applied_update_count = -1

        def _apply_signal(self, signal_json: str) -> None:
            signal = json.loads(signal_json)
            update_count = int(signal.get("online_update_count", -1))
            if update_count <= self.applied_update_count:
                return
            path = validate_model_update_signal(signal, settings.model_dir)
            payload = joblib.load(path)
            self.models = replace(
                self.models,
                sgd_model=payload["model"],
                model_version=str(payload.get("model_version", f"online-{update_count}")),
            )
            self.applied_update_count = update_count

        def process_broadcast_element(self, value: str, ctx: Any):
            self._apply_signal(value)
            ctx.get_broadcast_state(update_descriptor).put("champion", value)

        def process_element(self, value: str, ctx: Any):
            latest = ctx.get_broadcast_state(update_descriptor).get("champion")
            if latest:
                self._apply_signal(latest)
            envelope = json.loads(value)
            if envelope["route"] == "dlq":
                yield value
                return
            enriched = envelope["payload"]
            try:
                scores = score_event_features(enriched["features"], self.models)
                enriched["scores"] = scores
                enriched["model_version"] = self.models.model_version
                route = "anomaly" if scores["ensemble_score"] >= self.models.threshold else "metrics"
                yield json.dumps({"route": route, "payload": enriched})
            except Exception as exc:
                yield json.dumps(
                    {
                        "route": "dlq",
                        "payload": build_dlq_record(
                            str(exc),
                            enriched,
                            stage="scoring",
                            error_code="SCORING_ERROR",
                        ),
                    }
                )

    env = StreamExecutionEnvironment.get_execution_environment()
    env.enable_checkpointing(settings.checkpoint_interval_ms, CheckpointingMode.EXACTLY_ONCE)
    checkpoint_config = env.get_checkpoint_config()
    checkpoint_config.set_min_pause_between_checkpoints(5_000)
    checkpoint_config.set_checkpoint_timeout(60_000)
    checkpoint_config.set_max_concurrent_checkpoints(1)
    checkpoint_config.enable_externalized_checkpoints(ExternalizedCheckpointCleanup.RETAIN_ON_CANCELLATION)

    event_source = (
        KafkaSource.builder()
        .set_bootstrap_servers(settings.bootstrap_servers)
        .set_topics(settings.input_topic)
        .set_group_id(settings.group_id)
        .set_starting_offsets(KafkaOffsetsInitializer.committed_offsets(KafkaOffsetResetStrategy.EARLIEST))
        .set_value_only_deserializer(SimpleStringSchema())
        .build()
    )
    update_source = (
        KafkaSource.builder()
        .set_bootstrap_servers(settings.bootstrap_servers)
        .set_topics(settings.model_update_topic)
        .set_group_id(f"{settings.group_id}-model-updates")
        .set_starting_offsets(KafkaOffsetsInitializer.earliest())
        .set_value_only_deserializer(SimpleStringSchema())
        .build()
    )
    watermarks = WatermarkStrategy.for_bounded_out_of_orderness(
        Duration.of_millis(settings.out_of_order_ms)
    ).with_timestamp_assigner(_TimestampAssigner())
    events = env.from_source(event_source, watermarks, "transaction-events")
    updates = env.from_source(update_source, WatermarkStrategy.no_watermarks(), "model-updates")
    velocity_envelopes = events.key_by(event_user_key).process(_VelocityFunction(), output_type=Types.STRING())
    keyed_envelopes = velocity_envelopes.key_by(envelope_user_key)
    scored = keyed_envelopes.connect(updates.broadcast(update_descriptor)).process(
        _ScoringFunction(), output_type=Types.STRING()
    )

    def _sink(topic: str, prefix: str) -> Any:
        serializer = (
            KafkaRecordSerializationSchema.builder()
            .set_topic(topic)
            .set_value_serialization_schema(SimpleStringSchema())
            .build()
        )
        return (
            KafkaSink.builder()
            .set_bootstrap_servers(settings.bootstrap_servers)
            .set_record_serializer(serializer)
            .set_delivery_guarantee(DeliveryGuarantee.EXACTLY_ONCE)
            .set_transactional_id_prefix(prefix)
            .build()
        )

    for route, topic in (
        ("anomaly", settings.anomaly_topic),
        ("metrics", settings.metrics_topic),
        ("dlq", settings.dlq_topic),
    ):
        scored.filter(lambda raw, selected=route: json.loads(raw)["route"] == selected).map(
            lambda raw: json.dumps(json.loads(raw)["payload"]), output_type=Types.STRING()
        ).sink_to(_sink(topic, f"fraud-{route}-"))

    return env


def main() -> None:  # pragma: no cover - executed inside the Flink image
    parser = argparse.ArgumentParser(description="Run the checkpointed PyFlink fraud scoring job.")
    parser.add_argument("--job-name", default="realtime-fraud-scoring")
    args = parser.parse_args()
    build_flink_job(_settings_from_env()).execute(args.job_name)


if __name__ == "__main__":
    main()
