"""Minimal streaming pipeline skeleton (parse -> features -> route)."""

from __future__ import annotations

from time import perf_counter
from typing import Any

from src.common.metrics_stub import MetricsRegistry
from src.streaming.ensemble_scoring import EnsembleModels, route_score_to_topic, score_event_features
from src.streaming.event_parser import parse_and_validate_event, route_parse_result
from src.streaming.feature_extractor import enrich_event_with_features


def process_stream_payload(
    raw_payload: str | bytes | dict[str, Any],
    *,
    valid_topic: str = "feature-events",
    dlq_topic: str = "dead-letter",
    txn_velocity_1h: int = 1,
    models: EnsembleModels | None = None,
    score_weights: tuple[float, float, float] = (0.4, 0.3, 0.3),
    score_threshold: float = 0.5,
    anomaly_topic: str = "anomalies",
    normal_topic: str = "metrics",
    metrics: MetricsRegistry | None = None,
) -> tuple[str, dict[str, Any]]:
    started = perf_counter()
    if metrics is not None:
        metrics.inc("stream_events_in_total")

    parsed = parse_and_validate_event(raw_payload)
    if not parsed.ok:
        topic, routed = route_parse_result(parsed, valid_topic=valid_topic, dlq_topic=dlq_topic)
        if metrics is not None:
            metrics.inc("stream_events_dlq_total")
            latency_ms = (perf_counter() - started) * 1000.0
            metrics.inc("stream_process_latency_ms_total", latency_ms)
            metrics.set_gauge("stream_last_process_latency_ms", latency_ms)
        return topic, routed

    try:
        enriched = enrich_event_with_features(parsed.event or {}, txn_velocity_1h=txn_velocity_1h)
        if models is None:
            if metrics is not None:
                metrics.inc("stream_events_valid_total")
                latency_ms = (perf_counter() - started) * 1000.0
                metrics.inc("stream_process_latency_ms_total", latency_ms)
                metrics.set_gauge("stream_last_process_latency_ms", latency_ms)
            return valid_topic, enriched

        scores = score_event_features(enriched["features"], models, weights=score_weights)
        out = dict(enriched)
        out["scores"] = scores
        routed_topic = route_score_to_topic(
            scores["ensemble_score"],
            threshold=score_threshold,
            anomaly_topic=anomaly_topic,
            normal_topic=normal_topic,
        )
        if metrics is not None:
            if routed_topic == anomaly_topic:
                metrics.inc("stream_events_anomaly_total")
            elif routed_topic == normal_topic:
                metrics.inc("stream_events_normal_total")
            latency_ms = (perf_counter() - started) * 1000.0
            metrics.inc("stream_process_latency_ms_total", latency_ms)
            metrics.set_gauge("stream_last_process_latency_ms", latency_ms)
        return routed_topic, out
    except Exception as exc:
        # Preserve sanitized parser event for diagnostics when feature extraction fails.
        dlq = {
            "error": str(exc),
            "raw_event": parsed.event,
        }
        if metrics is not None:
            metrics.inc("stream_events_dlq_total")
            latency_ms = (perf_counter() - started) * 1000.0
            metrics.inc("stream_process_latency_ms_total", latency_ms)
            metrics.set_gauge("stream_last_process_latency_ms", latency_ms)
        return dlq_topic, dlq


def process_stream_batch(
    payloads: list[str | bytes | dict[str, Any]],
    *,
    valid_topic: str = "feature-events",
    dlq_topic: str = "dead-letter",
    txn_velocity_1h: int = 1,
    models: EnsembleModels | None = None,
    score_weights: tuple[float, float, float] = (0.4, 0.3, 0.3),
    score_threshold: float = 0.5,
    anomaly_topic: str = "anomalies",
    normal_topic: str = "metrics",
    metrics: MetricsRegistry | None = None,
) -> dict[str, list[dict[str, Any]]]:
    if models is None:
        out: dict[str, list[dict[str, Any]]] = {valid_topic: [], dlq_topic: []}
    else:
        out = {anomaly_topic: [], normal_topic: [], dlq_topic: []}
    for payload in payloads:
        topic, routed = process_stream_payload(
            payload,
            valid_topic=valid_topic,
            dlq_topic=dlq_topic,
            txn_velocity_1h=txn_velocity_1h,
            models=models,
            score_weights=score_weights,
            score_threshold=score_threshold,
            anomaly_topic=anomaly_topic,
            normal_topic=normal_topic,
            metrics=metrics,
        )
        out[topic].append(routed)
    return out
