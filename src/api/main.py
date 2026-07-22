"""Feedback API for analyst labeling events."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field

from src.api.feedback_publisher import FeedbackPublisher
from src.api.feedback_store import SqlFeedbackStore
from src.common.metrics_stub import MetricsRegistry
from src.common.structured_logging import log_event

logger = logging.getLogger(__name__)


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feedback_id: str | None = Field(default=None, min_length=1, max_length=64)
    anomaly_id: str = Field(min_length=1)
    label: Literal["true_positive", "false_positive"]
    analyst_id: str = Field(min_length=1)
    features: dict[str, Any] | None = None
    notes: str | None = None


class FeedbackResponse(BaseModel):
    status: str
    feedback_id: str
    duplicate: bool
    published_at: str


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def create_app(
    *,
    publisher: FeedbackPublisher | None = None,
    feedback_store: SqlFeedbackStore | None = None,
    metrics: MetricsRegistry | None = None,
    api_key: str | None = None,
) -> FastAPI:
    if publisher is None and feedback_store is None:
        raise ValueError("Either publisher or feedback_store is required.")
    app = FastAPI(title="Realtime Fraud Feedback API", version="0.1.0")
    metrics_registry = metrics or MetricsRegistry()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics_endpoint() -> str:
        return metrics_registry.render_prometheus()

    @app.post("/feedback", response_model=FeedbackResponse, status_code=202)
    def post_feedback(
        req: FeedbackRequest,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        provided_api_key: str | None = Header(default=None, alias="X-API-Key"),
    ) -> FeedbackResponse:
        if api_key is not None and provided_api_key != api_key:
            log_event(logger, "feedback_auth_rejected", level=logging.WARNING)
            raise HTTPException(status_code=401, detail="Invalid or missing API key.")
        metrics_registry.inc("feedback_requests_total")
        payload = req.model_dump()
        feedback_id = req.feedback_id or idempotency_key or str(uuid4())
        payload["feedback_id"] = feedback_id
        payload["received_at"] = _utc_now_iso()
        duplicate = False
        try:
            if feedback_store is not None:
                stored = feedback_store.accept(
                    feedback_id=feedback_id,
                    idempotency_key=idempotency_key,
                    payload=payload,
                )
                feedback_id = stored.feedback_id
                payload = stored.payload
                duplicate = stored.duplicate
                metrics_registry.inc("feedback_duplicates_total", 1.0 if duplicate else 0.0)
                metrics_registry.inc("feedback_durably_accepted_total", 0.0 if duplicate else 1.0)
            elif publisher is not None:
                publisher.publish(payload)
                metrics_registry.inc("feedback_published_total")
        except Exception as exc:  # noqa: BLE001
            metrics_registry.inc("feedback_publish_errors_total")
            log_event(
                logger,
                "feedback_accept_failed",
                correlation_id=feedback_id,
                level=logging.ERROR,
                error_type=type(exc).__name__,
            )
            raise HTTPException(status_code=503, detail=f"Publisher unavailable: {exc}") from exc

        log_event(
            logger,
            "feedback_accepted",
            correlation_id=feedback_id,
            anomaly_id=req.anomaly_id,
            duplicate=duplicate,
            durable=feedback_store is not None,
        )

        return FeedbackResponse(
            status="accepted",
            feedback_id=feedback_id,
            duplicate=duplicate,
            published_at=payload["received_at"],
        )

    return app
