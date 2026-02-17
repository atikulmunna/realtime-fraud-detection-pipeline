"""Feedback API for analyst labeling events."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field

from src.api.feedback_publisher import FeedbackPublisher
from src.common.metrics_stub import MetricsRegistry


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    anomaly_id: str = Field(min_length=1)
    label: Literal["true_positive", "false_positive"]
    analyst_id: str = Field(min_length=1)
    features: dict[str, Any] | None = None
    notes: str | None = None


class FeedbackResponse(BaseModel):
    status: str
    published_at: str


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def create_app(*, publisher: FeedbackPublisher, metrics: MetricsRegistry | None = None) -> FastAPI:
    app = FastAPI(title="Realtime Fraud Feedback API", version="0.1.0")
    metrics_registry = metrics or MetricsRegistry()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics_endpoint() -> str:
        return metrics_registry.render_prometheus()

    @app.post("/feedback", response_model=FeedbackResponse, status_code=202)
    def post_feedback(req: FeedbackRequest) -> FeedbackResponse:
        metrics_registry.inc("feedback_requests_total")
        payload = req.model_dump()
        payload["received_at"] = _utc_now_iso()
        try:
            publisher.publish(payload)
            metrics_registry.inc("feedback_published_total")
        except Exception as exc:  # noqa: BLE001
            metrics_registry.inc("feedback_publish_errors_total")
            raise HTTPException(status_code=503, detail=f"Publisher unavailable: {exc}") from exc

        return FeedbackResponse(status="accepted", published_at=payload["received_at"])

    return app
