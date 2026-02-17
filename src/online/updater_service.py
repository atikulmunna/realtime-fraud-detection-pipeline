"""Local updater service entrypoint for processing feedback messages."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field

from src.common.metrics_stub import MetricsRegistry
from src.common.feature_contract import FEATURES_V1
from src.online.online_sgd_updater import OnlineSGDUpdater, process_feedback_messages
from src.streaming.ensemble_scoring import EnsembleModels, load_ensemble_models
from src.streaming.pipeline_skeleton import process_stream_payload


def _load_feedback_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if isinstance(obj, dict):
            records.append(obj)
    return records


def run_updater_service_once(
    *,
    model_path: str | Path,
    batch_size: int,
    feedback_file: str | Path | None = None,
    force_flush: bool = True,
) -> dict[str, Any]:
    updater = OnlineSGDUpdater(model_path=model_path, batch_size=batch_size)

    messages: list[dict[str, Any]] = []
    if feedback_file is not None and Path(feedback_file).exists():
        messages = _load_feedback_jsonl(feedback_file)

    result = process_feedback_messages(messages, updater=updater, force_flush=force_flush)
    return {
        "messages_in": len(messages),
        "updated": result.updated,
        "batch_size": result.batch_size,
        "skipped": result.skipped,
        "signal": result.signal,
    }


def create_updater_metrics_app(*, metrics: MetricsRegistry | None = None) -> FastAPI:
    app = FastAPI(title="Realtime Fraud Online Updater Metrics", version="0.1.0")
    metrics_registry = metrics or MetricsRegistry()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics_endpoint() -> str:
        return metrics_registry.render_prometheus()

    return app


class _DemoIFModel:
    def decision_function(self, x):
        amount = x[:, 0]
        return -(amount - 100.0) / 100.0


class _DemoScaler:
    def transform(self, x):
        return x / 1000.0


class _DemoAEModel:
    def predict(self, x):
        import numpy as np

        out = np.array(x, copy=True)
        out[:, 0] = np.minimum(out[:, 0], 0.12)
        return out


class _DemoSGDModel:
    def predict_proba(self, x):
        import numpy as np

        amount = np.clip(x[:, 0] / 1000.0, 0.0, 1.0)
        p1 = np.clip(0.1 + (0.8 * amount), 0.0, 1.0)
        p0 = 1.0 - p1
        return np.vstack([p0, p1]).T


def _build_demo_models() -> EnsembleModels:
    return EnsembleModels(
        if_model=_DemoIFModel(),
        ae_model=_DemoAEModel(),
        ae_scaler=_DemoScaler(),
        ae_threshold_p99=0.01,
        sgd_model=_DemoSGDModel(),
    )


class FeedbackIngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    features: dict[str, Any]


class StreamEventRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event: dict[str, Any]
    txn_velocity_1h: int = Field(default=1, ge=1)
    score_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


def create_online_service_app(
    *,
    updater: OnlineSGDUpdater,
    metrics: MetricsRegistry,
    stream_models: EnsembleModels,
) -> FastAPI:
    app = FastAPI(title="Realtime Fraud Online Service", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics_endpoint() -> str:
        return metrics.render_prometheus()

    @app.post("/feedback")
    def ingest_feedback(req: FeedbackIngestRequest) -> dict[str, Any]:
        accepted = updater.add_feedback(req.model_dump())
        update = updater.flush(force=False) if updater.ready() else None
        return {
            "accepted": bool(accepted),
            "updated": bool(update.updated) if update is not None else False,
            "online_update_count": int(updater.online_update_count),
        }

    @app.post("/feedback/sample")
    def ingest_feedback_sample() -> dict[str, Any]:
        accepted_a = updater.add_feedback(
            {"label": "true_positive", "features": {k: 1.0 for k in FEATURES_V1}}
        )
        accepted_b = updater.add_feedback(
            {"label": "false_positive", "features": {k: 0.2 for k in FEATURES_V1}}
        )
        update = updater.flush(force=True)
        return {
            "accepted": bool(accepted_a and accepted_b),
            "updated": bool(update.updated),
            "online_update_count": int(updater.online_update_count),
        }

    @app.post("/stream/event")
    def ingest_stream_event(req: StreamEventRequest) -> dict[str, Any]:
        topic, payload = process_stream_payload(
            req.event,
            models=stream_models,
            score_threshold=float(req.score_threshold),
            txn_velocity_1h=int(req.txn_velocity_1h),
            anomaly_topic="anomalies",
            normal_topic="metrics",
            dlq_topic="dead-letter",
            metrics=metrics,
        )
        return {"topic": topic, "payload": payload}

    @app.post("/stream/sample")
    def ingest_stream_sample() -> dict[str, Any]:
        samples = [
            {
                "event_id": "svc-high-1",
                "timestamp": "2026-02-17T08:15:00Z",
                "user_id": "C100",
                "type": "TRANSFER",
                "amount": 900.0,
                "old_balance_orig": 1000.0,
                "new_balance_orig": 100.0,
            },
            {
                "event_id": "svc-low-1",
                "timestamp": "2026-02-17T08:16:00Z",
                "user_id": "C101",
                "type": "PAYMENT",
                "amount": 20.0,
                "old_balance_orig": 500.0,
                "new_balance_orig": 480.0,
            },
        ]
        out: dict[str, int] = {"anomalies": 0, "metrics": 0, "dead-letter": 0}
        for event in samples:
            topic, _ = process_stream_payload(
                event,
                models=stream_models,
                score_threshold=0.5,
                txn_velocity_1h=1,
                anomaly_topic="anomalies",
                normal_topic="metrics",
                dlq_topic="dead-letter",
                metrics=metrics,
            )
            if topic in out:
                out[topic] += 1
        return out

    @app.post("/flush")
    def flush(force: bool = True) -> dict[str, Any]:
        result = updater.flush(force=force)
        if not result.updated and result.batch_size > 0 and not force:
            raise HTTPException(status_code=409, detail="Not enough buffered feedback for update.")
        return {
            "updated": result.updated,
            "batch_size": result.batch_size,
            "skipped": result.skipped,
            "signal": result.signal,
        }

    return app


def app_factory() -> FastAPI:
    metrics = MetricsRegistry()
    model_path = os.getenv("ONLINE_MODEL_PATH", "models/sgd_classifier_v1.joblib")
    batch_size = int(os.getenv("ONLINE_BATCH_SIZE", "500"))
    updater = OnlineSGDUpdater(model_path=model_path, batch_size=batch_size, metrics=metrics)

    if_path = os.getenv("IF_MODEL_PATH", "models/isolation_forest_v1.joblib")
    ae_path = os.getenv("AE_MODEL_PATH", "models/autoencoder_v1.joblib")
    sgd_path = os.getenv("SGD_MODEL_PATH", "models/sgd_classifier_v1.joblib")
    try:
        stream_models = load_ensemble_models(
            if_model_path=if_path,
            ae_model_path=ae_path,
            sgd_model_path=sgd_path,
        )
    except Exception:  # noqa: BLE001
        stream_models = _build_demo_models()

    return create_online_service_app(updater=updater, metrics=metrics, stream_models=stream_models)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single local online-updater pass.")
    parser.add_argument("--model-path", default="models/sgd_classifier_v1.joblib")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--feedback-file", default=None, help="Optional JSONL feedback file path.")
    parser.add_argument("--no-force-flush", action="store_true")
    args = parser.parse_args()

    summary = run_updater_service_once(
        model_path=args.model_path,
        batch_size=args.batch_size,
        feedback_file=args.feedback_file,
        force_flush=not args.no_force_flush,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
