"""Online SGD updater skeleton for feedback-driven model updates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import joblib
import numpy as np

from src.common.feature_contract import FEATURES_V1
from src.common.metrics_stub import MetricsRegistry


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class ModelUpdatePublisher(Protocol):
    def publish(self, payload: dict[str, Any]) -> None:
        ...


@dataclass
class InMemoryModelUpdatePublisher:
    events: list[dict[str, Any]]

    def publish(self, payload: dict[str, Any]) -> None:
        self.events.append(payload)


@dataclass
class UpdateResult:
    updated: bool
    batch_size: int
    skipped: int
    signal: dict[str, Any] | None


class OnlineSGDUpdater:
    def __init__(
        self,
        *,
        model_path: str | Path = "models/sgd_classifier_v1.joblib",
        batch_size: int = 500,
        publisher: ModelUpdatePublisher | None = None,
        metrics: MetricsRegistry | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        self.publisher = publisher
        self.metrics = metrics or MetricsRegistry()

        payload = joblib.load(self.model_path)
        self.model = payload["model"]
        self.model_type = payload.get("model_type", "sgd_classifier")
        self.features_order = payload.get("features_order", FEATURES_V1)
        self.online_update_count = int(payload.get("online_update_count", 0))

        self._x_buffer: list[list[float]] = []
        self._y_buffer: list[int] = []
        self._skipped = 0
        self.metrics.set_gauge("online_updater_buffer_size", 0)

    def _vector_from_features(self, features: dict[str, Any]) -> list[float]:
        missing = [k for k in FEATURES_V1 if k not in features]
        if missing:
            raise ValueError(f"Missing required feedback feature keys: {missing}")
        return [float(features[k]) for k in FEATURES_V1]

    def add_feedback(self, feedback: dict[str, Any]) -> bool:
        self.metrics.inc("online_feedback_received_total")
        label = feedback.get("label")
        features = feedback.get("features")

        if label not in {"true_positive", "false_positive"}:
            self._skipped += 1
            self.metrics.inc("online_feedback_skipped_total")
            return False
        if not isinstance(features, dict):
            self._skipped += 1
            self.metrics.inc("online_feedback_skipped_total")
            return False

        try:
            x = self._vector_from_features(features)
        except Exception:
            self._skipped += 1
            self.metrics.inc("online_feedback_skipped_total")
            return False

        y = 1 if label == "true_positive" else 0
        self._x_buffer.append(x)
        self._y_buffer.append(y)
        self.metrics.inc("online_feedback_accepted_total")
        self.metrics.set_gauge("online_updater_buffer_size", len(self._x_buffer))
        return True

    def ready(self) -> bool:
        return len(self._x_buffer) >= self.batch_size

    def flush(self, *, force: bool = False) -> UpdateResult:
        n = len(self._x_buffer)
        if n == 0:
            return UpdateResult(updated=False, batch_size=0, skipped=self._skipped, signal=None)
        if not force and n < self.batch_size:
            return UpdateResult(updated=False, batch_size=n, skipped=self._skipped, signal=None)

        x = np.array(self._x_buffer, dtype=float)
        y = np.array(self._y_buffer, dtype=int)
        self.model.partial_fit(x, y, classes=np.array([0, 1], dtype=int))
        self.online_update_count += 1
        self.metrics.inc("online_updates_total")
        self.metrics.set_gauge("online_last_update_batch_size", n)
        self.metrics.set_gauge("online_update_count", self.online_update_count)

        payload = {
            "model_type": self.model_type,
            "model": self.model,
            "features_order": self.features_order,
            "online_update_count": self.online_update_count,
            "last_updated_at": _utc_now_iso(),
        }
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, self.model_path)

        signal = {
            "model_type": self.model_type,
            "model_path": str(self.model_path),
            "online_update_count": self.online_update_count,
            "batch_size": int(n),
            "updated_at": payload["last_updated_at"],
        }
        if self.publisher is not None:
            self.publisher.publish(signal)

        self._x_buffer.clear()
        self._y_buffer.clear()
        self.metrics.set_gauge("online_updater_buffer_size", 0)
        skipped = self._skipped
        self._skipped = 0
        return UpdateResult(updated=True, batch_size=n, skipped=skipped, signal=signal)


def process_feedback_messages(
    messages: list[dict[str, Any]],
    *,
    updater: OnlineSGDUpdater,
    force_flush: bool = False,
) -> UpdateResult:
    for msg in messages:
        updater.add_feedback(msg)
    return updater.flush(force=force_flush)
