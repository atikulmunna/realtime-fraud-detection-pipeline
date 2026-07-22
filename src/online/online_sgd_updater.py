"""Online SGD updater skeleton for feedback-driven model updates."""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any, Protocol

import joblib
import numpy as np

from src.common.feature_contract import FEATURES_V1
from src.common.metrics_stub import MetricsRegistry


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class ModelUpdatePublisher(Protocol):
    def publish(self, payload: dict[str, Any]) -> None: ...


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
        self.candidate_path = self.model_path.with_suffix(self.model_path.suffix + ".candidate")
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
        self.update_history = list(payload.get("update_history", []))
        self.processed_feedback_ids = set(payload.get("processed_feedback_ids", []))

        self._x_buffer: list[list[float]] = []
        self._y_buffer: list[int] = []
        self._feedback_id_buffer: list[str] = []
        self._pending_signal: dict[str, Any] | None = None
        self._skipped = 0
        self._lock = RLock()
        self.metrics.set_gauge("online_updater_buffer_size", 0)

    def _vector_from_features(self, features: dict[str, Any]) -> list[float]:
        missing = [k for k in FEATURES_V1 if k not in features]
        if missing:
            raise ValueError(f"Missing required feedback feature keys: {missing}")
        return [float(features[k]) for k in FEATURES_V1]

    def add_feedback(self, feedback: dict[str, Any]) -> bool:
        with self._lock:
            self.metrics.inc("online_feedback_received_total")
            feedback_id = str(feedback.get("feedback_id", ""))
            if feedback_id and feedback_id in self.processed_feedback_ids:
                self.metrics.inc("online_feedback_duplicate_total")
                return False
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
            self._feedback_id_buffer.append(feedback_id)
            self.metrics.inc("online_feedback_accepted_total")
            self.metrics.set_gauge("online_updater_buffer_size", len(self._x_buffer))
            return True

    def ready(self) -> bool:
        with self._lock:
            return len(self._x_buffer) >= self.batch_size

    @property
    def buffer_size(self) -> int:
        """Return the number of accepted records not yet durably applied."""
        with self._lock:
            return len(self._x_buffer)

    @property
    def model_age_seconds(self) -> float:
        """Return local champion artifact age for freshness monitoring."""
        return max(0.0, time.time() - self.model_path.stat().st_mtime)

    def flush(self, *, force: bool = False, stage_candidate: bool = False) -> UpdateResult:
        with self._lock:
            return self._flush_locked(force=force, stage_candidate=stage_candidate)

    def _flush_locked(self, *, force: bool, stage_candidate: bool) -> UpdateResult:
        n = len(self._x_buffer)
        if n == 0:
            return UpdateResult(updated=False, batch_size=0, skipped=self._skipped, signal=None)
        if not force and n < self.batch_size:
            return UpdateResult(updated=False, batch_size=n, skipped=self._skipped, signal=None)

        x = np.array(self._x_buffer, dtype=float)
        y = np.array(self._y_buffer, dtype=int)
        self.model.partial_fit(x, y, classes=np.array([0, 1], dtype=int))
        self.online_update_count += 1
        accepted_ids = [value for value in self._feedback_id_buffer if value]
        self.processed_feedback_ids.update(accepted_ids)
        batch_manifest = {
            "online_update_count": self.online_update_count,
            "batch_size": int(n),
            "feedback_ids": accepted_ids,
            "updated_at": _utc_now_iso(),
        }
        self.update_history = [*self.update_history[-99:], batch_manifest]
        self.metrics.inc("online_updates_total")
        self.metrics.set_gauge("online_last_update_batch_size", n)
        self.metrics.set_gauge("online_update_count", self.online_update_count)

        payload = {
            "model_type": self.model_type,
            "model": self.model,
            "features_order": self.features_order,
            "online_update_count": self.online_update_count,
            "last_updated_at": _utc_now_iso(),
            "update_history": self.update_history,
            "processed_feedback_ids": sorted(self.processed_feedback_ids)[-10000:],
        }
        target_path = self.candidate_path if stage_candidate else self.model_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = target_path.with_suffix(target_path.suffix + ".tmp")
        joblib.dump(payload, temporary_path)
        os.replace(temporary_path, target_path)

        signal = {
            "model_type": self.model_type,
            "model_path": str(target_path),
            "staged_candidate": stage_candidate,
            "online_update_count": self.online_update_count,
            "batch_size": int(n),
            "updated_at": payload["last_updated_at"],
        }
        if self.publisher is not None and not stage_candidate:
            self.publisher.publish(signal)
        elif stage_candidate:
            self._pending_signal = signal

        self._x_buffer.clear()
        self._y_buffer.clear()
        self._feedback_id_buffer.clear()
        self.metrics.set_gauge("online_updater_buffer_size", 0)
        skipped = self._skipped
        self._skipped = 0
        return UpdateResult(updated=True, batch_size=n, skipped=skipped, signal=signal)

    def promote_candidate(self) -> None:
        """Atomically make the staged candidate the local active artifact."""
        with self._lock:
            if not self.candidate_path.is_file():
                raise FileNotFoundError(f"Staged candidate does not exist: {self.candidate_path}")
            os.replace(self.candidate_path, self.model_path)
            if self.publisher is not None and self._pending_signal is not None:
                promoted_signal = {
                    **self._pending_signal,
                    "model_path": str(self.model_path),
                    "artifact_sha256": hashlib.sha256(self.model_path.read_bytes()).hexdigest(),
                    "staged_candidate": False,
                }
                self.publisher.publish(promoted_signal)
            self._pending_signal = None

    def reject_candidate(self, backup_payload: dict[str, Any]) -> None:
        """Discard a staged candidate and restore the in-memory active state."""
        with self._lock:
            self.model = backup_payload["model"]
            self.model_type = backup_payload.get("model_type", "sgd_classifier")
            self.features_order = backup_payload.get("features_order", FEATURES_V1)
            self.online_update_count = int(backup_payload.get("online_update_count", 0))
            self.update_history = list(backup_payload.get("update_history", []))
            self.processed_feedback_ids = set(backup_payload.get("processed_feedback_ids", []))
            self.candidate_path.unlink(missing_ok=True)
            self._pending_signal = None


def process_feedback_messages(
    messages: list[dict[str, Any]],
    *,
    updater: OnlineSGDUpdater,
    force_flush: bool = False,
) -> UpdateResult:
    for msg in messages:
        updater.add_feedback(msg)
    return updater.flush(force=force_flush)
