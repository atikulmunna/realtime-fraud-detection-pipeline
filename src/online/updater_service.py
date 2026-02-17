"""Local updater service entrypoint for processing feedback messages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.online.online_sgd_updater import OnlineSGDUpdater, process_feedback_messages


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
