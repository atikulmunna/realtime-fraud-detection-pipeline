"""Simple API healthcheck helpers for local observability smoke checks."""

from __future__ import annotations

import argparse
import json
from typing import Any, Callable

import requests


FetchFn = Callable[[str, float], tuple[int, str]]


def _default_fetch(url: str, timeout_s: float) -> tuple[int, str]:
    resp = requests.get(url, timeout=timeout_s)
    return int(resp.status_code), str(resp.text)


def run_healthchecks(
    *,
    base_url: str = "http://127.0.0.1:8000",
    timeout_s: float = 2.0,
    fetch_fn: FetchFn | None = None,
) -> dict[str, Any]:
    fetch = fetch_fn or _default_fetch
    checks: dict[str, Any] = {}

    health_url = f"{base_url.rstrip('/')}/health"
    metrics_url = f"{base_url.rstrip('/')}/metrics"

    try:
        status, body = fetch(health_url, timeout_s)
        checks["health"] = {"ok": status == 200, "status": status, "body": body[:300]}
    except Exception as exc:  # noqa: BLE001
        checks["health"] = {"ok": False, "status": 0, "error": str(exc)}

    try:
        status, body = fetch(metrics_url, timeout_s)
        checks["metrics"] = {
            "ok": status == 200,
            "status": status,
            "non_empty": len(body.strip()) > 0,
            "body": body[:300],
        }
    except Exception as exc:  # noqa: BLE001
        checks["metrics"] = {"ok": False, "status": 0, "error": str(exc)}

    checks["overall_ok"] = bool(checks["health"]["ok"] and checks["metrics"]["ok"])
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description="Run API /health and /metrics checks.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=2.0)
    args = parser.parse_args()

    result = run_healthchecks(base_url=args.base_url, timeout_s=args.timeout)
    print(json.dumps(result, indent=2))

    if not result["overall_ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
