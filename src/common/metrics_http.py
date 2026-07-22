"""Dependency-light HTTP endpoint for Prometheus metrics and liveness."""

from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import Any

from src.common.metrics_stub import MetricsRegistry


def start_metrics_http_server(
    *,
    metrics: MetricsRegistry,
    host: str = "0.0.0.0",
    port: int,
) -> ThreadingHTTPServer:
    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/metrics":
                body = metrics.render_prometheus().encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/plain; version=0.0.4")
            elif self.path == "/health":
                body = b'{"status":"ok"}'
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json")
            else:
                self.send_response(HTTPStatus.NOT_FOUND)
                self.end_headers()
                return
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return

    server = ThreadingHTTPServer((host, int(port)), _Handler)
    Thread(target=server.serve_forever, daemon=True).start()
    return server
