"""Minimal Prometheus-style in-memory metrics registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock


@dataclass
class MetricsRegistry:
    _counters: dict[str, float] = field(default_factory=dict)
    _gauges: dict[str, float] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)

    def inc(self, name: str, amount: float = 1.0) -> None:
        with self._lock:
            self._counters[name] = float(self._counters.get(name, 0.0) + amount)

    def set_gauge(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = float(value)

    def get_counter(self, name: str) -> float:
        with self._lock:
            return float(self._counters.get(name, 0.0))

    def get_gauge(self, name: str) -> float:
        with self._lock:
            return float(self._gauges.get(name, 0.0))

    def render_prometheus(self) -> str:
        lines: list[str] = []
        with self._lock:
            for name in sorted(self._counters.keys()):
                lines.append(f"# TYPE {name} counter")
                lines.append(f"{name} {self._counters[name]}")
            for name in sorted(self._gauges.keys()):
                lines.append(f"# TYPE {name} gauge")
                lines.append(f"{name} {self._gauges[name]}")
        return "\n".join(lines) + ("\n" if lines else "")
