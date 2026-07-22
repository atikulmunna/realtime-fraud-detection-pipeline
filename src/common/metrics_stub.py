"""Small compatibility wrapper around the production Prometheus client."""

from __future__ import annotations

from threading import RLock

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest


class MetricsRegistry:
    """Own an isolated collector registry while preserving the project's API."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry(auto_describe=True)
        self._counters: dict[str, Counter] = {}
        self._gauges: dict[str, Gauge] = {}
        self._histograms: dict[str, Histogram] = {}
        self._lock = RLock()

    def _counter(self, name: str) -> Counter:
        metric = self._counters.get(name)
        if metric is None:
            metric = Counter(name, f"Application counter {name}.", registry=self.registry)
            self._counters[name] = metric
        return metric

    def _gauge(self, name: str) -> Gauge:
        metric = self._gauges.get(name)
        if metric is None:
            metric = Gauge(name, f"Application gauge {name}.", registry=self.registry)
            self._gauges[name] = metric
        return metric

    def inc(self, name: str, amount: float = 1.0) -> None:
        with self._lock:
            self._counter(name).inc(float(amount))

    def set_gauge(self, name: str, value: float) -> None:
        with self._lock:
            self._gauge(name).set(float(value))

    def observe(self, name: str, value: float, *, buckets: tuple[float, ...] | None = None) -> None:
        with self._lock:
            metric = self._histograms.get(name)
            if metric is None:
                if buckets is None:
                    metric = Histogram(name, f"Application histogram {name}.", registry=self.registry)
                else:
                    metric = Histogram(
                        name,
                        f"Application histogram {name}.",
                        registry=self.registry,
                        buckets=buckets,
                    )
                self._histograms[name] = metric
            metric.observe(float(value))

    def get_counter(self, name: str) -> float:
        with self._lock:
            metric = self._counters.get(name)
            return 0.0 if metric is None else float(metric._value.get())

    def get_gauge(self, name: str) -> float:
        with self._lock:
            metric = self._gauges.get(name)
            return 0.0 if metric is None else float(metric._value.get())

    def render_prometheus(self) -> str:
        with self._lock:
            return generate_latest(self.registry).decode("utf-8")
