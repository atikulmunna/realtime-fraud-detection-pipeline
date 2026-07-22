from src.common.metrics_stub import MetricsRegistry


def test_metrics_registry_counter_and_gauge():
    m = MetricsRegistry()
    m.inc("a_total")
    m.inc("a_total", 2)
    m.set_gauge("b_gauge", 3.5)
    assert m.get_counter("a_total") == 3.0
    assert m.get_gauge("b_gauge") == 3.5


def test_metrics_registry_renders_prometheus_text():
    m = MetricsRegistry()
    m.inc("feedback_requests_total", 2)
    m.set_gauge("online_updater_buffer_size", 4)
    text = m.render_prometheus()
    assert "# TYPE feedback_requests_total counter" in text
    assert "feedback_requests_total 2.0" in text
    assert "# TYPE online_updater_buffer_size gauge" in text


def test_metrics_registry_exposes_real_prometheus_histogram():
    m = MetricsRegistry()
    m.observe("request_latency_ms", 12.0, buckets=(5.0, 25.0))

    rendered = m.render_prometheus()
    assert "# TYPE request_latency_ms histogram" in rendered
    assert 'request_latency_ms_bucket{le="25.0"} 1.0' in rendered
    assert "request_latency_ms_count 1.0" in rendered
