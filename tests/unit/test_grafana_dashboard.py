import json
from pathlib import Path


def test_grafana_dashboard_contains_stream_metrics_panels():
    path = Path("infra/grafana/dashboards/realtime_fraud_overview.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    panels = data["panels"]

    titles = {panel["title"] for panel in panels}
    assert "Stream Event Rates (5m)" in titles
    assert "Stream Last Processing Latency" in titles
    assert "Stream Avg Latency (5m)" in titles
    assert "Stream Anomaly Ratio (5m)" in titles

    queries = [target["expr"] for panel in panels for target in panel.get("targets", [])]
    assert any("stream_events_in_total" in q for q in queries)
    assert any("stream_events_anomaly_total" in q for q in queries)
    assert any("stream_process_latency_ms_total" in q for q in queries)

