from pathlib import Path


def test_prometheus_alert_rules_file_exists():
    path = Path("infra/prometheus/alerts.yml")
    assert path.exists()


def test_prometheus_alert_rules_contains_expected_alerts():
    text = Path("infra/prometheus/alerts.yml").read_text(encoding="utf-8")
    expected = [
        "alert: FeedbackPublishErrorsDetected",
        "alert: OnlineConsumerStalled",
        "alert: OnlineConsumerLagHigh",
        "alert: FeedbackOutboxBacklogHigh",
        "alert: StreamDlqRatioHigh",
        "alert: StreamP95LatencyHigh",
        "alert: OnlineModelStale",
        "alert: PromotionFailuresDetected",
        "alert: PromotionRollbacksDetected",
        "expr: increase(feedback_publish_errors_total[5m]) > 0",
        "expr: online_consumer_lag > 0 and increase(online_consumer_commits_total[10m]) == 0",
        "expr: increase(promotion_fail_total[15m]) > 0",
        "expr: increase(promotion_rollback_total[15m]) > 0",
    ]
    for token in expected:
        assert token in text
