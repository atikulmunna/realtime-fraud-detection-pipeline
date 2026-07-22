# Observability Runbook

Prometheus scrapes the feedback API, online updater, and feedback consumer. Grafana provisions the **Realtime Fraud Overview** dashboard from the repository. Application logs are JSON and use `correlation_id` to connect a feedback request to an online update.

## Triage order

1. Check `outbox_backlog`. A growing value means feedback is durable in Postgres but is not reaching Kafka. Inspect relay connectivity and `outbox_publish_failures_total`.
2. Check `online_consumer_lag`. Lag with no increase in `online_consumer_commits_total` means candidates are not reaching a durable promotion or rollback decision. Inspect consumer logs, MLflow, and the promotion counters.
3. Check `stream_events_dlq_total` by error records before changing a schema or model. A ratio above 1% triggers `StreamDlqRatioHigh`.
4. Check the latency histogram. `StreamP95LatencyHigh` fires when p95 exceeds 500 ms for ten minutes.
5. Check `online_model_age_seconds` and MLflow aliases. A rejected candidate is expected to increment both `promotion_fail_total` and `promotion_rollback_total`; repeated failures require reviewing holdout quality and thresholds.

## Alert responses

- `FeedbackPublishErrorsDetected`: confirm API availability, then move clients to the durable Postgres-backed API configuration.
- `FeedbackOutboxBacklogHigh`: keep the API online because writes are durable; restore Kafka/relay service and watch backlog drain.
- `OnlineConsumerStalled` or `OnlineConsumerLagHigh`: do not reset offsets. Fix the updater, holdout, or registry dependency and allow idempotent redelivery.
- `PromotionFailuresDetected`: inspect the rejection reasons in correlated JSON logs. Do not manually move the champion alias without validating the candidate.
- `OnlineModelStale`: verify fresh feedback exists before treating age alone as an incident.

## Useful PromQL

```promql
rate(stream_events_dlq_total[5m]) / clamp_min(rate(stream_events_in_total[5m]), 1e-9)
histogram_quantile(0.95, sum by (le) (rate(stream_event_processing_latency_ms_bucket[5m])))
rate(stream_events_anomaly_total[5m]) / clamp_min(rate(stream_events_valid_total[5m]), 1e-9)
```
