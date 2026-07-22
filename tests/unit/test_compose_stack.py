from pathlib import Path


def test_compose_defines_containerized_application_and_topic_bootstrap():
    text = Path("infra/docker-compose.yml").read_text(encoding="utf-8")
    for service in [
        "feedback-api:",
        "outbox-relay:",
        "online-updater:",
        "topic-init:",
        "mlflow:",
        "jobmanager:",
        "taskmanager:",
    ]:
        assert service in text
    assert "KAFKA_AUTO_CREATE_TOPICS_ENABLE: \"false\"" in text
    assert "condition: service_healthy" in text
    assert "feedback_api_key" in text
    assert "redis:" not in text
