from src.streaming.kafka_bootstrap import TopicSpec, ensure_topics


class _DummyAdmin:
    def __init__(self, topics: list[str]):
        self._topics = set(topics)
        self.created_calls = []
        self.closed = False

    def list_topics(self):
        return list(self._topics)

    def create_topics(self, new_topics, validate_only=False):
        self.created_calls.append((new_topics, validate_only))
        for t in new_topics:
            self._topics.add(t.name)

    def close(self):
        self.closed = True


def test_ensure_topics_creates_only_missing_topics():
    admin = _DummyAdmin(topics=["raw-events", "feedback"])
    specs = [
        TopicSpec("raw-events", 8),
        TopicSpec("feedback", 2),
        TopicSpec("dead-letter", 2),
    ]

    result = ensure_topics(
        bootstrap_servers="localhost:9092",
        specs=specs,
        admin_client=admin,
    )

    assert result["created"] == ["dead-letter"]
    assert "dead-letter" in result["existing"]
    assert len(admin.created_calls) == 1


def test_ensure_topics_no_creation_when_all_exist():
    admin = _DummyAdmin(topics=["raw-events", "feedback"])
    specs = [
        TopicSpec("raw-events", 8),
        TopicSpec("feedback", 2),
    ]

    result = ensure_topics(
        bootstrap_servers="localhost:9092",
        specs=specs,
        admin_client=admin,
    )

    assert result["created"] == []
    assert admin.created_calls == []
