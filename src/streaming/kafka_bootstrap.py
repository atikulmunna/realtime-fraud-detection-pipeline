"""Kafka topic bootstrap utilities for local/dev environments."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable

from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError


@dataclass(frozen=True)
class TopicSpec:
    name: str
    num_partitions: int
    replication_factor: int = 1


DEFAULT_TOPIC_SPECS = [
    TopicSpec(name="raw-events", num_partitions=8),
    TopicSpec(name="anomalies", num_partitions=4),
    TopicSpec(name="feedback", num_partitions=2),
    TopicSpec(name="metrics", num_partitions=4),
    TopicSpec(name="dead-letter", num_partitions=2),
    TopicSpec(name="model-updates", num_partitions=1),
]


def _to_new_topics(specs: Iterable[TopicSpec]) -> list[NewTopic]:
    return [
        NewTopic(
            name=spec.name,
            num_partitions=spec.num_partitions,
            replication_factor=spec.replication_factor,
        )
        for spec in specs
    ]


def ensure_topics(
    *,
    bootstrap_servers: str,
    specs: Iterable[TopicSpec] = DEFAULT_TOPIC_SPECS,
    client_id: str = "fraud-topic-bootstrap",
    admin_client: KafkaAdminClient | None = None,
) -> dict[str, list[str]]:
    own_client = admin_client is None
    admin = admin_client or KafkaAdminClient(bootstrap_servers=bootstrap_servers, client_id=client_id)

    try:
        existing = set(admin.list_topics())
        missing_specs = [spec for spec in specs if spec.name not in existing]
        created: list[str] = []

        if missing_specs:
            new_topics = _to_new_topics(missing_specs)
            try:
                admin.create_topics(new_topics=new_topics, validate_only=False)
                created = [spec.name for spec in missing_specs]
            except TopicAlreadyExistsError:
                # Handles race conditions if another process created topics after list_topics().
                created = []

        final_existing = existing.union(created)
        return {
            "existing": sorted(final_existing),
            "created": created,
        }
    finally:
        if own_client:
            admin.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ensure required Kafka topics exist.")
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    args = parser.parse_args()

    result = ensure_topics(bootstrap_servers=args.bootstrap_servers)
    print(f"created={result['created']}")
    print(f"existing_count={len(result['existing'])}")


if __name__ == "__main__":
    main()
