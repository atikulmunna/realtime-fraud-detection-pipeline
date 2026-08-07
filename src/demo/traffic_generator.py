"""Synthetic transaction traffic generator for live demo deployments.

The Flink job consumes ``raw-events`` but nothing in the offline or Compose
topology produces it, so a deployed stack would idle with an empty dashboard.
This module emits contract-valid events at a controlled rate to give a live
demo continuous, deterministic-on-request traffic.
"""

from __future__ import annotations

import argparse
import json
import random
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from confluent_kafka import Producer

# PaySim only ever labels TRANSFER and CASH_OUT as fraudulent, so draining
# patterns are drawn from those types and benign traffic from the rest.
BENIGN_TYPES = ("PAYMENT", "CASH_IN", "DEBIT")
DRAINING_TYPES = ("TRANSFER", "CASH_OUT")

# The user population is deliberately far larger than the number of events a demo
# will emit, so a user rarely transacts twice inside one UTC hour.
#
# This matters because txn_velocity_1h is degenerate in PaySim: it is 1 for
# essentially every training row (p99 = 1, max = 2). The models therefore treat
# any velocity above 1 as strongly anomalous, and a small population makes the
# Flink job's per-user counter climb until nearly all traffic is flagged. With
# this default about 99% of events carry velocity 1 and the observed anomaly rate
# tracks FRAUD_RATIO instead. Only users that actually transact create Flink
# state, so a large population costs roughly nothing.
DEFAULT_USERS = 1_000_000


class TransactionPublisher(Protocol):
    def publish(self, payload: dict[str, Any]) -> None: ...


@dataclass
class ConfluentTransactionPublisher:
    """Keyed producer for ``raw-events``.

    Partitioning by ``user_id`` keeps a user's events in one partition, which is
    what the Flink job's per-user velocity state assumes.
    """

    producer: Producer
    topic: str = "raw-events"
    key_field: str = "user_id"

    def publish(self, payload: dict[str, Any]) -> None:
        self.producer.produce(
            self.topic,
            value=json.dumps(payload).encode("utf-8"),
            key=str(payload[self.key_field]),
        )
        # poll(0) drains delivery callbacks without blocking the send loop;
        # flush() per event would cap throughput at one round trip per message.
        self.producer.poll(0)

    def close(self, timeout: float = 10.0) -> None:
        remaining = self.producer.flush(timeout)
        if remaining:
            raise RuntimeError(f"Kafka did not acknowledge {remaining} message(s).")


@dataclass
class RecordingTransactionPublisher:
    """In-memory publisher used by tests and dry runs."""

    events: list[dict[str, Any]] = field(default_factory=list)

    def publish(self, payload: dict[str, Any]) -> None:
        self.events.append(payload)

    def close(self, timeout: float = 10.0) -> None:
        return None


def generate_transaction(
    rng: random.Random,
    *,
    now: datetime,
    users: int = DEFAULT_USERS,
    fraud_ratio: float = 0.03,
) -> dict[str, Any]:
    """Build one contract-valid event.

    Draining events set ``new_balance_orig`` to zero against a large
    ``old_balance_orig``, which is the balance signature the ensemble keys on.
    """
    if users < 1:
        raise ValueError("users must be at least 1")
    if not 0.0 <= fraud_ratio <= 1.0:
        raise ValueError("fraud_ratio must be between 0.0 and 1.0")

    user_id = f"C{rng.randrange(users):07d}"
    draining = rng.random() < fraud_ratio

    if draining:
        old_balance = round(rng.uniform(5_000.0, 250_000.0), 2)
        amount = old_balance
        new_balance = 0.0
        txn_type = rng.choice(DRAINING_TYPES)
    else:
        old_balance = round(rng.uniform(100.0, 20_000.0), 2)
        amount = round(rng.uniform(1.0, min(old_balance, 2_500.0)), 2)
        new_balance = round(old_balance - amount, 2)
        txn_type = rng.choice(BENIGN_TYPES)

    return {
        "event_id": str(uuid.UUID(int=rng.getrandbits(128), version=4)),
        "timestamp": now.isoformat().replace("+00:00", "Z"),
        "user_id": user_id,
        "type": txn_type,
        "amount": amount,
        "old_balance_orig": old_balance,
        "new_balance_orig": new_balance,
    }


def run_generator(
    publisher: TransactionPublisher,
    *,
    rate: float = 5.0,
    duration: float | None = 60.0,
    users: int = DEFAULT_USERS,
    fraud_ratio: float = 0.03,
    seed: int | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
    now_factory: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> dict[str, Any]:
    """Publish events at ``rate`` per second until ``duration`` elapses.

    ``duration=None`` runs until interrupted. The clock and sleeper are
    injectable so tests exercise the pacing loop without real time passing.
    """
    if rate <= 0:
        raise ValueError("rate must be greater than 0")
    if duration is not None and duration < 0:
        raise ValueError("duration must not be negative")

    rng = random.Random(seed)
    interval = 1.0 / rate
    started = clock()
    published = 0
    draining = 0

    while duration is None or clock() - started < duration:
        event = generate_transaction(rng, now=now_factory(), users=users, fraud_ratio=fraud_ratio)
        publisher.publish(event)
        published += 1
        if event["type"] in DRAINING_TYPES and event["new_balance_orig"] == 0.0:
            draining += 1
        # Schedule against an absolute tick rather than sleeping a fixed
        # interval, so publish latency and float error do not accumulate drift.
        delay = (started + published * interval) - clock()
        if delay > 0:
            sleeper(delay)

    return {
        "published": published,
        "draining": draining,
        "elapsed_seconds": round(clock() - started, 3),
        "rate": rate,
    }


def build_confluent_publisher(
    *,
    bootstrap_servers: str = "localhost:9092",
    topic: str = "raw-events",
) -> ConfluentTransactionPublisher:
    producer = Producer(
        {
            "bootstrap.servers": bootstrap_servers,
            "enable.idempotence": True,
            "acks": "all",
            "retries": 10,
            "delivery.timeout.ms": 30000,
            "client.id": "fraud-traffic-generator",
        }
    )
    return ConfluentTransactionPublisher(producer=producer, topic=topic)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Publish synthetic transactions to the raw-events topic.")
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument("--topic", default="raw-events")
    parser.add_argument("--rate", type=float, default=5.0, help="Events per second.")
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Seconds to run. Omit to run until stopped.",
    )
    parser.add_argument("--users", type=int, default=DEFAULT_USERS)
    parser.add_argument("--fraud-ratio", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Generate without connecting to Kafka.")
    args = parser.parse_args(argv)

    publisher: ConfluentTransactionPublisher | RecordingTransactionPublisher
    if args.dry_run:
        publisher = RecordingTransactionPublisher()
    else:
        publisher = build_confluent_publisher(bootstrap_servers=args.bootstrap_servers, topic=args.topic)

    try:
        summary = run_generator(
            publisher,
            rate=args.rate,
            duration=args.duration,
            users=args.users,
            fraud_ratio=args.fraud_ratio,
            seed=args.seed,
        )
    except KeyboardInterrupt:
        summary = {"published": None, "interrupted": True}
    finally:
        publisher.close()

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
