"""Transactional feedback and outbox persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, Column, DateTime, MetaData, String, Table, create_engine, func, insert, select, update
from sqlalchemy.exc import IntegrityError

metadata = MetaData()
feedback_records = Table(
    "feedback_records",
    metadata,
    Column("feedback_id", String(64), primary_key=True),
    Column("idempotency_key", String(255), unique=True, nullable=True),
    Column("payload", JSON, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)
feedback_outbox = Table(
    "feedback_outbox",
    metadata,
    Column("feedback_id", String(64), primary_key=True),
    Column("payload", JSON, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("published_at", DateTime(timezone=True), nullable=True),
)


@dataclass(frozen=True)
class StoredFeedback:
    feedback_id: str
    payload: dict[str, Any]
    duplicate: bool


class SqlFeedbackStore:
    def __init__(self, database_url: str, *, create_schema: bool = False) -> None:
        self.engine = create_engine(database_url, pool_pre_ping=True)
        if create_schema:
            metadata.create_all(self.engine)

    def accept(
        self,
        *,
        feedback_id: str,
        idempotency_key: str | None,
        payload: dict[str, Any],
    ) -> StoredFeedback:
        now = datetime.now(UTC)
        try:
            with self.engine.begin() as connection:
                connection.execute(
                    insert(feedback_records).values(
                        feedback_id=feedback_id,
                        idempotency_key=idempotency_key,
                        payload=payload,
                        created_at=now,
                    )
                )
                connection.execute(
                    insert(feedback_outbox).values(
                        feedback_id=feedback_id,
                        payload=payload,
                        created_at=now,
                    )
                )
            return StoredFeedback(feedback_id=feedback_id, payload=payload, duplicate=False)
        except IntegrityError:
            with self.engine.connect() as connection:
                condition = (
                    feedback_records.c.idempotency_key == idempotency_key
                    if idempotency_key
                    else feedback_records.c.feedback_id == feedback_id
                )
                row = connection.execute(select(feedback_records).where(condition)).mappings().one()
            return StoredFeedback(
                feedback_id=str(row["feedback_id"]),
                payload=json.loads(json.dumps(row["payload"])),
                duplicate=True,
            )

    def pending(self, *, limit: int = 100) -> list[dict[str, Any]]:
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(feedback_outbox)
                .where(feedback_outbox.c.published_at.is_(None))
                .order_by(feedback_outbox.c.created_at)
                .limit(limit)
            ).mappings()
            return [dict(row) for row in rows]

    def mark_published(self, feedback_id: str) -> None:
        with self.engine.begin() as connection:
            connection.execute(
                update(feedback_outbox)
                .where(feedback_outbox.c.feedback_id == feedback_id)
                .values(published_at=datetime.now(UTC))
            )

    def pending_count(self) -> int:
        with self.engine.connect() as connection:
            count = connection.scalar(
                select(func.count()).select_from(feedback_outbox).where(feedback_outbox.c.published_at.is_(None))
            )
        return int(count or 0)
