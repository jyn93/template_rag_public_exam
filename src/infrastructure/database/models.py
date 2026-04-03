"""SQLAlchemy ORM models for study history."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import DateTime, Float, Integer, String, Uuid
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

__all__ = ["Base", "StudySession"]


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""


class StudySession(Base):
    """Record of a completed exam or chat study session.

    Attributes:
        id: UUID primary key.
        session_type: Either ``"exam"`` or ``"chat"``.
        subject: Optional subject label used when generating the session.
        topic: Topic or query string entered by the user.
        score_avg: Average score per question (0–10); ``None`` for chat sessions.
        num_questions: Number of questions in the exam; ``None`` for chat sessions.
        num_correct: Number of correct answers; ``None`` for chat sessions.
        percentage: Percentage of correct answers (0–100); ``None`` for chat.
        created_at: UTC timestamp when the session was recorded.
    """

    __tablename__ = "study_sessions"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    session_type: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    subject: Mapped[str | None] = mapped_column(String(256), nullable=True, index=True)
    topic: Mapped[str | None] = mapped_column(String(512), nullable=True)
    score_avg: Mapped[float | None] = mapped_column(Float, nullable=True)
    num_questions: Mapped[int | None] = mapped_column(Integer, nullable=True)
    num_correct: Mapped[int | None] = mapped_column(Integer, nullable=True)
    percentage: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )
