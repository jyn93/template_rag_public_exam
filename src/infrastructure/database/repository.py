"""Repository for study session persistence."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.infrastructure.database.models import StudySession

__all__ = ["StudySessionRepository"]


class StudySessionRepository:
    """CRUD repository for :class:`StudySession` records.

    Args:
        session: Open async SQLAlchemy session (injected via FastAPI Depends).

    Example:
        >>> async with session_factory() as session:
        ...     repo = StudySessionRepository(session)
        ...     session_obj = await repo.create(session_type="exam", topic="appeal")
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        session_type: str,
        topic: str | None = None,
        subject: str | None = None,
        score_avg: float | None = None,
        num_questions: int | None = None,
        num_correct: int | None = None,
        percentage: int | None = None,
    ) -> StudySession:
        """Persist a new study session record.

        Args:
            session_type: ``"exam"`` or ``"chat"``.
            topic: The query or topic text entered by the user.
            subject: Optional subject label.
            score_avg: Mean score per question (0–10).
            num_questions: Total number of questions.
            num_correct: Number of correctly answered questions.
            percentage: Percentage correct (0–100).

        Returns:
            The persisted :class:`StudySession` instance with a generated id.
        """
        record = StudySession(
            session_type=session_type,
            topic=topic,
            subject=subject,
            score_avg=score_avg,
            num_questions=num_questions,
            num_correct=num_correct,
            percentage=percentage,
            created_at=datetime.now(UTC),
        )
        self._session.add(record)
        await self._session.commit()
        await self._session.refresh(record)
        return record

    async def list_recent(
        self,
        limit: int = 50,
        subject: str | None = None,
        session_type: str | None = None,
    ) -> list[StudySession]:
        """Return recent study sessions, newest first.

        Args:
            limit: Maximum number of records to return (default 50).
            subject: If provided, filter by this subject label.
            session_type: If provided, filter by ``"exam"`` or ``"chat"``.

        Returns:
            List of :class:`StudySession` sorted by ``created_at`` descending.
        """
        stmt = select(StudySession).order_by(StudySession.created_at.desc())
        if subject:
            stmt = stmt.where(StudySession.subject == subject)
        if session_type:
            stmt = stmt.where(StudySession.session_type == session_type)
        stmt = stmt.limit(limit)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def get_by_id(self, session_id: uuid.UUID) -> StudySession | None:
        """Fetch a single session by its UUID primary key.

        Args:
            session_id: The UUID of the record to retrieve.

        Returns:
            The matching :class:`StudySession` or ``None`` if not found.
        """
        result = await self._session.execute(
            select(StudySession).where(StudySession.id == session_id)
        )
        return result.scalar_one_or_none()
