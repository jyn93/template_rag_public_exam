"""FastAPI router for study session history endpoints."""

from __future__ import annotations

import uuid
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from src.infrastructure.database.engine import get_async_session
from src.infrastructure.database.repository import StudySessionRepository

__all__ = ["router"]

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/history", tags=["history"])


# ── Request / Response models ─────────────────────────────────────────────────


class RecordSessionRequest(BaseModel):
    """Request body for recording a new study session."""

    session_type: str = Field(
        ...,
        pattern="^(exam|chat)$",
        description="Session type: 'exam' or 'chat'.",
    )
    topic: str | None = Field(default=None, description="Topic or query text.")
    subject: str | None = Field(default=None, description="Subject label.")
    score_avg: float | None = Field(
        default=None, ge=0.0, le=10.0, description="Average score per question."
    )
    num_questions: int | None = Field(
        default=None, ge=0, description="Total number of questions."
    )
    num_correct: int | None = Field(
        default=None, ge=0, description="Number of correct answers."
    )
    percentage: int | None = Field(
        default=None, ge=0, le=100, description="Percentage correct."
    )


class StudySessionResponse(BaseModel):
    """Response body for a single study session."""

    id: uuid.UUID
    session_type: str
    topic: str | None
    subject: str | None
    score_avg: float | None
    num_questions: int | None
    num_correct: int | None
    percentage: int | None
    created_at: str  # ISO 8601 string


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.post(
    "/sessions",
    summary="Record a completed study session",
    response_model=StudySessionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def record_session(
    body: RecordSessionRequest,
    db: Annotated[AsyncSession, Depends(get_async_session)],
) -> StudySessionResponse:
    """Persist a new study session record in PostgreSQL.

    Args:
        body: Session metadata to store.
        db: Injected async database session.

    Returns:
        The persisted :class:`StudySessionResponse` with generated id and timestamp.
    """
    logger.info(
        "history_record_session",
        session_type=body.session_type,
        subject=body.subject,
    )
    repo = StudySessionRepository(db)
    record = await repo.create(
        session_type=body.session_type,
        topic=body.topic,
        subject=body.subject,
        score_avg=body.score_avg,
        num_questions=body.num_questions,
        num_correct=body.num_correct,
        percentage=body.percentage,
    )
    return StudySessionResponse(
        id=record.id,
        session_type=record.session_type,
        topic=record.topic,
        subject=record.subject,
        score_avg=record.score_avg,
        num_questions=record.num_questions,
        num_correct=record.num_correct,
        percentage=record.percentage,
        created_at=record.created_at.isoformat(),
    )


@router.get(
    "/sessions",
    summary="List recent study sessions",
    response_model=list[StudySessionResponse],
    status_code=status.HTTP_200_OK,
)
async def list_sessions(
    db: Annotated[AsyncSession, Depends(get_async_session)],
    limit: int = 50,
    subject: str | None = None,
    session_type: str | None = None,
) -> list[StudySessionResponse]:
    """Return recent study sessions ordered by most recent first.

    Args:
        db: Injected async database session.
        limit: Maximum number of records to return (default 50, max 200).
        subject: Optional subject filter.
        session_type: Optional type filter (``"exam"`` or ``"chat"``).

    Returns:
        List of :class:`StudySessionResponse` sorted newest first.

    Raises:
        422: If ``session_type`` is not ``"exam"`` or ``"chat"``.
    """
    if session_type and session_type not in {"exam", "chat"}:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="session_type must be 'exam' or 'chat'.",
        )

    limit = min(limit, 200)
    repo = StudySessionRepository(db)
    records = await repo.list_recent(
        limit=limit,
        subject=subject,
        session_type=session_type,
    )
    return [
        StudySessionResponse(
            id=r.id,
            session_type=r.session_type,
            topic=r.topic,
            subject=r.subject,
            score_avg=r.score_avg,
            num_questions=r.num_questions,
            num_correct=r.num_correct,
            percentage=r.percentage,
            created_at=r.created_at.isoformat(),
        )
        for r in records
    ]
