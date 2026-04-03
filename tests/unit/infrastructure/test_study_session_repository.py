"""Unit tests for StudySessionRepository using in-memory SQLite."""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.infrastructure.database.models import Base
from src.infrastructure.database.repository import StudySessionRepository

# ── Fixtures ──────────────────────────────────────────────────────────────────

DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(scope="function")
async def db_session() -> AsyncSession:  # type: ignore[misc]
    """Yield a fresh in-memory SQLite session for each test."""
    engine = create_async_engine(DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with factory() as session:
        yield session  # type: ignore[misc]

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


# ── TestStudySessionRepository ────────────────────────────────────────────────


class TestStudySessionRepository:
    """Tests for StudySessionRepository CRUD operations."""

    async def test_create_returns_record_with_id(
        self, db_session: AsyncSession
    ) -> None:
        """Created record has a non-None UUID id."""
        repo = StudySessionRepository(db_session)
        record = await repo.create(session_type="exam", topic="Test topic")
        assert record.id is not None

    async def test_create_persists_session_type(
        self, db_session: AsyncSession
    ) -> None:
        """session_type is persisted correctly."""
        repo = StudySessionRepository(db_session)
        record = await repo.create(session_type="chat")
        assert record.session_type == "chat"

    async def test_create_persists_optional_fields(
        self, db_session: AsyncSession
    ) -> None:
        """All optional fields are stored and returned."""
        repo = StudySessionRepository(db_session)
        record = await repo.create(
            session_type="exam",
            topic="Administrative Law",
            subject="Law",
            score_avg=7.5,
            num_questions=10,
            num_correct=7,
            percentage=70,
        )
        assert record.topic == "Administrative Law"
        assert record.subject == "Law"
        assert record.score_avg == pytest.approx(7.5)
        assert record.num_questions == 10
        assert record.num_correct == 7
        assert record.percentage == 70

    async def test_create_sets_created_at(self, db_session: AsyncSession) -> None:
        """created_at is set to a non-None datetime."""
        repo = StudySessionRepository(db_session)
        record = await repo.create(session_type="exam")
        assert record.created_at is not None

    async def test_list_recent_returns_all_records(
        self, db_session: AsyncSession
    ) -> None:
        """list_recent returns all records when limit is large."""
        repo = StudySessionRepository(db_session)
        await repo.create(session_type="exam")
        await repo.create(session_type="chat")
        records = await repo.list_recent(limit=100)
        assert len(records) == 2

    async def test_list_recent_ordered_newest_first(
        self, db_session: AsyncSession
    ) -> None:
        """Records are returned newest first."""
        repo = StudySessionRepository(db_session)
        r1 = await repo.create(session_type="exam", topic="first")
        r2 = await repo.create(session_type="exam", topic="second")
        records = await repo.list_recent()
        # newest first: r2 before r1
        assert records[0].id == r2.id
        assert records[1].id == r1.id

    async def test_list_recent_respects_limit(
        self, db_session: AsyncSession
    ) -> None:
        """list_recent returns at most `limit` records."""
        repo = StudySessionRepository(db_session)
        for i in range(5):
            await repo.create(session_type="exam", topic=f"topic {i}")
        records = await repo.list_recent(limit=3)
        assert len(records) == 3

    async def test_list_recent_filters_by_subject(
        self, db_session: AsyncSession
    ) -> None:
        """subject filter returns only matching records."""
        repo = StudySessionRepository(db_session)
        await repo.create(session_type="exam", subject="Law")
        await repo.create(session_type="exam", subject="History")
        records = await repo.list_recent(subject="Law")
        assert all(r.subject == "Law" for r in records)
        assert len(records) == 1

    async def test_list_recent_filters_by_session_type(
        self, db_session: AsyncSession
    ) -> None:
        """session_type filter returns only matching records."""
        repo = StudySessionRepository(db_session)
        await repo.create(session_type="exam")
        await repo.create(session_type="chat")
        records = await repo.list_recent(session_type="chat")
        assert all(r.session_type == "chat" for r in records)
        assert len(records) == 1

    async def test_list_recent_empty_when_no_records(
        self, db_session: AsyncSession
    ) -> None:
        """Empty list returned when no records exist."""
        repo = StudySessionRepository(db_session)
        records = await repo.list_recent()
        assert records == []

    async def test_get_by_id_returns_correct_record(
        self, db_session: AsyncSession
    ) -> None:
        """get_by_id retrieves the record with the matching UUID."""
        repo = StudySessionRepository(db_session)
        created = await repo.create(session_type="exam", topic="Find me")
        fetched = await repo.get_by_id(created.id)
        assert fetched is not None
        assert fetched.id == created.id
        assert fetched.topic == "Find me"

    async def test_get_by_id_returns_none_for_unknown_id(
        self, db_session: AsyncSession
    ) -> None:
        """get_by_id returns None when the UUID does not exist."""
        import uuid

        repo = StudySessionRepository(db_session)
        result = await repo.get_by_id(uuid.uuid4())
        assert result is None

    async def test_create_with_null_optional_fields(
        self, db_session: AsyncSession
    ) -> None:
        """All optional fields default to None when not provided."""
        repo = StudySessionRepository(db_session)
        record = await repo.create(session_type="chat")
        assert record.topic is None
        assert record.subject is None
        assert record.score_avg is None
        assert record.num_questions is None
        assert record.num_correct is None
        assert record.percentage is None
