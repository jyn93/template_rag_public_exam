"""Async SQLAlchemy engine and session factory."""

from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.core.config.settings import get_settings

__all__ = ["get_async_session", "get_engine"]

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """Return the shared async database engine (created on first call).

    Returns:
        :class:`AsyncEngine` configured from ``Settings.database_url``.
    """
    global _engine  # noqa: PLW0603
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(
            settings.database_url,
            pool_pre_ping=True,
            echo=settings.debug,
        )
    return _engine


def _get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory  # noqa: PLW0603
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=get_engine(),
            expire_on_commit=False,
            class_=AsyncSession,
        )
    return _session_factory


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield an :class:`AsyncSession` from the shared session factory.

    Intended for use with FastAPI's ``Depends`` system.

    Yields:
        An open :class:`AsyncSession` that is committed and closed after use.
    """
    factory = _get_session_factory()
    async with factory() as session:
        yield session
