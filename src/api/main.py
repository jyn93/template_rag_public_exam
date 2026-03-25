"""FastAPI application entry point."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from src.api.routers.chat import router as chat_router
from src.api.routers.exam import router as exam_router
from src.api.routers.ingestion import router as ingestion_router
from src.core.config.logging_config import configure_logging
from src.core.config.settings import get_settings

settings = get_settings()

# Configure logging before any logger is instantiated.
configure_logging(log_level=settings.log_level, debug=settings.debug)

logger = structlog.get_logger(__name__)

_start_time: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:  # noqa: ARG001
    """Record the precise startup time when the application is ready."""
    global _start_time  # noqa: PLW0603
    _start_time = time.time()
    logger.info("api_startup", service=settings.app_name)
    yield
    logger.info("api_shutdown", service=settings.app_name)


app = FastAPI(
    title=settings.app_name,
    description="RAG system for public exam (oposiciones) preparation.",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next: Callable[..., Any]) -> Response:
    """Log every HTTP request with method, path, status code, and duration."""
    start = time.perf_counter()
    response: Response = await call_next(request)
    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    logger.info(
        "http_request",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=duration_ms,
    )
    return response


app.include_router(ingestion_router)
app.include_router(chat_router)
app.include_router(exam_router)


@app.get("/health", tags=["ops"])
async def health_check() -> dict[str, Any]:
    """Return service health status.

    Returns:
        Dict with status, uptime_seconds, and service name.
    """
    uptime = round(time.time() - _start_time, 2)
    logger.info("health_check", uptime_seconds=uptime)
    return {
        "status": "ok",
        "service": settings.app_name,
        "uptime_seconds": uptime,
    }


@app.get("/", tags=["ops"])
async def root() -> dict[str, str]:
    """Root endpoint — redirect hint for API consumers.

    Returns:
        Dict with message and docs URL.
    """
    return {
        "message": f"Welcome to {settings.app_name} API",
        "docs": "/docs",
    }
