"""FastAPI application entry point."""

from __future__ import annotations

import time
from typing import Any

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.core.config.settings import get_settings

logger = structlog.get_logger(__name__)

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description="RAG system for public exam (oposiciones) preparation.",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_start_time = time.time()


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
