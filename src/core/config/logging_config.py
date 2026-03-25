"""Structured logging configuration with caller context and environment-aware output.

Configures structlog with:
- Caller context processor (filename:ClassName.method:lineno)
- Full exception tracebacks via exc_info
- Dev mode: coloured human-readable console output
- Prod mode: JSON output suitable for log aggregators
- stdlib logging integration so third-party libraries are also captured
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import structlog
from structlog.types import EventDict, WrappedLogger

__all__ = ["configure_logging"]

# Modules whose frames should be skipped when walking the call stack.
_SKIP_MODULES: frozenset[str] = frozenset(
    {
        __name__,
        "structlog._base",
        "structlog._log_levels",
        "structlog.stdlib",
        "structlog._frames",
        "logging",
        "logging.handlers",
    }
)


def _add_caller_info(
    _logger: WrappedLogger,
    _method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Inject filename, class name, function name, and line number into every record.

    Walks the call stack upward, skipping structlog and logging internals, until
    it reaches the first application frame.  The result is stored as a single
    ``caller`` key in the format ``filename.py:ClassName.method:lineno`` (or
    ``filename.py:function:lineno`` when the call is made outside a class).

    Args:
        _logger: Unused — required by the structlog processor protocol.
        _method_name: Unused — required by the structlog processor protocol.
        event_dict: Mutable log record dictionary to enrich.

    Returns:
        The enriched *event_dict* with a ``caller`` key added.
    """
    frame = sys._getframe(0)  # noqa: SLF001
    while frame is not None:
        module: str = frame.f_globals.get("__name__", "")
        is_internal = any(module.startswith(skip) for skip in _SKIP_MODULES)
        if not is_internal and module:
            filename = Path(frame.f_code.co_filename).name
            funcname = frame.f_code.co_name
            lineno = frame.f_lineno
            local_self = frame.f_locals.get("self")
            if local_self is not None:
                cls_name = type(local_self).__name__
                event_dict["caller"] = f"{filename}:{cls_name}.{funcname}:{lineno}"
            else:
                event_dict["caller"] = f"{filename}:{funcname}:{lineno}"
            break
        frame = frame.f_back  # type: ignore[assignment]
    return event_dict


def configure_logging(log_level: str = "INFO", *, debug: bool = False) -> None:
    """Configure structlog for the whole application.

    Must be called **once** at application startup, before any logger is used.
    Subsequent calls are no-ops thanks to ``cache_logger_on_first_use=True``.

    Behaviour:
    - In *debug* mode (``DEBUG=true`` in settings): coloured, human-readable
      console output with pretty-printed tracebacks.
    - In production mode: single-line JSON records with exception info serialised
      under an ``exception`` key, suitable for log aggregators (Loki, ELK, etc.).
    - Both modes capture ``exc_info`` so full tracebacks appear whenever
      ``logger.exception()`` or ``logger.error(..., exc_info=True)`` is used.
    - stdlib ``logging`` is wired through structlog so third-party library
      warnings (e.g. qdrant-client version warnings) are formatted consistently.

    Args:
        log_level: Minimum log level string (``"DEBUG"``, ``"INFO"``, etc.).
            Case-insensitive.  Defaults to ``"INFO"``.
        debug: When ``True`` the console renderer is used instead of JSON.

    Example:
        >>> from src.core.config.logging_config import configure_logging
        >>> configure_logging(log_level="DEBUG", debug=True)
    """
    numeric_level: int = getattr(logging, log_level.upper(), logging.INFO)

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
        _add_caller_info,
    ]

    if debug:
        processors: list[Any] = [
            *shared_processors,
            structlog.dev.ConsoleRenderer(colors=True),
        ]
    else:
        processors = [
            *shared_processors,
            structlog.processors.ExceptionRenderer(),
            structlog.processors.JSONRenderer(),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Route stdlib logging (third-party libs) through the same structlog pipeline.
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=numeric_level,
        force=True,
    )
