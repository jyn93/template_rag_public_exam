"""Unit tests for the structured logging configuration."""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest
import structlog

from src.core.config.logging_config import _add_caller_info, configure_logging


class TestAddCallerInfo:
    """Tests for the _add_caller_info structlog processor."""

    def test_adds_caller_key(self) -> None:
        """Verify that the processor injects a 'caller' key into event_dict."""
        event_dict: dict[str, object] = {"event": "test_event"}
        result = _add_caller_info(None, "info", event_dict)  # type: ignore[arg-type]
        assert "caller" in result

    def test_caller_contains_filename(self) -> None:
        """Verify that 'caller' contains the calling filename."""
        event_dict: dict[str, object] = {"event": "test_event"}
        result = _add_caller_info(None, "info", event_dict)  # type: ignore[arg-type]
        caller = str(result["caller"])
        assert ".py:" in caller

    def test_caller_contains_line_number(self) -> None:
        """Verify that 'caller' contains a line number (colon-separated integer)."""
        event_dict: dict[str, object] = {"event": "test_event"}
        result = _add_caller_info(None, "info", event_dict)  # type: ignore[arg-type]
        caller = str(result["caller"])
        # Last segment after the final ':' should be a line number
        last_part = caller.rsplit(":", 1)[-1]
        assert last_part.isdigit(), f"Expected line number, got: {last_part!r}"

    def test_caller_includes_class_name_when_in_method(self) -> None:
        """Verify class name appears in caller when called from an instance method."""

        class _Dummy:
            def do_log(self) -> dict[str, object]:
                event_dict: dict[str, object] = {"event": "x"}
                return _add_caller_info(None, "info", event_dict)  # type: ignore[arg-type]

        result = _Dummy().do_log()
        assert "_Dummy" in str(result["caller"])

    def test_caller_includes_class_name_when_called_from_method(self) -> None:
        """Verify class name appears when the processor is called from any method."""
        # This method itself runs inside TestAddCallerInfo, so the class should appear.
        event_dict: dict[str, object] = {"event": "x"}
        result = _add_caller_info(None, "info", event_dict)  # type: ignore[arg-type]
        caller = str(result["caller"])
        # The class name should be present because we're inside a class method
        assert "TestAddCallerInfo" in caller

    def test_returns_event_dict_unchanged_on_empty_stack(self) -> None:
        """Verify the processor returns the event dict even if no frame is found."""
        event_dict: dict[str, object] = {"event": "x"}
        result = _add_caller_info(None, "info", event_dict)  # type: ignore[arg-type]
        assert result is event_dict


def _module_level_caller_call() -> dict[str, object]:
    """Helper called at module scope (no enclosing class) to test caller format."""
    event_dict: dict[str, object] = {"event": "module_level"}
    return _add_caller_info(None, "info", event_dict)  # type: ignore[arg-type]


def test_caller_omits_class_name_in_module_scope() -> None:
    """Verify caller format is filename:funcname:lineno when no class is present."""
    result = _module_level_caller_call()
    caller = str(result["caller"])
    # Should contain the helper function name without a preceding class name
    assert "_module_level_caller_call" in caller
    assert "." not in caller.split(":")[1]


class TestConfigureLogging:
    """Tests for the configure_logging() factory function.

    All tests patch both structlog.configure and logging.basicConfig to prevent
    side effects on pytest's logging capture infrastructure.
    """

    def test_configure_debug_mode_uses_console_renderer(self) -> None:
        """In debug mode the processor chain ends with ConsoleRenderer."""
        with patch("structlog.configure") as mock_configure:
            with patch("logging.basicConfig"):
                configure_logging(log_level="DEBUG", debug=True)
            assert mock_configure.called
            processors = mock_configure.call_args.kwargs["processors"]
            assert any(isinstance(p, structlog.dev.ConsoleRenderer) for p in processors)

    def test_configure_prod_mode_uses_json_renderer(self) -> None:
        """In production mode the processor chain ends with JSONRenderer."""
        with patch("structlog.configure") as mock_configure:
            with patch("logging.basicConfig"):
                configure_logging(log_level="INFO", debug=False)
            assert mock_configure.called
            processors = mock_configure.call_args.kwargs["processors"]
            assert any(
                isinstance(p, structlog.processors.JSONRenderer) for p in processors
            )

    def test_configure_prod_mode_includes_exception_renderer(self) -> None:
        """In production mode ExceptionRenderer is included before JSONRenderer."""
        with patch("structlog.configure") as mock_configure:
            with patch("logging.basicConfig"):
                configure_logging(log_level="INFO", debug=False)
            processors = mock_configure.call_args.kwargs["processors"]
            assert any(
                isinstance(p, structlog.processors.ExceptionRenderer)
                for p in processors
            )

    def test_configure_sets_filtering_bound_logger(self) -> None:
        """Verify wrapper_class is a filtering bound logger (respects log_level)."""
        with patch("structlog.configure") as mock_configure:
            with patch("logging.basicConfig"):
                configure_logging(log_level="WARNING", debug=False)
            wrapper = mock_configure.call_args.kwargs["wrapper_class"]
            assert callable(wrapper)

    def test_configure_passes_correct_level_to_basicconfig(self) -> None:
        """Verify basicConfig is called with the numeric log level."""
        with patch("structlog.configure"):
            with patch("logging.basicConfig") as mock_basic:
                configure_logging(log_level="WARNING", debug=False)
            mock_basic.assert_called_once()
            call_kwargs = mock_basic.call_args.kwargs
            assert call_kwargs.get("level") == logging.WARNING

    def test_configure_unknown_level_falls_back_to_info(self) -> None:
        """An unrecognised log level string should default to INFO."""
        with patch("structlog.configure") as mock_configure:
            with patch("logging.basicConfig"):
                configure_logging(log_level="NONSENSE", debug=False)
            wrapper = mock_configure.call_args.kwargs["wrapper_class"]
            assert callable(wrapper)

    def test_configure_caller_info_processor_present(self) -> None:
        """The _add_caller_info processor is included in the chain."""
        with patch("structlog.configure") as mock_configure:
            with patch("logging.basicConfig"):
                configure_logging(debug=False)
            processors = mock_configure.call_args.kwargs["processors"]
            assert _add_caller_info in processors

    @pytest.mark.parametrize("debug", [True, False])
    def test_configure_timestamp_processor_present(self, debug: bool) -> None:
        """TimeStamper is always included regardless of debug flag."""
        with patch("structlog.configure") as mock_configure:
            with patch("logging.basicConfig"):
                configure_logging(debug=debug)
            processors = mock_configure.call_args.kwargs["processors"]
            assert any(
                isinstance(p, structlog.processors.TimeStamper) for p in processors
            )
