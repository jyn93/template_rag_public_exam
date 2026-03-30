"""Unit tests for LangfuseService."""

from __future__ import annotations

import os
from unittest.mock import patch

import litellm
import pytest

from src.core.config.settings import Settings
from src.infrastructure.observability.langfuse_tracer import LangfuseService

# ── Helpers ───────────────────────────────────────────────────────────────────

_LANGFUSE_ENV_KEYS = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST")


def _settings(**overrides: object) -> Settings:
    """Build a minimal Settings instance with Langfuse fields set."""
    defaults: dict[str, object] = {
        "langfuse_enabled": True,
        "langfuse_public_key": "pk-test-key",
        "langfuse_secret_key": "sk-test-key",
        "langfuse_host": "http://langfuse.test:3000",
    }
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)  # type: ignore[call-arg]


@pytest.fixture(autouse=True)
def _clean_langfuse_env():
    """Remove Langfuse env vars before each test and restore litellm callbacks."""
    original_success = list(litellm.success_callback)
    original_failure = list(litellm.failure_callback)

    for key in _LANGFUSE_ENV_KEYS:
        os.environ.pop(key, None)

    yield

    # Restore litellm callbacks to the state before the test.
    litellm.success_callback = original_success
    litellm.failure_callback = original_failure

    for key in _LANGFUSE_ENV_KEYS:
        os.environ.pop(key, None)


# ── TestLangfuseServiceConfigure ─────────────────────────────────────────────


class TestLangfuseServiceConfigure:
    """Tests for LangfuseService.configure()."""

    def test_sets_public_key_env_var(self):
        """LANGFUSE_PUBLIC_KEY is written to the environment."""
        LangfuseService.configure(_settings())

        assert os.environ["LANGFUSE_PUBLIC_KEY"] == "pk-test-key"

    def test_sets_secret_key_env_var(self):
        """LANGFUSE_SECRET_KEY is written to the environment."""
        LangfuseService.configure(_settings())

        assert os.environ["LANGFUSE_SECRET_KEY"] == "sk-test-key"

    def test_sets_host_env_var(self):
        """LANGFUSE_HOST is written to the environment."""
        LangfuseService.configure(_settings())

        assert os.environ["LANGFUSE_HOST"] == "http://langfuse.test:3000"

    def test_registers_litellm_success_callback(self):
        """'langfuse' is added to litellm.success_callback."""
        LangfuseService.configure(_settings())

        assert "langfuse" in litellm.success_callback

    def test_registers_litellm_failure_callback(self):
        """'langfuse' is added to litellm.failure_callback."""
        LangfuseService.configure(_settings())

        assert "langfuse" in litellm.failure_callback

    def test_idempotent_success_callback(self):
        """Calling configure() twice does not duplicate the success callback."""
        LangfuseService.configure(_settings())
        LangfuseService.configure(_settings())

        assert litellm.success_callback.count("langfuse") == 1

    def test_idempotent_failure_callback(self):
        """Calling configure() twice does not duplicate the failure callback."""
        LangfuseService.configure(_settings())
        LangfuseService.configure(_settings())

        assert litellm.failure_callback.count("langfuse") == 1

    def test_noop_when_langfuse_disabled(self):
        """No environment variables are set when langfuse_enabled=False."""
        LangfuseService.configure(_settings(langfuse_enabled=False))

        assert "LANGFUSE_PUBLIC_KEY" not in os.environ
        assert "LANGFUSE_SECRET_KEY" not in os.environ

    def test_no_callbacks_when_langfuse_disabled(self):
        """LiteLLM callbacks are not modified when langfuse_enabled=False."""
        before = list(litellm.success_callback)
        LangfuseService.configure(_settings(langfuse_enabled=False))

        assert litellm.success_callback == before

    def test_noop_when_public_key_missing(self):
        """No setup is performed when langfuse_public_key is empty."""
        LangfuseService.configure(_settings(langfuse_public_key=""))

        assert "LANGFUSE_PUBLIC_KEY" not in os.environ

    def test_noop_when_secret_key_missing(self):
        """No setup is performed when langfuse_secret_key is empty."""
        LangfuseService.configure(_settings(langfuse_secret_key=""))

        assert "LANGFUSE_SECRET_KEY" not in os.environ

    def test_no_callbacks_when_keys_missing(self):
        """LiteLLM callbacks are not modified when keys are empty."""
        before_success = list(litellm.success_callback)
        before_failure = list(litellm.failure_callback)

        LangfuseService.configure(
            _settings(langfuse_public_key="", langfuse_secret_key="")
        )

        assert litellm.success_callback == before_success
        assert litellm.failure_callback == before_failure


# ── TestLangfuseServiceIsConfigured ──────────────────────────────────────────


class TestLangfuseServiceIsConfigured:
    """Tests for LangfuseService.is_configured()."""

    def test_returns_true_when_both_keys_set(self):
        """True when LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set."""
        with patch.dict(
            os.environ,
            {"LANGFUSE_PUBLIC_KEY": "pk-x", "LANGFUSE_SECRET_KEY": "sk-x"},
        ):
            assert LangfuseService.is_configured() is True

    def test_returns_false_when_public_key_absent(self):
        """False when LANGFUSE_PUBLIC_KEY is not in the environment."""
        env = {"LANGFUSE_SECRET_KEY": "sk-x"}
        # Ensure PUBLIC_KEY is absent
        with patch.dict(os.environ, env):
            os.environ.pop("LANGFUSE_PUBLIC_KEY", None)
            assert LangfuseService.is_configured() is False

    def test_returns_false_when_secret_key_absent(self):
        """False when LANGFUSE_SECRET_KEY is not in the environment."""
        env = {"LANGFUSE_PUBLIC_KEY": "pk-x"}
        with patch.dict(os.environ, env):
            os.environ.pop("LANGFUSE_SECRET_KEY", None)
            assert LangfuseService.is_configured() is False

    def test_returns_false_when_public_key_empty(self):
        """False when LANGFUSE_PUBLIC_KEY is an empty string."""
        with patch.dict(
            os.environ,
            {"LANGFUSE_PUBLIC_KEY": "", "LANGFUSE_SECRET_KEY": "sk-x"},
        ):
            assert LangfuseService.is_configured() is False

    def test_returns_false_when_secret_key_empty(self):
        """False when LANGFUSE_SECRET_KEY is an empty string."""
        with patch.dict(
            os.environ,
            {"LANGFUSE_PUBLIC_KEY": "pk-x", "LANGFUSE_SECRET_KEY": ""},
        ):
            assert LangfuseService.is_configured() is False

    def test_true_after_configure(self):
        """is_configured() returns True after a successful configure() call."""
        LangfuseService.configure(_settings())

        assert LangfuseService.is_configured() is True

    def test_false_before_configure(self):
        """is_configured() returns False before configure() is called."""
        assert LangfuseService.is_configured() is False
