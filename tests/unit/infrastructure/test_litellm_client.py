"""Unit tests for LiteLLMClient."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.exceptions import GenerationError
from src.infrastructure.llm.litellm_client import LiteLLMClient

# ── Helpers ───────────────────────────────────────────────────────────────────


def make_litellm_response(content: str = "Test answer", total_tokens: int = 42):
    """Build a minimal mock that mimics a litellm completion response."""
    message = MagicMock()
    message.content = content

    choice = MagicMock()
    choice.message = message

    usage = MagicMock()
    usage.total_tokens = total_tokens

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def client() -> LiteLLMClient:
    """Default LiteLLMClient for tests."""
    return LiteLLMClient(model="gpt-4o-mini", temperature=0.1)


# ── TestLiteLLMClientInit ─────────────────────────────────────────────────────


class TestLiteLLMClientInit:
    """Tests for LiteLLMClient construction."""

    def test_model_stored(self):
        """Model identifier is stored."""
        c = LiteLLMClient(model="gpt-4o")
        assert c._model == "gpt-4o"

    def test_temperature_stored(self):
        """Temperature is stored."""
        c = LiteLLMClient(model="gpt-4o", temperature=0.7)
        assert c._temperature == pytest.approx(0.7)

    def test_default_max_tokens(self):
        """Default max_tokens is 4096."""
        c = LiteLLMClient(model="gpt-4o")
        assert c._max_tokens == 4096

    def test_default_timeout(self):
        """Default timeout is 60 seconds."""
        c = LiteLLMClient(model="gpt-4o")
        assert c._timeout == pytest.approx(60.0)


# ── TestComplete ──────────────────────────────────────────────────────────────


class TestComplete:
    """Tests for LiteLLMClient.complete."""

    async def test_returns_response_text(self, client):
        """complete() returns the model's text content."""
        mock_response = make_litellm_response("Administrative law answer")

        with patch(
            "src.infrastructure.llm.litellm_client.litellm.acompletion",
            new=AsyncMock(return_value=mock_response),
        ):
            result = await client.complete("What is administrative law?")

        assert result == "Administrative law answer"

    async def test_sends_user_message(self, client):
        """User prompt is included in the messages list."""
        mock_response = make_litellm_response()
        acompletion = AsyncMock(return_value=mock_response)

        with patch(
            "src.infrastructure.llm.litellm_client.litellm.acompletion",
            new=acompletion,
        ):
            await client.complete("Hello model")

        call_kwargs = acompletion.call_args.kwargs
        messages = call_kwargs["messages"]
        user_msgs = [m for m in messages if m["role"] == "user"]
        assert len(user_msgs) == 1
        assert user_msgs[0]["content"] == "Hello model"

    async def test_includes_system_prompt_when_provided(self, client):
        """System prompt is prepended as a system-role message."""
        mock_response = make_litellm_response()
        acompletion = AsyncMock(return_value=mock_response)

        with patch(
            "src.infrastructure.llm.litellm_client.litellm.acompletion",
            new=acompletion,
        ):
            await client.complete("Query", system_prompt="Be concise.")

        messages = acompletion.call_args.kwargs["messages"]
        system_msgs = [m for m in messages if m["role"] == "system"]
        assert system_msgs[0]["content"] == "Be concise."

    async def test_omits_system_message_when_none(self, client):
        """No system message is added when system_prompt is None."""
        mock_response = make_litellm_response()
        acompletion = AsyncMock(return_value=mock_response)

        with patch(
            "src.infrastructure.llm.litellm_client.litellm.acompletion",
            new=acompletion,
        ):
            await client.complete("Query")

        messages = acompletion.call_args.kwargs["messages"]
        assert not any(m["role"] == "system" for m in messages)

    async def test_passes_model_to_litellm(self, client):
        """The configured model name is forwarded to litellm."""
        mock_response = make_litellm_response()
        acompletion = AsyncMock(return_value=mock_response)

        with patch(
            "src.infrastructure.llm.litellm_client.litellm.acompletion",
            new=acompletion,
        ):
            await client.complete("Query")

        assert acompletion.call_args.kwargs["model"] == "gpt-4o-mini"

    async def test_raises_generation_error_on_litellm_failure(self, client):
        """Any exception from litellm is wrapped in GenerationError."""
        with patch(
            "src.infrastructure.llm.litellm_client.litellm.acompletion",
            new=AsyncMock(side_effect=ConnectionError("network error")),
        ):
            with pytest.raises(GenerationError, match="LiteLLM completion failed"):
                await client.complete("Query")

    async def test_error_chains_original_cause(self, client):
        """The original exception is chained on the GenerationError."""
        cause = RuntimeError("timeout")

        with patch(
            "src.infrastructure.llm.litellm_client.litellm.acompletion",
            new=AsyncMock(side_effect=cause),
        ):
            with pytest.raises(GenerationError) as exc_info:
                await client.complete("Query")

        assert exc_info.value.__cause__ is cause

    async def test_propagates_generation_error_unchanged(self, client):
        """GenerationError from deeper layers is re-raised as-is."""
        original = GenerationError("already wrapped")

        with patch(
            "src.infrastructure.llm.litellm_client.litellm.acompletion",
            new=AsyncMock(side_effect=original),
        ):
            with pytest.raises(GenerationError) as exc_info:
                await client.complete("Query")

        assert exc_info.value is original

    async def test_handles_empty_content_gracefully(self, client):
        """None content from the model is converted to an empty string."""
        mock_response = make_litellm_response(content=None)  # type: ignore[arg-type]

        with patch(
            "src.infrastructure.llm.litellm_client.litellm.acompletion",
            new=AsyncMock(return_value=mock_response),
        ):
            result = await client.complete("Query")

        assert result == ""
