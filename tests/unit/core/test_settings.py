"""Unit tests for application settings."""

from __future__ import annotations

import pytest

from src.core.config.settings import LLMProvider, RetrieverType, Settings


class TestSettings:
    """Tests for Settings configuration model."""

    def test_default_values(self) -> None:
        """Verify Settings has sensible defaults without an .env file."""
        settings = Settings(
            _env_file=None,  # type: ignore[call-arg]
        )
        assert settings.app_name == "RAG Oposiciones"
        assert settings.debug is False
        assert settings.llm_provider == LLMProvider.GROQ
        assert settings.retriever_type == RetrieverType.HYBRID
        assert settings.top_k == 5
        assert settings.chunk_size == 512

    def test_override_via_constructor(self) -> None:
        """Verify settings can be overridden programmatically."""
        settings = Settings(
            _env_file=None,  # type: ignore[call-arg]
            debug=True,
            llm_provider="openai",
            top_k=10,
        )
        assert settings.debug is True
        assert settings.llm_provider == LLMProvider.OPENAI
        assert settings.top_k == 10

    def test_llm_provider_enum_values(self) -> None:
        """Verify LLMProvider enum contains expected values."""
        assert LLMProvider.ANTHROPIC == "anthropic"
        assert LLMProvider.OPENAI == "openai"
        assert LLMProvider.OLLAMA == "ollama"
        assert LLMProvider.GROQ == "groq"

    def test_retriever_type_enum_values(self) -> None:
        """Verify RetrieverType enum contains expected values."""
        assert RetrieverType.DENSE == "dense"
        assert RetrieverType.HYBRID == "hybrid"

    @pytest.mark.parametrize(
        "provider",
        [
            LLMProvider.ANTHROPIC,
            LLMProvider.OPENAI,
            LLMProvider.OLLAMA,
            LLMProvider.GROQ,
        ],
    )
    def test_all_providers_accepted(self, provider: LLMProvider) -> None:
        """Verify all LLM providers are accepted by Settings."""
        settings = Settings(
            _env_file=None,  # type: ignore[call-arg]
            llm_provider=provider,
        )
        assert settings.llm_provider == provider

    def test_groq_api_key_accepted_via_constructor(self) -> None:
        """Verify groq_api_key can be set programmatically."""
        settings = Settings(
            _env_file=None,  # type: ignore[call-arg]
            groq_api_key="gsk_test123",
        )
        assert settings.groq_api_key == "gsk_test123"
