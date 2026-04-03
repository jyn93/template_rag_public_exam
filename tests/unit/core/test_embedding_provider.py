"""Unit tests for EmbeddingProvider settings and _build_embedding_model factory."""

from __future__ import annotations

import pytest

from src.api.dependencies import _build_embedding_model
from src.core.config.settings import EmbeddingProvider, Settings

# ── TestEmbeddingProviderEnum ─────────────────────────────────────────────────


class TestEmbeddingProviderEnum:
    """Tests for the EmbeddingProvider StrEnum."""

    def test_openai_value(self) -> None:
        """OPENAI member maps to 'openai' string."""
        assert EmbeddingProvider.OPENAI == "openai"

    def test_ollama_value(self) -> None:
        """OLLAMA member maps to 'ollama' string."""
        assert EmbeddingProvider.OLLAMA == "ollama"

    @pytest.mark.parametrize(
        "provider", [EmbeddingProvider.OPENAI, EmbeddingProvider.OLLAMA]
    )
    def test_all_providers_accepted_by_settings(self, provider: EmbeddingProvider) -> None:  # noqa: E501
        """All EmbeddingProvider values are accepted by Settings."""
        settings = Settings(
            _env_file=None,  # type: ignore[call-arg]
            embedding_provider=provider,
        )
        assert settings.embedding_provider == provider


# ── TestSettingsEmbeddingFields ───────────────────────────────────────────────


class TestSettingsEmbeddingFields:
    """Tests for embedding-related Settings fields."""

    def test_default_provider_is_openai(self) -> None:
        """embedding_provider defaults to ollama."""
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        assert settings.embedding_provider == EmbeddingProvider.OLLAMA

    def test_default_ollama_embedding_model(self) -> None:
        """ollama_embedding_model defaults to nomic-embed-text."""
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        assert settings.ollama_embedding_model == "nomic-embed-text"

    def test_ollama_embedding_model_override(self) -> None:
        """ollama_embedding_model can be overridden via constructor."""
        settings = Settings(
            _env_file=None,  # type: ignore[call-arg]
            ollama_embedding_model="mxbai-embed-large",
        )
        assert settings.ollama_embedding_model == "mxbai-embed-large"


# ── TestBuildEmbeddingModel ───────────────────────────────────────────────────


class TestBuildEmbeddingModel:
    """Tests for the _build_embedding_model() factory function."""

    def _make_settings(self, **kwargs: object) -> Settings:
        return Settings(_env_file=None, **kwargs)  # type: ignore[call-arg]

    def test_returns_openai_embedding_when_provider_is_openai(self) -> None:
        """Returns OpenAIEmbedding when embedding_provider=openai."""
        from llama_index.embeddings.openai import OpenAIEmbedding

        settings = self._make_settings(
            embedding_provider=EmbeddingProvider.OPENAI,
            embedding_model="text-embedding-3-small",
            openai_api_key="sk-test",
        )
        model = _build_embedding_model(settings)
        assert isinstance(model, OpenAIEmbedding)

    def test_returns_ollama_embedding_when_provider_is_ollama(self) -> None:
        """Returns OllamaEmbedding when embedding_provider=ollama."""
        from llama_index.embeddings.ollama import OllamaEmbedding

        settings = self._make_settings(
            embedding_provider=EmbeddingProvider.OLLAMA,
            ollama_embedding_model="nomic-embed-text",
            ollama_base_url="http://localhost:11434",
        )
        model = _build_embedding_model(settings)
        assert isinstance(model, OllamaEmbedding)

    def test_openai_embedding_uses_configured_model(self) -> None:
        """OpenAIEmbedding is initialised with the configured model name."""
        from llama_index.embeddings.openai import OpenAIEmbedding

        settings = self._make_settings(
            embedding_provider=EmbeddingProvider.OPENAI,
            embedding_model="text-embedding-3-large",
            openai_api_key="sk-test",
        )
        model = _build_embedding_model(settings)
        assert isinstance(model, OpenAIEmbedding)
        assert model.model_name == "text-embedding-3-large"

    def test_ollama_embedding_uses_configured_model(self) -> None:
        """OllamaEmbedding is initialised with the configured model name."""
        from llama_index.embeddings.ollama import OllamaEmbedding

        settings = self._make_settings(
            embedding_provider=EmbeddingProvider.OLLAMA,
            ollama_embedding_model="mxbai-embed-large",
            ollama_base_url="http://localhost:11434",
        )
        model = _build_embedding_model(settings)
        assert isinstance(model, OllamaEmbedding)
        assert model.model_name == "mxbai-embed-large"

    def test_ollama_embedding_uses_configured_base_url(self) -> None:
        """OllamaEmbedding is initialised with the configured Ollama base URL."""
        from llama_index.embeddings.ollama import OllamaEmbedding

        settings = self._make_settings(
            embedding_provider=EmbeddingProvider.OLLAMA,
            ollama_embedding_model="nomic-embed-text",
            ollama_base_url="http://ollama-server:11434",
        )
        model = _build_embedding_model(settings)
        assert isinstance(model, OllamaEmbedding)
        assert model.base_url == "http://ollama-server:11434"
