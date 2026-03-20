"""Unit tests for RAGGenerator and Generator base class."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.core.exceptions import GenerationError
from src.core.generation.base import GenerationInput, GenerationOutput
from src.core.generation.rag_generator import RAGGenerator

# ── Helpers ───────────────────────────────────────────────────────────────────


def make_input(
    query: str = "What is an administrative appeal?",
    context: list[str] | None = None,
    metadata: dict[str, str] | None = None,
) -> GenerationInput:
    return GenerationInput(
        query=query,
        context=context if context is not None else ["An appeal is a legal remedy."],
        metadata=metadata or {"subject": "Administrative Law"},
    )


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_llm() -> AsyncMock:
    """LLM client mock that returns a fixed answer."""
    llm = AsyncMock()
    llm.complete = AsyncMock(
        return_value="Administrative appeal is a formal challenge."
    )
    return llm


@pytest.fixture
def generator(mock_llm) -> RAGGenerator:
    """RAGGenerator with a mocked LLM client."""
    return RAGGenerator(llm_client=mock_llm)


# ── TestGenerationInput ───────────────────────────────────────────────────────


class TestGenerationInput:
    """Tests for GenerationInput dataclass."""

    def test_default_metadata_is_empty_dict(self):
        """metadata defaults to an empty dict, not a shared mutable default."""
        a = GenerationInput(query="q", context=["c"])
        b = GenerationInput(query="q", context=["c"])
        a.metadata["key"] = "value"
        assert "key" not in b.metadata

    def test_stores_query_and_context(self):
        """query and context are stored correctly."""
        inp = GenerationInput(query="my query", context=["chunk1", "chunk2"])
        assert inp.query == "my query"
        assert inp.context == ["chunk1", "chunk2"]


# ── TestGenerationOutput ──────────────────────────────────────────────────────


class TestGenerationOutput:
    """Tests for GenerationOutput dataclass."""

    def test_default_tokens_used_is_zero(self):
        """tokens_used defaults to 0."""
        out = GenerationOutput(content="answer", sources=["source"])
        assert out.tokens_used == 0

    def test_accepts_dict_content(self):
        """content can be a dict for structured outputs."""
        out = GenerationOutput(content={"questions": []}, sources=[])
        assert isinstance(out.content, dict)


# ── TestGeneratorValidate ─────────────────────────────────────────────────────


class TestGeneratorValidate:
    """Tests for Generator._validate (via RAGGenerator)."""

    async def test_raises_value_error_for_empty_context(self, generator):
        """ValueError raised when context list is empty."""
        inp = GenerationInput(query="test", context=[])
        with pytest.raises(ValueError, match="context"):
            await generator.generate(inp)

    async def test_does_not_raise_for_single_chunk(self, generator):
        """Single-chunk context passes validation."""
        inp = make_input(context=["Only chunk."])
        result = await generator.generate(inp)
        assert result is not None


# ── TestRAGGeneratorBuildPrompt ───────────────────────────────────────────────


class TestRAGGeneratorBuildPrompt:
    """Tests for RAGGenerator._build_prompt."""

    def test_prompt_contains_query(self, generator):
        """The assembled prompt includes the query."""
        inp = make_input(query="What is habeas corpus?")
        prompt = generator._build_prompt(inp)
        assert "What is habeas corpus?" in prompt

    def test_prompt_contains_all_context_chunks(self, generator):
        """All context chunks appear in the assembled prompt."""
        inp = make_input(context=["First chunk.", "Second chunk."])
        prompt = generator._build_prompt(inp)
        assert "First chunk." in prompt
        assert "Second chunk." in prompt

    def test_chunks_are_numbered(self, generator):
        """Chunks are prefixed with a [1], [2], … marker."""
        inp = make_input(context=["Alpha.", "Beta."])
        prompt = generator._build_prompt(inp)
        assert "[1]" in prompt
        assert "[2]" in prompt


# ── TestRAGGeneratorCallLLM ───────────────────────────────────────────────────


class TestRAGGeneratorCallLLM:
    """Tests for RAGGenerator._call_llm."""

    async def test_passes_system_prompt_to_llm(self, generator, mock_llm):
        """The RAG system prompt is forwarded to the LLM client."""
        await generator._call_llm("some prompt")
        call_kwargs = mock_llm.complete.call_args.kwargs
        assert call_kwargs["system_prompt"] is not None
        assert len(call_kwargs["system_prompt"]) > 0

    async def test_returns_llm_response_text(self, generator, mock_llm):
        """_call_llm returns the raw text from the LLM client."""
        mock_llm.complete.return_value = "raw answer"
        result = await generator._call_llm("prompt")
        assert result == "raw answer"

    async def test_wraps_unexpected_exception(self, generator, mock_llm):
        """Non-GenerationError exceptions are wrapped in GenerationError."""
        mock_llm.complete.side_effect = ValueError("bad input")
        with pytest.raises(GenerationError, match="RAG generation failed"):
            await generator._call_llm("prompt")

    async def test_propagates_generation_error_unchanged(self, generator, mock_llm):
        """GenerationError from the LLM client is re-raised as-is."""
        original = GenerationError("llm down")
        mock_llm.complete.side_effect = original
        with pytest.raises(GenerationError) as exc_info:
            await generator._call_llm("prompt")
        assert exc_info.value is original


# ── TestRAGGeneratorParseResponse ─────────────────────────────────────────────


class TestRAGGeneratorParseResponse:
    """Tests for RAGGenerator._parse_response."""

    def test_content_is_raw_llm_text(self, generator):
        """content equals the raw model response."""
        inp = make_input(context=["chunk"])
        out = generator._parse_response("The answer is X.", inp)
        assert out.content == "The answer is X."

    def test_sources_are_context_previews(self, generator):
        """Sources contain previews of the context chunks."""
        long_chunk = "A" * 300
        inp = make_input(context=[long_chunk])
        out = generator._parse_response("answer", inp)
        assert len(out.sources) == 1
        assert len(out.sources[0]) <= 200

    def test_sources_count_matches_context_count(self, generator):
        """Number of sources equals number of context chunks."""
        inp = make_input(context=["chunk1", "chunk2", "chunk3"])
        out = generator._parse_response("answer", inp)
        assert len(out.sources) == 3


# ── TestRAGGeneratorGenerate ──────────────────────────────────────────────────


class TestRAGGeneratorGenerate:
    """End-to-end tests for RAGGenerator.generate."""

    async def test_returns_generation_output(self, generator):
        """generate() returns a GenerationOutput instance."""
        result = await generator.generate(make_input())
        assert isinstance(result, GenerationOutput)

    async def test_content_comes_from_llm(self, generator, mock_llm):
        """The output content is the LLM's response."""
        mock_llm.complete.return_value = "My final answer."
        result = await generator.generate(make_input())
        assert result.content == "My final answer."

    async def test_llm_called_exactly_once(self, generator, mock_llm):
        """The LLM is called once per generate() invocation."""
        await generator.generate(make_input())
        mock_llm.complete.assert_called_once()

    async def test_multiple_context_chunks_used(self, generator, mock_llm):
        """All context chunks are passed to the prompt."""
        chunks = ["Chunk A.", "Chunk B.", "Chunk C."]
        await generator.generate(make_input(context=chunks))
        prompt_arg = mock_llm.complete.call_args.args[0]
        for chunk in chunks:
            assert chunk in prompt_arg
