"""Unit tests for ExamGenerator."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from src.core.exceptions import GenerationError
from src.core.generation.base import GenerationInput, GenerationOutput
from src.core.generation.exam_generator import (
    _DEFAULT_DIFFICULTY,
    _DEFAULT_EXAM_TYPE,
    _DEFAULT_NUM_QUESTIONS,
    ExamGenerator,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

VALID_EXAM_JSON = json.dumps(
    {
        "questions": [
            {
                "id": 1,
                "type": "test",
                "question": "What is an appeal?",
                "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
                "correct_answer": "A",
                "explanation": "An appeal is...",
            }
        ]
    }
)


def make_input(
    query: str = "Administrative appeal procedure",
    context: list[str] | None = None,
    metadata: dict[str, str] | None = None,
) -> GenerationInput:
    return GenerationInput(
        query=query,
        context=context
        if context is not None
        else ["An appeal is filed within 30 days."],
        metadata=metadata or {"subject": "Administrative Law"},
    )


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_llm() -> AsyncMock:
    """LLM client mock returning valid exam JSON."""
    llm = AsyncMock()
    llm.complete = AsyncMock(return_value=VALID_EXAM_JSON)
    return llm


@pytest.fixture
def generator(mock_llm) -> ExamGenerator:
    """ExamGenerator with default parameters and mocked LLM."""
    return ExamGenerator(llm_client=mock_llm)


# ── TestExamGeneratorInit ─────────────────────────────────────────────────────


class TestExamGeneratorInit:
    """Tests for ExamGenerator construction."""

    def test_default_num_questions(self, mock_llm):
        """Default num_questions matches constant."""
        g = ExamGenerator(llm_client=mock_llm)
        assert g._num_questions == _DEFAULT_NUM_QUESTIONS

    def test_default_exam_type(self, mock_llm):
        """Default exam_type matches constant."""
        g = ExamGenerator(llm_client=mock_llm)
        assert g._exam_type == _DEFAULT_EXAM_TYPE

    def test_default_difficulty(self, mock_llm):
        """Default difficulty matches constant."""
        g = ExamGenerator(llm_client=mock_llm)
        assert g._difficulty == _DEFAULT_DIFFICULTY

    def test_custom_params_stored(self, mock_llm):
        """Custom parameters are stored correctly."""
        g = ExamGenerator(
            llm_client=mock_llm,
            num_questions=3,
            exam_type="desarrollo",
            difficulty="dificil",
        )
        assert g._num_questions == 3
        assert g._exam_type == "desarrollo"
        assert g._difficulty == "dificil"


# ── TestBuildPrompt ───────────────────────────────────────────────────────────


class TestBuildPrompt:
    """Tests for ExamGenerator._build_prompt."""

    def test_prompt_contains_num_questions(self, generator):
        """num_questions appears in the assembled prompt."""
        inp = make_input()
        prompt = generator._build_prompt(inp)
        assert str(generator._num_questions) in prompt

    def test_prompt_contains_exam_type(self, generator):
        """exam_type appears in the assembled prompt."""
        inp = make_input()
        prompt = generator._build_prompt(inp)
        assert generator._exam_type in prompt

    def test_prompt_contains_difficulty(self, generator):
        """difficulty appears in the assembled prompt."""
        inp = make_input()
        prompt = generator._build_prompt(inp)
        assert generator._difficulty in prompt

    def test_prompt_contains_all_context_chunks(self, generator):
        """All context chunks are included in the prompt."""
        inp = make_input(context=["First chunk.", "Second chunk."])
        prompt = generator._build_prompt(inp)
        assert "First chunk." in prompt
        assert "Second chunk." in prompt

    def test_chunks_separated_by_divider(self, generator):
        """Context chunks are separated with the divider string."""
        inp = make_input(context=["Alpha.", "Beta."])
        prompt = generator._build_prompt(inp)
        assert "---" in prompt


# ── TestCallLLM ───────────────────────────────────────────────────────────────


class TestCallLLM:
    """Tests for ExamGenerator._call_llm."""

    async def test_returns_llm_response(self, generator, mock_llm):
        """_call_llm returns the raw text from the LLM."""
        mock_llm.complete.return_value = "raw json string"
        result = await generator._call_llm("prompt")
        assert result == "raw json string"

    async def test_calls_llm_once(self, generator, mock_llm):
        """LLM is called exactly once per _call_llm invocation."""
        await generator._call_llm("prompt")
        mock_llm.complete.assert_called_once()

    async def test_wraps_unexpected_exception(self, generator, mock_llm):
        """Unexpected exceptions are wrapped in GenerationError."""
        mock_llm.complete.side_effect = ConnectionError("network")
        with pytest.raises(GenerationError, match="Exam generation LLM call failed"):
            await generator._call_llm("prompt")

    async def test_propagates_generation_error(self, generator, mock_llm):
        """GenerationError from the LLM is re-raised as-is."""
        original = GenerationError("llm error")
        mock_llm.complete.side_effect = original
        with pytest.raises(GenerationError) as exc_info:
            await generator._call_llm("prompt")
        assert exc_info.value is original

    async def test_error_chains_cause(self, generator, mock_llm):
        """Original exception is chained on the wrapped GenerationError."""
        cause = RuntimeError("timeout")
        mock_llm.complete.side_effect = cause
        with pytest.raises(GenerationError) as exc_info:
            await generator._call_llm("prompt")
        assert exc_info.value.__cause__ is cause


# ── TestExtractJson ───────────────────────────────────────────────────────────


class TestExtractJson:
    """Tests for ExamGenerator._extract_json (static helper)."""

    def test_parses_clean_json(self):
        """Direct JSON string is parsed correctly."""
        raw = '{"questions": [{"id": 1}]}'
        result = ExamGenerator._extract_json(raw)
        assert result == {"questions": [{"id": 1}]}

    def test_extracts_json_from_markdown_prose(self):
        """JSON embedded in prose (markdown fence) is extracted via regex."""
        raw = 'Here is the exam:\n```json\n{"questions": []}\n```'
        result = ExamGenerator._extract_json(raw)
        assert result == {"questions": []}

    def test_returns_empty_questions_on_failure(self):
        """Completely unparseable input returns safe fallback."""
        result = ExamGenerator._extract_json("No JSON here at all.")
        assert result == {"questions": []}

    def test_returns_empty_questions_on_malformed_json(self):
        """Malformed JSON in a brace block returns safe fallback."""
        result = ExamGenerator._extract_json("{not valid json}")
        assert result == {"questions": []}

    def test_extracts_nested_json(self):
        """Nested JSON structures are parsed correctly."""
        data = {"questions": [{"id": 1, "options": {"A": "opt1"}}]}
        result = ExamGenerator._extract_json(json.dumps(data))
        assert result == data


# ── TestParseResponse ─────────────────────────────────────────────────────────


class TestParseResponse:
    """Tests for ExamGenerator._parse_response."""

    def test_content_is_dict(self, generator):
        """content is a dict (JSON-parsed exam)."""
        inp = make_input()
        out = generator._parse_response(VALID_EXAM_JSON, inp)
        assert isinstance(out.content, dict)

    def test_content_has_questions_key(self, generator):
        """Parsed content has a 'questions' key."""
        inp = make_input()
        out = generator._parse_response(VALID_EXAM_JSON, inp)
        assert "questions" in out.content

    def test_sources_capped_at_max(self, generator):
        """Sources are capped at _MAX_SOURCES (3) chunks."""
        chunks = ["c1", "c2", "c3", "c4", "c5"]
        inp = make_input(context=chunks)
        out = generator._parse_response(VALID_EXAM_JSON, inp)
        assert len(out.sources) == 3

    def test_sources_come_from_context(self, generator):
        """Sources are the first chunks from the input context."""
        inp = make_input(context=["first", "second", "third"])
        out = generator._parse_response(VALID_EXAM_JSON, inp)
        assert out.sources == ["first", "second", "third"]

    def test_fallback_on_invalid_json(self, generator):
        """Invalid JSON returns fallback without raising."""
        inp = make_input()
        out = generator._parse_response("not json at all", inp)
        assert out.content == {"questions": []}


# ── TestGenerate ──────────────────────────────────────────────────────────────


class TestGenerate:
    """End-to-end tests for ExamGenerator.generate."""

    async def test_returns_generation_output(self, generator):
        """generate() returns a GenerationOutput instance."""
        result = await generator.generate(make_input())
        assert isinstance(result, GenerationOutput)

    async def test_content_is_dict_with_questions(self, generator):
        """Output content is a dict containing a questions list."""
        result = await generator.generate(make_input())
        assert isinstance(result.content, dict)
        assert "questions" in result.content

    async def test_raises_value_error_for_empty_context(self, generator):
        """ValueError raised when context is empty (from base class)."""
        with pytest.raises(ValueError, match="context"):
            await generator.generate(make_input(context=[]))

    async def test_llm_called_once_per_generate(self, generator, mock_llm):
        """LLM is called exactly once per generate() call."""
        await generator.generate(make_input())
        mock_llm.complete.assert_called_once()

    async def test_handles_llm_returning_prose_with_json(self, generator, mock_llm):
        """JSON embedded in LLM prose output is parsed correctly."""
        mock_llm.complete.return_value = (
            f"Sure! Here is the exam:\n{VALID_EXAM_JSON}\nHope this helps."
        )
        result = await generator.generate(make_input())
        assert "questions" in result.content

    async def test_handles_malformed_llm_response_gracefully(self, generator, mock_llm):
        """Completely malformed LLM response returns empty questions without raising."""
        mock_llm.complete.return_value = "I cannot generate an exam."
        result = await generator.generate(make_input())
        assert result.content == {"questions": []}

    @pytest.mark.parametrize(
        "exam_type,difficulty",
        [
            ("test", "facil"),
            ("desarrollo", "media"),
            ("mixto", "dificil"),
        ],
    )
    async def test_all_exam_type_difficulty_combinations(
        self, mock_llm, exam_type, difficulty
    ):
        """All exam_type / difficulty combinations produce valid output."""
        g = ExamGenerator(
            llm_client=mock_llm,
            num_questions=2,
            exam_type=exam_type,
            difficulty=difficulty,
        )
        result = await g.generate(make_input())
        assert isinstance(result, GenerationOutput)
