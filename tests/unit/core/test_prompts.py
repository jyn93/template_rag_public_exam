"""Unit tests for prompt templates."""

from __future__ import annotations

from src.core.config.prompts import (
    ANSWER_EVALUATION_PROMPT,
    EXAM_GENERATION_PROMPT,
    RAG_SYSTEM_PROMPT,
    RAG_USER_PROMPT_TEMPLATE,
)


class TestPromptTemplates:
    """Tests for LLM prompt template constants."""

    def test_rag_system_prompt_is_non_empty(self) -> None:
        """Verify RAG_SYSTEM_PROMPT is a non-empty string."""
        assert isinstance(RAG_SYSTEM_PROMPT, str)
        assert len(RAG_SYSTEM_PROMPT) > 0

    def test_rag_user_prompt_template_has_placeholders(self) -> None:
        """Verify RAG_USER_PROMPT_TEMPLATE contains required placeholders."""
        assert "{context}" in RAG_USER_PROMPT_TEMPLATE
        assert "{question}" in RAG_USER_PROMPT_TEMPLATE

    def test_exam_generation_prompt_has_placeholders(self) -> None:
        """Verify EXAM_GENERATION_PROMPT contains required placeholders."""
        assert "{num_questions}" in EXAM_GENERATION_PROMPT
        assert "{exam_type}" in EXAM_GENERATION_PROMPT
        assert "{difficulty}" in EXAM_GENERATION_PROMPT
        assert "{context}" in EXAM_GENERATION_PROMPT

    def test_answer_evaluation_prompt_has_placeholders(self) -> None:
        """Verify ANSWER_EVALUATION_PROMPT contains required placeholders."""
        assert "{question}" in ANSWER_EVALUATION_PROMPT
        assert "{correct_answer}" in ANSWER_EVALUATION_PROMPT
        assert "{student_answer}" in ANSWER_EVALUATION_PROMPT
