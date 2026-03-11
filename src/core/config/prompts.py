"""Centralized prompt templates for all LLM interactions."""

from __future__ import annotations

__all__ = [
    "RAG_SYSTEM_PROMPT",
    "RAG_USER_PROMPT_TEMPLATE",
    "EXAM_GENERATION_PROMPT",
    "ANSWER_EVALUATION_PROMPT",
]

RAG_SYSTEM_PROMPT = """You are an expert assistant specializing in Spanish public
administration exams (oposiciones). Your role is to help candidates prepare by
answering questions based exclusively on the provided study material.

Rules:
- Answer only using the provided context.
- If the context does not contain the answer, say so explicitly.
- Be precise and cite relevant sections when possible.
- Use clear, formal Spanish suitable for administrative exams.
"""

RAG_USER_PROMPT_TEMPLATE = """Context:
{context}

Question: {question}

Answer based only on the context above:"""

EXAM_GENERATION_PROMPT = """Generate a {num_questions}-question {exam_type} exam
based on the following study material. Difficulty: {difficulty}.

Study material:
{context}

Return a valid JSON object with this structure:
{{
  "questions": [
    {{
      "id": 1,
      "type": "{exam_type}",
      "question": "...",
      "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
      "correct_answer": "A",
      "explanation": "..."
    }}
  ]
}}"""

ANSWER_EVALUATION_PROMPT = """Evaluate the following student answer for an
oposiciones exam question. Be rigorous but constructive.

Question: {question}
Correct answer: {correct_answer}
Student answer: {student_answer}

Return a JSON object:
{{
  "score": <0-10>,
  "is_correct": <true/false>,
  "feedback": "...",
  "missing_points": ["..."],
  "strengths": ["..."]
}}"""
