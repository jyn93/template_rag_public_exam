"""Streamlit study progress dashboard — session history and statistics."""

from __future__ import annotations

from collections import defaultdict
from typing import cast

import streamlit as st

from src.frontend.api_client import call_history_list_api

st.set_page_config(
    page_title="Historial de Progreso — Oposiciones",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Historial de Progreso")
st.caption("Track your study sessions and monitor improvement over time.")

# ── Filters sidebar ───────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Filters")
    filter_type = st.selectbox(
        "Session type",
        options=["All", "exam", "chat"],
        index=0,
    )
    filter_subject = st.text_input(
        "Subject filter",
        value="",
        placeholder="e.g. Administrative Law",
    )
    limit = st.slider("Max sessions to load", min_value=10, max_value=200, value=50)
    if st.button("🔄 Refresh", use_container_width=True):
        st.rerun()

# ── Load data ─────────────────────────────────────────────────────────────────

sessions, error = call_history_list_api(
    limit=limit,
    subject=filter_subject.strip() or None,
    session_type=filter_type if filter_type != "All" else None,
)

if error:
    st.error(f"❌ Could not load history: {error}")
    st.stop()

if not sessions:
    st.info("No study sessions recorded yet. Complete an exam or chat session first.")
    st.stop()

# ── Summary metrics ───────────────────────────────────────────────────────────

exam_sessions = [s for s in sessions if s.get("session_type") == "exam"]
chat_sessions = [s for s in sessions if s.get("session_type") == "chat"]

scores = [
    float(s["score_avg"])  # type: ignore[arg-type]
    for s in exam_sessions
    if s.get("score_avg") is not None
]
percentages = [
    int(s["percentage"])  # type: ignore[arg-type]
    for s in exam_sessions
    if s.get("percentage") is not None
]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total sessions", len(sessions))
col2.metric("Exam sessions", len(exam_sessions))
col3.metric(
    "Avg score",
    f"{round(sum(scores) / len(scores), 1)}/10" if scores else "—",
)
col4.metric(
    "Avg percentage",
    f"{round(sum(percentages) / len(percentages))}%" if percentages else "—",
)

st.divider()

# ── Score trend chart ─────────────────────────────────────────────────────────

if exam_sessions:
    st.subheader("Score trend (exam sessions)")
    chart_data = [
        {
            "date": str(s.get("created_at", ""))[:10],
            "score": float(s["score_avg"])  # type: ignore[arg-type]
            if s.get("score_avg") is not None
            else 0.0,
            "percentage": int(s["percentage"])  # type: ignore[arg-type]
            if s.get("percentage") is not None
            else 0,
        }
        for s in reversed(exam_sessions)  # oldest first for chart
    ]
    if chart_data:
        import pandas as pd

        df = pd.DataFrame(chart_data)
        st.line_chart(df.set_index("date")[["score", "percentage"]])

# ── Per-subject breakdown ─────────────────────────────────────────────────────

st.divider()
st.subheader("Progress by subject")

subject_data: dict[str, list[int]] = defaultdict(list)
for s in exam_sessions:
    subj = str(s.get("subject") or "Unspecified")
    pct = s.get("percentage")
    if pct is not None:
        subject_data[subj].append(int(cast(int, pct)))

if subject_data:
    for subj, pcts in sorted(subject_data.items()):
        avg_pct = round(sum(pcts) / len(pcts))
        color = "green" if avg_pct >= 50 else "red"
        st.markdown(
            f"**{subj}** — {len(pcts)} session(s) — "
            f"avg **:{color}[{avg_pct}%]**"
        )
        st.progress(avg_pct / 100)
else:
    st.info("No exam sessions with subject data yet.")

# ── Session log ───────────────────────────────────────────────────────────────

st.divider()
st.subheader("Session log")

for s in sessions:
    s_type = str(s.get("session_type", ""))
    topic = str(s.get("topic") or "—")
    subject = str(s.get("subject") or "—")
    created = str(s.get("created_at", ""))[:16].replace("T", " ")
    icon = "📝" if s_type == "exam" else "💬"

    with st.expander(f"{icon} {s_type.capitalize()} — {topic[:60]} — {created}"):
        col_a, col_b = st.columns(2)
        col_a.markdown(f"**Subject:** {subject}")
        col_a.markdown(f"**Topic:** {topic}")
        if s_type == "exam":
            score = s.get("score_avg")
            pct = s.get("percentage")
            n_q = s.get("num_questions")
            n_c = s.get("num_correct")
            col_b.markdown(
                f"**Score:** {score}/10" if score is not None else "**Score:** —"
            )
            col_b.markdown(
                f"**Result:** {n_c}/{n_q} correct ({pct}%)"
                if n_q is not None
                else "**Result:** —"
            )
