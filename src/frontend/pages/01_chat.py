"""Streamlit chat page — RAG-grounded question answering."""

from __future__ import annotations

import streamlit as st

from src.frontend.api_client import call_chat_api

_DEFAULT_TOP_K = 5

st.set_page_config(page_title="Chat RAG — Oposiciones", page_icon="💬", layout="wide")

st.title("💬 Chat RAG")
st.caption("Ask questions about your study material and get grounded answers.")

# ── Session state ─────────────────────────────────────────────────────────────

if "chat_history" not in st.session_state:
    st.session_state.chat_history: list[dict[str, object]] = []

# ── Sidebar controls ──────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Settings")
    subject = st.text_input(
        "Subject filter",
        value="",
        placeholder="e.g. Administrative Law",
        help="Optional: filter retrieved context to a specific subject.",
    )
    top_k = st.slider(
        "Context chunks (top-k)",
        min_value=1,
        max_value=20,
        value=_DEFAULT_TOP_K,
        help="Number of document chunks retrieved per query.",
    )
    if st.button("🗑️ Clear history", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# ── Chat history display ──────────────────────────────────────────────────────

for message in st.session_state.chat_history:
    with st.chat_message(str(message["role"])):
        st.markdown(str(message["content"]))
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📎 Sources", expanded=False):
                for i, src in enumerate(list(message["sources"]), 1):  # type: ignore[arg-type]
                    st.markdown(f"**[{i}]** {src}")

# ── Chat input ────────────────────────────────────────────────────────────────

if query := st.chat_input("Ask a question about the study material…"):
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching and generating answer…"):
            answer, sources, error = call_chat_api(query, subject, top_k)

        if error:
            st.error(f"❌ {error}")
            st.session_state.chat_history.append(
                {"role": "assistant", "content": f"Error: {error}", "sources": []}
            )
        else:
            st.markdown(answer)
            if sources:
                with st.expander("📎 Sources", expanded=False):
                    for i, src in enumerate(sources, 1):
                        st.markdown(f"**[{i}]** {src}")
            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
