from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st

from version import APP_NAME, __version__

API_BASE_URL = os.getenv("RAG_API_BASE_URL", "http://127.0.0.1:8000")
QUERY_ENDPOINT = f"{API_BASE_URL.rstrip('/')}/api/v1/query"
CHAT_ENDPOINT = f"{API_BASE_URL.rstrip('/')}/api/v1/chat"

st.set_page_config(page_title=APP_NAME, page_icon="PI", layout="wide")

if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title(APP_NAME)
st.caption(f"Version {__version__} | API: {API_BASE_URL}")

with st.sidebar:
    mode = st.radio("Mode", ["Chat", "Single query"], horizontal=True)
    top_k = st.slider("Retrieved contexts", min_value=1, max_value=10, value=5)
    show_contexts = st.toggle("Show retrieved contexts", value=True)
    if st.button("Clear conversation"):
        st.session_state.session_id = None
        st.session_state.messages = []
        st.rerun()


def post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()


def render_contexts(contexts: list[dict[str, Any]]) -> None:
    if not show_contexts or not contexts:
        return
    with st.expander("Retrieved context", expanded=False):
        for index, context in enumerate(contexts, start=1):
            metadata = context.get("metadata", {})
            extra = metadata.get("extra", {})
            source_file = extra.get("source_file") or metadata.get("source") or "Unknown source"
            page = metadata.get("page")
            score = context.get("score", 0.0)
            st.markdown(f"**{index}. {source_file}**" + (f" page {page}" if page else ""))
            st.caption(f"Score: {score:.4f}")
            st.write(context.get("content", ""))


if mode == "Chat":
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant":
                render_contexts(message.get("contexts", []))

    prompt = st.chat_input("Ask about companies, eligibility, rounds, packages, or preparation")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        payload: dict[str, Any] = {"query": prompt}
        if st.session_state.session_id:
            payload["session_id"] = st.session_state.session_id

        with st.chat_message("assistant"):
            try:
                data = post_json(CHAT_ENDPOINT, payload)
                st.session_state.session_id = data.get("session_id") or st.session_state.session_id

                answer = data.get("answer", "")
                st.write(answer)
                sources = data.get("sources", [])
                if sources:
                    st.caption("Sources: " + ", ".join(sources))

                context_data = post_json(QUERY_ENDPOINT, {"query": prompt, "top_k": top_k, "modality": "text"})
                contexts = context_data.get("contexts", [])
                render_contexts(contexts)
                st.session_state.messages.append({"role": "assistant", "content": answer, "contexts": contexts})
            except requests.RequestException as exc:
                st.error(f"API request failed: {exc}")
else:
    query = st.text_area("Query", placeholder="What is Amazon eligibility criteria?", height=120)
    if st.button("Run query", type="primary") and query.strip():
        try:
            data = post_json(QUERY_ENDPOINT, {"query": query.strip(), "top_k": top_k, "modality": "text"})
            st.subheader("Answer")
            st.write(data.get("answer", ""))
            sources = data.get("sources", [])
            if sources:
                st.caption("Sources: " + ", ".join(sources))
            st.caption(
                f"Model: {data.get('model')} | Retrieval: {data.get('retrieval_time_ms', 0)} ms | "
                f"Generation: {data.get('generation_time_ms', 0)} ms"
            )
            render_contexts(data.get("contexts", []))
        except requests.RequestException as exc:
            st.error(f"API request failed: {exc}")
