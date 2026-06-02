from __future__ import annotations

import html
import os
from typing import Any

import requests
import streamlit as st

from version import APP_NAME, __version__

API_BASE_URL = os.getenv("RAG_API_BASE_URL", "http://127.0.0.1:8000")
QUERY_ENDPOINT = f"{API_BASE_URL.rstrip('/')}/api/v1/query"
CHAT_ENDPOINT = f"{API_BASE_URL.rstrip('/')}/api/v1/chat"

st.set_page_config(page_title=APP_NAME, page_icon="PI", layout="wide")


def inject_global_styles() -> None:
    st.markdown(
        """
        <style>
            :root {
                --app-bg: #f6f7fb;
                --panel: #ffffff;
                --panel-soft: #f9fafb;
                --ink: #111827;
                --muted: #6b7280;
                --line: #e5e7eb;
                --brand: #2563eb;
                --brand-dark: #1e40af;
                --brand-soft: #eff6ff;
                --success-soft: #ecfdf5;
                --success: #047857;
                --shadow: 0 18px 45px rgba(15, 23, 42, 0.08);
                --radius: 8px;
            }

            html {
                scroll-behavior: smooth;
            }

            .stApp {
                background:
                    linear-gradient(180deg, rgba(37, 99, 235, 0.08), rgba(246, 247, 251, 0) 280px),
                    var(--app-bg);
                color: var(--ink);
            }

            .block-container {
                max-width: 1120px;
                padding-top: 1.25rem;
                padding-bottom: 7rem;
            }

            h1, h2, h3, h4, p, li {
                letter-spacing: 0;
            }

            .app-header {
                background: rgba(255, 255, 255, 0.92);
                border: 1px solid rgba(229, 231, 235, 0.9);
                border-radius: var(--radius);
                padding: 1.25rem 1.35rem;
                box-shadow: var(--shadow);
                margin-bottom: 1.25rem;
            }

            .header-row {
                display: flex;
                align-items: flex-start;
                justify-content: space-between;
                gap: 1rem;
                flex-wrap: wrap;
            }

            .brand-kicker {
                color: var(--brand-dark);
                font-size: 0.78rem;
                font-weight: 700;
                text-transform: uppercase;
                margin-bottom: 0.3rem;
            }

            .brand-title {
                color: var(--ink);
                font-size: 2.35rem;
                line-height: 1.08;
                font-weight: 780;
                margin: 0;
            }

            .brand-subtitle {
                color: var(--muted);
                font-size: 0.98rem;
                max-width: 720px;
                margin-top: 0.55rem;
            }

            .pill-row {
                display: flex;
                flex-wrap: wrap;
                gap: 0.5rem;
                margin-top: 0.95rem;
            }

            .status-pill {
                border: 1px solid var(--line);
                border-radius: 999px;
                background: var(--panel-soft);
                color: #374151;
                font-size: 0.78rem;
                font-weight: 650;
                padding: 0.35rem 0.65rem;
                white-space: nowrap;
            }

            .status-pill.green {
                border-color: #bbf7d0;
                color: var(--success);
                background: var(--success-soft);
            }

            .metric-card {
                border: 1px solid var(--line);
                border-radius: var(--radius);
                padding: 0.8rem 0.9rem;
                background: var(--panel);
                min-width: 150px;
            }

            .metric-label {
                color: var(--muted);
                font-size: 0.76rem;
                font-weight: 650;
            }

            .metric-value {
                color: var(--ink);
                font-size: 0.96rem;
                font-weight: 750;
                margin-top: 0.2rem;
            }

            .welcome-grid {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 0.8rem;
                margin: 1.1rem 0 0.75rem;
            }

            .welcome-card {
                background: var(--panel);
                border: 1px solid var(--line);
                border-radius: var(--radius);
                padding: 1rem;
                min-height: 128px;
                box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
            }

            .welcome-card-title {
                color: var(--ink);
                font-weight: 750;
                margin-bottom: 0.45rem;
            }

            .welcome-card-text {
                color: var(--muted);
                font-size: 0.92rem;
                line-height: 1.5;
            }

            .section-title {
                color: var(--ink);
                font-size: 1rem;
                font-weight: 760;
                margin: 0.5rem 0 0.35rem;
            }

            .source-row {
                display: flex;
                flex-wrap: wrap;
                gap: 0.4rem;
                margin: 0.6rem 0 0.25rem;
            }

            .source-chip {
                display: inline-flex;
                align-items: center;
                border-radius: 999px;
                border: 1px solid #bfdbfe;
                background: var(--brand-soft);
                color: #1d4ed8;
                font-size: 0.78rem;
                font-weight: 650;
                padding: 0.28rem 0.55rem;
            }

            .context-card {
                border: 1px solid var(--line);
                border-radius: var(--radius);
                background: #ffffff;
                padding: 0.85rem 0.95rem;
                margin: 0.55rem 0;
            }

            .context-meta {
                color: var(--muted);
                font-size: 0.78rem;
                font-weight: 650;
                margin-bottom: 0.5rem;
            }

            .context-title {
                color: var(--ink);
                font-size: 0.94rem;
                font-weight: 760;
                margin-bottom: 0.2rem;
            }

            .timing-row {
                display: flex;
                flex-wrap: wrap;
                gap: 0.45rem;
                margin: 0.65rem 0 0.1rem;
            }

            .timing-pill {
                border: 1px solid var(--line);
                border-radius: 999px;
                color: var(--muted);
                background: #ffffff;
                font-size: 0.76rem;
                padding: 0.26rem 0.55rem;
            }

            section[data-testid="stSidebar"] {
                background: #ffffff;
                border-right: 1px solid var(--line);
            }

            section[data-testid="stSidebar"] .block-container {
                padding-top: 1.25rem;
                padding-bottom: 1.25rem;
            }

            div[data-testid="stSidebarUserContent"] {
                color: var(--ink);
            }

            div[data-testid="stChatMessage"] {
                border-radius: var(--radius);
                border: 1px solid rgba(229, 231, 235, 0.9);
                background: rgba(255, 255, 255, 0.92);
                box-shadow: 0 10px 24px rgba(15, 23, 42, 0.04);
                padding: 0.85rem 1rem;
                margin: 0.65rem 0;
            }

            div[data-testid="stChatMessage"] p {
                line-height: 1.62;
            }

            div[data-testid="stChatInput"] {
                background: rgba(255, 255, 255, 0.95);
                border-top: 1px solid rgba(229, 231, 235, 0.8);
                padding-top: 0.65rem;
            }

            div[data-testid="stChatInput"] textarea {
                border-radius: 8px;
                border-color: #cbd5e1;
                box-shadow: 0 12px 28px rgba(15, 23, 42, 0.08);
            }

            .stButton > button {
                border-radius: 8px;
                font-weight: 700;
            }

            .stTextArea textarea {
                border-radius: 8px;
                border-color: #cbd5e1;
                background: #ffffff;
            }

            .streamlit-expanderHeader {
                font-weight: 700;
            }

            @media (max-width: 760px) {
                .block-container {
                    padding-left: 1rem;
                    padding-right: 1rem;
                    padding-bottom: 6rem;
                }

                .app-header {
                    padding: 1rem;
                }

                .brand-title {
                    font-size: 1.7rem;
                }

                .welcome-grid {
                    grid-template-columns: 1fr;
                }

                .metric-card {
                    width: 100%;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(mode: str) -> None:
    safe_api_url = html.escape(API_BASE_URL)
    safe_mode = html.escape(mode)
    st.markdown(
        f"""
        <div class="app-header">
            <div class="header-row">
                <div>
                    <div class="brand-kicker">Placement Intelligence</div>
                    <h1 class="brand-title">{html.escape(APP_NAME)}</h1>
                    <div class="brand-subtitle">
                        Ask grounded questions about eligibility, packages, interview rounds,
                        hiring patterns, and preparation using the placement RAG corpus.
                    </div>
                    <div class="pill-row">
                        <span class="status-pill green">Grounded retrieval enabled</span>
                        <span class="status-pill">Mode: {safe_mode}</span>
                        <span class="status-pill">Version {html.escape(__version__)}</span>
                        <span class="status-pill">API {safe_api_url}</span>
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Session</div>
                    <div class="metric-value">{len(st.session_state.messages)} messages</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_welcome() -> None:
    st.markdown(
        """
        <div class="welcome-grid">
            <div class="welcome-card">
                <div class="welcome-card-title">Eligibility checks</div>
                <div class="welcome-card-text">
                    Compare CGPA, backlog, bond, package, and role constraints across companies.
                </div>
            </div>
            <div class="welcome-card">
                <div class="welcome-card-title">Interview prep</div>
                <div class="welcome-card-text">
                    Ask about rounds, focus areas, technical topics, and company-specific tips.
                </div>
            </div>
            <div class="welcome-card">
                <div class="welcome-card-title">Grounded answers</div>
                <div class="welcome-card-text">
                    Responses include source citations and retrieved context when enabled.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sources(sources: list[str]) -> None:
    if not sources:
        return
    chips = "".join(f'<span class="source-chip">{html.escape(source)}</span>' for source in sources)
    st.markdown(f'<div class="source-row">{chips}</div>', unsafe_allow_html=True)


def render_timing(data: dict[str, Any]) -> None:
    model = data.get("model")
    retrieval_time = data.get("retrieval_time_ms")
    generation_time = data.get("generation_time_ms")
    if model is None and retrieval_time is None and generation_time is None:
        return

    pills: list[str] = []
    if model:
        pills.append(f'<span class="timing-pill">Model: {html.escape(str(model))}</span>')
    if retrieval_time is not None:
        pills.append(f'<span class="timing-pill">Retrieval: {retrieval_time} ms</span>')
    if generation_time is not None:
        pills.append(f'<span class="timing-pill">Generation: {generation_time} ms</span>')
    st.markdown(f'<div class="timing-row">{"".join(pills)}</div>', unsafe_allow_html=True)


inject_global_styles()

if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.markdown("### Controls")
    mode = st.radio("Mode", ["Chat", "Single query"], horizontal=True, label_visibility="collapsed")
    st.markdown('<div class="section-title">Retrieval</div>', unsafe_allow_html=True)
    top_k = st.slider("Retrieved contexts", min_value=1, max_value=10, value=5)
    show_contexts = st.toggle("Show retrieved contexts", value=True)
    st.markdown('<div class="section-title">Session</div>', unsafe_allow_html=True)
    active_session = st.session_state.session_id or "New conversation"
    st.caption(f"Session: {active_session}")
    st.caption(f"Messages: {len(st.session_state.messages)}")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.session_id = None
        st.session_state.messages = []
        st.rerun()

render_header(mode)


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
            page_text = f" page {page}" if page else ""
            st.markdown(
                f"""
                <div class="context-card">
                    <div class="context-title">{index}. {html.escape(str(source_file))}{html.escape(page_text)}</div>
                    <div class="context-meta">Score {score:.4f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(context.get("content", ""))


if mode == "Chat":
    if not st.session_state.messages:
        render_welcome()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_sources(message.get("sources", []))
                render_timing(message.get("metrics", {}))
                render_contexts(message.get("contexts", []))

    prompt = st.chat_input("Ask about companies, eligibility, rounds, packages, or preparation")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        payload: dict[str, Any] = {"query": prompt}
        if st.session_state.session_id:
            payload["session_id"] = st.session_state.session_id

        with st.chat_message("assistant"):
            try:
                with st.spinner("Retrieving sources and drafting a grounded answer..."):
                    data = post_json(CHAT_ENDPOINT, payload)
                st.session_state.session_id = data.get("session_id") or st.session_state.session_id

                answer = data.get("answer", "")
                st.markdown(answer)
                sources = data.get("sources", [])
                render_sources(sources)

                with st.spinner("Preparing retrieved context..."):
                    context_data = post_json(QUERY_ENDPOINT, {"query": prompt, "top_k": top_k, "modality": "text"})
                contexts = context_data.get("contexts", [])
                render_timing(context_data)
                render_contexts(contexts)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "metrics": context_data,
                        "contexts": contexts,
                    }
                )
            except requests.RequestException as exc:
                st.error(f"API request failed: {exc}")
else:
    st.markdown('<div class="section-title">Single query</div>', unsafe_allow_html=True)
    query = st.text_area(
        "Query",
        placeholder="What is Amazon eligibility criteria?",
        height=140,
        label_visibility="collapsed",
    )
    if st.button("Run query", type="primary", use_container_width=True) and query.strip():
        try:
            with st.spinner("Retrieving sources and generating an answer..."):
                data = post_json(QUERY_ENDPOINT, {"query": query.strip(), "top_k": top_k, "modality": "text"})
            st.markdown('<div class="section-title">Answer</div>', unsafe_allow_html=True)
            st.markdown(data.get("answer", ""))
            sources = data.get("sources", [])
            render_sources(sources)
            render_timing(data)
            render_contexts(data.get("contexts", []))
        except requests.RequestException as exc:
            st.error(f"API request failed: {exc}")
