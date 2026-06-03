from __future__ import annotations

import html
import os
import re
from datetime import datetime
from textwrap import shorten
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
                --user-bg: #2563eb;
                --user-ink: #ffffff;
                --assistant-bg: #ffffff;
                --assistant-ink: #111827;
                --success-soft: #ecfdf5;
                --success: #047857;
                --warning-soft: #fffbeb;
                --warning: #b45309;
                --danger-soft: #fef2f2;
                --danger: #b91c1c;
                --shadow: 0 18px 45px rgba(15, 23, 42, 0.08);
                --radius: 8px;
            }

            @media (prefers-color-scheme: dark) {
                :root {
                    --app-bg: #0f172a;
                    --panel: #111827;
                    --panel-soft: #1f2937;
                    --ink: #f8fafc;
                    --muted: #cbd5e1;
                    --line: #334155;
                    --brand: #60a5fa;
                    --brand-dark: #bfdbfe;
                    --brand-soft: #172554;
                    --user-bg: #2563eb;
                    --user-ink: #ffffff;
                    --assistant-bg: #111827;
                    --assistant-ink: #f8fafc;
                    --success-soft: #052e1a;
                    --success: #86efac;
                    --warning-soft: #451a03;
                    --warning: #fbbf24;
                    --danger-soft: #450a0a;
                    --danger: #fca5a5;
                    --shadow: 0 18px 45px rgba(0, 0, 0, 0.28);
                }
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

            @media (prefers-color-scheme: dark) {
                .stApp {
                    background:
                        linear-gradient(180deg, rgba(96, 165, 250, 0.12), rgba(15, 23, 42, 0) 280px),
                        var(--app-bg);
                }

                .app-header {
                    background: rgba(17, 24, 39, 0.94);
                    border-color: rgba(51, 65, 85, 0.95);
                }
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

            .welcome-hero {
                background: var(--panel);
                border: 1px solid var(--line);
                border-radius: var(--radius);
                box-shadow: var(--shadow);
                padding: 1.35rem;
                margin: 0.75rem 0 1rem;
            }

            .welcome-title {
                color: var(--ink);
                font-size: 1.55rem;
                font-weight: 780;
                margin-bottom: 0.4rem;
            }

            .welcome-subtitle {
                color: var(--muted);
                font-size: 0.98rem;
                line-height: 1.55;
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

            .source-card {
                display: inline-flex;
                flex-direction: column;
                gap: 0.16rem;
                min-width: min(220px, 100%);
                align-items: center;
                justify-content: center;
                text-decoration: none !important;
                border-radius: var(--radius);
                border: 1px solid #bfdbfe;
                background: var(--brand-soft);
                color: #1d4ed8;
                padding: 0.6rem 0.7rem;
                box-shadow: 0 8px 18px rgba(37, 99, 235, 0.08);
                transition: transform 160ms ease, border-color 160ms ease, box-shadow 160ms ease;
            }

            .source-card:hover {
                transform: translateY(-1px);
                border-color: var(--brand);
                box-shadow: 0 12px 24px rgba(37, 99, 235, 0.14);
            }

            .source-label {
                color: var(--muted);
                font-size: 0.7rem;
                font-weight: 750;
                text-transform: uppercase;
            }

            .source-name {
                color: var(--brand-dark);
                font-size: 0.82rem;
                font-weight: 750;
                line-height: 1.3;
                text-align: center;
            }

            .context-card {
                border: 1px solid var(--line);
                border-radius: var(--radius);
                background: var(--panel);
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
                background: var(--panel);
                font-size: 0.76rem;
                padding: 0.26rem 0.55rem;
            }

            .stats-grid {
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 0.55rem;
                margin: 0.65rem 0 0.35rem;
            }

            .stat-card {
                border: 1px solid var(--line);
                border-radius: var(--radius);
                background: var(--panel);
                padding: 0.7rem 0.75rem;
            }

            .stat-label {
                color: var(--muted);
                font-size: 0.72rem;
                font-weight: 720;
                text-transform: uppercase;
            }

            .stat-value {
                color: var(--ink);
                font-size: 1rem;
                font-weight: 780;
                margin-top: 0.18rem;
            }

            .confidence-badge {
                border-radius: 999px;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                min-width: 72px;
                padding: 0.28rem 0.55rem;
                font-size: 0.78rem;
                font-weight: 780;
            }

            .confidence-high {
                color: var(--success);
                background: var(--success-soft);
                border: 1px solid rgba(4, 120, 87, 0.22);
            }

            .confidence-medium {
                color: var(--warning);
                background: var(--warning-soft);
                border: 1px solid rgba(180, 83, 9, 0.22);
            }

            .confidence-low {
                color: var(--danger);
                background: var(--danger-soft);
                border: 1px solid rgba(185, 28, 28, 0.22);
            }

            .chat-row {
                display: flex;
                width: 100%;
                margin: 0.75rem 0;
            }

            .chat-row.user {
                justify-content: flex-end;
            }

            .chat-row.assistant {
                justify-content: flex-start;
            }

            .chat-bubble {
                max-width: min(760px, 82%);
                border: 1px solid var(--line);
                border-radius: 18px;
                padding: 0.85rem 0.95rem;
                box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
            }

            .chat-bubble.user {
                background: var(--user-bg);
                color: var(--user-ink);
                border-color: rgba(37, 99, 235, 0.3);
                border-bottom-right-radius: 8px;
            }

            .chat-bubble.assistant {
                background: var(--assistant-bg);
                color: var(--assistant-ink);
                border-bottom-left-radius: 8px;
            }

            .chat-bubble .message-text {
                white-space: pre-wrap;
                line-height: 1.58;
                font-size: 0.96rem;
            }

            .message-meta {
                font-size: 0.72rem;
                font-weight: 720;
                opacity: 0.74;
                margin-bottom: 0.35rem;
            }

            .message-meta.user {
                text-align: right;
            }

            .typing {
                display: inline-flex;
                align-items: center;
                gap: 0.35rem;
            }

            .typing span {
                width: 0.42rem;
                height: 0.42rem;
                border-radius: 999px;
                background: var(--muted);
                animation: typingPulse 1.15s infinite ease-in-out;
            }

            .typing span:nth-child(2) {
                animation-delay: 0.16s;
            }

            .typing span:nth-child(3) {
                animation-delay: 0.32s;
            }

            @keyframes typingPulse {
                0%, 80%, 100% {
                    opacity: 0.28;
                    transform: translateY(0);
                }
                40% {
                    opacity: 1;
                    transform: translateY(-3px);
                }
            }

            section[data-testid="stSidebar"] {
                background: var(--panel);
                border-right: 1px solid var(--line);
            }

            section[data-testid="stSidebar"] .block-container {
                padding-top: 1.25rem;
                padding-bottom: 1.25rem;
            }

            div[data-testid="stSidebarUserContent"] {
                color: var(--ink);
            }

            div[data-testid="stChatInput"] {
                position: sticky;
                bottom: 0;
                z-index: 50;
                background: color-mix(in srgb, var(--panel) 94%, transparent);
                border-top: 1px solid rgba(229, 231, 235, 0.8);
                padding-top: 0.65rem;
            }

            div[data-testid="stChatInput"] textarea {
                border-radius: 8px;
                border-color: #cbd5e1;
                box-shadow: 0 12px 28px rgba(15, 23, 42, 0.08);
            }

            @media (prefers-color-scheme: dark) {
                div[data-testid="stChatInput"] {
                    border-top-color: rgba(51, 65, 85, 0.9);
                }

                div[data-testid="stChatInput"] textarea,
                .stTextArea textarea {
                    background: #0f172a;
                    color: #f8fafc;
                    border-color: #475569;
                }
            }

            .stButton > button {
                border-radius: 8px;
                font-weight: 700;
            }

            .stTextArea textarea {
                border-radius: 8px;
                border-color: #cbd5e1;
                background: var(--panel);
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

                .stats-grid {
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                }

                .chat-bubble {
                    max-width: 94%;
                }

                .metric-card {
                    width: 100%;
                }
            }

            @media (max-width: 480px) {
                .stats-grid {
                    grid-template-columns: 1fr;
                }

                .chat-bubble {
                    max-width: 100%;
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
        f"""
        <div class="welcome-hero">
            <div class="welcome-title">{html.escape(APP_NAME)}</div>
            <div class="welcome-subtitle">
                A grounded placement assistant for eligibility, interview prep, package comparisons,
                hiring trends, and document-backed answers.
            </div>
        </div>
        <div class="welcome-grid">
            <div class="welcome-card">
                <div class="welcome-card-title">Can I apply to Amazon with 7.6 CGPA and 1 backlog?</div>
                <div class="welcome-card-text">
                    Check eligibility against CGPA, backlog, bond, package, and source context.
                </div>
            </div>
            <div class="welcome-card">
                <div class="welcome-card-title">What should I prepare for Microsoft interviews?</div>
                <div class="welcome-card-text">
                    Get round details, focus areas, and grounded preparation guidance.
                </div>
            </div>
            <div class="welcome-card">
                <div class="welcome-card-title">Which bond-free companies offer more than 40 LPA?</div>
                <div class="welcome-card-text">
                    Ask multi-hop questions and inspect citations before trusting the answer.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def current_timestamp() -> str:
    return datetime.now().strftime("%I:%M %p").lstrip("0")


def message_timestamp(message: dict[str, Any]) -> str:
    timestamp = message.get("timestamp")
    return str(timestamp) if timestamp else current_timestamp()


def highest_context_score(contexts: list[dict[str, Any]]) -> float:
    scores = [float(context.get("score") or 0.0) for context in contexts]
    return max(scores, default=0.0)


def confidence_level(contexts: list[dict[str, Any]]) -> str:
    score = highest_context_score(contexts)
    if score >= 0.55:
        return "High"
    if score >= 0.35:
        return "Medium"
    return "Low"


def confidence_class(level: str) -> str:
    return f"confidence-{level.lower()}"


def total_response_time(data: dict[str, Any]) -> int | None:
    retrieval_time = data.get("retrieval_time_ms")
    generation_time = data.get("generation_time_ms")
    if retrieval_time is None and generation_time is None:
        return None
    return int(retrieval_time or 0) + int(generation_time or 0)


SOURCE_CARD_RE = re.compile(
    r'<a\b[^>]*class=["\'][^"\']*\bsource-card\b[^"\']*["\'][\s\S]*?</a>',
    re.IGNORECASE,
)
SOURCE_ROW_RE = re.compile(
    r'<div\b[^>]*class=["\'][^"\']*\bsource-row\b[^"\']*["\'][\s\S]*?</div>',
    re.IGNORECASE,
)
HTML_TAG_RE = re.compile(r"</?[^>\n]+>")
FINAL_ANSWER_RE = re.compile(
    r"^\s*(?:final\s+answer|final\s+response|user-facing\s+answer)\s*:\s*(.*)$",
    re.IGNORECASE,
)
ANSWER_LABEL_RE = re.compile(r"^\s*answer\s*:\s*(.*)$", re.IGNORECASE)
INTERNAL_MARKER_RE = re.compile(
    r"^\s*(?:step\s*\d+|reasoning\s+chain|hop\s+type|sources\s+required|internal\s+reasoning(?:\s+traces?)?)\s*:",
    re.IGNORECASE,
)
INTERNAL_SECTION_RE = re.compile(
    r"^\s*(?:reasoning\s+chain|sources\s+required|internal\s+reasoning(?:\s+traces?)?)\s*:",
    re.IGNORECASE,
)
URL_RE = re.compile(r"^(?:https?://|mailto:)", re.IGNORECASE)


def clean_user_facing_answer(answer: Any) -> str:
    """Strip display-only artifacts while preserving Markdown structure."""
    text = str(answer or "").strip()
    if not text:
        return ""

    text = html.unescape(text)
    text = SOURCE_ROW_RE.sub("", text)
    text = SOURCE_CARD_RE.sub("", text)

    lines = text.splitlines()
    for index, line in enumerate(lines):
        match = FINAL_ANSWER_RE.match(line)
        if match:
            lines = [match.group(1), *lines[index + 1 :]]
            break

    cleaned_lines: list[str] = []
    for line in lines:
        final_match = FINAL_ANSWER_RE.match(line)
        if final_match:
            if final_match.group(1).strip():
                cleaned_lines.append(final_match.group(1).strip())
            continue

        answer_match = ANSWER_LABEL_RE.match(line)
        if answer_match and not cleaned_lines:
            if answer_match.group(1).strip():
                cleaned_lines.append(answer_match.group(1).strip())
            continue

        if INTERNAL_MARKER_RE.match(line):
            continue

        cleaned_lines.append(line.rstrip())

    cleaned = "\n".join(cleaned_lines).strip()
    cleaned = HTML_TAG_RE.sub("", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def clean_context_content(content: Any) -> str:
    text = str(content or "").strip()
    text = html.unescape(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def source_target(source: str) -> str:
    if URL_RE.match(source.strip()):
        return source.strip()
    return "#retrieved-context"


def render_chat_message(role: str, content: str, timestamp: str) -> None:
    role_label = "You" if role == "user" else "Assistant"
    if role == "assistant":
        with st.chat_message("assistant"):
            st.caption(f"{role_label} | {timestamp}")
            cleaned_content = clean_user_facing_answer(content)
            raw_content = str(content or "").strip()
            st.markdown(cleaned_content or raw_content or "_No answer returned._")
        return

    safe_content = html.escape(content or "").replace("\n", "<br>")
    st.markdown(
        f"""
        <div class="chat-row {html.escape(role)}">
            <div class="chat-bubble {html.escape(role)}">
                <div class="message-meta {html.escape(role)}">{role_label} | {html.escape(timestamp)}</div>
                <div class="message-text">{safe_content}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_typing_animation() -> None:
    st.markdown(
        """
        <div class="chat-row assistant">
            <div class="chat-bubble assistant">
                <div class="message-meta">Assistant is thinking</div>
                <div class="typing" aria-label="Generating response">
                    <span></span><span></span><span></span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sources(sources: list[str]) -> None:
    cleaned_sources = [str(source).strip() for source in sources if str(source).strip()]
    if not cleaned_sources:
        return

    st.markdown("#### Sources")
    columns = st.columns(min(3, len(cleaned_sources)))
    for index, source in enumerate(cleaned_sources, start=1):
        label = f"Source {index}: {shorten(source, width=52, placeholder='...')}"
        with columns[(index - 1) % len(columns)]:
            st.link_button(
                label,
                source_target(source),
                help=source,
                width="stretch",
            )


def render_retrieval_stats(data: dict[str, Any], contexts: list[dict[str, Any]]) -> None:
    if not data and not contexts:
        return

    retrieved_count = len(contexts)
    reranked_count = len(contexts)
    response_time = total_response_time(data)
    confidence = confidence_level(contexts)
    response_label = f"{response_time} ms" if response_time is not None else "N/A"
    st.markdown(
        f"""
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Retrieved chunks</div>
                <div class="stat-value">{retrieved_count}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Reranked chunks</div>
                <div class="stat-value">{reranked_count}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Response time</div>
                <div class="stat-value">{html.escape(response_label)}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Confidence</div>
                <div class="stat-value">
                    <span class="confidence-badge {confidence_class(confidence)}">{confidence}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
    st.markdown("#### Retrieved Context")
    for index, context in enumerate(contexts, start=1):
        metadata = context.get("metadata", {})
        extra = metadata.get("extra", {})
        source_file = extra.get("source_file") or metadata.get("source") or "Unknown source"
        page = metadata.get("page")
        page_text = f" page {page}" if page else ""
        try:
            score = float(context.get("score", 0.0) or 0.0)
        except (TypeError, ValueError):
            score = 0.0

        with st.expander(f"Source {index}: {source_file}{page_text}", expanded=False):
            st.caption(f"Score {score:.4f}")
            st.markdown(clean_context_content(context.get("content", "")) or "_No context text returned._")


if mode == "Chat":
    if not st.session_state.messages:
        render_welcome()

    for message in st.session_state.messages:
        render_chat_message(message["role"], message["content"], message_timestamp(message))
        if message["role"] == "assistant":
            contexts = message.get("contexts", [])
            metrics = message.get("metrics", {})
            render_sources(message.get("sources", []))
            render_retrieval_stats(metrics, contexts)
            render_timing(metrics)
            render_contexts(contexts)

    prompt = st.chat_input("Ask about companies, eligibility, rounds, packages, or preparation")
    if prompt:
        user_timestamp = current_timestamp()
        st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": user_timestamp})
        render_chat_message("user", prompt, user_timestamp)

        payload: dict[str, Any] = {"query": prompt}
        if st.session_state.session_id:
            payload["session_id"] = st.session_state.session_id

        assistant_placeholder = st.empty()
        try:
            with assistant_placeholder.container():
                render_typing_animation()
            with st.spinner("Retrieving sources and drafting a grounded answer..."):
                data = post_json(CHAT_ENDPOINT, payload)
            st.session_state.session_id = data.get("session_id") or st.session_state.session_id
            
            answer = data.get("answer", "")
            print(data)
            print(answer)
            print(type(answer))
            sources = data.get("sources", [])

            with st.spinner("Preparing retrieved context..."):
                context_data = post_json(QUERY_ENDPOINT, {"query": prompt, "top_k": top_k, "modality": "text"})
            contexts = context_data.get("contexts", [])
            assistant_timestamp = current_timestamp()

            assistant_placeholder.empty()
            render_chat_message("assistant", answer, assistant_timestamp)
            render_sources(sources)
            render_retrieval_stats(context_data, contexts)
            render_timing(context_data)
            render_contexts(contexts)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "timestamp": assistant_timestamp,
                    "sources": sources,
                    "metrics": context_data,
                    "contexts": contexts,
                }
            )
        except requests.RequestException as exc:
            assistant_placeholder.empty()
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
            contexts = data.get("contexts", [])
            answer = data.get("answer", "")
            print(data)
            print(answer)
            print(type(answer))
            st.markdown('<div class="section-title">Answer</div>', unsafe_allow_html=True)
            st.markdown(clean_user_facing_answer(answer) or "_No answer returned._")
            sources = data.get("sources", [])
            render_sources(sources)
            render_retrieval_stats(data, contexts)
            render_timing(data)
            render_contexts(contexts)
        except requests.RequestException as exc:
            st.error(f"API request failed: {exc}")
