import os
from typing import Any

import requests
import streamlit as st

from version import __version__


DEFAULT_API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000/api/v1")


def ask_backend(api_url: str, question: str, conversation_id: str | None) -> dict[str, Any]:
    payload: dict[str, Any] = {"query": question}
    if conversation_id:
        payload["session_id"] = conversation_id

    response = requests.post(f"{api_url.rstrip('/')}/chat", json=payload, timeout=60)
    response.raise_for_status()
    return response.json()


def main() -> None:
    st.set_page_config(page_title="Placement Intelligence RAG", layout="wide")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None

    with st.sidebar:
        st.title("Placement RAG")
        api_url = st.text_input("FastAPI URL", value=DEFAULT_API_URL)
        if st.button("New chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_id = None
        st.caption(f"Version {__version__}")

    st.title("Placement Intelligence RAG")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask about eligibility, companies, roles, packages, or preparation.")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving placement context..."):
            try:
                data = ask_backend(api_url, prompt, st.session_state.conversation_id)
            except requests.RequestException as exc:
                answer = f"Could not reach the FastAPI backend: {exc}"
                st.error(answer)
            else:
                st.session_state.conversation_id = data.get("session_id", st.session_state.conversation_id)
                answer = data.get("answer") or "No answer returned."
                st.markdown(answer)

                contexts = data.get("contexts") or data.get("sources") or []
                if contexts:
                    with st.expander("Retrieved context"):
                        for index, context in enumerate(contexts, start=1):
                            content = context.get("content", "")
                            score = context.get("score")
                            label = f"{index}. score={score:.4f}" if isinstance(score, float) else f"{index}."
                            st.markdown(f"**{label}**")
                            st.write(content)

    st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
