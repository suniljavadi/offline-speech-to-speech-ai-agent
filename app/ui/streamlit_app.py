"""Streamlit UI that calls the loopback FastAPI service."""

from __future__ import annotations

import ipaddress
import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from typing import Any

import streamlit as st


class APIError(RuntimeError):
    """Raised when the local API rejects or cannot complete a UI request."""


def _api_base_url() -> str:
    value = os.environ.get("APP_API_URL", "http://127.0.0.1:8000").rstrip("/")
    try:
        parsed = urlsplit(value)
        parsed.port
    except ValueError as exc:
        raise APIError("APP_API_URL is invalid.") from exc
    if parsed.scheme != "http" or parsed.hostname is None:
        raise APIError("APP_API_URL must be an HTTP loopback URL.")
    try:
        loopback = parsed.hostname == "localhost" or ipaddress.ip_address(
            parsed.hostname
        ).is_loopback
    except ValueError:
        loopback = parsed.hostname == "localhost"
    if (
        not loopback
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise APIError("APP_API_URL must point to a loopback host only.")
    return value


def _api_request(
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Call the local API and preserve its error detail for the user."""
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        _api_base_url() + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    try:
        with urlopen(request, timeout=180) as response:
            body = response.read()
    except HTTPError as exc:
        try:
            error_body = json.loads(exc.read().decode("utf-8"))
            detail = error_body.get("detail", str(exc))
        except (UnicodeDecodeError, json.JSONDecodeError):
            detail = f"Local API returned HTTP {exc.code}."
        raise APIError(str(detail)) from exc
    except URLError as exc:
        raise APIError(
            "Could not reach the local API. Start it with `python -m app.api.main`."
        ) from exc
    try:
        result = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise APIError("The local API returned an invalid response.") from exc
    if not isinstance(result, dict):
        raise APIError("The local API returned an unexpected response shape.")
    return result


def _clear_pending() -> None:
    st.session_state.pending_id = None
    st.session_state.pending_tool = None
    st.session_state.approval_request = None


def main() -> None:
    st.set_page_config(page_title="Offline Voice Coding Assistant")
    st.title("Offline Voice & Coding Assistant")
    st.caption(
        "Local microphone, speech models, Ollama, and workspace tools. "
        "No cloud inference. Microphone starts only when you activate a turn."
    )

    for key, default in (
        ("conversation_id", None),
        ("pending_id", None),
        ("pending_tool", None),
        ("approval_request", None),
        ("last_result", None),
    ):
        st.session_state.setdefault(key, default)

    try:
        conversations = _api_request("GET", "/conversations?limit=50")["conversations"]
    except (APIError, KeyError) as exc:
        conversations = None
        st.sidebar.error(str(exc))

    if conversations is not None:
        options = [None, *(item["id"] for item in conversations)]
        if (
            st.session_state.conversation_id is not None
            and st.session_state.conversation_id not in options
        ):
            options.append(st.session_state.conversation_id)
        selected = st.sidebar.selectbox(
            "Conversation",
            options,
            index=options.index(st.session_state.conversation_id)
            if st.session_state.conversation_id in options
            else 0,
            format_func=lambda value: "New conversation"
            if value is None
            else next(
                (
                    f"{item['id'][:8]} - {item['updated_at'][:19]}"
                    for item in conversations
                    if item["id"] == value
                ),
                value,
            ),
            disabled=st.session_state.pending_id is not None,
        )
        if selected != st.session_state.conversation_id:
            st.session_state.conversation_id = selected
            _clear_pending()
            st.session_state.last_result = None
    else:
        selected = st.session_state.conversation_id
    if (
        selected is not None
        and st.session_state.pending_id is None
        and st.sidebar.button("Delete this conversation")
    ):
        try:
            _api_request("DELETE", f"/conversations/{selected}")
            st.session_state.conversation_id = None
            _clear_pending()
            st.session_state.last_result = None
            st.rerun()
        except APIError as exc:
            st.sidebar.error(str(exc))

    mode = st.sidebar.radio("Assistant mode", ["Voice chat", "Coding assistant"])
    latest_user_message = None
    latest_assistant_message = None
    if st.session_state.conversation_id:
        try:
            detail = _api_request(
                "GET", f"/conversations/{st.session_state.conversation_id}"
            )
            for message in detail["messages"]:
                if message["role"] in {"user", "assistant"}:
                    if message["role"] == "user":
                        latest_user_message = message["content"]
                    else:
                        latest_assistant_message = message["content"]
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
        except (APIError, KeyError) as exc:
            st.error(str(exc))

    if st.session_state.last_result:
        result = st.session_state.last_result
        if (
            result.get("transcript")
            and result["transcript"] != latest_user_message
        ):
            st.info(f"Recognized: {result['transcript']}")
        if (
            result.get("response")
            and result["response"] != latest_assistant_message
        ):
            with st.chat_message("assistant"):
                st.markdown(result["response"])
        if result.get("status") in {"no_speech", "no_transcript"}:
            st.warning("No speech was recognized. Try again when ready.")

    if st.session_state.pending_id:
        st.warning(
            "This operation has not run. Review its exact arguments before "
            "approving."
        )
        if st.session_state.approval_request is None:
            if st.button("Review exact proposed change"):
                try:
                    request = _api_request(
                        "POST",
                        "/coding/approval/prepare",
                        {"pending_id": st.session_state.pending_id},
                    )
                    st.session_state.approval_request = request
                    st.rerun()
                except APIError as exc:
                    st.error(str(exc))
        else:
            request = st.session_state.approval_request
            st.subheader(f"Proposed operation: {request['tool_name']}")
            st.json(request["arguments"])
            st.caption(
                f"This confirmation expires in "
                f"{request['expires_in_seconds']:.0f} seconds."
            )
            approved = st.checkbox(
                "I approve exactly this operation and these arguments."
            )
            if st.button("Approve and continue", disabled=not approved):
                try:
                    token = _api_request(
                        "POST",
                        "/coding/approval/confirm",
                        {
                            "pending_id": st.session_state.pending_id,
                            "request_id": request["request_id"],
                        },
                    )["approval_token"]
                    result = _api_request(
                        "POST",
                        "/coding/approval/resume",
                        {
                            "pending_id": st.session_state.pending_id,
                            "approval_token": token,
                        },
                    )
                    st.session_state.last_result = result
                    _clear_pending()
                    next_pending = result.get("pending_tool")
                    if next_pending is not None:
                        st.session_state.pending_id = next_pending["pending_id"]
                        st.session_state.pending_tool = next_pending
                    st.rerun()
                except APIError as exc:
                    st.error(str(exc))
    else:
        if mode == "Coding assistant":
            st.info(
                "Ask about the configured workspace or request a change. "
                "File modifications require review and approval."
            )
        if st.button("Start listening", type="primary"):
            st.session_state.last_result = None
            with st.spinner("Listening and processing locally..."):
                try:
                    path = (
                        "/coding/turn"
                        if mode == "Coding assistant"
                        else "/voice/turn"
                    )
                    result = _api_request(
                        "POST",
                        path,
                        {"conversation_id": st.session_state.conversation_id},
                    )
                    st.session_state.last_result = result
                    st.session_state.conversation_id = (
                        result.get("conversation_id")
                        or st.session_state.conversation_id
                    )
                    pending_tool = result.get("pending_tool")
                    if pending_tool:
                        st.session_state.pending_id = pending_tool["pending_id"]
                        st.session_state.pending_tool = pending_tool
                except APIError as exc:
                    st.error(str(exc))
            st.rerun()


if __name__ == "__main__":
    main()
