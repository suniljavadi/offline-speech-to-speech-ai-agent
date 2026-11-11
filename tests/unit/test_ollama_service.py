from __future__ import annotations

import json
import socket
import threading
from urllib.error import URLError
from urllib.request import Request

import pytest

from app.services.ollama_service import (
    ChatMessage,
    OllamaConfig,
    OllamaModelUnavailableError,
    OllamaRequestError,
    OllamaService,
    OllamaToolCall,
    OllamaUnavailableError,
)


class FakeResponse:
    def __init__(self, chunks: list[dict] | None = None, body: dict | None = None):
        self.lines = [
            json.dumps(chunk).encode("utf-8") + b"\n" for chunk in (chunks or [])
        ]
        self.body = json.dumps(body or {}).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def __iter__(self):
        return iter(self.lines)

    def read(self):
        return self.body


def make_service(opener, config: OllamaConfig | None = None) -> OllamaService:
    return OllamaService(config=config, opener=opener)


def test_streams_local_chat_response_and_uses_configured_context() -> None:
    response = FakeResponse(
        chunks=[
            {"message": {"content": "Hello"}, "done": False},
            {"message": {"content": " there."}, "done": True},
        ]
    )
    captured = {}

    def opener(request: Request, timeout: float):
        captured["request"] = request
        captured["timeout"] = timeout
        return response

    service = make_service(opener)

    text = service.chat("Hi", [ChatMessage("user", "Earlier")])

    payload = json.loads(captured["request"].data)
    assert text == "Hello there."
    assert captured["timeout"] == 120
    assert captured["request"].full_url == "http://127.0.0.1:11434/api/chat"
    assert payload["model"] == "llama3.1"
    assert payload["stream"] is True
    assert payload["messages"] == [
        {"role": "system", "content": service.config.system_prompt},
        {"role": "user", "content": "Earlier"},
        {"role": "user", "content": "Hi"},
    ]


def test_context_is_bounded_by_message_count_and_characters() -> None:
    service = make_service(
        lambda *_args, **_kwargs: FakeResponse(),
        OllamaConfig(
            system_prompt="sys",
            max_history_messages=2,
            max_context_characters=10,
        ),
    )

    messages = service.build_messages(
        [ChatMessage("user", "one"), ChatMessage("assistant", "two")],
        "now",
    )

    assert messages == [
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "now"},
    ]


def test_rejects_empty_prompt_and_system_messages_in_history() -> None:
    service = make_service(lambda *_args, **_kwargs: FakeResponse())

    with pytest.raises(ValueError, match="required"):
        service.build_messages([], "  ")
    with pytest.raises(ValueError, match="system messages"):
        service.build_messages([ChatMessage("system", "override")], "Hi")
    with pytest.raises(ValueError, match="System prompt"):
        service.build_messages([], "Hi", system_prompt="")


def test_build_messages_accepts_a_mode_specific_system_prompt() -> None:
    service = make_service(lambda *_args, **_kwargs: FakeResponse())

    messages = service.build_messages([], "Explain this project", system_prompt="Coding mode")

    assert messages[0] == {"role": "system", "content": "Coding mode"}


@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1:11434",
        "http://api.example.com:11434",
        "http://user:pass@127.0.0.1:11434",
        "http://127.0.0.1:11434/proxy",
        "http://localhost:invalid",
    ],
)
def test_rejects_non_local_or_invalid_ollama_urls(url: str) -> None:
    with pytest.raises(ValueError):
        OllamaConfig(base_url=url)


def test_health_checks_version_and_model_availability() -> None:
    responses = iter(
        [
            FakeResponse(body={"version": "0.5.0"}),
            FakeResponse(body={"models": [{"name": "llama3.1:latest"}]}),
        ]
    )
    service = make_service(lambda *_args, **_kwargs: next(responses))

    health = service.check_health()

    assert health.version == "0.5.0"
    assert health.model == "llama3.1"
    assert health.model_available is True


def test_health_reports_missing_model_and_require_raises_guidance() -> None:
    responses = iter(
        [
            FakeResponse(body={"version": "0.5.0"}),
            FakeResponse(body={"models": []}),
            FakeResponse(body={"version": "0.5.0"}),
            FakeResponse(body={"models": []}),
        ]
    )
    service = make_service(lambda *_args, **_kwargs: next(responses))

    assert service.check_health().model_available is False
    with pytest.raises(OllamaModelUnavailableError, match="ollama pull llama3.1"):
        service.require_model_available()


def test_connection_and_timeout_failures_are_explicit() -> None:
    def offline(*_args, **_kwargs):
        raise URLError("connection refused")

    service = make_service(offline)

    with pytest.raises(OllamaUnavailableError, match="local Ollama"):
        service.list_models()

    def timeout(*_args, **_kwargs):
        raise socket.timeout("timed out")

    with pytest.raises(OllamaUnavailableError):
        make_service(timeout).list_models()


def test_streamed_model_error_is_not_a_success_response() -> None:
    service = make_service(
        lambda *_args, **_kwargs: FakeResponse(
            chunks=[{"error": "model 'llama3.1' not found"}]
        )
    )

    with pytest.raises(OllamaModelUnavailableError, match="not found"):
        service.chat("Hi")


def test_stream_can_be_cancelled_between_chunks() -> None:
    event = threading.Event()

    class CancelAfterFirstChunk(FakeResponse):
        def __iter__(self):
            yield json.dumps({"message": {"content": "first"}, "done": False}).encode()
            event.set()
            yield json.dumps({"message": {"content": "second"}, "done": True}).encode()

    service = make_service(lambda *_args, **_kwargs: CancelAfterFirstChunk())

    assert list(service.stream_chat("Hi", stop_event=event)) == ["first"]


def test_invalid_stream_json_raises_request_error() -> None:
    class InvalidResponse(FakeResponse):
        def __iter__(self):
            yield b"{not-json}\n"

    service = make_service(lambda *_args, **_kwargs: InvalidResponse())

    with pytest.raises(OllamaRequestError, match="invalid streaming"):
        service.chat("Hi")


def test_incomplete_stream_raises_request_error() -> None:
    service = make_service(
        lambda *_args, **_kwargs: FakeResponse(
            chunks=[{"message": {"content": "partial"}, "done": False}]
        )
    )

    with pytest.raises(OllamaRequestError, match="before completing"):
        service.chat("Hi")


def test_chat_with_tools_parses_registered_function_calls() -> None:
    payload = {
        "message": {
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "function": {
                        "name": "get_current_time",
                        "arguments": {"timezone": "UTC"},
                    },
                }
            ],
        }
    }
    captured = {}

    def opener(request: Request, timeout: float):
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeResponse(body=payload)

    service = make_service(opener)
    result = service.chat_with_tools(
        [{"role": "user", "content": "What time is it?"}],
        [{"type": "function", "function": {"name": "get_current_time"}}],
    )

    assert result.content == ""
    assert result.tool_calls == (
        OllamaToolCall(
            "get_current_time", {"timezone": "UTC"}, "call-1"
        ),
    )
    request_payload = json.loads(captured["request"].data)
    assert request_payload["stream"] is False
    assert request_payload["tools"][0]["function"]["name"] == "get_current_time"


@pytest.mark.parametrize(
    "message",
    [
        {"content": "bad", "tool_calls": "not-a-list"},
        {
            "content": "",
            "tool_calls": [{"function": {"name": "tool", "arguments": []}}],
        },
        {"content": 1},
    ],
)
def test_chat_with_tools_rejects_malformed_response(message: dict) -> None:
    service = make_service(
        lambda *_args, **_kwargs: FakeResponse(body={"message": message})
    )

    with pytest.raises(OllamaRequestError):
        service.chat_with_tools(
            [{"role": "user", "content": "Run a tool"}],
            [{"type": "function", "function": {"name": "tool"}}],
        )
