"""Loopback-only Ollama chat client with bounded context and streaming."""

from __future__ import annotations

import json
import logging
import socket
import threading
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


class OllamaError(RuntimeError):
    """Base error for local Ollama integration."""


class OllamaUnavailableError(OllamaError):
    """Raised when the local Ollama service cannot be reached."""


class OllamaModelUnavailableError(OllamaError):
    """Raised when the configured model is not available locally."""


class OllamaRequestError(OllamaError):
    """Raised when Ollama returns an invalid or unsuccessful response."""


@dataclass(frozen=True)
class ChatMessage:
    """A validated Ollama chat message."""

    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"Unsupported chat role: {self.role}")
        if not isinstance(self.content, str):
            raise ValueError("Chat message content must be a string.")


@dataclass(frozen=True)
class OllamaConfig:
    """Local Ollama endpoint, model, timeout, and context limits."""

    base_url: str = "http://127.0.0.1:11434"
    model: str = "llama3.1"
    system_prompt: str = (
        "You are a helpful, concise assistant. Do not claim to perform actions "
        "that you have not performed, and say when you are uncertain."
    )
    request_timeout_seconds: float = 120.0
    max_history_messages: int = 20
    max_context_characters: int = 24000
    context_window_tokens: int = 8192
    max_response_tokens: int = 1024

    def __post_init__(self) -> None:
        parsed = urlsplit(self.base_url)
        if (
            parsed.scheme != "http"
            or parsed.hostname is None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
        ):
            raise ValueError(
                "Ollama must use an HTTP URL on a loopback host without credentials, "
                "path, query, or fragment."
            )
        try:
            parsed.port
        except ValueError as exc:
            raise ValueError("Ollama base_url must contain a valid port.") from exc
        try:
            is_loopback = parsed.hostname == "localhost" or ip_address_is_loopback(
                parsed.hostname
            )
        except ValueError:
            is_loopback = False
        if not is_loopback:
            raise ValueError("Ollama base_url must resolve to a loopback address.")
        if not self.model.strip():
            raise ValueError("Ollama model cannot be empty.")
        if not self.system_prompt.strip():
            raise ValueError("System prompt cannot be empty.")
        if self.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be greater than zero.")
        if self.max_history_messages < 1:
            raise ValueError("max_history_messages must be at least one.")
        if self.max_context_characters < len(self.system_prompt):
            raise ValueError("Context limit must fit the configured system prompt.")
        if self.context_window_tokens < 1 or self.max_response_tokens < 1:
            raise ValueError("Token limits must be at least one.")


@dataclass(frozen=True)
class OllamaHealth:
    """Reachability and configured-model availability."""

    version: str
    model: str
    model_available: bool


@dataclass(frozen=True)
class OllamaToolCall:
    """One named tool call returned by a local Ollama model."""

    name: str
    arguments: dict[str, Any]
    call_id: str


@dataclass(frozen=True)
class OllamaToolResponse:
    """Model text and validated tool calls from a non-streaming tool request."""

    content: str
    tool_calls: tuple[OllamaToolCall, ...]


UrlOpener = Callable[..., Any]


def ip_address_is_loopback(hostname: str) -> bool:
    """Check literal IP addresses without resolving or contacting the network."""
    import ipaddress

    return ipaddress.ip_address(hostname).is_loopback


class OllamaService:
    """Call an Ollama server on loopback; never use a remote inference API."""

    def __init__(
        self,
        config: OllamaConfig | None = None,
        opener: UrlOpener | None = None,
    ) -> None:
        self.config = config or OllamaConfig()
        self._opener = opener or urlopen

    def build_messages(
        self,
        history: Sequence[ChatMessage],
        prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> list[dict[str, str]]:
        """Create a system-first request and trim oldest context to configured limits."""
        if not prompt.strip():
            raise ValueError("A user message is required.")

        active_system_prompt = (
            self.config.system_prompt if system_prompt is None else system_prompt
        )
        if not active_system_prompt.strip():
            raise ValueError("System prompt cannot be empty.")
        messages = [ChatMessage("system", active_system_prompt)]
        for item in history:
            message = item if isinstance(item, ChatMessage) else ChatMessage(**item)
            if message.role == "system":
                raise ValueError("Conversation history must not contain system messages.")
            messages.append(message)
        messages.append(ChatMessage("user", prompt))

        user_message = messages[-1]
        if len(user_message.content) + len(messages[0].content) > self.config.max_context_characters:
            raise ValueError("User message exceeds the configured context limit.")

        messages = [
            messages[0],
            *messages[1:][-self.config.max_history_messages :],
        ]
        while self._message_characters(messages) > self.config.max_context_characters:
            if len(messages) == 2:
                raise ValueError("Request exceeds the configured context limit.")
            messages.pop(1)
        return [{"role": message.role, "content": message.content} for message in messages]

    def stream_chat(
        self,
        prompt: str,
        history: Sequence[ChatMessage] = (),
        stop_event: threading.Event | None = None,
    ) -> Iterator[str]:
        """Yield response text deltas from Ollama's newline-delimited chat stream."""
        messages = self.build_messages(history, prompt)
        payload = {
            "model": self.config.model,
            "messages": messages,
            "stream": True,
            "options": {
                "num_ctx": self.config.context_window_tokens,
                "num_predict": self.config.max_response_tokens,
            },
        }
        request = Request(
            self._endpoint("/api/chat"),
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with self._opener(
                request, timeout=self.config.request_timeout_seconds
            ) as response:
                completed = False
                for raw_line in response:
                    if stop_event is not None and stop_event.is_set():
                        return
                    if not raw_line.strip():
                        continue
                    try:
                        chunk = json.loads(raw_line)
                    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                        raise OllamaRequestError(
                            "Ollama returned an invalid streaming response."
                        ) from exc
                    if not isinstance(chunk, dict):
                        raise OllamaRequestError(
                            "Ollama returned an invalid streaming response."
                        )
                    if chunk.get("error"):
                        self._raise_response_error(str(chunk["error"]))
                    message = chunk.get("message", {})
                    if not isinstance(message, dict):
                        raise OllamaRequestError(
                            "Ollama returned an invalid message chunk."
                        )
                    content = message.get("content", "")
                    if not isinstance(content, str):
                        raise OllamaRequestError(
                            "Ollama returned a non-text response chunk."
                        )
                    if content:
                        yield content
                    if chunk.get("done") is True:
                        completed = True
                        return
                if not completed:
                    raise OllamaRequestError(
                        "Ollama closed the response before completing the stream."
                    )
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            self._raise_http_error(exc.code, body)
        except (URLError, TimeoutError, socket.timeout, ConnectionError, OSError) as exc:
            logger.warning("Local Ollama request failed: %s", exc.__class__.__name__)
            raise OllamaUnavailableError(
                "Could not reach the local Ollama service. Start Ollama and verify "
                "its loopback address."
            ) from exc

    def chat(
        self,
        prompt: str,
        history: Sequence[ChatMessage] = (),
        stop_event: threading.Event | None = None,
    ) -> str:
        """Return the complete response, collecting streamed text deltas."""
        return "".join(self.stream_chat(prompt, history, stop_event))

    def chat_with_tools(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
    ) -> OllamaToolResponse:
        """Request structured Ollama tool calls using only supplied tool schemas."""
        if not messages:
            raise ValueError("At least one chat message is required.")
        if not tools:
            raise ValueError("At least one tool schema is required.")
        payload = {
            "model": self.config.model,
            "messages": list(messages),
            "tools": list(tools),
            "stream": False,
            "options": {
                "num_ctx": self.config.context_window_tokens,
                "num_predict": self.config.max_response_tokens,
            },
        }
        request = Request(
            self._endpoint("/api/chat"),
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self._opener(
                request, timeout=self.config.request_timeout_seconds
            ) as response:
                try:
                    data = json.loads(response.read())
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    raise OllamaRequestError(
                        "Ollama returned invalid tool-call JSON."
                    ) from exc
        except HTTPError as exc:
            self._raise_http_error(
                exc.code, exc.read().decode("utf-8", errors="replace")
            )
        except (URLError, TimeoutError, socket.timeout, ConnectionError, OSError) as exc:
            logger.warning("Local Ollama tool request failed: %s", exc.__class__.__name__)
            raise OllamaUnavailableError(
                "Could not reach the local Ollama service. Start Ollama and verify "
                "its loopback address."
            ) from exc

        if not isinstance(data, dict):
            raise OllamaRequestError("Ollama returned an invalid tool response.")
        if data.get("error"):
            self._raise_response_error(str(data["error"]))
        message = data.get("message")
        if not isinstance(message, dict):
            raise OllamaRequestError("Ollama response did not contain a message.")
        content = message.get("content", "")
        if not isinstance(content, str):
            raise OllamaRequestError("Ollama returned non-text message content.")
        raw_calls = message.get("tool_calls", [])
        if not isinstance(raw_calls, list):
            raise OllamaRequestError("Ollama returned an invalid tool-call list.")
        tool_calls: list[OllamaToolCall] = []
        for index, raw_call in enumerate(raw_calls):
            if not isinstance(raw_call, dict):
                raise OllamaRequestError("Ollama returned an invalid tool call.")
            function = raw_call.get("function")
            if not isinstance(function, dict):
                raise OllamaRequestError(
                    "Ollama tool call did not contain a function."
                )
            name = function.get("name")
            arguments = function.get("arguments", {})
            if not isinstance(name, str) or not name:
                raise OllamaRequestError("Ollama returned an unnamed tool call.")
            if not isinstance(arguments, dict):
                raise OllamaRequestError(
                    "Ollama tool arguments must be a JSON object."
                )
            call_id = raw_call.get("id", f"ollama-call-{index}")
            if not isinstance(call_id, str):
                raise OllamaRequestError("Ollama returned an invalid tool call ID.")
            tool_calls.append(OllamaToolCall(name, arguments, call_id))
        return OllamaToolResponse(content, tuple(tool_calls))

    def list_models(self) -> list[str]:
        """Return model names available on the configured local Ollama service."""
        data = self._get_json("/api/tags")
        models = data.get("models")
        if not isinstance(models, list):
            raise OllamaRequestError("Ollama returned an invalid model list.")
        names = [
            item["name"]
            for item in models
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        ]
        return names

    def check_health(self) -> OllamaHealth:
        """Check local service version and whether the configured model is present."""
        version_data = self._get_json("/api/version")
        version = version_data.get("version")
        if not isinstance(version, str) or not version:
            raise OllamaRequestError("Ollama returned an invalid version response.")
        models = self.list_models()
        return OllamaHealth(
            version=version,
            model=self.config.model,
            model_available=self._model_is_available(models, self.config.model),
        )

    def require_model_available(self) -> None:
        """Raise an actionable error if the configured model has not been pulled."""
        health = self.check_health()
        if not health.model_available:
            raise OllamaModelUnavailableError(
                f"Ollama is running, but model '{self.config.model}' is not available. "
                f"Pull it explicitly with: ollama pull {self.config.model}"
            )

    def _get_json(self, path: str) -> dict[str, Any]:
        request = Request(self._endpoint(path), headers={"Accept": "application/json"})
        try:
            with self._opener(
                request, timeout=self.config.request_timeout_seconds
            ) as response:
                try:
                    data = json.loads(response.read())
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    raise OllamaRequestError(
                        "Ollama returned invalid JSON."
                    ) from exc
        except HTTPError as exc:
            self._raise_http_error(
                exc.code, exc.read().decode("utf-8", errors="replace")
            )
        except (URLError, TimeoutError, socket.timeout, ConnectionError, OSError) as exc:
            logger.warning("Local Ollama request failed: %s", exc.__class__.__name__)
            raise OllamaUnavailableError(
                "Could not reach the local Ollama service. Start Ollama and verify "
                "its loopback address."
            ) from exc
        if not isinstance(data, dict):
            raise OllamaRequestError("Ollama returned an invalid JSON object.")
        if data.get("error"):
            self._raise_response_error(str(data["error"]))
        return data

    def _endpoint(self, path: str) -> str:
        return f"{self.config.base_url.rstrip('/')}{path}"

    @staticmethod
    def _message_characters(messages: Sequence[ChatMessage]) -> int:
        return sum(len(message.content) for message in messages)

    @staticmethod
    def _model_is_available(models: Sequence[str], configured_model: str) -> bool:
        def normalized(name: str) -> str:
            return name if ":" in name else f"{name}:latest"

        target = normalized(configured_model)
        return any(normalized(name) == target for name in models)

    @staticmethod
    def _raise_response_error(message: str) -> None:
        if "not found" in message.casefold() or "pull" in message.casefold():
            raise OllamaModelUnavailableError(message)
        raise OllamaRequestError(f"Ollama returned an error: {message}")

    @classmethod
    def _raise_http_error(cls, status: int, body: str) -> None:
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            data = {}
        message = data.get("error") if isinstance(data, dict) else None
        if isinstance(message, str) and (
            "not found" in message.casefold() or "pull" in message.casefold()
        ):
            raise OllamaModelUnavailableError(
                f"Configured Ollama model is unavailable: {message}"
            )
        raise OllamaRequestError(
            f"Ollama returned HTTP {status}."
            + (f" {message}" if isinstance(message, str) else "")
        )
