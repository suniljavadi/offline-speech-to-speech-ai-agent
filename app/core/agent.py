"""Approval-aware local agent loop for registered workspace tools."""

from __future__ import annotations

import json
import secrets
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from app.core.permissions import ApprovalError, ApprovalRequest
from app.core.tool_registry import ToolRegistry
from app.services.ollama_service import (
    ChatMessage,
    OllamaService,
    OllamaToolCall,
)

AgentStatus = Literal["completed", "approval_required", "max_tool_calls"]


@dataclass(frozen=True)
class PendingAgentTool:
    """Tool call awaiting confirmation, safe to display in the approval UI."""

    pending_id: str
    tool_name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class AgentTurnResult:
    """Final agent response or the next user approval request."""

    status: AgentStatus
    response: str
    pending_tool: PendingAgentTool | None = None
    tool_calls_executed: int = 0


@dataclass
class _PendingState:
    messages: list[dict[str, Any]]
    remaining_calls: list[OllamaToolCall]
    tool_schemas: list[dict[str, Any]]
    response_parts: list[str]
    conversation_id: str | None
    executed_count: int
    expires_at: float
    approval_request_id: str | None = None


class LocalAgent:
    """Select and invoke only typed local tools, pausing for risky actions."""

    def __init__(
        self,
        llm: OllamaService,
        registry: ToolRegistry,
        *,
        max_tool_calls: int = 8,
        pending_ttl_seconds: float = 600.0,
        clock: Any = time.monotonic,
    ) -> None:
        if max_tool_calls < 1:
            raise ValueError("max_tool_calls must be at least one.")
        if pending_ttl_seconds <= 0:
            raise ValueError("pending_ttl_seconds must be greater than zero.")
        self.llm = llm
        self.registry = registry
        self.max_tool_calls = max_tool_calls
        self.pending_ttl_seconds = pending_ttl_seconds
        self._clock = clock
        self._pending: dict[str, _PendingState] = {}
        self._lock = threading.Lock()

    def run(
        self,
        prompt: str,
        history: Sequence[ChatMessage] = (),
        conversation_id: str | None = None,
        *,
        system_prompt: str | None = None,
    ) -> AgentTurnResult:
        """Run local model/tool rounds until a final answer or approval is needed."""
        messages = self.llm.build_messages(
            history, prompt, system_prompt=system_prompt
        )
        tool_schemas = [self._ollama_schema(tool) for tool in self.registry.available_tools()]
        if not tool_schemas:
            raise RuntimeError("No agent tools are available in the local configuration.")
        response_parts: list[str] = []
        executed_count = 0

        while executed_count < self.max_tool_calls:
            response = self.llm.chat_with_tools(messages, tool_schemas)
            if response.content:
                response_parts.append(response.content)
            if not response.tool_calls:
                return AgentTurnResult(
                    "completed", "".join(response_parts).strip(), None, executed_count
                )
            tool_messages = [
                self._assistant_tool_message(response.content, response.tool_calls)
            ]
            messages.extend(tool_messages)
            result = self._process_calls(
                messages,
                list(response.tool_calls),
                tool_schemas,
                response_parts,
                conversation_id,
                executed_count,
            )
            if isinstance(result, AgentTurnResult):
                return result
            messages, response_parts, executed_count = result

        return AgentTurnResult(
            "max_tool_calls",
            "".join(response_parts).strip(),
            None,
            executed_count,
        )

    def prepare_pending_approval(self, pending_id: str) -> ApprovalRequest:
        """Prepare the exact tool operation for display in a confirmation UI."""
        state = self._get_pending(pending_id)
        current_call = state.remaining_calls[0]
        request = self.registry.prepare_approval(
            current_call.name, current_call.arguments
        )
        state.approval_request_id = request.request_id
        return request

    def confirm_pending(self, pending_id: str, approval_request_id: str) -> str:
        """Issue a token only for the operation currently displayed for approval."""
        state = self._get_pending(pending_id)
        if (
            state.approval_request_id is None
            or state.approval_request_id != approval_request_id
        ):
            raise ApprovalError("Approval request does not match the pending operation.")
        token = self.registry.confirm_approval(approval_request_id)
        state.approval_request_id = None
        return token

    def resume(
        self, pending_id: str, approval_token: str
    ) -> AgentTurnResult:
        """Execute the confirmed pending call and continue the local agent loop."""
        with self._lock:
            state = self._pending.pop(pending_id, None)
        if state is None or self._clock() >= state.expires_at:
            raise ApprovalError("Pending agent operation is missing or expired.")

        call = state.remaining_calls[0]
        try:
            result = self.registry.execute(
                call.name,
                call.arguments,
                approval_token=approval_token,
                conversation_id=state.conversation_id,
            )
            self._append_tool_result(state.messages, call, result.result)
        except ApprovalError:
            with self._lock:
                self._pending[pending_id] = state
            raise
        except Exception as exc:
            self._append_tool_result(
                state.messages, call, {"error": type(exc).__name__}
            )
        state.remaining_calls.pop(0)
        state.executed_count += 1
        if state.remaining_calls:
            outcome = self._process_calls(
                state.messages,
                state.remaining_calls,
                state.tool_schemas,
                state.response_parts,
                state.conversation_id,
                state.executed_count,
            )
            if isinstance(outcome, AgentTurnResult):
                return outcome
            messages, response_parts, executed_count = outcome
            return self._continue(
                messages,
                state.tool_schemas,
                response_parts,
                state.conversation_id,
                executed_count,
            )
        return self._continue(
            state.messages,
            state.tool_schemas,
            state.response_parts,
            state.conversation_id,
            state.executed_count,
        )

    def _continue(
        self,
        messages: list[dict[str, Any]],
        tool_schemas: list[dict[str, Any]],
        response_parts: list[str],
        conversation_id: str | None,
        executed_count: int,
    ) -> AgentTurnResult:
        while executed_count < self.max_tool_calls:
            response = self.llm.chat_with_tools(messages, tool_schemas)
            if response.content:
                response_parts.append(response.content)
            if not response.tool_calls:
                return AgentTurnResult(
                    "completed", "".join(response_parts).strip(), None, executed_count
                )
            messages.append(
                self._assistant_tool_message(response.content, response.tool_calls)
            )
            outcome = self._process_calls(
                messages,
                list(response.tool_calls),
                tool_schemas,
                response_parts,
                conversation_id,
                executed_count,
            )
            if isinstance(outcome, AgentTurnResult):
                return outcome
            messages, response_parts, executed_count = outcome
        return AgentTurnResult(
            "max_tool_calls",
            "".join(response_parts).strip(),
            None,
            executed_count,
        )

    def _process_calls(
        self,
        messages: list[dict[str, Any]],
        calls: list[OllamaToolCall],
        tool_schemas: list[dict[str, Any]],
        response_parts: list[str],
        conversation_id: str | None,
        executed_count: int,
    ) -> tuple[list[dict[str, Any]], list[str], int] | AgentTurnResult:
        for index, call in enumerate(calls):
            if executed_count >= self.max_tool_calls:
                return AgentTurnResult(
                    "max_tool_calls",
                    "".join(response_parts).strip(),
                    None,
                    executed_count,
                )
            descriptor = next(
                (
                    tool
                    for tool in self.registry.available_tools()
                    if tool.name == call.name
                ),
                None,
            )
            if descriptor is None:
                self._append_tool_result(
                    messages, call, {"error": "Unknown or unavailable tool"}
                )
                executed_count += 1
                continue
            if descriptor.requires_approval:
                pending_id = secrets.token_urlsafe(24)
                pending_state = _PendingState(
                    messages=messages,
                    remaining_calls=calls[index:],
                    tool_schemas=tool_schemas,
                    response_parts=response_parts,
                    conversation_id=conversation_id,
                    executed_count=executed_count,
                    expires_at=self._clock() + self.pending_ttl_seconds,
                )
                with self._lock:
                    now = self._clock()
                    expired = [
                        key
                        for key, state in self._pending.items()
                        if now >= state.expires_at
                    ]
                    for key in expired:
                        self._pending.pop(key, None)
                    self._pending[pending_id] = pending_state
                return AgentTurnResult(
                    "approval_required",
                    "".join(response_parts).strip(),
                    PendingAgentTool(
                        pending_id, call.name, json.loads(json.dumps(call.arguments))
                    ),
                    executed_count,
                )
            try:
                result = self.registry.execute(
                    call.name,
                    call.arguments,
                    conversation_id=conversation_id,
                )
                self._append_tool_result(messages, call, result.result)
            except Exception as exc:
                self._append_tool_result(
                    messages, call, {"error": type(exc).__name__}
                )
            executed_count += 1
        return messages, response_parts, executed_count

    def _get_pending(self, pending_id: str) -> _PendingState:
        with self._lock:
            state = self._pending.get(pending_id)
            if state is None or self._clock() >= state.expires_at:
                self._pending.pop(pending_id, None)
                raise ApprovalError("Pending agent operation is missing or expired.")
            return state

    @staticmethod
    def _append_tool_result(
        messages: list[dict[str, Any]],
        call: OllamaToolCall,
        result: Any,
    ) -> None:
        messages.append(
            {
                "role": "tool",
                "tool_name": call.name,
                "content": json.dumps(result, ensure_ascii=False, default=str),
            }
        )

    @staticmethod
    def _assistant_tool_message(
        content: str, calls: Sequence[OllamaToolCall]
    ) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": content,
            "tool_calls": [
                {
                    "function": {"name": call.name, "arguments": call.arguments}
                }
                for call in calls
            ],
        }

    @staticmethod
    def _ollama_schema(descriptor: Any) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": descriptor.name,
                "description": descriptor.description,
                "parameters": descriptor.arguments_schema,
            },
        }
