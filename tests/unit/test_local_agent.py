from __future__ import annotations

from pathlib import Path

import pytest

from app.core.agent import LocalAgent
from app.core.permissions import ApprovalError
from app.core.tool_registry import ToolRegistry
from app.memory.conversation_store import ConversationStore
from app.services.ollama_service import (
    ChatMessage,
    OllamaToolCall,
    OllamaToolResponse,
)
from app.tools.workspace import Workspace
from app.tools.workspace_tools import WorkspaceTools


class FakeLLM:
    def __init__(self, responses: list[OllamaToolResponse]) -> None:
        self.responses = list(responses)
        self.requests = []

    def build_messages(self, history, prompt, *, system_prompt=None):
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(
            {"role": item.role, "content": item.content} for item in history
        )
        return messages + [{"role": "user", "content": prompt}]

    def chat_with_tools(self, messages, tools):
        self.requests.append((list(messages), list(tools)))
        return self.responses.pop(0)


def make_registry(tmp_path: Path, *, allow_code_execution: bool = False):
    root = tmp_path / "workspace"
    root.mkdir(exist_ok=True)
    store = ConversationStore(":memory:")
    workspace_tools = WorkspaceTools(
        Workspace([root]),
        summarize_text=lambda _text: "summary",
        generate_text=lambda _text: "generated code",
        explain_text=lambda _text: "explanation",
        allow_code_execution=allow_code_execution,
    )
    return ToolRegistry(workspace_tools, audit_store=store), root, store


def test_agent_executes_registered_read_tool_then_returns_final_answer(
    tmp_path: Path,
) -> None:
    registry, _, store = make_registry(tmp_path)
    llm = FakeLLM(
        [
            OllamaToolResponse(
                "",
                (
                    OllamaToolCall(
                        "get_current_time", {"timezone": "UTC"}, "call-1"
                    ),
                ),
            ),
            OllamaToolResponse("It is currently 12:00 UTC.", ()),
        ]
    )
    agent = LocalAgent(llm, registry)

    result = agent.run(
        "What time is it?",
        history=[ChatMessage("assistant", "Hello")],
    )

    assert result.status == "completed"
    assert result.response == "It is currently 12:00 UTC."
    assert result.tool_calls_executed == 1
    first_messages, offered_tools = llm.requests[0]
    assert first_messages[0]["content"] == "Hello"
    assert "execute_python_code" not in {
        tool["function"]["name"] for tool in offered_tools
    }
    second_messages, _ = llm.requests[1]
    assert second_messages[-1]["role"] == "tool"
    assert second_messages[-1]["tool_name"] == "get_current_time"
    assert store.list_tool_audit()[0].status == "succeeded"
    store.close()


def test_agent_pauses_for_write_approval_and_resumes_after_confirmation(
    tmp_path: Path,
) -> None:
    registry, root, store = make_registry(tmp_path)
    llm = FakeLLM(
        [
            OllamaToolResponse(
                "",
                (
                    OllamaToolCall(
                        "write_file",
                        {"path": "notes.txt", "content": "approved text"},
                        "call-write",
                    ),
                ),
            ),
            OllamaToolResponse("The file is written.", ()),
        ]
    )
    agent = LocalAgent(llm, registry)

    pending = agent.run("Write a note")

    assert pending.status == "approval_required"
    assert pending.pending_tool is not None
    assert not (root / "notes.txt").exists()
    pending.pending_tool.arguments["content"] = "tampered after display"
    approval_request = agent.prepare_pending_approval(
        pending.pending_tool.pending_id
    )
    assert approval_request.arguments["content"] == "approved text"
    approval_request.arguments["content"] = "tampered in approval UI"
    token = agent.confirm_pending(
        pending.pending_tool.pending_id, approval_request.request_id
    )
    final = agent.resume(pending.pending_tool.pending_id, token)

    assert final.status == "completed"
    assert (root / "notes.txt").read_text(encoding="utf-8") == "approved text"
    assert final.tool_calls_executed == 1
    store.close()


def test_agent_keeps_pending_operation_after_invalid_approval_token(
    tmp_path: Path,
) -> None:
    registry, root, store = make_registry(tmp_path)
    llm = FakeLLM(
        [
            OllamaToolResponse(
                "",
                (
                    OllamaToolCall(
                        "write_file", {"path": "x.txt", "content": "x"}, "call"
                    ),
                ),
            ),
            OllamaToolResponse("Done.", ()),
        ]
    )
    agent = LocalAgent(llm, registry)
    pending = agent.run("Write x")
    pending_id = pending.pending_tool.pending_id

    with pytest.raises(ApprovalError):
        agent.resume(pending_id, "invalid-token")
    approval_request = agent.prepare_pending_approval(pending_id)
    token = agent.confirm_pending(pending_id, approval_request.request_id)
    result = agent.resume(pending_id, token)

    assert result.status == "completed"
    assert (root / "x.txt").read_text(encoding="utf-8") == "x"
    store.close()


def test_agent_rejects_unknown_tool_names_without_execution(tmp_path: Path) -> None:
    registry, _, store = make_registry(tmp_path)
    llm = FakeLLM(
        [
            OllamaToolResponse(
                "",
                (
                    OllamaToolCall(
                        "shell", {"command": "whoami"}, "not-registered"
                    ),
                ),
            ),
            OllamaToolResponse("I could not use that tool.", ()),
        ]
    )
    result = LocalAgent(llm, registry).run("Run shell")

    assert result.status == "completed"
    assert llm.requests[1][0][-1]["role"] == "tool"
    assert "Unknown or unavailable" in llm.requests[1][0][-1]["content"]
    assert store.list_tool_audit() == ()
    store.close()


def test_agent_stops_after_configured_maximum_tool_calls(tmp_path: Path) -> None:
    registry, _, store = make_registry(tmp_path)
    call = OllamaToolCall("get_current_time", {"timezone": "UTC"}, "call")
    llm = FakeLLM(
        [
            OllamaToolResponse("", (call,)),
            OllamaToolResponse("", (call,)),
        ]
    )
    result = LocalAgent(llm, registry, max_tool_calls=1).run("time")

    assert result.status == "max_tool_calls"
    assert result.tool_calls_executed == 1
    store.close()
