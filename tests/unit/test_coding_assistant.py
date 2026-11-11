from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest

from app.core.agent import LocalAgent
from app.core.coding_assistant import CODING_SYSTEM_PROMPT, VoiceCodingAssistant
from app.core.permissions import ApprovalError
from app.core.tool_registry import ToolRegistry
from app.memory.conversation_store import ConversationStore
from app.memory.memory_manager import ConversationMemoryManager
from app.services.ollama_service import OllamaToolCall, OllamaToolResponse
from app.services.stt_service import TranscriptionResult
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


class FakeCapture:
    def __init__(self, audio: np.ndarray | None = None) -> None:
        self.audio = np.zeros(16, dtype=np.float32) if audio is None else audio

    def capture_utterance(self, stop_event=None):
        if stop_event is not None and stop_event.is_set():
            return None
        return self.audio


class FakeRecognizer:
    def __init__(self, transcript: str) -> None:
        self.transcript = transcript

    def transcribe(self, audio):
        return TranscriptionResult(self.transcript, "en", 0.99, 1.0, 1)


class FakeSpeechOutput:
    def __init__(self) -> None:
        self.spoken = []

    def speak(self, text: str, stop_event=None) -> None:
        self.spoken.append(text)


def make_assistant(
    tmp_path: Path,
    responses: list[OllamaToolResponse],
    *,
    transcript: str = "Explain this project",
    with_memory: bool = False,
):
    root = tmp_path / "workspace"
    root.mkdir(exist_ok=True)
    store = ConversationStore(":memory:")
    workspace_tools = WorkspaceTools(
        Workspace([root]),
        summarize_text=lambda _text: "summary",
        generate_text=lambda _text: "generated code",
        explain_text=lambda _text: "explanation",
    )
    registry = ToolRegistry(workspace_tools, audit_store=store)
    llm = FakeLLM(responses)
    output = FakeSpeechOutput()
    memory = ConversationMemoryManager(store) if with_memory else None
    assistant = VoiceCodingAssistant(
        FakeCapture(),
        FakeRecognizer(transcript),
        LocalAgent(llm, registry),
        output,
        memory=memory,
    )
    return assistant, root, llm, output, store, memory


def test_voice_coding_turn_uses_coding_prompt_and_read_only_file_tool(
    tmp_path: Path,
) -> None:
    assistant, root, llm, output, store, _ = make_assistant(
        tmp_path,
        [
            OllamaToolResponse(
                "",
                (
                    OllamaToolCall(
                        "read_file",
                        {"path": "app.py", "max_characters": 1000},
                        "read-source",
                    ),
                ),
            ),
            OllamaToolResponse("The module defines the application entry point.", ()),
        ],
    )
    (root / "app.py").write_text("def main(): pass\n", encoding="utf-8")

    result = assistant.run_turn()

    assert result.status == "completed"
    assert result.transcript == "Explain this project"
    assert result.response == "The module defines the application entry point."
    assert output.spoken == [result.response]
    assert llm.requests[0][0][0] == {
        "role": "system",
        "content": CODING_SYSTEM_PROMPT,
    }
    assert "write_file" in {
        tool["function"]["name"] for tool in llm.requests[0][1]
    }
    assert store.list_tool_audit()[0].status == "succeeded"
    store.close()


def test_voice_coding_change_waits_for_exact_user_approval_and_persists(
    tmp_path: Path,
) -> None:
    assistant, root, llm, output, store, memory = make_assistant(
        tmp_path,
        [
            OllamaToolResponse(
                "",
                (
                    OllamaToolCall(
                        "write_file",
                        {"path": "fix.py", "content": "value = 1\n"},
                        "write-source",
                    ),
                ),
            ),
            OllamaToolResponse("The requested file was updated.", ()),
        ],
        transcript="Create fix.py with value set to one",
        with_memory=True,
    )

    pending = assistant.run_turn()

    assert pending.status == "approval_required"
    assert pending.agent_result is not None
    pending_id = pending.agent_result.pending_tool.pending_id
    assert not (root / "fix.py").exists()
    request = assistant.prepare_pending_approval(pending_id)
    assert request.arguments == {
        "path": "fix.py",
        "content": "value = 1\n",
        "overwrite": False,
    }

    with pytest.raises(ApprovalError):
        assistant.confirm_pending(pending_id, "wrong-request")
    token = assistant.confirm_pending(pending_id, request.request_id)
    completed = assistant.resume_pending(pending_id, token)

    assert completed.status == "completed"
    assert (root / "fix.py").read_text(encoding="utf-8") == "value = 1\n"
    assert output.spoken[-1] == "The requested file was updated."
    snapshot = memory.restore(pending.conversation_id)
    assert [message.role for message in snapshot.messages] == ["user", "assistant"]
    store.close()


def test_voice_coding_turn_reports_empty_transcript_without_calling_agent(
    tmp_path: Path,
) -> None:
    assistant, _, llm, output, store, _ = make_assistant(
        tmp_path, [], transcript="   "
    )

    result = assistant.run_turn()

    assert result.status == "no_transcript"
    assert llm.requests == []
    assert output.spoken == []
    store.close()


def test_voice_coding_turn_stops_before_transcription_if_cancelled(
    tmp_path: Path,
) -> None:
    assistant, _, llm, output, store, _ = make_assistant(tmp_path, [])
    stop_event = threading.Event()
    stop_event.set()

    result = assistant.run_turn(stop_event=stop_event)

    assert result.status == "interrupted"
    assert llm.requests == []
    assert output.spoken == []
    store.close()
