from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from app.api.application import create_app
from app.core.agent import AgentTurnResult, PendingAgentTool
from app.core.coding_assistant import VoiceCodingTurnResult
from app.core.permissions import ApprovalError, ApprovalRequest
from app.core.voice_interaction import VoiceInteractionResult
from app.memory.conversation_store import ConversationStore
from app.services.audio_service import AudioCaptureError
from app.ui.streamlit_app import APIError, _api_base_url


class FakeVoiceInteraction:
    def __init__(self) -> None:
        self.calls = []

    def run_turn(self, *, conversation_id=None):
        self.calls.append(conversation_id)
        return VoiceInteractionResult(
            "completed", "recognized request", "spoken answer", "en",
            conversation_id or "voice-session",
        )


class FailingVoiceInteraction(FakeVoiceInteraction):
    def run_turn(self, *, conversation_id=None):
        raise AudioCaptureError("Check microphone access and audio permissions.")


class FakeCodingAssistant:
    def __init__(self) -> None:
        self.confirmed = []

    def run_turn(self, *, conversation_id=None):
        pending = PendingAgentTool(
            "pending-123",
            "write_file",
            {"path": "notes.txt", "content": "approved text"},
        )
        return VoiceCodingTurnResult(
            "approval_required",
            "Write a note",
            "Review the exact change before it runs.",
            "en",
            conversation_id or "coding-session",
            AgentTurnResult("approval_required", "", pending, 0),
        )

    def prepare_pending_approval(self, pending_id):
        assert pending_id == "pending-123"
        return ApprovalRequest(
            "request-123",
            "write_file",
            {"path": "notes.txt", "content": "approved text", "overwrite": False},
            120,
        )

    def confirm_pending(self, pending_id, request_id):
        if pending_id != "pending-123" or request_id != "request-123":
            raise ApprovalError("Approval request does not match the pending operation.")
        self.confirmed.append((pending_id, request_id))
        return "token-123"

    def resume_pending(self, pending_id, token):
        assert pending_id == "pending-123"
        assert token == "token-123"
        return AgentTurnResult("completed", "The note was written.", None, 1)


def make_client(tmp_path: Path):
    store = ConversationStore(tmp_path / "conversations.sqlite3")
    voice = FakeVoiceInteraction()
    coding = FakeCodingAssistant()
    app = create_app(voice, coding, store)
    return (
        TestClient(
            app,
            base_url="http://127.0.0.1",
            client=("127.0.0.1", 50000),
        ),
        voice,
        coding,
        store,
    )


def test_health_and_voice_route_are_local_and_forward_session(tmp_path: Path) -> None:
    client, voice, _coding, store = make_client(tmp_path)

    assert client.get("/health").json() == {
        "status": "ok",
        "inference": "local",
    }
    response = client.post(
        "/voice/turn", json={"conversation_id": "existing-session"}
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "completed",
        "transcript": "recognized request",
        "response": "spoken answer",
        "language": "en",
        "conversation_id": "existing-session",
    }
    assert voice.calls == ["existing-session"]
    store.close()


def test_coding_api_exposes_exact_approval_then_resume_flow(tmp_path: Path) -> None:
    client, _voice, coding, store = make_client(tmp_path)

    turn = client.post("/coding/turn", json={})
    assert turn.status_code == 200
    assert turn.json()["status"] == "approval_required"
    pending = turn.json()["pending_tool"]
    assert pending["tool_name"] == "write_file"

    prepared = client.post(
        "/coding/approval/prepare",
        json={"pending_id": pending["pending_id"]},
    )
    assert prepared.status_code == 200
    assert prepared.json()["arguments"] == {
        "path": "notes.txt",
        "content": "approved text",
        "overwrite": False,
    }

    rejected = client.post(
        "/coding/approval/confirm",
        json={"pending_id": "pending-123", "request_id": "wrong"},
    )
    assert rejected.status_code == 409
    confirmed = client.post(
        "/coding/approval/confirm",
        json={
            "pending_id": "pending-123",
            "request_id": prepared.json()["request_id"],
        },
    )
    assert confirmed.status_code == 200
    resumed = client.post(
        "/coding/approval/resume",
        json={
            "pending_id": "pending-123",
            "approval_token": confirmed.json()["approval_token"],
        },
    )
    assert resumed.status_code == 200
    assert resumed.json()["response"] == "The note was written."
    assert coding.confirmed == [("pending-123", "request-123")]
    store.close()


def test_conversation_routes_list_get_and_delete_local_history(
    tmp_path: Path,
) -> None:
    client, _voice, _coding, store = make_client(tmp_path)
    conversation = store.create_conversation("session-a")
    store.add_message(conversation.id, "user", "question")
    store.add_message(conversation.id, "assistant", "answer")

    listed = client.get("/conversations?limit=10&offset=0")
    detail = client.get("/conversations/session-a")
    deleted = client.delete("/conversations/session-a")

    assert listed.status_code == 200
    assert [item["id"] for item in listed.json()["conversations"]] == ["session-a"]
    assert [item["content"] for item in detail.json()["messages"]] == [
        "question",
        "answer",
    ]
    assert deleted.json() == {"deleted": True}
    assert client.get("/conversations/session-a").status_code == 404
    store.close()


def test_api_rejects_non_loopback_host_origin_and_unknown_fields(
    tmp_path: Path,
) -> None:
    client, _voice, _coding, store = make_client(tmp_path)

    assert client.get("/health", headers={"host": "example.com"}).status_code == 400
    assert client.get(
        "/health",
        headers={"origin": "https://example.com"},
    ).status_code == 403
    assert client.post(
        "/voice/turn",
        json={"unexpected": True},
    ).status_code == 422
    store.close()


def test_api_rejects_non_loopback_client_address(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "conversations.sqlite3")
    client = TestClient(
        create_app(FakeVoiceInteraction(), FakeCodingAssistant(), store),
        base_url="http://127.0.0.1",
        client=("192.0.2.1", 50000),
    )

    assert client.get("/health").status_code == 403
    store.close()


def test_api_returns_actionable_local_service_error(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "conversations.sqlite3")
    client = TestClient(
        create_app(
            FailingVoiceInteraction(), FakeCodingAssistant(), store
        ),
        base_url="http://127.0.0.1",
        client=("127.0.0.1", 50000),
    )

    response = client.post("/voice/turn", json={})

    assert response.status_code == 503
    assert "microphone access" in response.json()["detail"]
    store.close()


def test_streamlit_api_url_must_be_loopback(monkeypatch) -> None:
    monkeypatch.setenv("APP_API_URL", "http://127.0.0.1:8000")
    assert _api_base_url() == "http://127.0.0.1:8000"

    monkeypatch.setenv("APP_API_URL", "http://api.example.com")
    with pytest.raises(APIError, match="loopback"):
        _api_base_url()

    monkeypatch.setenv("APP_API_URL", "http://localhost:invalid")
    with pytest.raises(APIError, match="invalid"):
        _api_base_url()
