"""Validated request and response models for the local HTTP API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.core.agent import AgentTurnResult, PendingAgentTool
from app.core.coding_assistant import VoiceCodingTurnResult
from app.core.permissions import ApprovalRequest
from app.core.voice_interaction import VoiceInteractionResult
from app.memory.conversation_store import (
    Conversation,
    ConversationSnapshot,
    StoredMessage,
)


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VoiceTurnRequest(APIModel):
    conversation_id: str | None = Field(default=None, min_length=1, max_length=128)


class VoiceTurnResponse(APIModel):
    status: Literal["completed", "interrupted", "no_speech", "no_transcript"]
    transcript: str
    response: str
    language: str | None = None
    conversation_id: str | None = None

    @classmethod
    def from_result(cls, result: VoiceInteractionResult) -> VoiceTurnResponse:
        return cls(**result.__dict__)


class CodingTurnResponse(APIModel):
    status: Literal[
        "completed",
        "approval_required",
        "max_tool_calls",
        "interrupted",
        "no_speech",
        "no_transcript",
    ]
    transcript: str
    response: str
    language: str | None = None
    conversation_id: str | None = None
    pending_tool: PendingAgentTool | None = None

    @classmethod
    def from_result(cls, result: VoiceCodingTurnResult) -> CodingTurnResponse:
        return cls(
            status=result.status,
            transcript=result.transcript,
            response=result.response,
            language=result.language,
            conversation_id=result.conversation_id,
            pending_tool=(
                result.agent_result.pending_tool
                if result.agent_result is not None
                else None
            ),
        )


class PendingApprovalRequest(APIModel):
    pending_id: str = Field(min_length=1, max_length=256)


class ApprovalConfirmationRequest(APIModel):
    pending_id: str = Field(min_length=1, max_length=256)
    request_id: str = Field(min_length=1, max_length=256)


class ApprovalResumeRequest(APIModel):
    pending_id: str = Field(min_length=1, max_length=256)
    approval_token: str = Field(min_length=1, max_length=256)


class ApprovalRequestResponse(APIModel):
    request_id: str
    tool_name: str
    arguments: dict[str, Any]
    expires_in_seconds: float

    @classmethod
    def from_request(cls, request: ApprovalRequest) -> ApprovalRequestResponse:
        return cls(**request.__dict__)


class ApprovalTokenResponse(APIModel):
    approval_token: str


class AgentTurnResponse(APIModel):
    status: Literal["completed", "approval_required", "max_tool_calls"]
    response: str
    pending_tool: PendingAgentTool | None = None
    tool_calls_executed: int

    @classmethod
    def from_result(cls, result: AgentTurnResult) -> AgentTurnResponse:
        return cls(**result.__dict__)


class ConversationResponse(APIModel):
    id: str
    created_at: str
    updated_at: str
    summary: str | None
    summary_through_message_id: int | None
    revision: int

    @classmethod
    def from_conversation(cls, value: Conversation) -> ConversationResponse:
        return cls(**value.__dict__)


class StoredMessageResponse(APIModel):
    id: int
    role: Literal["user", "assistant", "tool"]
    content: str
    created_at: str

    @classmethod
    def from_message(cls, value: StoredMessage) -> StoredMessageResponse:
        return cls(
            id=value.id,
            role=value.role,
            content=value.content,
            created_at=value.created_at,
        )


class ConversationDetailResponse(APIModel):
    conversation: ConversationResponse
    messages: list[StoredMessageResponse]

    @classmethod
    def from_snapshot(
        cls, value: ConversationSnapshot
    ) -> ConversationDetailResponse:
        return cls(
            conversation=ConversationResponse.from_conversation(value.conversation),
            messages=[
                StoredMessageResponse.from_message(message)
                for message in value.messages
            ],
        )


class ConversationListResponse(APIModel):
    conversations: list[ConversationResponse]


class DeleteConversationResponse(APIModel):
    deleted: bool


class DeleteAllConversationsResponse(APIModel):
    deleted_count: int
