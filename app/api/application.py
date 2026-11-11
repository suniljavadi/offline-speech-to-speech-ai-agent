"""FastAPI application factory for injected local assistant services."""

from __future__ import annotations

import ipaddress
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from app.core.coding_assistant import VoiceCodingAssistant
from app.core.permissions import ApprovalError
from app.core.voice_interaction import VoiceInteractionService
from app.memory.conversation_store import (
    ConversationNotFoundError,
    ConversationStoreError,
    ConversationStore,
)
from app.services.audio_service import AudioCaptureError
from app.services.ollama_service import OllamaError
from app.services.stt_service import TranscriptionError
from app.services.tts_service import TextToSpeechError
from app.core.tool_registry import ToolRegistryError

from .schemas import (
    AgentTurnResponse,
    ApprovalConfirmationRequest,
    ApprovalRequestResponse,
    ApprovalResumeRequest,
    ApprovalTokenResponse,
    CodingTurnResponse,
    ConversationDetailResponse,
    ConversationListResponse,
    ConversationResponse,
    DeleteAllConversationsResponse,
    DeleteConversationResponse,
    PendingApprovalRequest,
    VoiceTurnRequest,
    VoiceTurnResponse,
)

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


def _is_loopback_host(host: str | None) -> bool:
    if not host:
        return False
    if host.casefold() in _LOCAL_HOSTS:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _is_loopback_origin(origin: str) -> bool:
    try:
        parsed = urlsplit(origin)
        return parsed.scheme in {"http", "https"} and _is_loopback_host(
            parsed.hostname
        )
    except ValueError:
        return False


def create_app(
    voice_interaction: VoiceInteractionService,
    coding_assistant: VoiceCodingAssistant,
    conversation_store: ConversationStore,
) -> FastAPI:
    """Create an API for local services without opening hardware at import time."""
    app = FastAPI(
        title="Offline Speech-to-Speech AI Agent",
        version="0.1.0",
        description="Loopback-only API for local voice and coding assistance.",
    )

    @app.middleware("http")
    async def restrict_to_loopback(request: Request, call_next):
        if not _is_loopback_host(request.url.hostname):
            return JSONResponse(
                status_code=400, content={"detail": "A loopback Host is required."}
            )
        if request.client is not None and not _is_loopback_host(request.client.host):
            return JSONResponse(
                status_code=403,
                content={"detail": "This service accepts loopback clients only."},
            )
        origin = request.headers.get("origin")
        if origin is not None and not _is_loopback_origin(origin):
            return JSONResponse(
                status_code=403,
                content={"detail": "Only loopback browser origins are allowed."},
            )
        return await call_next(request)

    @app.exception_handler(ApprovalError)
    async def approval_error_handler(_request: Request, exc: ApprovalError):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(ConversationNotFoundError)
    async def conversation_not_found_handler(
        _request: Request, exc: ConversationNotFoundError
    ):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    async def local_service_error_handler(_request: Request, exc: Exception):
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    for error_type in (
        AudioCaptureError,
        OllamaError,
        TextToSpeechError,
        ToolRegistryError,
        TranscriptionError,
    ):
        app.add_exception_handler(error_type, local_service_error_handler)

    @app.exception_handler(ConversationStoreError)
    async def conversation_store_error_handler(
        _request: Request, exc: ConversationStoreError
    ):
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "inference": "local"}

    @app.post("/voice/turn", response_model=VoiceTurnResponse)
    def run_voice_turn(payload: VoiceTurnRequest) -> VoiceTurnResponse:
        result = voice_interaction.run_turn(
            conversation_id=payload.conversation_id
        )
        return VoiceTurnResponse.from_result(result)

    @app.post("/coding/turn", response_model=CodingTurnResponse)
    def run_coding_turn(payload: VoiceTurnRequest) -> CodingTurnResponse:
        result = coding_assistant.run_turn(
            conversation_id=payload.conversation_id
        )
        return CodingTurnResponse.from_result(result)

    @app.post(
        "/coding/approval/prepare",
        response_model=ApprovalRequestResponse,
    )
    def prepare_coding_approval(
        payload: PendingApprovalRequest,
    ) -> ApprovalRequestResponse:
        request = coding_assistant.prepare_pending_approval(payload.pending_id)
        return ApprovalRequestResponse.from_request(request)

    @app.post(
        "/coding/approval/confirm",
        response_model=ApprovalTokenResponse,
    )
    def confirm_coding_approval(
        payload: ApprovalConfirmationRequest,
    ) -> ApprovalTokenResponse:
        token = coding_assistant.confirm_pending(
            payload.pending_id, payload.request_id
        )
        return ApprovalTokenResponse(approval_token=token)

    @app.post(
        "/coding/approval/resume",
        response_model=AgentTurnResponse,
    )
    def resume_coding_approval(
        payload: ApprovalResumeRequest,
    ) -> AgentTurnResponse:
        result = coding_assistant.resume_pending(
            payload.pending_id, payload.approval_token
        )
        return AgentTurnResponse.from_result(result)

    @app.get("/conversations", response_model=ConversationListResponse)
    def list_conversations(
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
    ) -> ConversationListResponse:
        return ConversationListResponse(
            conversations=[
                ConversationResponse.from_conversation(conversation)
                for conversation in conversation_store.list_conversations(
                    limit=limit, offset=offset
                )
            ]
        )

    @app.get(
        "/conversations/{conversation_id}",
        response_model=ConversationDetailResponse,
    )
    def get_conversation(
        conversation_id: str,
        limit: int = Query(default=200, ge=1, le=1000),
    ) -> ConversationDetailResponse:
        return ConversationDetailResponse.from_snapshot(
            conversation_store.get_conversation(conversation_id, message_limit=limit)
        )

    @app.delete(
        "/conversations/{conversation_id}",
        response_model=DeleteConversationResponse,
    )
    def delete_conversation(conversation_id: str) -> DeleteConversationResponse:
        deleted = conversation_store.delete_conversation(conversation_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Conversation not found.")
        return DeleteConversationResponse(deleted=True)

    @app.delete(
        "/conversations",
        response_model=DeleteAllConversationsResponse,
    )
    def delete_all_conversations() -> DeleteAllConversationsResponse:
        return DeleteAllConversationsResponse(
            deleted_count=conversation_store.delete_all_conversations()
        )

    return app
