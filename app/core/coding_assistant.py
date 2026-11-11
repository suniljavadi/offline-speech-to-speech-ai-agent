"""Voice-driven coding assistance using workspace-scoped agent tools."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from app.core.agent import AgentTurnResult, LocalAgent
from app.core.permissions import ApprovalRequest
from app.core.voice_interaction import (
    AudioCapture,
    ConversationMemory,
    SpeechOutput,
    SpeechRecognizer,
)
from app.services.ollama_service import ChatMessage

CODING_SYSTEM_PROMPT = (
    "You are a local voice coding assistant working only in the configured "
    "workspace. For coding questions, inspect relevant project context and source "
    "before answering; prefer read-only search, file reading, and code explanation. "
    "Do not modify files unless the user explicitly requested a change. Any file "
    "write requires the application's explicit user approval; describe the exact "
    "change and never imply it happened before the write tool succeeds. Use code "
    "generation for proposals when appropriate, but do not claim generated code "
    "was tested. Python and pytest execution may be unavailable; never attempt "
    "execution unless the registered tool is available, and do not claim tests "
    "passed unless the test tool actually succeeded. Never use or invent shell "
    "commands or tools outside the registered workspace tools. Be concise when "
    "speaking, explain uncertainty, and do not repeat large source excerpts."
)

CodingTurnStatus = Literal[
    "completed",
    "approval_required",
    "max_tool_calls",
    "interrupted",
    "no_speech",
    "no_transcript",
]


@dataclass(frozen=True)
class VoiceCodingTurnResult:
    """Outcome of one coding request captured through the microphone."""

    status: CodingTurnStatus
    transcript: str
    response: str
    language: str | None = None
    conversation_id: str | None = None
    agent_result: AgentTurnResult | None = None


class VoiceCodingAssistant:
    """Capture a spoken coding request and route it through the local agent."""

    def __init__(
        self,
        capture: AudioCapture,
        recognizer: SpeechRecognizer,
        agent: LocalAgent,
        speech_output: SpeechOutput,
        *,
        memory: ConversationMemory | None = None,
    ) -> None:
        self.capture = capture
        self.recognizer = recognizer
        self.agent = agent
        self.speech_output = speech_output
        self.memory = memory
        self._turn_lock = threading.Lock()
        self._pending_lock = threading.Lock()
        self._pending_conversations: dict[str, str | None] = {}

    def run_turn(
        self,
        history: Sequence[ChatMessage] | None = None,
        conversation_id: str | None = None,
        stop_event: threading.Event | None = None,
    ) -> VoiceCodingTurnResult:
        """Capture, transcribe, handle, and speak one local coding request."""
        if not self._turn_lock.acquire(blocking=False):
            raise RuntimeError("A voice coding turn is already in progress.")
        try:
            audio = self.capture.capture_utterance(stop_event=stop_event)
            if stop_event is not None and stop_event.is_set():
                return VoiceCodingTurnResult("interrupted", "", "")
            if audio is None:
                return VoiceCodingTurnResult("no_speech", "", "")

            transcription = self.recognizer.transcribe(audio)
            transcript = transcription.text.strip()
            if stop_event is not None and stop_event.is_set():
                return VoiceCodingTurnResult(
                    "interrupted",
                    transcript,
                    "",
                    transcription.language,
                    conversation_id,
                )
            if not transcript:
                return VoiceCodingTurnResult(
                    "no_transcript",
                    "",
                    "",
                    transcription.language,
                    conversation_id,
                )

            effective_history = list(history or ())
            if self.memory is not None:
                conversation = self.memory.start_conversation(conversation_id)
                conversation_id = conversation.id
                context = self.memory.load_context(conversation_id)
                if history is None:
                    effective_history = list(context.history)
                self.memory.append_message(conversation_id, "user", transcript)

            result = self.agent.run(
                transcript,
                history=effective_history,
                conversation_id=conversation_id,
                system_prompt=CODING_SYSTEM_PROMPT,
            )
            self._track_pending(result, conversation_id)
            response = self._speak_result(result, stop_event)
            if result.status == "completed" and self.memory is not None and response:
                self.memory.append_message(conversation_id, "assistant", response)
            return VoiceCodingTurnResult(
                result.status,
                transcript,
                response,
                transcription.language,
                conversation_id,
                result,
            )
        finally:
            self._turn_lock.release()

    def prepare_pending_approval(self, pending_id: str) -> ApprovalRequest:
        """Return the exact proposed change for the confirmation interface."""
        return self.agent.prepare_pending_approval(pending_id)

    def confirm_pending(
        self, pending_id: str, approval_request_id: str
    ) -> str:
        """Confirm only the request currently displayed to the user."""
        return self.agent.confirm_pending(pending_id, approval_request_id)

    def resume_pending(
        self,
        pending_id: str,
        approval_token: str,
        stop_event: threading.Event | None = None,
    ) -> AgentTurnResult:
        """Resume an approved operation and speak the resulting agent response."""
        with self._pending_lock:
            conversation_id = self._pending_conversations.get(pending_id)
        result = self.agent.resume(pending_id, approval_token)
        with self._pending_lock:
            self._pending_conversations.pop(pending_id, None)
        self._track_pending(result, conversation_id)
        response = self._speak_result(result, stop_event)
        if result.status == "completed" and self.memory is not None and response:
            if conversation_id is not None:
                self.memory.append_message(
                    conversation_id, "assistant", response
                )
        return result

    def _track_pending(
        self, result: AgentTurnResult, conversation_id: str | None
    ) -> None:
        if result.status == "approval_required" and result.pending_tool is not None:
            with self._pending_lock:
                self._pending_conversations[
                    result.pending_tool.pending_id
                ] = conversation_id

    def _speak_result(
        self,
        result: AgentTurnResult,
        stop_event: threading.Event | None = None,
    ) -> str:
        response = result.response.strip()
        if result.status == "approval_required" and not response:
            response = "A change is ready for your review. Nothing has been changed yet."
        if response and (stop_event is None or not stop_event.is_set()):
            self.speech_output.speak(response, stop_event=stop_event)
        return response
