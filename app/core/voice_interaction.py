"""End-to-end coordination of one local speech-to-speech turn."""

from __future__ import annotations

import re
import threading
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

from app.services.ollama_service import ChatMessage
from app.services.stt_service import TranscriptionResult
from app.memory.conversation_store import Conversation, StoredMessage
from app.memory.memory_manager import ConversationContext

InteractionStatus = Literal["completed", "interrupted", "no_speech", "no_transcript"]


class AudioCapture(Protocol):
    """Capture one VAD-delimited utterance."""

    def capture_utterance(
        self, stop_event: threading.Event | None = None
    ) -> np.ndarray | None:
        """Return mono audio, or ``None`` if no qualifying speech was captured."""
        ...


class SpeechRecognizer(Protocol):
    """Transcribe local mono audio."""

    def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        """Return transcript and detected-language information."""
        ...


class ResponseGenerator(Protocol):
    """Generate an assistant response from local conversation context."""

    def stream_chat(
        self,
        prompt: str,
        history: Sequence[ChatMessage] = (),
        stop_event: threading.Event | None = None,
    ) -> Iterator[str]:
        """Yield response text deltas."""
        ...


class SpeechOutput(Protocol):
    """Speak response text through the local TTS and playback services."""

    def speak(
        self, text: str, stop_event: threading.Event | None = None
    ) -> None:
        """Synthesize and play response text, optionally interruptibly."""
        ...


class ConversationMemory(Protocol):
    """Persist and retrieve local conversation turns."""

    def start_conversation(self, conversation_id: str | None = None) -> Conversation:
        """Create a new conversation or restore the requested one."""
        ...

    def load_context(self, conversation_id: str) -> ConversationContext:
        """Return bounded prior context for the local model."""
        ...

    def append_message(
        self, conversation_id: str, role: str, content: str
    ) -> StoredMessage:
        """Persist a conversation message."""
        ...


@dataclass(frozen=True)
class VoiceInteractionResult:
    """Result of one completed, interrupted, or empty voice turn."""

    status: InteractionStatus
    transcript: str
    response: str
    language: str | None = None
    conversation_id: str | None = None


class _SentenceBuffer:
    """Release complete response sentences while limiting unpunctuated text."""

    _BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")

    def __init__(self, maximum_characters: int) -> None:
        if maximum_characters < 1:
            raise ValueError("maximum_characters must be at least one.")
        self.maximum_characters = maximum_characters
        self._pending = ""

    def feed(self, text: str, flush: bool = False) -> list[str]:
        """Add a response delta and return speakable sentence-sized pieces."""
        self._pending += text
        pieces: list[str] = []
        while True:
            match = self._BOUNDARY.search(self._pending)
            if match is None:
                break
            sentence = self._pending[: match.start()].strip()
            self._pending = self._pending[match.end() :]
            if sentence:
                pieces.append(sentence)

        while len(self._pending) > self.maximum_characters:
            split_at = self._pending.rfind(" ", 0, self.maximum_characters + 1)
            if split_at < 1:
                split_at = self.maximum_characters
            piece = self._pending[:split_at].strip()
            self._pending = self._pending[split_at:].lstrip()
            if piece:
                pieces.append(piece)

        if flush and self._pending.strip():
            pieces.append(self._pending.strip())
            self._pending = ""
        return pieces


class VoiceInteractionService:
    """Run one user-activated, offline speech-to-speech conversation turn."""

    def __init__(
        self,
        capture: AudioCapture,
        recognizer: SpeechRecognizer,
        response_generator: ResponseGenerator,
        speech_output: SpeechOutput,
        *,
        maximum_response_chunk_characters: int = 500,
        memory: ConversationMemory | None = None,
    ) -> None:
        self.capture = capture
        self.recognizer = recognizer
        self.response_generator = response_generator
        self.speech_output = speech_output
        self.maximum_response_chunk_characters = maximum_response_chunk_characters
        self.memory = memory
        if maximum_response_chunk_characters < 1:
            raise ValueError(
                "maximum_response_chunk_characters must be at least one."
            )
        self._turn_lock = threading.Lock()

    def run_turn(
        self,
        history: Sequence[ChatMessage] | None = None,
        conversation_id: str | None = None,
        stop_event: threading.Event | None = None,
    ) -> VoiceInteractionResult:
        """Capture, transcribe, generate, and speak one local assistant turn."""
        if not self._turn_lock.acquire(blocking=False):
            raise RuntimeError("A voice interaction is already in progress.")
        try:
            return self._run_turn(history, conversation_id, stop_event)
        finally:
            self._turn_lock.release()

    def _run_turn(
        self,
        history: Sequence[ChatMessage] | None,
        conversation_id: str | None,
        stop_event: threading.Event | None,
    ) -> VoiceInteractionResult:
        audio = self.capture.capture_utterance(stop_event=stop_event)
        if stop_event is not None and stop_event.is_set():
            return VoiceInteractionResult("interrupted", "", "")
        if audio is None:
            return VoiceInteractionResult("no_speech", "", "")

        transcription = self.recognizer.transcribe(audio)
        transcript = transcription.text.strip()
        if stop_event is not None and stop_event.is_set():
            return VoiceInteractionResult(
                "interrupted",
                transcript,
                "",
                transcription.language,
                conversation_id,
            )
        if not transcript:
            return VoiceInteractionResult(
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

        response_parts: list[str] = []
        sentence_buffer = _SentenceBuffer(
            self.maximum_response_chunk_characters
        )
        for delta in self.response_generator.stream_chat(
            transcript, history=effective_history, stop_event=stop_event
        ):
            if stop_event is not None and stop_event.is_set():
                break
            response_parts.append(delta)
            for sentence in sentence_buffer.feed(delta):
                self.speech_output.speak(sentence, stop_event=stop_event)
                if stop_event is not None and stop_event.is_set():
                    break
            if stop_event is not None and stop_event.is_set():
                break

        if stop_event is None or not stop_event.is_set():
            for sentence in sentence_buffer.feed("", flush=True):
                self.speech_output.speak(sentence, stop_event=stop_event)

        response = "".join(response_parts).strip()
        if self.memory is not None and response:
            self.memory.append_message(conversation_id, "assistant", response)

        return VoiceInteractionResult(
            status=(
                "interrupted"
                if stop_event is not None and stop_event.is_set()
                else "completed"
            ),
            transcript=transcript,
            response=response,
            language=transcription.language,
            conversation_id=conversation_id,
        )
