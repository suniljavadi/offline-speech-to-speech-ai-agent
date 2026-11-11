from __future__ import annotations

import threading

import numpy as np

from app.core.voice_interaction import VoiceInteractionService
from app.memory.conversation_store import ConversationStore
from app.memory.memory_manager import ConversationMemoryManager
from app.services.ollama_service import ChatMessage
from app.services.stt_service import TranscriptionResult


class FakeCapture:
    def __init__(self, audio: np.ndarray | None) -> None:
        self.audio = audio
        self.calls = 0

    def capture_utterance(self, stop_event=None):
        self.calls += 1
        return self.audio


class FakeRecognizer:
    def __init__(self, transcript: str = "Turn left at the next street.") -> None:
        self.transcript = transcript
        self.received = []

    def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        self.received.append(audio)
        return TranscriptionResult(
            self.transcript, "en", 0.99, audio.size / 16000, 1
        )


class FakeResponseGenerator:
    def __init__(self, deltas: list[str]) -> None:
        self.deltas = deltas
        self.requests = []

    def stream_chat(self, prompt, history=(), stop_event=None):
        self.requests.append((prompt, history))
        yield from self.deltas


class FakeSpeechOutput:
    def __init__(self, stop_after_first: bool = False) -> None:
        self.spoken = []
        self.stop_after_first = stop_after_first

    def speak(self, text, stop_event=None) -> None:
        self.spoken.append(text)
        if self.stop_after_first:
            stop_event.set()


def test_full_voice_turn_connects_capture_llm_and_sentence_playback() -> None:
    audio = np.ones(512, dtype=np.float32)
    capture = FakeCapture(audio)
    recognizer = FakeRecognizer()
    llm = FakeResponseGenerator(["Hello ", "there. How ", "can I help?"])
    playback = FakeSpeechOutput()
    history = [ChatMessage("assistant", "Previous answer")]
    service = VoiceInteractionService(capture, recognizer, llm, playback)

    result = service.run_turn(history=history)

    assert result.status == "completed"
    assert result.transcript == "Turn left at the next street."
    assert result.response == "Hello there. How can I help?"
    assert result.language == "en"
    assert llm.requests == [(result.transcript, history)]
    assert playback.spoken == ["Hello there.", "How can I help?"]


def test_silence_skips_transcription_and_response_services() -> None:
    capture = FakeCapture(None)
    recognizer = FakeRecognizer()
    llm = FakeResponseGenerator(["unused"])
    playback = FakeSpeechOutput()

    result = VoiceInteractionService(capture, recognizer, llm, playback).run_turn()

    assert result.status == "no_speech"
    assert recognizer.received == []
    assert llm.requests == []
    assert playback.spoken == []


def test_empty_transcript_does_not_send_empty_prompt_to_llm() -> None:
    service = VoiceInteractionService(
        FakeCapture(np.ones(100, dtype=np.float32)),
        FakeRecognizer("  "),
        FakeResponseGenerator(["unused"]),
        FakeSpeechOutput(),
    )

    result = service.run_turn()

    assert result.status == "no_transcript"
    assert result.response == ""


def test_interrupted_response_stops_speaking_later_sentences() -> None:
    stop_event = threading.Event()
    service = VoiceInteractionService(
        FakeCapture(np.ones(512, dtype=np.float32)),
        FakeRecognizer(),
        FakeResponseGenerator(["First sentence. ", "Second sentence."]),
        FakeSpeechOutput(stop_after_first=True),
    )

    result = service.run_turn(stop_event=stop_event)

    assert result.status == "interrupted"
    assert service.speech_output.spoken == ["First sentence."]


def test_interruption_during_capture_returns_without_transcribing() -> None:
    stop_event = threading.Event()
    stop_event.set()
    capture = FakeCapture(np.ones(512, dtype=np.float32))
    recognizer = FakeRecognizer()

    result = VoiceInteractionService(
        capture, recognizer, FakeResponseGenerator([]), FakeSpeechOutput()
    ).run_turn(stop_event=stop_event)

    assert result.status == "interrupted"
    assert recognizer.received == []


def test_voice_turn_persists_messages_and_restores_history(tmp_path) -> None:
    store = ConversationStore(tmp_path / "voice-memory.sqlite3")
    memory = ConversationMemoryManager(store)
    llm = FakeResponseGenerator(["First answer."])
    service = VoiceInteractionService(
        FakeCapture(np.ones(512, dtype=np.float32)),
        FakeRecognizer("First question."),
        llm,
        FakeSpeechOutput(),
        memory=memory,
    )

    first = service.run_turn()
    second = service.run_turn(
        conversation_id=first.conversation_id,
        history=None,
    )
    restored = memory.restore(first.conversation_id)

    assert first.conversation_id
    assert second.conversation_id == first.conversation_id
    assert len(restored.messages) == 4
    assert [message.role for message in restored.messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert llm.requests[1][1] == [
        ChatMessage("user", "First question."),
        ChatMessage("assistant", "First answer."),
    ]
    store.close()
