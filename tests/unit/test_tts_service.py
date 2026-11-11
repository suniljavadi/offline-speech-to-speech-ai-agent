from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.services.tts_service import (
    AudioPlaybackError,
    AudioPlaybackService,
    PiperSpeechOutput,
    PiperTTSConfig,
    PiperTTSService,
    SpeechAudio,
    TextToSpeechError,
    VoiceModelUnavailableError,
    split_sentences,
)


class FakePiperVoice:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def synthesize(self, text: str):
        self.calls.append(text)
        yield SimpleNamespace(
            sample_rate=22050,
            audio_float_array=np.array([0.1, 0.2], dtype=np.float32),
        )
        yield SimpleNamespace(
            sample_rate=22050,
            audio_float_array=np.array([0.3], dtype=np.float32),
        )


class FakeOutputStream:
    def __init__(self, chunks: list[np.ndarray]) -> None:
        self.chunks = chunks

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def write(self, block: np.ndarray) -> None:
        self.chunks.append(block.copy())


def make_tts(tmp_path: Path, voice: FakePiperVoice | None = None):
    model_path = tmp_path / "voice.onnx"
    config_path = tmp_path / "voice.onnx.json"
    model_path.touch()
    config_path.touch()
    voice = voice or FakePiperVoice()
    loaded = []

    def loader(*args, **kwargs):
        loaded.append((args, kwargs))
        return voice

    return (
        PiperTTSService(
            PiperTTSConfig(model_path=model_path, config_path=config_path),
            voice_loader=loader,
        ),
        voice,
        loaded,
    )


def test_synthesizes_each_sentence_as_mono_float_audio(tmp_path: Path) -> None:
    service, voice, loaded = make_tts(tmp_path)

    chunks = list(service.synthesize_sentences("First sentence. Second one!"))

    assert voice.calls == ["First sentence.", "Second one!"]
    assert len(chunks) == 2
    assert chunks[0].sample_rate == 22050
    assert chunks[0].samples.dtype == np.float32
    assert chunks[0].samples.tolist() == pytest.approx([0.1, 0.2, 0.3])
    assert len(loaded) == 1


def test_synthesize_combines_sentence_audio(tmp_path: Path) -> None:
    service, _, _ = make_tts(tmp_path)

    result = service.synthesize("One. Two.")

    assert result.sample_rate == 22050
    assert result.samples.tolist() == pytest.approx([0.1, 0.2, 0.3, 0.1, 0.2, 0.3])


def test_empty_text_does_not_load_voice(tmp_path: Path) -> None:
    service, _, loaded = make_tts(tmp_path)

    result = service.synthesize(" \n ")

    assert result.samples.size == 0
    assert loaded == []


def test_missing_voice_model_is_an_explicit_error(tmp_path: Path) -> None:
    service = PiperTTSService(
        PiperTTSConfig(model_path=tmp_path / "missing.onnx"),
        voice_loader=lambda *_args, **_kwargs: pytest.fail("must not load"),
    )

    with pytest.raises(VoiceModelUnavailableError, match="missing"):
        service.synthesize("Hello.")


def test_voice_loading_failure_is_explicit(tmp_path: Path) -> None:
    model_path = tmp_path / "voice.onnx"
    model_path.touch()
    Path(f"{model_path}.json").touch()
    service = PiperTTSService(
        PiperTTSConfig(model_path=model_path),
        voice_loader=lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("bad")),
    )

    with pytest.raises(VoiceModelUnavailableError, match="Could not load"):
        service.synthesize("Hello.")


def test_synthesis_failure_is_not_returned_as_audio(tmp_path: Path) -> None:
    class BrokenVoice:
        def synthesize(self, _text: str):
            raise RuntimeError("decoder failed")

    service, _, _ = make_tts(tmp_path, BrokenVoice())

    with pytest.raises(TextToSpeechError, match="could not synthesize"):
        list(service.synthesize_sentences("Hello."))


def test_sentence_splitter_handles_boundaries_and_long_sentences() -> None:
    assert split_sentences("  One. Two!\nThree?  ") == ["One.", "Two!", "Three?"]
    assert split_sentences("one two three four", max_characters=7) == [
        "one two",
        "three",
        "four",
    ]
    assert split_sentences("   ") == []


def test_playback_writes_short_blocks_and_can_be_interrupted() -> None:
    blocks = []
    event = threading.Event()

    class StopAfterFirstBlock(FakeOutputStream):
        def write(self, block: np.ndarray) -> None:
            super().write(block)
            event.set()

    audio = SpeechAudio(np.ones(100, dtype=np.float32), 1000)
    playback = AudioPlaybackService(
        output_stream_factory=lambda **_kwargs: StopAfterFirstBlock(blocks),
        block_duration_seconds=0.01,
    )

    playback.play(audio, stop_event=event)

    assert len(blocks) == 1
    assert blocks[0].shape == (10,)


def test_playback_errors_are_explicit() -> None:
    audio = SpeechAudio(np.ones(10, dtype=np.float32), 1000)
    playback = AudioPlaybackService(
        output_stream_factory=lambda **_kwargs: (_ for _ in ()).throw(
            OSError("no output")
        )
    )

    with pytest.raises(AudioPlaybackError, match="Could not play speech"):
        playback.play(audio)


def test_speech_output_stops_between_sentences(tmp_path: Path) -> None:
    service, voice, _ = make_tts(tmp_path)
    played = []

    class StopAfterFirstPlayback:
        def play(self, audio: SpeechAudio, stop_event=None) -> None:
            played.append(audio)
            stop_event.set()

    event = threading.Event()
    output = PiperSpeechOutput(service, StopAfterFirstPlayback())

    output.speak("First sentence. Second sentence.", stop_event=event)

    assert len(played) == 1
    assert voice.calls == ["First sentence."]
