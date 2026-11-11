from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from app.services.stt_service import (
    FasterWhisperConfig,
    FasterWhisperService,
    TranscriptionError,
    WhisperModelUnavailableError,
)


class FakeWhisperModel:
    def __init__(self, segments, info) -> None:
        self.segments = segments
        self.info = info
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        return iter(self.segments), self.info


def test_transcribes_audio_and_reports_language_metadata() -> None:
    model = FakeWhisperModel(
        [SimpleNamespace(text=" Hello"), SimpleNamespace(text=" there.")],
        SimpleNamespace(
            language="en", language_probability=0.98, duration=1.5
        ),
    )
    service = FasterWhisperService(model_factory=lambda *_args, **_kwargs: model)

    result = service.transcribe(np.ones(16000, dtype=np.float32))

    assert result.text == "Hello there."
    assert result.language == "en"
    assert result.language_probability == 0.98
    assert result.duration_seconds == 1.5
    assert result.segment_count == 2
    assert model.calls[0][1] == {
        "language": None,
        "beam_size": 5,
        "vad_filter": True,
    }


def test_empty_audio_returns_empty_result_without_loading_model() -> None:
    service = FasterWhisperService(
        model_factory=lambda *_args, **_kwargs: pytest.fail("model should not load")
    )

    result = service.transcribe(np.array([], dtype=np.float32))

    assert result.text == ""
    assert result.language is None
    assert result.duration_seconds == 0


def test_rejects_stereo_audio() -> None:
    service = FasterWhisperService()

    with pytest.raises(ValueError, match="mono"):
        service.transcribe(np.zeros((2, 512), dtype=np.float32))


def test_rejects_non_finite_samples() -> None:
    service = FasterWhisperService()

    with pytest.raises(ValueError, match="finite"):
        service.transcribe(np.array([np.nan], dtype=np.float32))


def test_rejects_mismatched_sample_rate() -> None:
    service = FasterWhisperService()

    with pytest.raises(ValueError, match="Expected 16000 Hz"):
        service.transcribe(np.ones(512, dtype=np.float32), sample_rate=8000)


def test_runtime_downloads_cannot_be_enabled() -> None:
    with pytest.raises(ValueError, match="disabled"):
        FasterWhisperConfig(local_files_only=False)


def test_model_load_failure_is_actionable() -> None:
    def fail_model_load(*_args, **_kwargs):
        raise OSError("model missing")

    service = FasterWhisperService(model_factory=fail_model_load)

    with pytest.raises(WhisperModelUnavailableError, match="runtime downloads"):
        service.transcribe(np.ones(512, dtype=np.float32))


def test_transcription_failure_is_not_returned_as_success() -> None:
    class BrokenModel:
        def transcribe(self, *_args, **_kwargs):
            raise RuntimeError("decoder failed")

    service = FasterWhisperService(model_factory=lambda *_args, **_kwargs: BrokenModel())

    with pytest.raises(TranscriptionError, match="could not transcribe"):
        service.transcribe(np.ones(512, dtype=np.float32))
