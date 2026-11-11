from __future__ import annotations

import threading

import numpy as np
import pytest

from app.services.audio_service import (
    AudioCaptureConfig,
    AudioCaptureError,
    CaptureAlreadyRunningError,
    MicrophoneCapture,
)


class ThresholdVAD:
    def reset(self) -> None:
        pass

    def is_speech(self, frame: np.ndarray) -> bool:
        return bool(np.max(np.abs(frame)) > 0.5)


class FakeInputStream:
    def __init__(self, callback, samples: list[np.ndarray]) -> None:
        self.callback = callback
        self.samples = samples

    def __enter__(self):
        for sample in self.samples:
            self.callback(sample[:, None], sample.size, None, None)
        return self

    def __exit__(self, *_args) -> None:
        return None


def capture_with(samples: list[np.ndarray], config: AudioCaptureConfig) -> MicrophoneCapture:
    def stream_factory(**kwargs):
        return FakeInputStream(kwargs["callback"], samples)

    return MicrophoneCapture(ThresholdVAD(), config, stream_factory)


def test_capture_returns_speech_and_discards_trailing_silence() -> None:
    config = AudioCaptureConfig(
        silence_duration_seconds=0.06,
        minimum_speech_seconds=0.03,
        preroll_seconds=0.032,
    )
    silent = np.zeros(512, dtype=np.float32)
    speech = np.ones(512, dtype=np.float32)
    capture = capture_with([silent, speech, speech, silent, silent], config)

    result = capture.capture_utterance()

    assert result is not None
    assert result.shape == (1536,)
    assert result.dtype == np.float32


def test_capture_returns_none_when_no_speech_is_detected() -> None:
    config = AudioCaptureConfig(maximum_listen_seconds=0.01)
    silent = np.zeros(512, dtype=np.float32)
    capture = capture_with([silent], config)

    assert capture.capture_utterance() is None


def test_capture_returns_partial_speech_when_stopped() -> None:
    config = AudioCaptureConfig(minimum_speech_seconds=0.03)
    speech = np.ones(512, dtype=np.float32)
    stop_event = threading.Event()

    class StopAfterSpeechVAD(ThresholdVAD):
        def is_speech(self, frame: np.ndarray) -> bool:
            is_speech = super().is_speech(frame)
            if is_speech:
                stop_event.set()
            return is_speech

    capture = MicrophoneCapture(
        StopAfterSpeechVAD(),
        config,
        lambda **kwargs: FakeInputStream(kwargs["callback"], [speech]),
    )

    result = capture.capture_utterance(stop_event)

    assert result is not None
    assert result.shape == (512,)


def test_capture_rejects_invalid_silero_frame_size() -> None:
    with pytest.raises(ValueError, match="Frame size must be 512"):
        AudioCaptureConfig(frame_samples=256)


def test_capture_surfaces_microphone_startup_errors() -> None:
    def broken_stream_factory(**_kwargs):
        raise OSError("no input device")

    capture = MicrophoneCapture(
        ThresholdVAD(), stream_factory=broken_stream_factory
    )

    with pytest.raises(AudioCaptureError, match="Microphone capture failed"):
        capture.capture_utterance()


def test_capture_prevents_parallel_recording() -> None:
    capture = MicrophoneCapture(ThresholdVAD())
    assert capture._capture_lock.acquire(blocking=False)
    try:
        with pytest.raises(CaptureAlreadyRunningError):
            capture.capture_utterance()
    finally:
        capture._capture_lock.release()
