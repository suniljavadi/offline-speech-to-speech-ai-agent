"""Explicitly activated microphone capture and speech segmentation."""

from __future__ import annotations

import importlib
import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from app.services.vad_service import VoiceActivityDetector

logger = logging.getLogger(__name__)


class AudioCaptureError(RuntimeError):
    """Raised when microphone capture cannot complete safely."""


class CaptureAlreadyRunningError(AudioCaptureError):
    """Raised when a second microphone capture is requested concurrently."""


@dataclass(frozen=True)
class AudioCaptureConfig:
    """Capture and utterance-boundary settings."""

    sample_rate: int = 16000
    frame_samples: int = 512
    silence_duration_seconds: float = 0.8
    minimum_speech_seconds: float = 0.2
    preroll_seconds: float = 0.2
    maximum_utterance_seconds: float = 30.0
    maximum_listen_seconds: float = 60.0
    queue_frames: int = 128

    def __post_init__(self) -> None:
        if self.sample_rate not in {8000, 16000}:
            raise ValueError("Capture sample rate must be 8000 Hz or 16000 Hz.")
        expected_frame_size = 256 if self.sample_rate == 8000 else 512
        if self.frame_samples != expected_frame_size:
            raise ValueError(
                f"Frame size must be {expected_frame_size} samples at "
                f"{self.sample_rate} Hz for Silero VAD."
            )
        for field_name in (
            "silence_duration_seconds",
            "minimum_speech_seconds",
            "maximum_utterance_seconds",
            "maximum_listen_seconds",
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be greater than zero.")
        if self.preroll_seconds < 0:
            raise ValueError("preroll_seconds cannot be negative.")
        if self.queue_frames < 1:
            raise ValueError("queue_frames must be at least one.")


StreamFactory = Callable[..., Any]


class MicrophoneCapture:
    """Capture a single utterance after explicit invocation by the caller."""

    def __init__(
        self,
        vad: VoiceActivityDetector,
        config: AudioCaptureConfig | None = None,
        stream_factory: StreamFactory | None = None,
    ) -> None:
        self.vad = vad
        self.config = config or AudioCaptureConfig()
        self._stream_factory = stream_factory or self._default_stream_factory
        self._capture_lock = threading.Lock()

    def capture_utterance(
        self, stop_event: threading.Event | None = None
    ) -> np.ndarray | None:
        """Record one VAD-delimited utterance, or return ``None`` for no speech."""
        if not self._capture_lock.acquire(blocking=False):
            raise CaptureAlreadyRunningError(
                "Microphone capture is already running."
            )

        try:
            return self._capture(stop_event)
        finally:
            self._capture_lock.release()

    def _capture(self, stop_event: threading.Event | None) -> np.ndarray | None:
        frames: queue.Queue[np.ndarray] = queue.Queue(
            maxsize=self.config.queue_frames
        )
        callback_errors: queue.Queue[str] = queue.Queue(maxsize=1)

        def audio_callback(
            indata: np.ndarray, _frame_count: int, _time_info: object, status: object
        ) -> None:
            if status:
                try:
                    callback_errors.put_nowait(str(status))
                except queue.Full:
                    pass
                return
            try:
                frames.put_nowait(np.asarray(indata[:, 0], dtype=np.float32).copy())
            except queue.Full:
                try:
                    callback_errors.put_nowait(
                        "Audio callback queue is full; captured frames were dropped."
                    )
                except queue.Full:
                    pass

        preroll_count = int(
            self.config.preroll_seconds
            * self.config.sample_rate
            / self.config.frame_samples
        )
        preroll: deque[np.ndarray] = deque(maxlen=preroll_count or 1)
        recording: list[np.ndarray] = []
        trailing_silence: list[np.ndarray] = []
        speech_samples = 0
        recorded_samples = 0
        silence_samples = 0
        started_at = time.monotonic()
        minimum_speech_samples = int(
            self.config.minimum_speech_seconds * self.config.sample_rate
        )
        silence_limit_samples = int(
            self.config.silence_duration_seconds * self.config.sample_rate
        )
        utterance_limit_samples = int(
            self.config.maximum_utterance_seconds * self.config.sample_rate
        )

        try:
            self.vad.reset()
            with self._stream_factory(
                samplerate=self.config.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self.config.frame_samples,
                callback=audio_callback,
            ):
                while True:
                    if stop_event is not None and stop_event.is_set():
                        return self._finish(recording, speech_samples, minimum_speech_samples)
                    if time.monotonic() - started_at >= self.config.maximum_listen_seconds:
                        return self._finish(recording, speech_samples, minimum_speech_samples)

                    try:
                        callback_error = callback_errors.get_nowait()
                    except queue.Empty:
                        callback_error = None
                    if callback_error is not None:
                        raise AudioCaptureError(callback_error)

                    timeout = min(
                        0.1,
                        max(
                            0.001,
                            self.config.maximum_listen_seconds
                            - (time.monotonic() - started_at),
                        ),
                    )
                    try:
                        frame = frames.get(timeout=timeout)
                    except queue.Empty:
                        continue

                    if frame.size != self.config.frame_samples:
                        raise AudioCaptureError(
                            "Microphone returned an unexpected audio frame size."
                        )
                    if not np.isfinite(frame).all():
                        raise AudioCaptureError(
                            "Microphone returned non-finite audio samples."
                        )

                    speech_detected = self.vad.is_speech(frame)
                    if not recording:
                        if speech_detected:
                            recording.extend(preroll)
                            recording.append(frame)
                            recorded_samples = sum(
                                part.size for part in recording
                            )
                            speech_samples = frame.size
                            trailing_silence.clear()
                            silence_samples = 0
                        else:
                            preroll.append(frame)
                        continue

                    if speech_detected:
                        recording.extend(trailing_silence)
                        recorded_samples += silence_samples
                        trailing_silence.clear()
                        recording.append(frame)
                        recorded_samples += frame.size
                        speech_samples += frame.size
                        silence_samples = 0
                    else:
                        trailing_silence.append(frame)
                        silence_samples += frame.size
                        if silence_samples >= silence_limit_samples:
                            if speech_samples >= minimum_speech_samples:
                                return np.concatenate(recording).astype(
                                    np.float32, copy=False
                                )
                            recording.clear()
                            trailing_silence.clear()
                            recorded_samples = 0
                            speech_samples = 0
                            silence_samples = 0
                            preroll.clear()

                    if (
                        recorded_samples + silence_samples >= utterance_limit_samples
                    ):
                        return self._finish(
                            recording, speech_samples, minimum_speech_samples
                        )
        except AudioCaptureError:
            raise
        except Exception as exc:
            logger.exception("Microphone capture failed")
            raise AudioCaptureError(
                "Microphone capture failed. Check the selected input device and "
                "audio permissions."
            ) from exc

    @staticmethod
    def _finish(
        recording: list[np.ndarray],
        speech_samples: int,
        minimum_speech_samples: int,
    ) -> np.ndarray | None:
        if not recording or speech_samples < minimum_speech_samples:
            return None
        return np.concatenate(recording).astype(np.float32, copy=False)

    @staticmethod
    def _default_stream_factory(**kwargs: Any) -> Any:
        try:
            sounddevice = importlib.import_module("sounddevice")
        except ImportError as exc:
            raise AudioCaptureError(
                "The sounddevice package is required for microphone capture."
            ) from exc
        return sounddevice.InputStream(**kwargs)
