"""Local Faster-Whisper transcription service."""

from __future__ import annotations

import importlib
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class TranscriptionError(RuntimeError):
    """Raised when local speech transcription fails."""


class WhisperModelUnavailableError(TranscriptionError):
    """Raised when the requested local Whisper model cannot be loaded."""


@dataclass(frozen=True)
class FasterWhisperConfig:
    """Model and decoding options for a local Faster-Whisper instance."""

    model_size_or_path: str = "base"
    device: str = "cpu"
    compute_type: str = "int8"
    language: str | None = None
    beam_size: int = 5
    sample_rate: int = 16000
    model_cache_dir: Path = Path("models/whisper")
    local_files_only: bool = True

    def __post_init__(self) -> None:
        if not self.model_size_or_path.strip():
            raise ValueError("model_size_or_path cannot be empty.")
        if self.beam_size < 1:
            raise ValueError("beam_size must be at least one.")
        if self.sample_rate not in {8000, 16000, 22050, 44100, 48000}:
            raise ValueError("Unsupported transcription sample rate.")
        if not self.local_files_only:
            raise ValueError(
                "Runtime model downloads are disabled; use a separate setup step."
            )


@dataclass(frozen=True)
class TranscriptionResult:
    """Transcript and language metadata returned by the STT service."""

    text: str
    language: str | None
    language_probability: float | None
    duration_seconds: float
    segment_count: int


ModelFactory = Callable[..., Any]


class FasterWhisperService:
    """Load Faster-Whisper lazily and transcribe local mono audio."""

    def __init__(
        self,
        config: FasterWhisperConfig | None = None,
        model_factory: ModelFactory | None = None,
    ) -> None:
        self.config = config or FasterWhisperConfig()
        self._model_factory = model_factory
        self._model: Any | None = None
        self._model_lock = threading.Lock()

    def transcribe(
        self, audio: np.ndarray, sample_rate: int | None = None
    ) -> TranscriptionResult:
        """Transcribe a mono waveform without downloading models at runtime."""
        waveform = np.asarray(audio, dtype=np.float32)
        if waveform.ndim != 1:
            raise ValueError("Transcription input must be a mono 1-D waveform.")
        if not np.isfinite(waveform).all():
            raise ValueError("Transcription input must contain finite samples only.")

        actual_sample_rate = sample_rate or self.config.sample_rate
        if actual_sample_rate != self.config.sample_rate:
            raise ValueError(
                f"Expected {self.config.sample_rate} Hz audio, received "
                f"{actual_sample_rate} Hz."
            )
        if waveform.size == 0:
            return TranscriptionResult("", None, None, 0.0, 0)

        model = self._get_model()
        try:
            with self._model_lock:
                segments, info = model.transcribe(
                    waveform,
                    language=self.config.language,
                    beam_size=self.config.beam_size,
                    vad_filter=True,
                )
                materialized_segments = list(segments)
        except Exception as exc:
            logger.exception("Local speech transcription failed")
            raise TranscriptionError(
                "Faster-Whisper could not transcribe the captured audio."
            ) from exc

        text = "".join(segment.text for segment in materialized_segments).strip()
        duration = getattr(info, "duration", waveform.size / actual_sample_rate)
        probability = getattr(info, "language_probability", None)
        return TranscriptionResult(
            text=text,
            language=getattr(info, "language", None),
            language_probability=(
                float(probability) if probability is not None else None
            ),
            duration_seconds=float(duration),
            segment_count=len(materialized_segments),
        )

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        with self._model_lock:
            if self._model is not None:
                return self._model
            try:
                factory = self._model_factory or self._default_model_factory()
                self._model = factory(
                    self.config.model_size_or_path,
                    device=self.config.device,
                    compute_type=self.config.compute_type,
                    download_root=str(self.config.model_cache_dir),
                    local_files_only=True,
                )
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                raise WhisperModelUnavailableError(
                    f"Could not load local Faster-Whisper model "
                    f"'{self.config.model_size_or_path}'. Download it during setup "
                    "or point the configuration at an existing local model; "
                    "runtime downloads are disabled."
                ) from exc
        return self._model

    @staticmethod
    def _default_model_factory() -> ModelFactory:
        try:
            module = importlib.import_module("faster_whisper")
        except ImportError as exc:
            raise WhisperModelUnavailableError(
                "The faster-whisper package is required for local transcription."
            ) from exc
        return getattr(module, "WhisperModel")
