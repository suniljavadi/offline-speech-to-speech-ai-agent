"""Silero-backed voice activity detection."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np


class VoiceActivityDetector(Protocol):
    """Classify fixed-size, mono audio frames as speech or non-speech."""

    def reset(self) -> None:
        """Reset any streaming state before a new recording."""
        ...

    def is_speech(self, frame: np.ndarray) -> bool:
        """Return whether a single frame contains speech."""
        ...


class SileroVADService:
    """Lazy-loaded Silero VAD adapter for streaming mono audio."""

    _FRAME_SIZES = {8000: 256, 16000: 512}

    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        model_loader: Callable[[], object] | None = None,
    ) -> None:
        if sample_rate not in self._FRAME_SIZES:
            raise ValueError("Silero VAD supports 8000 Hz or 16000 Hz audio.")
        if not 0.0 < threshold < 1.0:
            raise ValueError("VAD threshold must be between 0 and 1.")

        self.sample_rate = sample_rate
        self.threshold = threshold
        self._model_loader = model_loader
        self._model: object | None = None
        self._torch: object | None = None

    def reset(self) -> None:
        """Reset model state before detecting a new utterance."""
        self._load_model()
        reset_states = getattr(self._model, "reset_states", None)
        if callable(reset_states):
            reset_states()

    def is_speech(self, frame: np.ndarray) -> bool:
        """Classify one Silero-sized audio frame."""
        values = np.asarray(frame, dtype=np.float32)
        expected_size = self._FRAME_SIZES[self.sample_rate]
        if values.ndim != 1 or values.size != expected_size:
            raise ValueError(
                f"Expected a mono frame of {expected_size} samples at "
                f"{self.sample_rate} Hz."
            )
        if not np.isfinite(values).all():
            raise ValueError("Audio frame must contain only finite values.")

        self._load_model()
        from_numpy = getattr(self._torch, "from_numpy")
        tensor = from_numpy(values.copy())
        probability = getattr(self._model, "__call__")(tensor, self.sample_rate)
        score = float(getattr(probability, "item")())
        return score >= self.threshold

    def _load_model(self) -> None:
        if self._model is not None:
            return

        try:
            import torch

            if self._model_loader is None:
                from silero_vad import load_silero_vad

                model = load_silero_vad()
            else:
                model = self._model_loader()
        except (ImportError, OSError, RuntimeError) as exc:
            raise RuntimeError(
                "Could not load the local Silero VAD model. Install the project "
                "speech dependencies and make the packaged model available."
            ) from exc

        self._torch = torch
        self._model = model
