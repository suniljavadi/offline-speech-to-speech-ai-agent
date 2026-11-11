"""Local Piper speech synthesis and interruptible audio playback."""

from __future__ import annotations

import importlib
import logging
import re
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class TextToSpeechError(RuntimeError):
    """Raised when local speech synthesis or playback fails."""


class VoiceModelUnavailableError(TextToSpeechError):
    """Raised when the configured local Piper voice is missing or cannot load."""


class AudioPlaybackError(TextToSpeechError):
    """Raised when generated speech cannot be played locally."""


@dataclass(frozen=True)
class PiperTTSConfig:
    """Local Piper model, synthesis, and sentence chunk settings."""

    model_path: Path = Path("models/piper/en_US-lessac-medium.onnx")
    config_path: Path | None = None
    use_cuda: bool = False
    max_sentence_characters: int = 500

    def __post_init__(self) -> None:
        if self.max_sentence_characters < 1:
            raise ValueError("max_sentence_characters must be at least one.")

    @property
    def resolved_config_path(self) -> Path:
        """Return the explicit voice config or Piper's `<model>.onnx.json` path."""
        return self.config_path or Path(f"{self.model_path}.json")


@dataclass(frozen=True)
class SpeechAudio:
    """Mono float32 audio produced by a local Piper voice."""

    samples: np.ndarray
    sample_rate: int

    def __post_init__(self) -> None:
        if self.samples.ndim != 1:
            raise ValueError("Speech audio must be a mono 1-D waveform.")
        if self.samples.dtype != np.float32:
            raise ValueError("Speech audio samples must use float32.")
        if not np.isfinite(self.samples).all():
            raise ValueError("Speech audio must contain only finite samples.")
        if self.sample_rate <= 0:
            raise ValueError("Speech audio sample rate must be positive.")


VoiceLoader = Callable[..., Any]
OutputStreamFactory = Callable[..., Any]


class PiperTTSService:
    """Load Piper lazily and synthesize text without runtime downloads."""

    def __init__(
        self,
        config: PiperTTSConfig | None = None,
        voice_loader: VoiceLoader | None = None,
    ) -> None:
        self.config = config or PiperTTSConfig()
        self._voice_loader = voice_loader
        self._voice: Any | None = None
        self._voice_lock = threading.Lock()

    def synthesize(self, text: str) -> SpeechAudio:
        """Synthesize all sentences and return one contiguous mono waveform."""
        sentence_audio = list(self.synthesize_sentences(text))
        if not sentence_audio:
            return SpeechAudio(np.zeros(0, dtype=np.float32), 22050)
        sample_rates = {audio.sample_rate for audio in sentence_audio}
        if len(sample_rates) != 1:
            raise TextToSpeechError("Piper voice returned inconsistent sample rates.")
        return SpeechAudio(
            np.concatenate([audio.samples for audio in sentence_audio]).astype(
                np.float32, copy=False
            ),
            sentence_audio[0].sample_rate,
        )

    def synthesize_sentences(
        self,
        text: str,
        stop_event: threading.Event | None = None,
    ) -> Iterator[SpeechAudio]:
        """Yield locally synthesized sentence chunks for incremental playback."""
        sentences = split_sentences(
            text, max_characters=self.config.max_sentence_characters
        )
        if not sentences or (stop_event is not None and stop_event.is_set()):
            return

        voice = self._get_voice()
        for sentence in sentences:
            if stop_event is not None and stop_event.is_set():
                return
            try:
                chunks = list(voice.synthesize(sentence))
                if not chunks:
                    continue
                sample_rates = {int(chunk.sample_rate) for chunk in chunks}
                if len(sample_rates) != 1:
                    raise TextToSpeechError(
                        "Piper returned inconsistent sample rates within a sentence."
                    )
                samples = np.concatenate(
                    [np.asarray(chunk.audio_float_array, dtype=np.float32).reshape(-1)
                     for chunk in chunks]
                )
                if not np.isfinite(samples).all():
                    raise TextToSpeechError(
                        "Piper returned non-finite audio samples."
                    )
                yield SpeechAudio(samples, sample_rates.pop())
            except TextToSpeechError:
                raise
            except Exception as exc:
                logger.exception("Local Piper speech synthesis failed")
                raise TextToSpeechError(
                    "Piper could not synthesize the response text."
                ) from exc

    def _get_voice(self) -> Any:
        if self._voice is not None:
            return self._voice
        with self._voice_lock:
            if self._voice is not None:
                return self._voice
            model_path = self.config.model_path
            config_path = self.config.resolved_config_path
            if not model_path.is_file() or not config_path.is_file():
                raise VoiceModelUnavailableError(
                    "Piper voice model or config is missing. Download a voice during "
                    "setup and configure its .onnx and .onnx.json paths."
                )
            try:
                loader = self._voice_loader or self._default_voice_loader()
                self._voice = loader(
                    str(model_path),
                    config_path=str(config_path),
                    use_cuda=self.config.use_cuda,
                )
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                logger.exception("Could not load local Piper voice")
                raise VoiceModelUnavailableError(
                    f"Could not load local Piper voice at '{model_path}'. Verify the "
                    "model, matching config, and installed Piper runtime."
                ) from exc
        return self._voice

    @staticmethod
    def _default_voice_loader() -> VoiceLoader:
        try:
            module = importlib.import_module("piper")
        except ImportError as exc:
            raise VoiceModelUnavailableError(
                "The piper-tts package is required for local speech synthesis."
            ) from exc
        voice_type = getattr(module, "PiperVoice")
        return voice_type.load


class AudioPlaybackService:
    """Play mono audio locally in short cancellable blocks."""

    def __init__(
        self,
        output_stream_factory: OutputStreamFactory | None = None,
        block_duration_seconds: float = 0.05,
    ) -> None:
        if block_duration_seconds <= 0:
            raise ValueError("block_duration_seconds must be greater than zero.")
        self._output_stream_factory = (
            output_stream_factory or self._default_output_stream_factory
        )
        self.block_duration_seconds = block_duration_seconds
        self._playback_lock = threading.Lock()

    def play(
        self,
        audio: SpeechAudio,
        stop_event: threading.Event | None = None,
    ) -> None:
        """Play audio and return promptly when cancellation is requested."""
        if audio.samples.size == 0:
            return
        if not self._playback_lock.acquire(blocking=False):
            raise AudioPlaybackError("Audio playback is already in progress.")

        try:
            block_samples = max(
                1, int(audio.sample_rate * self.block_duration_seconds)
            )
            with self._output_stream_factory(
                samplerate=audio.sample_rate, channels=1, dtype="float32"
            ) as stream:
                for start in range(0, audio.samples.size, block_samples):
                    if stop_event is not None and stop_event.is_set():
                        return
                    stream.write(audio.samples[start : start + block_samples])
        except AudioPlaybackError:
            raise
        except Exception as exc:
            logger.exception("Local audio playback failed")
            raise AudioPlaybackError(
                "Could not play speech. Check the selected output device and "
                "audio permissions."
            ) from exc
        finally:
            self._playback_lock.release()

    @staticmethod
    def _default_output_stream_factory(**kwargs: Any) -> Any:
        try:
            sounddevice = importlib.import_module("sounddevice")
        except ImportError as exc:
            raise AudioPlaybackError(
                "The sounddevice package is required for audio playback."
            ) from exc
        return sounddevice.OutputStream(**kwargs)


class PiperSpeechOutput:
    """Coordinate sentence-level Piper synthesis with cancellable playback."""

    def __init__(
        self,
        tts: PiperTTSService,
        playback: AudioPlaybackService,
    ) -> None:
        self.tts = tts
        self.playback = playback

    def speak(
        self,
        text: str,
        stop_event: threading.Event | None = None,
    ) -> None:
        """Synthesize and play response sentences until complete or interrupted."""
        for sentence_audio in self.tts.synthesize_sentences(
            text, stop_event=stop_event
        ):
            self.playback.play(sentence_audio, stop_event=stop_event)


def split_sentences(text: str, max_characters: int = 500) -> list[str]:
    """Split at sentence/newline boundaries and cap long sentences by words."""
    if max_characters < 1:
        raise ValueError("max_characters must be at least one.")
    normalized = text.strip()
    if not normalized:
        return []

    sentence_parts = [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+|\n+", normalized)
        if part.strip()
    ]
    result: list[str] = []
    for sentence in sentence_parts:
        if len(sentence) <= max_characters:
            result.append(sentence)
            continue
        words = sentence.split()
        chunk: list[str] = []
        chunk_length = 0
        for word in words:
            if chunk and chunk_length + 1 + len(word) > max_characters:
                result.append(" ".join(chunk))
                chunk = []
                chunk_length = 0
            chunk.append(word)
            chunk_length += len(word) + (1 if len(chunk) > 1 else 0)
        if chunk:
            result.append(" ".join(chunk))
    return result
