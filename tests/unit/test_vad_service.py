from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from app.services.vad_service import SileroVADService


class FakeModel:
    def __init__(self, score: float) -> None:
        self.score = score
        self.reset_count = 0
        self.frames = []

    def reset_states(self) -> None:
        self.reset_count += 1

    def __call__(self, frame, sample_rate: int):
        self.frames.append((frame, sample_rate))
        return SimpleNamespace(item=lambda: self.score)


def install_fake_torch(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(from_numpy=lambda values: values),
    )


def test_silero_adapter_loads_lazily_and_resets_streaming_state(monkeypatch) -> None:
    install_fake_torch(monkeypatch)
    model = FakeModel(0.8)
    service = SileroVADService(model_loader=lambda: model)

    assert model.reset_count == 0
    service.reset()
    result = service.is_speech(np.zeros(512, dtype=np.float32))

    assert result is True
    assert model.reset_count == 1
    assert model.frames[0][1] == 16000


def test_silero_adapter_rejects_wrong_frame_size() -> None:
    service = SileroVADService(model_loader=lambda: FakeModel(0.9))

    with pytest.raises(ValueError, match="512 samples"):
        service.is_speech(np.zeros(256, dtype=np.float32))


def test_silero_adapter_rejects_unsupported_sample_rate() -> None:
    with pytest.raises(ValueError, match="8000 Hz or 16000 Hz"):
        SileroVADService(sample_rate=44100)


def test_silero_adapter_rejects_non_finite_samples() -> None:
    service = SileroVADService(model_loader=lambda: FakeModel(0.9))

    with pytest.raises(ValueError, match="finite"):
        service.is_speech(np.full(512, np.nan, dtype=np.float32))
