"""Explicit setup-time downloader for a Faster-Whisper model."""

from __future__ import annotations

import argparse
from pathlib import Path


MODEL_SIZES = (
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large-v3",
    "large-v3-turbo",
)


def main() -> int:
    """Download and initialize a model during setup, never during inference."""
    parser = argparse.ArgumentParser(
        description="Download a Faster-Whisper model for offline use."
    )
    parser.add_argument("--model", choices=MODEL_SIZES, default="base")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "models" / "whisper",
    )
    args = parser.parse_args()

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise SystemExit(
            "faster-whisper is not installed. Install the project dependencies first."
        ) from exc

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Downloading or verifying '{args.model}' in '{args.cache_dir}'. "
        "This setup command may access the model host."
    )
    WhisperModel(
        args.model,
        device="cpu",
        compute_type="int8",
        download_root=str(args.cache_dir),
        local_files_only=False,
    )
    print(
        "Model is ready. Runtime transcription uses local_files_only=True and "
        "will not download models."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
