"""Default local service composition and Uvicorn entry point."""

from __future__ import annotations

import os
from pathlib import Path

import uvicorn

from app.api.application import create_app
from app.core.agent import LocalAgent
from app.core.coding_assistant import VoiceCodingAssistant
from app.core.tool_registry import ToolRegistry
from app.core.voice_interaction import VoiceInteractionService
from app.memory.conversation_store import ConversationStore
from app.memory.memory_manager import ConversationMemoryManager
from app.services.audio_service import MicrophoneCapture
from app.services.ollama_service import OllamaConfig, OllamaService
from app.services.stt_service import FasterWhisperConfig, FasterWhisperService
from app.services.tts_service import (
    AudioPlaybackService,
    PiperSpeechOutput,
    PiperTTSConfig,
    PiperTTSService,
)
from app.services.vad_service import SileroVADService
from app.tools.workspace import Workspace
from app.tools.workspace_tools import WorkspaceTools


def build_local_app():
    """Build the default local-only service graph; models and devices stay lazy."""
    host = os.environ.get("APP_HOST", "127.0.0.1")
    if host.casefold() not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("APP_HOST must be localhost or a loopback IP address.")
    port = int(os.environ.get("APP_PORT", "8000"))
    if not 1 <= port <= 65535:
        raise ValueError("APP_PORT must be between 1 and 65535.")

    roots_setting = os.environ.get("WORKSPACE_ROOTS", str(Path.cwd()))
    roots = [
        Path(entry).expanduser().resolve()
        for entry in roots_setting.split(os.pathsep)
        if entry.strip()
    ]
    if not roots:
        raise ValueError("WORKSPACE_ROOTS must contain at least one path.")

    store = ConversationStore(
        os.environ.get("DATABASE_PATH", "data/conversations.sqlite3")
    )
    memory = ConversationMemoryManager(store)
    llm = OllamaService(
        OllamaConfig(
            base_url=os.environ.get(
                "OLLAMA_BASE_URL", "http://127.0.0.1:11434"
            ),
            model=os.environ.get("OLLAMA_MODEL", "llama3.1"),
        )
    )
    capture = MicrophoneCapture(vad=SileroVADService())
    recognizer = FasterWhisperService(
        FasterWhisperConfig(
            model_size_or_path=os.environ.get("WHISPER_MODEL", "base"),
            device=os.environ.get("WHISPER_DEVICE", "cpu"),
            compute_type=os.environ.get("WHISPER_COMPUTE_TYPE", "int8"),
            language=os.environ.get("STT_LANGUAGE") or None,
        )
    )
    speech_output = PiperSpeechOutput(
        PiperTTSService(
            PiperTTSConfig(
                model_path=Path(
                    os.environ.get(
                        "PIPER_MODEL_PATH",
                        "models/piper/en_US-lessac-medium.onnx",
                    )
                ),
                config_path=(
                    Path(os.environ["PIPER_CONFIG_PATH"])
                    if os.environ.get("PIPER_CONFIG_PATH")
                    else None
                ),
            )
        ),
        AudioPlaybackService(),
    )
    workspace_tools = WorkspaceTools(
        Workspace(roots),
        summarize_text=llm.chat,
        generate_text=llm.chat,
        explain_text=llm.chat,
    )
    agent = LocalAgent(
        llm, ToolRegistry(workspace_tools, audit_store=store)
    )
    voice_interaction = VoiceInteractionService(
        capture,
        recognizer,
        llm,
        speech_output,
        memory=memory,
    )
    coding_assistant = VoiceCodingAssistant(
        capture,
        recognizer,
        agent,
        speech_output,
        memory=memory,
    )
    return create_app(voice_interaction, coding_assistant, store)


def main() -> None:
    """Run the local API on its configured loopback interface."""
    host = os.environ.get("APP_HOST", "127.0.0.1")
    port = int(os.environ.get("APP_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


app = build_local_app()


if __name__ == "__main__":
    main()
