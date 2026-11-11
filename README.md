# Offline Speech-to-Speech AI Agent

This project is being implemented milestone by milestone. Milestones 1–9 cover
local microphone capture, Silero VAD, Faster-Whisper transcription, Ollama chat,
Piper speech output, memory, approved workspace tools, voice coding mode, and
local FastAPI/Streamlit integration. Quality measurements and further
optimization remain future work.

## Local setup

Requirements:

- Python 3.11 or newer. Faster-Whisper/CTranslate2 wheels depend on your Python
  version and platform; Python 3.11 is the initial target for deployment checks.
- A working microphone and audio input permissions.
- Internet access only for initial package/model setup. Normal transcription
  requires the dependencies and model to already be installed locally.

In PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python scripts\download_whisper_model.py --model base
python -m piper.download_voices en_US-lessac-medium --data-dir models\piper
```

The download command is a separate, explicit setup operation. Select another
supported model size with `--model` if your machine has different latency or
memory constraints. Model files are stored under `models\whisper\` and excluded
from version control.

## Using the local services

```python
from app.services.audio_service import MicrophoneCapture
from app.services.ollama_service import OllamaService
from app.services.stt_service import FasterWhisperService
from app.services.tts_service import (
    AudioPlaybackService,
    PiperSpeechOutput,
    PiperTTSConfig,
    PiperTTSService,
)
from app.services.vad_service import SileroVADService
from app.core.voice_interaction import VoiceInteractionService

voice = VoiceInteractionService(
    capture=MicrophoneCapture(vad=SileroVADService()),
    recognizer=FasterWhisperService(),
    response_generator=OllamaService(),
    speech_output=PiperSpeechOutput(
        PiperTTSService(
            PiperTTSConfig(
                model_path="models/piper/en_US-lessac-medium.onnx",
                config_path="models/piper/en_US-lessac-medium.onnx.json",
            )
        ),
        AudioPlaybackService(),
    ),
)
result = voice.run_turn()  # Opens microphone only when explicitly invoked.
print(result.transcript, "->", result.response)
```

`run_turn()` returns `no_speech`, `no_transcript`, `completed`, or `interrupted`.
Pass a `threading.Event` to interrupt capture, response generation, or audio
playback. For conversation context, supply prior `ChatMessage` items as
`history`, or attach `ConversationMemoryManager` and resume a returned
`conversation_id` to store and restore turns across process restarts.

### Persistent conversation memory

The default SQLite database is `data/conversations.sqlite3`; override it by
passing a different path to `ConversationStore`. The database uses schema
versioning, transactions, foreign-key cascading, and a bounded context window.
On POSIX systems the database file is restricted to owner read/write. On Windows,
the database inherits permissions from the containing directory and user
profile. The database is not encrypted, and stored transcripts remain until you
delete the conversation; no automatic transcript-retention policy is enabled.
Audio is never stored by this memory layer.

```python
from app.memory.conversation_store import ConversationStore
from app.memory.memory_manager import ConversationMemoryManager

store = ConversationStore("data/conversations.sqlite3")
memory = ConversationMemoryManager(store)
voice = VoiceInteractionService(
    capture=MicrophoneCapture(vad=SileroVADService()),
    recognizer=FasterWhisperService(),
    response_generator=OllamaService(),
    speech_output=PiperSpeechOutput(
        PiperTTSService(PiperTTSConfig()),
        AudioPlaybackService(),
    ),
    memory=memory,
)

result = voice.run_turn()
# Persist result.conversation_id in the future UI to restore this session.
if result.conversation_id:
    snapshot = memory.restore(result.conversation_id)
```

Conversation deletion is available through
`memory.delete_conversation(conversation_id)` and
`memory.delete_all_conversations()`. Optional summaries use an explicitly
injected local generator; for example, pass a callable that invokes
`OllamaService.chat` and returns only a concise summary. Summaries are saved
against the message revision they describe, so a concurrent conversation
change rejects a stale summary rather than replacing newer context.

### Agent tools and approvals

The registry provides typed schemas for time, workspace search/read, document
summary, approved file writes, local code generation, approved Python execution
and pytest, code explanation, and project-context search. File tools resolve
canonical paths against configured workspace roots and accept only bounded
UTF-8 text files.

Construct a registry with an audit store and explicit workspace:

```python
from pathlib import Path

from app.core.agent import LocalAgent
from app.core.tool_registry import ToolRegistry
from app.memory.conversation_store import ConversationStore
from app.services.ollama_service import OllamaService
from app.tools.workspace import Workspace
from app.tools.workspace_tools import WorkspaceTools

database = ConversationStore("data/conversations.sqlite3")
tools = WorkspaceTools(
    Workspace([Path.cwd()]),
    summarize_text=lambda prompt: OllamaService().chat(prompt),
    generate_text=lambda prompt: OllamaService().chat(prompt),
    explain_text=lambda prompt: OllamaService().chat(prompt),
)
registry = ToolRegistry(tools, audit_store=database)
agent = LocalAgent(OllamaService(), registry)
result = agent.run("Find where the project config is loaded.")
```

The agent advertises only tools with their required local model/config available.
File writes, code execution, and test runs pause at `approval_required`. The
confirmation UI should show the exact `pending_tool.tool_name` and
`pending_tool.arguments`, then call `agent.prepare_pending_approval(pending_id)`
to create the exact confirmation request to display. Only after the user
confirms should the application call
`agent.confirm_pending(pending_id, approval_request.request_id)` and then
`agent.resume(pending_id, token)`. Tokens are one-use, short-lived, and bound
to normalized arguments.
Tool audit records retain a SHA-256 argument digest and outcome, not raw code,
arguments, prompts, or file contents.

Python execution and pytest are disabled by default; enabling them through
`WorkspaceTools(..., allow_code_execution=True)` is an explicit trusted-local
configuration choice. They use fixed argument-vector subprocess calls, never a
shell command supplied by the model, with timeouts and workspace working
directories. **This is not an operating-system sandbox:** approved Python can
still access resources available to the current user. Do not enable execution
for untrusted code or on workspaces you do not control. A confirmation is not a
security boundary.

### Voice coding assistant mode

`VoiceCodingAssistant` reuses the explicit microphone capture, local speech
recognizer, and speech output with `LocalAgent`. It applies a coding-specific
system prompt that favors workspace context and read-only inspection, avoids
unrequested modifications, and does not claim code was tested unless tests
actually ran. Its tools remain confined to the configured workspace. Any file
write pauses for the same exact-request approval flow described above.

```python
from app.core.coding_assistant import VoiceCodingAssistant

coding_assistant = VoiceCodingAssistant(
    capture=MicrophoneCapture(vad=SileroVADService()),
    recognizer=FasterWhisperService(),
    agent=agent,  # LocalAgent configured with a workspace-scoped ToolRegistry
    speech_output=voice.speech_output,
    memory=memory,
)

result = coding_assistant.run_turn()
if result.status == "approval_required":
    pending = result.agent_result.pending_tool
    request = coding_assistant.prepare_pending_approval(pending.pending_id)
    # Display request.tool_name and request.arguments, then confirm only on user action.
    token = coding_assistant.confirm_pending(pending.pending_id, request.request_id)
    result = coding_assistant.resume_pending(pending.pending_id, token)
```

The approval comment represents a required user-interface decision; applications
must not confirm automatically. The returned status and pending tool are available
to a future UI/API integration. The agent loop is non-streaming, so its response
is spoken after the tool loop completes or pauses for approval.

### FastAPI and Streamlit local application

The API and UI are separate local processes. In two PowerShell terminals, from
the project root, start the API and UI:

```powershell
python -m app.api.main
streamlit run app\ui\streamlit_app.py
```

By default, the API binds to `127.0.0.1:8000` and the Streamlit UI is pinned to
`127.0.0.1` by `.streamlit/config.toml`.
The API accepts only loopback clients, loopback Host headers, and loopback
browser origins. Do not expose either service through a public interface or
reverse proxy. The health endpoint (`http://127.0.0.1:8000/health`) confirms
the API process only; microphone access, speech models, Piper voice, and Ollama
are loaded or checked only when used. Conversation details return at most the
most recent 200 messages by default.

The UI starts microphone capture only after **Start listening**. Select Voice
chat for ordinary conversation or Coding assistant for workspace questions.
Coding changes pause before execution; use **Review exact proposed change** to
inspect the tool and arguments, then affirmatively approve that exact request.
Conversation history is stored in the configured local SQLite database. Use
the conversation selector and delete control to inspect or remove a session.

Set `APP_HOST` only to `127.0.0.1`, `localhost`, or `::1`; `APP_PORT`,
`APP_API_URL`, `DATABASE_PATH`, `WORKSPACE_ROOTS`, `OLLAMA_BASE_URL`,
`OLLAMA_MODEL`, `WHISPER_MODEL`, `WHISPER_DEVICE`, `WHISPER_COMPUTE_TYPE`,
`STT_LANGUAGE`, `PIPER_MODEL_PATH`, and `PIPER_CONFIG_PATH` customize local
services. `WORKSPACE_ROOTS` is a path-separated list (semicolon-separated on
Windows). Workspace tools are confined to these roots.

### Ollama setup and chat

Install Ollama separately and pull a model during setup, for example:

```powershell
ollama pull llama3.1
ollama serve
```

The default client talks only to `http://127.0.0.1:11434`; non-loopback hosts
are rejected. It does not download models. Check local service/model status and
stream a response with:

```python
from app.services.ollama_service import OllamaService

llm = OllamaService()
llm.require_model_available()

for text_delta in llm.stream_chat("What time is it?"):
    print(text_delta, end="", flush=True)
```

Configure another installed model, local port, system prompt, request timeout,
context limits, or response length with `OllamaConfig`. Conversation history is
trimmed to configured message and character budgets before being sent.

### Piper speech output

The setup command above downloads the example `en_US-lessac-medium` voice and
its config into `models\piper\`. Voice model downloads happen only during setup.
Review the selected voice's model card and license before using or redistributing
it. Configure another voice by pointing `PiperTTSConfig` at its local ONNX model
and matching `.onnx.json` config.

```python
import threading

from app.services.tts_service import (
    AudioPlaybackService,
    PiperSpeechOutput,
    PiperTTSConfig,
    PiperTTSService,
)

voice = PiperTTSService(
    PiperTTSConfig(
        model_path="models/piper/en_US-lessac-medium.onnx",
        config_path="models/piper/en_US-lessac-medium.onnx.json",
    )
)
speech = PiperSpeechOutput(voice, AudioPlaybackService())
interrupt = threading.Event()
speech.speak("This response is spoken locally, one sentence at a time.", interrupt)
```

`AudioPlaybackService.play` accepts a stop event and checks it between short
playback blocks. Pass the same event to `PiperSpeechOutput.speak` to stop before
the next sentence as well. If the required voice files, Piper runtime, or output
device are unavailable, the service raises an explicit error.

`capture_utterance` returns `None` when no qualifying speech is captured. Pass a
`threading.Event` to stop listening; already captured speech is returned only if
it meets the configured minimum duration. Capture and transcription errors are
raised explicitly.

The STT service defaults to CPU/int8, 16 kHz audio, automatic language
detection, and local-only model loading. It refuses to download a missing model
during inference. Configure another model, device, compute type, language, or
cache directory through `FasterWhisperConfig`.

## Tests

```powershell
python -m pytest -q
```

The tests mock the microphone, VAD, Whisper, Ollama, and Piper models. A real
microphone/model smoke test still needs to be performed on the target machine.

## Current limitations

- No FastAPI or Streamlit interface is implemented yet.
- Voice coding mode is not implemented yet; standalone typed tools are available
  through the registry but are not yet exposed in the UI/API.
- Noise handling currently consists of Silero VAD and Faster-Whisper's VAD
  filtering; there is no dedicated denoising stage.
- Model-size latency and memory benchmarks have not yet been measured.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the architecture and
milestone acceptance criteria, and [DEVELOPMENT_PROGRESS.md](DEVELOPMENT_PROGRESS.md)
for current status.
