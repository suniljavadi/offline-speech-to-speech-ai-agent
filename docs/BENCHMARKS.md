# Local synthetic benchmark

## Reproduce

From the project root, run:

```powershell
python scripts\benchmark_local.py --iterations 100 --files 50
```

The script builds temporary synthetic Python files and an in-memory SQLite
database, then measures workspace search/read, Pydantic argument validation,
and append-plus-bounded-context retrieval. It does not access the network,
microphone, speech models, or Ollama. Timing is collected without allocation
tracing; peak traced Python allocations are measured in a separate single
operation. The allocation figure is not total process or native/model memory.

## Recorded baseline

Run on Windows 11, Python 3.14.3, with 50 synthetic files and 100 iterations
per operation. Times are milliseconds; the run is a local baseline, not a
performance guarantee or CI threshold.

| Operation | Median | p95 | Single-operation peak traced bytes |
|---|---:|---:|---:|
| Workspace search | 14.7772 | 32.2524 | 161,385 |
| Workspace read | 1.0926 | 3.4325 | 133,334 |
| Tool schema validation | 0.0019 | 0.0025 | 312 |
| SQLite append + bounded context | 0.1642 | 0.6627 | 10,650 |

Filesystem cache state, system load, storage, and machine configuration affect
these values; these numbers are one warmed-cache run. The search figures
describe only this 50-file synthetic corpus.
No inference, microphone, STT, TTS, or end-to-end voice latency/resource
baseline is available; collect those on the target machine with the selected
local models and devices.
