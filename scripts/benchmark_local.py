"""Measure local workspace, schema-validation, and SQLite operations."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import tempfile
import time
import tracemalloc
from pathlib import Path
from typing import Callable

from app.memory.conversation_store import ConversationStore
from app.tools.schemas import ReadFileArguments
from app.tools.workspace import Workspace


def _measure(
    operation: Callable[[], object],
    iterations: int,
) -> dict[str, float | int]:
    samples: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        operation()
        samples.append((time.perf_counter() - started) * 1000)

    tracemalloc.start()
    try:
        operation()
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    ordered = sorted(samples)
    p95_index = max(0, min(len(ordered) - 1, (95 * len(ordered) + 99) // 100 - 1))
    return {
        "iterations": iterations,
        "median_ms": round(statistics.median(ordered), 4),
        "p95_ms": round(ordered[p95_index], 4),
        "single_operation_peak_traced_bytes": peak_bytes,
    }


def run_benchmarks(iterations: int = 100, files: int = 50) -> dict[str, object]:
    """Run deterministic, synthetic benchmarks without devices or network calls."""
    if iterations < 1:
        raise ValueError("iterations must be at least one.")
    if files < 1:
        raise ValueError("files must be at least one.")

    with tempfile.TemporaryDirectory(prefix="offline-assistant-benchmark-") as temp:
        root = Path(temp)
        for index in range(files):
            (root / f"module_{index:04d}.py").write_text(
                f'"""Synthetic module {index}."""\nVALUE = {index}\n'
                "def target_function():\n    return 'benchmark-token'\n",
                encoding="utf-8",
            )

        workspace = Workspace([root])
        target_path = "module_0000.py"
        target_file = root / target_path
        read_arguments = ReadFileArguments(path=target_path, max_characters=20000)

        store = ConversationStore(":memory:")
        conversation = store.create_conversation()
        try:
            results = {
                "workspace_search": _measure(
                    lambda: workspace.search_files(
                        "benchmark-token", max_results=20
                    ),
                    iterations,
                ),
                "workspace_read": _measure(
                    lambda: workspace.read_text(target_file, 20000),
                    iterations,
                ),
                "tool_schema_validation": _measure(
                    lambda: ReadFileArguments.model_validate(
                        {
                            "path": target_path,
                            "max_characters": 20000,
                        }
                    ),
                    iterations,
                ),
                "sqlite_append_and_bounded_context": _measure(
                    lambda: (
                        store.add_message(
                            conversation.id, "user", "Measure local context access."
                        ),
                        store.get_messages_after(conversation.id, 0, 20),
                    ),
                    iterations,
                ),
            }
        finally:
            store.close()

    return {
        "scope": "synthetic local operations; excludes inference, audio, and network",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor() or "unspecified",
        },
        "dataset": {"files": files, "iterations_per_case": iterations},
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--files", type=int, default=50)
    arguments = parser.parse_args()
    print(
        json.dumps(
            run_benchmarks(arguments.iterations, arguments.files),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
