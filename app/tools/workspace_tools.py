"""Workspace-scoped read, write, search, and coding tool implementations."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.tools.schemas import (
    ExecutePythonCodeArguments,
    ExplainCodeArguments,
    GeneratePythonCodeArguments,
    GetCurrentTimeArguments,
    ReadFileArguments,
    RunTestsArguments,
    SearchLocalFilesArguments,
    SearchProjectContextArguments,
    SummarizeDocumentArguments,
    WriteFileArguments,
)
from app.tools.workspace import Workspace, WorkspaceAccessError

LocalTextGenerator = Callable[[str], str]


class WorkspaceTools:
    """Implement tools that are restricted to the configured project roots."""

    def __init__(
        self,
        workspace: Workspace,
        *,
        summarize_text: LocalTextGenerator | None = None,
        generate_text: LocalTextGenerator | None = None,
        explain_text: LocalTextGenerator | None = None,
        allow_code_execution: bool = False,
    ) -> None:
        self.workspace = workspace
        self.summarize_text = summarize_text
        self.generate_text = generate_text
        self.explain_text = explain_text
        self.allow_code_execution = allow_code_execution

    def get_current_time(self, arguments: GetCurrentTimeArguments) -> dict[str, str]:
        """Return local time or a configured IANA time zone."""
        if arguments.timezone.casefold() == "local":
            current = datetime.now().astimezone()
        elif arguments.timezone.upper() == "UTC":
            current = datetime.now(timezone.utc)
        else:
            try:
                current = datetime.now(ZoneInfo(arguments.timezone))
            except (ZoneInfoNotFoundError, ValueError) as exc:
                raise ValueError(
                    f"Unknown time zone: {arguments.timezone}"
                ) from exc
        return {"timezone": str(current.tzinfo), "iso8601": current.isoformat()}

    def search_local_files(
        self, arguments: SearchLocalFilesArguments
    ) -> list[dict[str, str]]:
        """Search workspace text files for paths matching a query."""
        return self.workspace.search_files(
            arguments.query,
            glob_pattern=arguments.glob_pattern,
            max_results=arguments.max_results,
            root=arguments.root,
        )

    def read_file(self, arguments: ReadFileArguments) -> dict[str, str]:
        """Read a bounded UTF-8 source or documentation file."""
        path = self.workspace.resolve(arguments.path)
        return {
            "path": str(path),
            "content": self.workspace.read_text(path, arguments.max_characters),
        }

    def summarize_document(
        self, arguments: SummarizeDocumentArguments
    ) -> dict[str, str]:
        """Summarize a local document through an injected local model callable."""
        if self.summarize_text is None:
            raise RuntimeError("No local document summarizer is configured.")
        path = self.workspace.resolve(arguments.path)
        content = self.workspace.read_text(path, arguments.max_characters)
        summary = self.summarize_text(
            "Summarize the following local document concisely:\n\n" + content
        )
        if not summary.strip():
            raise RuntimeError("The local summarizer returned an empty summary.")
        return {"path": str(path), "summary": summary.strip()}

    def write_file(self, arguments: WriteFileArguments) -> dict[str, str]:
        """Write a workspace file atomically; overwrite must be explicitly requested."""
        path = self.workspace.resolve(arguments.path)
        if path.suffix.casefold() not in self.workspace.TEXT_EXTENSIONS:
            raise WorkspaceAccessError("File type is not allowed for text writes.")
        if len(arguments.content.encode("utf-8")) > self.workspace.max_file_bytes:
            raise WorkspaceAccessError(
                f"Content exceeds the {self.workspace.max_file_bytes}-byte limit."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.workspace.resolve(arguments.path) != path:
            raise WorkspaceAccessError(
                "Target path changed while preparing the workspace write."
            )
        if path.exists() and not arguments.overwrite:
            raise FileExistsError(
                "Target file exists; set overwrite=true and approve that exact operation."
            )
        if path.exists() and not path.is_file():
            raise WorkspaceAccessError("Target is not a regular file.")

        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
                newline="",
            ) as temporary:
                temp_path = Path(temporary.name)
                temporary.write(arguments.content)
                temporary.flush()
                os.fsync(temporary.fileno())
            if path.exists() and not arguments.overwrite:
                raise FileExistsError(
                    "Target file appeared during write; refusing to overwrite."
                )
            os.replace(temp_path, path)
            temp_path = None
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
        return {"path": str(path), "status": "written"}

    def generate_python_code(
        self, arguments: GeneratePythonCodeArguments
    ) -> dict[str, str]:
        """Generate code text with an injected local model; never executes it."""
        if self.generate_text is None:
            raise RuntimeError("No local code generation model is configured.")
        prompt = (
            "Generate Python code for this requirement. Return code and a brief "
            "explanation; do not claim to execute it.\n"
            f"Requirement:\n{arguments.requirement}\n"
            f"Context:\n{arguments.context}"
        )
        return {"code": self.generate_text(prompt)}

    def execute_python_code(
        self, arguments: ExecutePythonCodeArguments
    ) -> dict[str, Any]:
        """Execute approved Python in a child process (not an OS security sandbox)."""
        if not self.allow_code_execution:
            raise PermissionError(
                "Python execution is disabled. Enable it explicitly in trusted local "
                "configuration before requesting execution."
            )
        root = self.workspace.roots[0]
        result = self._run_child_process(
            [sys.executable, "-I", "-S", "{script}"],
            cwd=root,
            timeout_seconds=arguments.timeout_seconds,
            source=arguments.code,
        )
        return self._process_result(result)

    def run_tests(self, arguments: RunTestsArguments) -> dict[str, Any]:
        """Run only pytest in the configured workspace, with explicit approval."""
        if not self.allow_code_execution:
            raise PermissionError(
                "Test execution is disabled. Enable it explicitly in trusted local "
                "configuration before requesting tests."
            )
        target = self.workspace.resolve(arguments.target)
        if not target.exists():
            raise FileNotFoundError(f"Test target does not exist: {arguments.target}")
        if target.is_file() and target.suffix.casefold() != ".py":
            raise WorkspaceAccessError("A test-file target must be a Python file.")
        root = self.workspace.find_root(target)
        result = self._run_child_process(
            [sys.executable, "-m", "pytest", str(target)],
            cwd=root,
            timeout_seconds=arguments.timeout_seconds,
            source=None,
        )
        return self._process_result(result)

    def explain_code(self, arguments: ExplainCodeArguments) -> dict[str, str]:
        """Explain bounded workspace source using an injected local model."""
        if self.explain_text is None:
            raise RuntimeError("No local code explanation model is configured.")
        path = self.workspace.resolve(arguments.path)
        source = self.workspace.read_text(path, arguments.max_characters)
        explanation = self.explain_text(
            f"Explain this code and answer the question.\n"
            f"Question: {arguments.question}\n"
            f"File: {path.name}\n\n{source}"
        )
        return {"path": str(path), "explanation": explanation}

    def search_project_context(
        self, arguments: SearchProjectContextArguments
    ) -> list[dict[str, str]]:
        """Return bounded relevant source snippets without sending them externally."""
        candidates = self.workspace.search_files(
            arguments.query,
            glob_pattern="**/*",
            max_results=arguments.max_results * 4,
        )
        results: list[dict[str, str]] = []
        seen: set[str] = set()
        for candidate in candidates:
            if (
                "content" not in candidate["match"].split(",")
                or candidate["path"] in seen
            ):
                continue
            seen.add(candidate["path"])
            path = self.workspace.resolve(candidate["path"])
            content = self.workspace.read_text(path, 30000)
            index = content.casefold().find(arguments.query.casefold())
            start = max(0, index - 400)
            end = min(len(content), index + len(arguments.query) + 600)
            snippet = re.sub(r"\s+", " ", content[start:end]).strip()
            results.append({"path": candidate["path"], "snippet": snippet})
            if len(results) >= arguments.max_results:
                break
        return results

    @staticmethod
    def _run_child_process(
        command: list[str],
        *,
        cwd: Path,
        timeout_seconds: int,
        source: str | None,
    ) -> subprocess.CompletedProcess[str]:
        """Run an approved fixed-argv Python operation without a shell."""
        script_path: Path | None = None
        if source is not None:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                suffix=".py",
                prefix="approved-python-",
                delete=False,
            ) as script:
                script.write(source)
                script_path = Path(script.name)
            command = [str(script_path) if part == "{script}" else part for part in command]
        environment = {
            key: value
            for key, value in os.environ.items()
            if key.casefold()
            in {
                "path",
                "systemroot",
                "windir",
                "temp",
                "tmp",
                "home",
                "userprofile",
            }
        }
        try:
            return subprocess.run(
                command,
                cwd=cwd,
                env=environment,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
                shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(
                f"Approved Python operation exceeded {timeout_seconds} seconds."
            ) from exc
        except OSError as exc:
            raise RuntimeError("Could not start the approved Python operation.") from exc
        finally:
            if script_path is not None:
                script_path.unlink(missing_ok=True)

    @staticmethod
    def _process_result(
        result: subprocess.CompletedProcess[str],
    ) -> dict[str, Any]:
        return {
            "return_code": result.returncode,
            "stdout": result.stdout[-20000:],
            "stderr": result.stderr[-20000:],
        }
