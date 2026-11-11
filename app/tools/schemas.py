"""Pydantic request schemas for registered assistant tools."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints
NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ToolArguments(BaseModel):
    """Strict base model shared by tool arguments."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)


class GetCurrentTimeArguments(ToolArguments):
    timezone: str = "local"


class SearchLocalFilesArguments(ToolArguments):
    root: Path | None = None
    query: str = ""
    glob_pattern: str = "**/*"
    max_results: int = Field(default=30, ge=1, le=200)


class ReadFileArguments(ToolArguments):
    path: NonEmptyText
    max_characters: int = Field(default=20000, ge=1, le=100000)


class SummarizeDocumentArguments(ToolArguments):
    path: NonEmptyText
    max_characters: int = Field(default=40000, ge=1, le=100000)


class WriteFileArguments(ToolArguments):
    path: NonEmptyText
    content: str
    overwrite: bool = False


class GeneratePythonCodeArguments(ToolArguments):
    requirement: NonEmptyText
    context: str = Field(default="", max_length=30000)


class ExecutePythonCodeArguments(ToolArguments):
    code: NonEmptyText = Field(max_length=50000)
    timeout_seconds: int = Field(default=10, ge=1, le=30)


class RunTestsArguments(ToolArguments):
    target: str = "."
    timeout_seconds: int = Field(default=120, ge=1, le=600)


class ExplainCodeArguments(ToolArguments):
    path: NonEmptyText
    question: NonEmptyText
    max_characters: int = Field(default=40000, ge=1, le=100000)


class SearchProjectContextArguments(ToolArguments):
    query: NonEmptyText
    max_results: int = Field(default=8, ge=1, le=50)


ToolName = Literal[
    "get_current_time",
    "search_local_files",
    "read_file",
    "summarize_document",
    "write_file",
    "generate_python_code",
    "execute_python_code",
    "run_tests",
    "explain_code",
    "search_project_context",
]
