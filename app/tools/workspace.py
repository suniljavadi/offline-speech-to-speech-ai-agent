"""Canonical workspace path validation and bounded text-file operations."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


class WorkspaceAccessError(PermissionError):
    """Raised for workspace traversal or unsupported file access."""


class Workspace:
    """Restrict text-file operations to explicitly configured directories."""

    TEXT_EXTENSIONS = {
        ".c",
        ".cfg",
        ".cpp",
        ".cs",
        ".css",
        ".go",
        ".h",
        ".htm",
        ".html",
        ".ini",
        ".java",
        ".js",
        ".json",
        ".jsx",
        ".md",
        ".mjs",
        ".php",
        ".ps1",
        ".py",
        ".pyi",
        ".rst",
        ".rs",
        ".scss",
        ".sh",
        ".sql",
        ".svelte",
        ".toml",
        ".ts",
        ".tsx",
        ".txt",
        ".vue",
        ".xml",
        ".yaml",
        ".yml",
    }

    def __init__(self, roots: Iterable[str | Path], *, max_file_bytes: int = 1_000_000):
        canonical_roots = tuple(Path(root).expanduser().resolve() for root in roots)
        if not canonical_roots:
            raise ValueError("At least one workspace root is required.")
        if any(not root.is_dir() for root in canonical_roots):
            raise ValueError("Every workspace root must be an existing directory.")
        if max_file_bytes < 1:
            raise ValueError("max_file_bytes must be positive.")
        self.roots = canonical_roots
        self.max_file_bytes = max_file_bytes

    def resolve(self, value: str | Path, root: str | Path | None = None) -> Path:
        """Resolve a file path and reject paths outside configured workspace roots."""
        requested = Path(value).expanduser()
        if requested.is_absolute():
            candidate = requested.resolve()
        else:
            selected_root = self._resolve_root(root)
            candidate = (selected_root / requested).resolve()
        if not any(self._is_relative_to(candidate, allowed) for allowed in self.roots):
            raise WorkspaceAccessError("Path is outside the configured workspace.")
        return candidate

    def find_root(self, path: Path) -> Path:
        """Return the configured workspace root containing a canonical path."""
        canonical = path.resolve()
        for root in self.roots:
            if self._is_relative_to(canonical, root):
                return root
        raise WorkspaceAccessError("Path is outside the configured workspace.")

    def read_text(self, path: Path, max_characters: int) -> str:
        """Read a bounded UTF-8 text file with an extension and size allowlist."""
        canonical = self.resolve(path)
        if not canonical.is_file():
            raise FileNotFoundError(f"Workspace file does not exist: {path}")
        if canonical.suffix.casefold() not in self.TEXT_EXTENSIONS:
            raise WorkspaceAccessError("File type is not allowed for text operations.")
        size = canonical.stat().st_size
        if size > self.max_file_bytes:
            raise WorkspaceAccessError(
                f"File exceeds the {self.max_file_bytes}-byte text limit."
            )
        try:
            return canonical.read_text(encoding="utf-8")[:max_characters]
        except UnicodeDecodeError as exc:
            raise WorkspaceAccessError(
                "File is not valid UTF-8 text."
            ) from exc

    def search_files(
        self,
        query: str,
        glob_pattern: str = "**/*",
        max_results: int = 30,
        root: str | Path | None = None,
    ) -> list[dict[str, str]]:
        """Search allowed workspace text files by name or case-insensitive content."""
        if not glob_pattern or Path(glob_pattern).is_absolute() or ".." in Path(glob_pattern).parts:
            raise WorkspaceAccessError("Glob pattern must be workspace-relative.")
        selected_roots = (
            (self._resolve_root(root),) if root is not None else self.roots
        )
        matches_by_path: dict[str, set[str]] = {}
        normalized_query = query.casefold()
        for workspace_root in selected_roots:
            for path in workspace_root.glob(glob_pattern):
                try:
                    canonical = self.resolve(path)
                    if (
                        not canonical.is_file()
                        or canonical.suffix.casefold() not in self.TEXT_EXTENSIONS
                        or canonical.stat().st_size > self.max_file_bytes
                    ):
                        continue
                    relative = canonical.relative_to(workspace_root).as_posix()
                    if normalized_query in relative.casefold():
                        matches_by_path.setdefault(relative, set()).add("filename")
                    if normalized_query:
                        try:
                            content = canonical.read_text(encoding="utf-8")
                        except UnicodeDecodeError:
                            continue
                        if normalized_query in content.casefold():
                            matches_by_path.setdefault(relative, set()).add("content")
                    if len(matches_by_path) >= max_results:
                        return [
                            {"path": name, "match": ",".join(sorted(kinds))}
                            for name, kinds in matches_by_path.items()
                        ]
                except (OSError, WorkspaceAccessError):
                    continue
        return [
            {"path": name, "match": ",".join(sorted(kinds))}
            for name, kinds in matches_by_path.items()
        ]

    def _resolve_root(self, root: str | Path | None) -> Path:
        if root is None:
            return self.roots[0]
        canonical = Path(root).expanduser().resolve()
        for allowed in self.roots:
            if canonical == allowed:
                return allowed
        raise WorkspaceAccessError("Selected root is not an allowed workspace.")

    @staticmethod
    def _is_relative_to(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False
