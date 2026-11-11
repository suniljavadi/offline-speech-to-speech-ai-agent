from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.core.permissions import ApprovalError, ApprovalManager
from app.core.tool_registry import (
    ToolExecutionError,
    ToolRegistry,
    UnknownToolError,
)
from app.memory.conversation_store import ConversationStore
from app.tools.workspace import Workspace, WorkspaceAccessError
from app.tools.workspace_tools import WorkspaceTools


def make_registry(
    tmp_path: Path,
    *,
    approval_manager: ApprovalManager | None = None,
    audit_store: ConversationStore | None = None,
    allow_code_execution: bool = False,
) -> tuple[ToolRegistry, Path]:
    root = tmp_path / "workspace"
    root.mkdir(exist_ok=True)
    tools = WorkspaceTools(
        Workspace([root]),
        summarize_text=lambda text: "summary:" + text[-20:],
        generate_text=lambda prompt: "print('generated')",
        explain_text=lambda prompt: "local explanation",
        allow_code_execution=allow_code_execution,
    )
    return (
        ToolRegistry(
            tools,
            approvals=approval_manager,
            audit_store=audit_store or ConversationStore(":memory:"),
        ),
        root,
    )


def approve(registry: ToolRegistry, tool: str, arguments: dict) -> str:
    request = registry.prepare_approval(tool, arguments)
    assert request.tool_name == tool
    return registry.confirm_approval(request.request_id)


def test_registry_describes_ten_typed_tools(tmp_path: Path) -> None:
    tools, _ = make_registry(tmp_path)

    descriptors = tools.list_tools()

    assert len(descriptors) == 10
    assert {descriptor.name for descriptor in descriptors} == {
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
    }
    assert {
        descriptor.name
        for descriptor in descriptors
        if descriptor.requires_approval
    } == {"write_file", "execute_python_code", "run_tests"}


def test_read_search_summarize_and_explain_tools_are_workspace_scoped(
    tmp_path: Path,
) -> None:
    registry, root = make_registry(tmp_path)
    source = root / "example.py"
    source.write_text("def greet():\n    return 'hello'\n", encoding="utf-8")

    assert registry.execute(
        "search_local_files", {"query": "greet"}
    ).result[0]["path"] == "example.py"
    assert "def greet" in registry.execute(
        "read_file", {"path": "example.py"}
    ).result["content"]
    assert registry.execute(
        "summarize_document", {"path": "example.py"}
    ).result["summary"].startswith("summary:")
    assert registry.execute(
        "explain_code",
        {"path": "example.py", "question": "What does this do?"},
    ).result["explanation"] == "local explanation"
    assert registry.execute(
        "search_project_context", {"query": "greet"}
    ).result[0]["path"] == "example.py"


def test_paths_outside_workspace_and_symlink_escape_are_rejected(
    tmp_path: Path,
) -> None:
    registry, root = make_registry(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")

    with pytest.raises(ToolExecutionError, match="failed"):
        registry.execute("read_file", {"path": str(outside)})
    link = root / "outside-link.txt"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation is not permitted on this Windows system.")
    with pytest.raises(WorkspaceAccessError):
        registry.tools.workspace.resolve(link)


def test_write_file_requires_exact_one_use_approval_and_never_overwrites_by_default(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path / "audit.sqlite3")
    conversation = store.create_conversation()
    registry, root = make_registry(tmp_path, audit_store=store)
    arguments = {"path": "new.txt", "content": "first"}

    with pytest.raises(ApprovalError, match="required"):
        registry.execute(
            "write_file", arguments, conversation_id=conversation.id
        )
    token = approve(registry, "write_file", arguments)
    result = registry.execute(
        "write_file",
        arguments,
        approval_token=token,
        conversation_id=conversation.id,
    )
    assert (root / "new.txt").read_text(encoding="utf-8") == "first"
    assert result.audit_id > 0

    with pytest.raises(ToolExecutionError, match="failed"):
        registry.execute(
            "write_file",
            arguments,
            approval_token=approve(registry, "write_file", arguments),
            conversation_id=conversation.id,
        )
    overwrite = {**arguments, "content": "second", "overwrite": True}
    result = registry.execute(
        "write_file",
        overwrite,
        approval_token=approve(registry, "write_file", overwrite),
        conversation_id=conversation.id,
    )
    assert (root / "new.txt").read_text(encoding="utf-8") == "second"
    assert result.audit_id > 0

    audit = store.list_tool_audit(conversation.id)
    assert [item.status for item in audit] == [
        "succeeded",
        "failed",
        "succeeded",
        "denied",
    ]
    assert all(len(item.argument_digest) == 64 for item in audit)
    store.close()


def test_approval_is_rejected_if_arguments_change_or_token_is_reused(
    tmp_path: Path,
) -> None:
    registry, _ = make_registry(tmp_path)
    arguments = {"path": "one.txt", "content": "original"}
    token = approve(registry, "write_file", arguments)

    with pytest.raises(ApprovalError, match="changed"):
        registry.execute(
            "write_file",
            {**arguments, "content": "changed"},
            approval_token=token,
        )
    with pytest.raises(ApprovalError, match="valid user approval"):
        registry.execute(
            "write_file", arguments, approval_token=token
        )


def test_expired_approval_cannot_be_used(tmp_path: Path) -> None:
    current_time = [10.0]
    approvals = ApprovalManager(ttl_seconds=2, clock=lambda: current_time[0])
    registry, _ = make_registry(tmp_path, approval_manager=approvals)
    arguments = {"path": "one.txt", "content": "original"}
    token = approve(registry, "write_file", arguments)
    current_time[0] = 12.0

    with pytest.raises(ApprovalError, match="expired"):
        registry.execute(
            "write_file", arguments, approval_token=token
        )


def test_tool_schemas_reject_unknown_fields_and_execution_is_off_by_default(
    tmp_path: Path,
) -> None:
    registry, _ = make_registry(tmp_path)

    with pytest.raises(ToolExecutionError, match="Invalid arguments"):
        registry.execute(
            "get_current_time", {"timezone": "UTC", "shell": "forbidden"}
        )
    with pytest.raises(ToolExecutionError, match="PermissionError"):
        registry.execute(
            "execute_python_code",
            {"code": "print('hello')"},
            approval_token=approve(
                registry, "execute_python_code", {"code": "print('hello')"}
            ),
        )
    with pytest.raises(UnknownToolError):
        registry.execute("arbitrary_shell", {"command": "whoami"})


def test_execution_uses_fixed_argv_no_shell_and_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, root = make_registry(tmp_path, allow_code_execution=True)
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        if command[0] != sys.executable:
            raise AssertionError("Unexpected executable")
        return subprocess.CompletedProcess(command, 0, "ok", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    arguments = {"code": "print('hello')", "timeout_seconds": 5}
    result = registry.execute(
        "execute_python_code",
        arguments,
        approval_token=approve(registry, "execute_python_code", arguments),
    )

    assert result.result["return_code"] == 0
    assert captured["shell"] is False
    assert captured["cwd"] == root
    assert captured["timeout"] == 5
    assert "-I" in captured["command"]
    assert "print('hello')" not in str(captured)


def test_run_tests_only_targets_workspace_and_fixed_pytest_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, root = make_registry(tmp_path, allow_code_execution=True)
    test_file = root / "test_sample.py"
    test_file.write_text("def test_ok(): pass\n", encoding="utf-8")
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return subprocess.CompletedProcess(command, 0, "1 passed", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    arguments = {"target": "test_sample.py", "timeout_seconds": 10}
    result = registry.execute(
        "run_tests",
        arguments,
        approval_token=approve(registry, "run_tests", arguments),
    )

    assert result.result["stdout"] == "1 passed"
    assert captured["command"][1:3] == ["-m", "pytest"]
    assert captured["command"][3] == str(test_file)
    assert captured["shell"] is False
    assert captured["cwd"] == root


def test_code_generation_returns_text_without_execution(tmp_path: Path) -> None:
    registry, _ = make_registry(tmp_path)

    result = registry.execute(
        "generate_python_code", {"requirement": "Create a greeting function"}
    )

    assert result.result["code"] == "print('generated')"


def test_current_time_supports_utc_and_rejects_unknown_zone(tmp_path: Path) -> None:
    registry, _ = make_registry(tmp_path)

    result = registry.execute("get_current_time", {"timezone": "UTC"})

    assert result.result["timezone"] == "UTC"
    assert result.result["iso8601"].endswith("+00:00")
    with pytest.raises(ToolExecutionError, match="ValueError"):
        registry.execute("get_current_time", {"timezone": "Not/AZone"})


def test_workspace_rejects_binary_extensions_and_oversized_files(
    tmp_path: Path,
) -> None:
    workspace = Workspace([tmp_path], max_file_bytes=5)
    binary = tmp_path / "data.bin"
    binary.write_bytes(b"xx")
    large = tmp_path / "large.txt"
    large.write_text("123456", encoding="utf-8")

    with pytest.raises(WorkspaceAccessError, match="File type"):
        workspace.read_text(binary, 100)
    with pytest.raises(WorkspaceAccessError, match="text limit"):
        workspace.read_text(large, 100)


def test_audit_does_not_store_raw_arguments_or_contents(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "audit-privacy.sqlite3")
    secret_text = "private document contents"

    record = store.record_tool_execution(
        conversation_id=None,
        tool_name="read_file",
        arguments={"path": "private.txt", "token": secret_text},
        approved=True,
        status="succeeded",
    )
    assert record.argument_digest != secret_text
    assert secret_text not in store._connection.execute(
        "SELECT * FROM tool_audit"
    ).fetchall()[0]["argument_digest"]
    store.close()


def test_deleting_conversation_preserves_minimized_audit_without_session_link(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path / "audit-delete.sqlite3")
    conversation = store.create_conversation()
    store.record_tool_execution(
        conversation_id=conversation.id,
        tool_name="read_file",
        arguments={"path": "example.py"},
        approved=True,
        status="succeeded",
    )

    store.delete_conversation(conversation.id)
    audit = store.list_tool_audit()

    assert len(audit) == 1
    assert audit[0].conversation_id is None
    store.close()


def test_python_execution_is_timeout_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, _ = make_registry(tmp_path, allow_code_execution=True)

    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("python", 1)

    monkeypatch.setattr(subprocess, "run", timeout)
    arguments = {"code": "while True: pass", "timeout_seconds": 1}

    with pytest.raises(ToolExecutionError, match="TimeoutError"):
        registry.execute(
            "execute_python_code",
            arguments,
            approval_token=approve(registry, "execute_python_code", arguments),
        )
