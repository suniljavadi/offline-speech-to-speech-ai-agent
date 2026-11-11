"""Typed tool registry with approval checks and privacy-minimized auditing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from app.core.permissions import ApprovalError, ApprovalManager, ApprovalRequest
from app.memory.conversation_store import ConversationStore
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
from app.tools.workspace_tools import WorkspaceTools


class ToolRegistryError(RuntimeError):
    """Base error for registered tool handling."""


class UnknownToolError(ToolRegistryError):
    """Raised when an unregistered tool name is requested."""


class ToolExecutionError(ToolRegistryError):
    """Raised when a registered tool fails."""


@dataclass(frozen=True)
class ToolDescriptor:
    """Public tool metadata for local agent planning."""

    name: str
    description: str
    arguments_schema: dict[str, Any]
    requires_approval: bool
    enabled: bool = True


@dataclass(frozen=True)
class ToolExecutionResult:
    """Successful, audited tool result."""

    tool_name: str
    result: Any
    audit_id: int


@dataclass(frozen=True)
class _ToolDefinition:
    name: str
    description: str
    arguments_model: type[BaseModel]
    handler: Callable[[Any], Any]
    requires_approval: bool


class ToolRegistry:
    """Validate all tool input and enforce exact user confirmation for risky tools."""

    def __init__(
        self,
        tools: WorkspaceTools,
        *,
        approvals: ApprovalManager | None = None,
        audit_store: ConversationStore,
    ) -> None:
        self.tools = tools
        self.approvals = approvals or ApprovalManager()
        self.audit_store = audit_store
        self._definitions = {
            definition.name: definition
            for definition in self._build_definitions()
        }

    def list_tools(self) -> tuple[ToolDescriptor, ...]:
        """Return descriptions and JSON schemas for agent planning/UI display."""
        return tuple(
            ToolDescriptor(
                name=definition.name,
                description=definition.description,
                arguments_schema=definition.arguments_model.model_json_schema(),
                requires_approval=definition.requires_approval,
                enabled=self._is_enabled(definition.name),
            )
            for definition in self._definitions.values()
        )

    def available_tools(self) -> tuple[ToolDescriptor, ...]:
        """Return only tools whose required local dependencies/configuration exist."""
        return tuple(
            descriptor
            for descriptor in self.list_tools()
            if descriptor.enabled
        )

    def prepare_approval(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> ApprovalRequest:
        """Validate and return exact arguments for a confirmation UI to display."""
        definition = self._get_definition(tool_name)
        if not definition.requires_approval:
            raise ValueError(f"Tool '{tool_name}' does not require user approval.")
        parsed = definition.arguments_model.model_validate(arguments)
        return self.approvals.prepare_request(
            tool_name, parsed.model_dump(mode="json")
        )

    def confirm_approval(self, request_id: str) -> str:
        """Issue a token for the exact prepared request after user confirmation."""
        return self.approvals.confirm_request(request_id)

    def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        approval_token: str | None = None,
        conversation_id: str | None = None,
    ) -> ToolExecutionResult:
        """Validate, authorize, execute, and audit one registered tool call."""
        definition = self._get_definition(tool_name)
        try:
            parsed = definition.arguments_model.model_validate(arguments)
        except ValidationError as exc:
            self._record(
                conversation_id,
                tool_name,
                self._safe_arguments(arguments),
                approved=False,
                status="denied",
                error_type="ValidationError",
            )
            raise ToolExecutionError(
                f"Invalid arguments for tool '{tool_name}'."
            ) from exc

        normalized_arguments = parsed.model_dump(mode="json")
        approved = not definition.requires_approval
        if definition.requires_approval:
            try:
                if approval_token is None:
                    raise ApprovalError("A valid user approval is required.")
                self.approvals.consume(
                    approval_token, tool_name, normalized_arguments
                )
                approved = True
            except ApprovalError as exc:
                self._record(
                    conversation_id,
                    tool_name,
                    normalized_arguments,
                    approved=False,
                    status="denied",
                    error_type=type(exc).__name__,
                )
                raise

        audit_id = self._record(
            conversation_id,
            tool_name,
            normalized_arguments,
            approved=approved,
            status="started",
            error_type=None,
        )
        try:
            result = definition.handler(parsed)
        except Exception as exc:
            self.audit_store.finish_tool_execution(
                audit_id, "failed", type(exc).__name__
            )
            raise ToolExecutionError(
                f"Tool '{tool_name}' failed ({type(exc).__name__})."
            ) from exc

        audit = self.audit_store.finish_tool_execution(audit_id, "succeeded")
        return ToolExecutionResult(tool_name, result, audit.id)

    def _record(
        self,
        conversation_id: str | None,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        approved: bool,
        status: str,
        error_type: str | None,
    ) -> int:
        record = self.audit_store.record_tool_execution(
            conversation_id=conversation_id,
            tool_name=tool_name,
            arguments=arguments,
            approved=approved,
            status=status,
            error_type=error_type,
        )
        return record.id

    @staticmethod
    def _safe_arguments(arguments: object) -> dict[str, Any]:
        if not isinstance(arguments, dict):
            return {"invalid_input_type": type(arguments).__name__}

        def normalize(value: Any) -> Any:
            if value is None or isinstance(value, (str, int, float, bool)):
                return value
            if isinstance(value, list):
                return [normalize(item) for item in value]
            if isinstance(value, dict):
                return {str(key): normalize(item) for key, item in value.items()}
            return {"unsupported_type": type(value).__name__}

        return {str(key): normalize(value) for key, value in arguments.items()}

    def _get_definition(self, tool_name: str) -> _ToolDefinition:
        try:
            return self._definitions[tool_name]
        except KeyError as exc:
            raise UnknownToolError(f"Unknown tool '{tool_name}'.") from exc

    def _is_enabled(self, tool_name: str) -> bool:
        if tool_name in {"execute_python_code", "run_tests"}:
            return self.tools.allow_code_execution
        if tool_name == "summarize_document":
            return self.tools.summarize_text is not None
        if tool_name == "generate_python_code":
            return self.tools.generate_text is not None
        if tool_name == "explain_code":
            return self.tools.explain_text is not None
        return True

    def _build_definitions(self) -> list[_ToolDefinition]:
        return [
            _ToolDefinition(
                "get_current_time",
                "Get the current local, UTC, or IANA time-zone time.",
                GetCurrentTimeArguments,
                self.tools.get_current_time,
                False,
            ),
            _ToolDefinition(
                "search_local_files",
                "Search configured workspace text files by name or content.",
                SearchLocalFilesArguments,
                self.tools.search_local_files,
                False,
            ),
            _ToolDefinition(
                "read_file",
                "Read a bounded UTF-8 text file inside an allowed workspace.",
                ReadFileArguments,
                self.tools.read_file,
                False,
            ),
            _ToolDefinition(
                "summarize_document",
                "Summarize a local workspace document with a configured local model.",
                SummarizeDocumentArguments,
                self.tools.summarize_document,
                False,
            ),
            _ToolDefinition(
                "write_file",
                "Write a text file inside an allowed workspace.",
                WriteFileArguments,
                self.tools.write_file,
                True,
            ),
            _ToolDefinition(
                "generate_python_code",
                "Generate Python source using a local model; this does not execute it.",
                GeneratePythonCodeArguments,
                self.tools.generate_python_code,
                False,
            ),
            _ToolDefinition(
                "execute_python_code",
                "Execute supplied Python in a child process; requires opt-in and approval.",
                ExecutePythonCodeArguments,
                self.tools.execute_python_code,
                True,
            ),
            _ToolDefinition(
                "run_tests",
                "Run pytest on an allowed workspace target; requires opt-in and approval.",
                RunTestsArguments,
                self.tools.run_tests,
                True,
            ),
            _ToolDefinition(
                "explain_code",
                "Explain a bounded source file using a configured local model.",
                ExplainCodeArguments,
                self.tools.explain_code,
                False,
            ),
            _ToolDefinition(
                "search_project_context",
                "Find relevant local source snippets for project questions.",
                SearchProjectContextArguments,
                self.tools.search_project_context,
                False,
            ),
        ]
