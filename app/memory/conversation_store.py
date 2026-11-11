"""Versioned, local SQLite storage for conversations and messages."""

from __future__ import annotations

import os
import hashlib
import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


class ConversationStoreError(RuntimeError):
    """Base error for conversation persistence failures."""


class ConversationNotFoundError(ConversationStoreError):
    """Raised when an operation references a missing conversation."""


class ConversationChangedError(ConversationStoreError):
    """Raised when conversation content changed during summarization."""


class UnsupportedSchemaVersionError(ConversationStoreError):
    """Raised when the database schema is newer than this application."""


@dataclass(frozen=True)
class Conversation:
    """Persisted conversation metadata."""

    id: str
    created_at: str
    updated_at: str
    summary: str | None
    summary_through_message_id: int | None
    revision: int


@dataclass(frozen=True)
class StoredMessage:
    """Persisted conversation message."""

    id: int
    conversation_id: str
    role: str
    content: str
    created_at: str


@dataclass(frozen=True)
class ConversationSnapshot:
    """Conversation metadata with its selected messages."""

    conversation: Conversation
    messages: tuple[StoredMessage, ...]


@dataclass(frozen=True)
class ToolAuditRecord:
    """Privacy-minimized result of a registered tool invocation."""

    id: int
    conversation_id: str | None
    tool_name: str
    argument_digest: str
    approved: bool
    status: str
    error_type: str | None
    created_at: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


class ConversationStore:
    """Thread-safe SQLite store with transactional writes and schema versioning."""

    SCHEMA_VERSION = 2
    _MESSAGE_ROLES = {"user", "assistant", "tool"}

    def __init__(
        self,
        database_path: str | Path = Path("data/conversations.sqlite3"),
        timeout_seconds: float = 5.0,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero.")
        self.database_path = str(database_path)
        if self.database_path != ":memory:":
            path = Path(self.database_path).expanduser()
            try:
                path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            except OSError as exc:
                raise ConversationStoreError(
                    "Could not create the local conversation database directory."
                ) from exc
            self.database_path = str(path)

        self._lock = threading.RLock()
        try:
            self._connection = sqlite3.connect(
                self.database_path,
                timeout=timeout_seconds,
                check_same_thread=False,
                isolation_level=None,
            )
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute(
                f"PRAGMA busy_timeout = {int(timeout_seconds * 1000)}"
            )
            self._connection.execute("PRAGMA journal_mode = WAL")
            self._initialize_schema()
            self._restrict_database_file_permissions()
        except ConversationStoreError:
            connection = getattr(self, "_connection", None)
            if connection is not None:
                connection.close()
            raise
        except sqlite3.Error as exc:
            connection = getattr(self, "_connection", None)
            if connection is not None:
                connection.close()
            raise ConversationStoreError(
                "Could not initialize the local conversation database."
            ) from exc

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            self._connection.close()

    def __enter__(self) -> ConversationStore:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def create_conversation(self, conversation_id: str | None = None) -> Conversation:
        """Create and return an empty conversation with a stable identifier."""
        identifier = conversation_id or str(uuid.uuid4())
        if not identifier.strip():
            raise ValueError("conversation_id cannot be empty.")
        now = _utc_now()
        try:
            with self._transaction() as connection:
                connection.execute(
                    """
                    INSERT INTO conversations (
                        id, created_at, updated_at, summary,
                        summary_through_message_id, revision
                    ) VALUES (?, ?, ?, NULL, NULL, 0)
                    """,
                    (identifier, now, now),
                )
                row = connection.execute(
                    "SELECT * FROM conversations WHERE id = ?", (identifier,)
                ).fetchone()
                return self._conversation_from_row(row)
        except sqlite3.IntegrityError as exc:
            raise ConversationStoreError(
                "Conversation identifier already exists."
            ) from exc
        except sqlite3.Error as exc:
            raise ConversationStoreError(
                "Could not create the local conversation."
            ) from exc

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        created_at: str | None = None,
    ) -> StoredMessage:
        """Append a user, assistant, or tool message atomically."""
        if role not in self._MESSAGE_ROLES:
            raise ValueError(f"Unsupported persisted message role: {role}")
        if not content.strip():
            raise ValueError("Message content cannot be empty.")
        timestamp = created_at or _utc_now()
        try:
            with self._transaction() as connection:
                cursor = connection.execute(
                    """
                    INSERT INTO messages (conversation_id, role, content, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (conversation_id, role, content, timestamp),
                )
                connection.execute(
                    """
                    UPDATE conversations
                    SET updated_at = ?, revision = revision + 1
                    WHERE id = ?
                    """,
                    (timestamp, conversation_id),
                )
                if connection.execute(
                    "SELECT changes()"
                ).fetchone()[0] != 1:
                    raise ConversationNotFoundError(
                        f"Conversation '{conversation_id}' does not exist."
                    )
                row = connection.execute(
                    "SELECT * FROM messages WHERE id = ?", (cursor.lastrowid,)
                ).fetchone()
                return self._message_from_row(row)
        except ConversationNotFoundError:
            raise
        except sqlite3.IntegrityError as exc:
            raise ConversationNotFoundError(
                f"Conversation '{conversation_id}' does not exist."
            ) from exc
        except sqlite3.Error as exc:
            raise ConversationStoreError(
                "Could not save the local conversation message."
            ) from exc

    def get_conversation(
        self, conversation_id: str, message_limit: int | None = None
    ) -> ConversationSnapshot:
        """Restore conversation metadata and messages in chronological order."""
        if message_limit is not None and message_limit < 1:
            raise ValueError("message_limit must be at least one.")
        with self._lock:
            try:
                conversation_row = self._connection.execute(
                    "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
                ).fetchone()
                if conversation_row is None:
                    raise ConversationNotFoundError(
                        f"Conversation '{conversation_id}' does not exist."
                    )
                query = (
                    "SELECT * FROM messages WHERE conversation_id = ? "
                    "ORDER BY id ASC"
                )
                parameters: tuple[object, ...] = (conversation_id,)
                if message_limit is not None:
                    query = (
                        "SELECT * FROM ("
                        "SELECT * FROM messages WHERE conversation_id = ? "
                        "ORDER BY id DESC LIMIT ?"
                        ") ORDER BY id ASC"
                    )
                    parameters = (conversation_id, message_limit)
                message_rows = self._connection.execute(
                    query, parameters
                ).fetchall()
                return ConversationSnapshot(
                    self._conversation_from_row(conversation_row),
                    tuple(self._message_from_row(row) for row in message_rows),
                )
            except ConversationNotFoundError:
                raise
            except sqlite3.Error as exc:
                raise ConversationStoreError(
                    "Could not restore the local conversation."
                ) from exc

    def get_messages_after(
        self,
        conversation_id: str,
        message_id: int,
        limit: int,
    ) -> tuple[StoredMessage, ...]:
        """Get messages following a summary boundary, newest bounded by id."""
        if message_id < 0:
            raise ValueError("message_id cannot be negative.")
        if limit < 1:
            raise ValueError("limit must be at least one.")
        with self._lock:
            try:
                exists = self._connection.execute(
                    "SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)
                ).fetchone()
                if exists is None:
                    raise ConversationNotFoundError(
                        f"Conversation '{conversation_id}' does not exist."
                    )
                rows = self._connection.execute(
                    """
                    SELECT * FROM (
                        SELECT * FROM messages
                        WHERE conversation_id = ? AND id > ?
                        ORDER BY id DESC
                        LIMIT ?
                    ) ORDER BY id ASC
                    """,
                    (conversation_id, message_id, limit),
                ).fetchall()
                return tuple(self._message_from_row(row) for row in rows)
            except ConversationNotFoundError:
                raise
            except sqlite3.Error as exc:
                raise ConversationStoreError(
                    "Could not retrieve local conversation context."
                ) from exc

    def list_conversations(
        self, limit: int = 50, offset: int = 0
    ) -> tuple[Conversation, ...]:
        """List most recently active conversations, newest first."""
        if limit < 1:
            raise ValueError("limit must be at least one.")
        if offset < 0:
            raise ValueError("offset cannot be negative.")
        with self._lock:
            try:
                rows = self._connection.execute(
                    """
                    SELECT * FROM conversations
                    ORDER BY updated_at DESC, id DESC
                    LIMIT ? OFFSET ?
                    """,
                    (limit, offset),
                ).fetchall()
                return tuple(self._conversation_from_row(row) for row in rows)
            except sqlite3.Error as exc:
                raise ConversationStoreError(
                    "Could not list local conversations."
                ) from exc

    def save_summary(
        self,
        conversation_id: str,
        summary: str,
        through_message_id: int,
        expected_revision: int,
    ) -> Conversation:
        """Save a summary only if its source messages have not changed."""
        if not summary.strip():
            raise ValueError("Conversation summary cannot be empty.")
        if through_message_id < 1:
            raise ValueError("through_message_id must be at least one.")
        if expected_revision < 0:
            raise ValueError("expected_revision cannot be negative.")
        try:
            with self._transaction() as connection:
                conversation = connection.execute(
                    "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
                ).fetchone()
                if conversation is None:
                    raise ConversationNotFoundError(
                        f"Conversation '{conversation_id}' does not exist."
                    )
                if conversation["revision"] != expected_revision:
                    raise ConversationChangedError(
                        "Conversation changed while its summary was being generated."
                    )
                previous_boundary = conversation["summary_through_message_id"]
                if (
                    previous_boundary is not None
                    and through_message_id < previous_boundary
                ):
                    raise ValueError(
                        "A conversation summary cannot move its message boundary backward."
                    )
                source_message = connection.execute(
                    "SELECT 1 FROM messages WHERE id = ? AND conversation_id = ?",
                    (through_message_id, conversation_id),
                ).fetchone()
                if source_message is None:
                    raise ValueError(
                        "Summary boundary must reference a message in this conversation."
                    )
                connection.execute(
                    """
                    UPDATE conversations
                    SET summary = ?, summary_through_message_id = ?
                    WHERE id = ?
                    """,
                    (summary.strip(), through_message_id, conversation_id),
                )
                row = connection.execute(
                    "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
                ).fetchone()
                return self._conversation_from_row(row)
        except (ConversationNotFoundError, ConversationChangedError, ValueError):
            raise
        except sqlite3.Error as exc:
            raise ConversationStoreError(
                "Could not save the local conversation summary."
            ) from exc

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and all its messages; return whether it existed."""
        with self._lock:
            try:
                with self._transaction() as connection:
                    cursor = connection.execute(
                        "DELETE FROM conversations WHERE id = ?", (conversation_id,)
                    )
                    return cursor.rowcount > 0
            except sqlite3.Error as exc:
                raise ConversationStoreError(
                    "Could not delete the local conversation."
                ) from exc

    def delete_all_conversations(self) -> int:
        """Delete every stored conversation and cascade-delete all messages."""
        with self._lock:
            try:
                with self._transaction() as connection:
                    cursor = connection.execute("DELETE FROM conversations")
                    return cursor.rowcount
            except sqlite3.Error as exc:
                raise ConversationStoreError(
                    "Could not delete local conversations."
                ) from exc

    def record_tool_execution(
        self,
        *,
        conversation_id: str | None,
        tool_name: str,
        arguments: dict[str, object],
        approved: bool,
        status: str,
        error_type: str | None = None,
    ) -> ToolAuditRecord:
        """Store tool outcome and a digest, never raw arguments or file contents."""
        if status not in {"started", "succeeded", "failed", "denied"}:
            raise ValueError("Unsupported tool execution status.")
        if not tool_name.strip():
            raise ValueError("tool_name cannot be empty.")
        if error_type is not None and not error_type.strip():
            raise ValueError("error_type cannot be empty when provided.")
        canonical_arguments = json.dumps(
            arguments, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        argument_digest = hashlib.sha256(
            canonical_arguments.encode("utf-8")
        ).hexdigest()
        created_at = _utc_now()
        try:
            with self._transaction() as connection:
                cursor = connection.execute(
                    """
                    INSERT INTO tool_audit (
                        conversation_id, tool_name, argument_digest,
                        approved, status, error_type, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        conversation_id,
                        tool_name,
                        argument_digest,
                        int(approved),
                        status,
                        error_type,
                        created_at,
                    ),
                )
                row = connection.execute(
                    "SELECT * FROM tool_audit WHERE id = ?", (cursor.lastrowid,)
                ).fetchone()
                return self._tool_audit_from_row(row)
        except sqlite3.IntegrityError as exc:
            raise ConversationNotFoundError(
                f"Conversation '{conversation_id}' does not exist."
            ) from exc
        except sqlite3.Error as exc:
            raise ConversationStoreError(
                "Could not record the local tool audit event."
            ) from exc

    def list_tool_audit(
        self, conversation_id: str | None = None, limit: int = 100
    ) -> tuple[ToolAuditRecord, ...]:
        """Read recent privacy-minimized tool audit records."""
        if limit < 1:
            raise ValueError("limit must be at least one.")
        with self._lock:
            try:
                if conversation_id is None:
                    rows = self._connection.execute(
                        "SELECT * FROM tool_audit ORDER BY id DESC LIMIT ?",
                        (limit,),
                    ).fetchall()
                else:
                    rows = self._connection.execute(
                        """
                        SELECT * FROM tool_audit
                        WHERE conversation_id = ?
                        ORDER BY id DESC LIMIT ?
                        """,
                        (conversation_id, limit),
                    ).fetchall()
                return tuple(self._tool_audit_from_row(row) for row in rows)
            except sqlite3.Error as exc:
                raise ConversationStoreError(
                    "Could not retrieve local tool audit records."
                ) from exc

    def finish_tool_execution(
        self, audit_id: int, status: str, error_type: str | None = None
    ) -> ToolAuditRecord:
        """Finalize a started tool audit event without changing its digest."""
        if status not in {"succeeded", "failed"}:
            raise ValueError("Final tool status must be succeeded or failed.")
        if audit_id < 1:
            raise ValueError("audit_id must be positive.")
        try:
            with self._transaction() as connection:
                cursor = connection.execute(
                    """
                    UPDATE tool_audit SET status = ?, error_type = ?
                    WHERE id = ? AND status = 'started'
                    """,
                    (status, error_type, audit_id),
                )
                if cursor.rowcount != 1:
                    raise ConversationStoreError(
                        "Tool audit event is missing or already finalized."
                    )
                row = connection.execute(
                    "SELECT * FROM tool_audit WHERE id = ?", (audit_id,)
                ).fetchone()
                return self._tool_audit_from_row(row)
        except ConversationStoreError:
            raise
        except sqlite3.Error as exc:
            raise ConversationStoreError(
                "Could not finalize the local tool audit event."
            ) from exc

    def _initialize_schema(self) -> None:
        with self._lock:
            self._connection.execute("BEGIN EXCLUSIVE")
            try:
                version = self._connection.execute(
                    "PRAGMA user_version"
                ).fetchone()[0]
                if version > self.SCHEMA_VERSION:
                    raise UnsupportedSchemaVersionError(
                        f"Database schema version {version} is newer than the "
                        f"supported version {self.SCHEMA_VERSION}."
                    )
                if version == 0:
                    self._connection.execute(
                        """
                        CREATE TABLE conversations (
                            id TEXT PRIMARY KEY,
                            created_at TEXT NOT NULL,
                            updated_at TEXT NOT NULL,
                            summary TEXT,
                            summary_through_message_id INTEGER,
                            revision INTEGER NOT NULL DEFAULT 0 CHECK (revision >= 0)
                        )
                        """
                    )
                    self._connection.execute(
                        """
                        CREATE TABLE messages (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            conversation_id TEXT NOT NULL
                                REFERENCES conversations(id) ON DELETE CASCADE,
                            role TEXT NOT NULL CHECK (
                                role IN ('user', 'assistant', 'tool')
                            ),
                            content TEXT NOT NULL CHECK (length(trim(content)) > 0),
                            created_at TEXT NOT NULL
                        )
                        """
                    )
                    self._connection.execute(
                        """
                        CREATE INDEX messages_conversation_id_id
                            ON messages(conversation_id, id)
                        """
                    )
                    version = 1
                if version == 1:
                    self._connection.execute(
                        """
                        CREATE TABLE tool_audit (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            conversation_id TEXT REFERENCES conversations(id)
                                ON DELETE SET NULL,
                            tool_name TEXT NOT NULL,
                            argument_digest TEXT NOT NULL,
                            approved INTEGER NOT NULL CHECK (approved IN (0, 1)),
                            status TEXT NOT NULL CHECK (
                                status IN ('started', 'succeeded', 'failed', 'denied')
                            ),
                            error_type TEXT,
                            created_at TEXT NOT NULL
                        )
                        """
                    )
                    self._connection.execute(
                        """
                        CREATE INDEX tool_audit_conversation_id_id
                            ON tool_audit(conversation_id, id)
                        """
                    )
                    self._connection.execute("PRAGMA user_version = 2")
                self._connection.commit()
            except (sqlite3.Error, UnsupportedSchemaVersionError):
                self._connection.rollback()
                raise

    def _restrict_database_file_permissions(self) -> None:
        if os.name == "nt" or self.database_path == ":memory:":
            return
        try:
            os.chmod(self.database_path, 0o600)
        except OSError as exc:
            raise ConversationStoreError(
                "Could not restrict permissions on the local conversation database."
            ) from exc

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield self._connection
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise

    @staticmethod
    def _conversation_from_row(row: sqlite3.Row) -> Conversation:
        return Conversation(
            id=row["id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            summary=row["summary"],
            summary_through_message_id=row["summary_through_message_id"],
            revision=row["revision"],
        )

    @staticmethod
    def _message_from_row(row: sqlite3.Row) -> StoredMessage:
        return StoredMessage(
            id=row["id"],
            conversation_id=row["conversation_id"],
            role=row["role"],
            content=row["content"],
            created_at=row["created_at"],
        )

    @staticmethod
    def _tool_audit_from_row(row: sqlite3.Row) -> ToolAuditRecord:
        return ToolAuditRecord(
            id=row["id"],
            conversation_id=row["conversation_id"],
            tool_name=row["tool_name"],
            argument_digest=row["argument_digest"],
            approved=bool(row["approved"]),
            status=row["status"],
            error_type=row["error_type"],
            created_at=row["created_at"],
        )
