from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from app.memory.conversation_store import (
    ConversationChangedError,
    ConversationNotFoundError,
    ConversationStore,
    ConversationStoreError,
    UnsupportedSchemaVersionError,
)


def test_persists_conversations_and_messages_across_reopen(tmp_path: Path) -> None:
    database = tmp_path / "data" / "conversations.sqlite3"
    store = ConversationStore(database)
    conversation = store.create_conversation()
    user = store.add_message(conversation.id, "user", "Hello")
    store.add_message(conversation.id, "assistant", "Hi there")
    store.close()

    reopened = ConversationStore(database)
    snapshot = reopened.get_conversation(conversation.id)

    assert snapshot.conversation.id == conversation.id
    assert snapshot.conversation.revision == 2
    assert [message.content for message in snapshot.messages] == [
        "Hello",
        "Hi there",
    ]
    assert snapshot.messages[0].id == user.id
    assert snapshot.messages[0].created_at
    reopened.close()


def test_latest_message_limit_preserves_chronological_order(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path / "messages.sqlite3")
    conversation = store.create_conversation()
    for content in ("first", "second", "third"):
        store.add_message(conversation.id, "user", content)

    snapshot = store.get_conversation(conversation.id, message_limit=2)

    assert [message.content for message in snapshot.messages] == ["second", "third"]
    store.close()


def test_lists_and_deletes_conversations_with_cascade(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "delete.sqlite3")
    first = store.create_conversation()
    second = store.create_conversation()
    store.add_message(first.id, "user", "remove me")
    store.add_message(second.id, "user", "keep me")

    listed = store.list_conversations(limit=1)
    assert len(listed) == 1
    assert store.delete_conversation(first.id) is True
    assert store.delete_conversation(first.id) is False
    assert store.delete_all_conversations() == 1
    assert store.list_conversations() == ()

    with pytest.raises(ConversationNotFoundError):
        store.get_conversation(first.id)
    store.close()


@pytest.mark.parametrize(
    ("role", "content"),
    [("system", "not a stored turn"), ("user", " "), ("invalid", "hello")],
)
def test_message_validation(role: str, content: str, tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "validation.sqlite3")
    conversation = store.create_conversation()

    with pytest.raises(ValueError):
        store.add_message(conversation.id, role, content)
    store.close()


def test_missing_conversation_and_invalid_message_are_explicit(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path / "missing.sqlite3")

    with pytest.raises(ConversationNotFoundError):
        store.add_message("missing", "user", "Hello")
    store.close()


def test_summary_uses_revision_guard_and_validates_boundary(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path / "summary.sqlite3")
    conversation = store.create_conversation()
    message = store.add_message(conversation.id, "user", "remember this")
    updated = store.save_summary(
        conversation.id,
        "A short summary.",
        through_message_id=message.id,
        expected_revision=1,
    )

    assert updated.summary == "A short summary."
    assert updated.summary_through_message_id == message.id
    with pytest.raises(ConversationChangedError):
        store.save_summary(
            conversation.id,
            "stale summary",
            through_message_id=message.id,
            expected_revision=0,
        )
    with pytest.raises(ValueError, match="message in this conversation"):
        store.save_summary(
            conversation.id,
            "wrong boundary",
            through_message_id=999,
            expected_revision=1,
        )
    store.close()


def test_rejects_newer_database_schema(tmp_path: Path) -> None:
    database = tmp_path / "future.sqlite3"
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA user_version = 999")
    connection.close()

    with pytest.raises(UnsupportedSchemaVersionError, match="newer"):
        ConversationStore(database)


def test_database_file_permissions_are_restricted_on_posix(tmp_path: Path) -> None:
    if __import__("os").name == "nt":
        pytest.skip("POSIX permission bits are not portable to Windows.")
    database = tmp_path / "private.sqlite3"
    store = ConversationStore(database)

    assert database.stat().st_mode & 0o777 == 0o600
    store.close()


def test_database_path_error_is_reported(tmp_path: Path) -> None:
    file_path = tmp_path / "not-a-directory"
    file_path.write_text("not a folder", encoding="utf-8")

    with pytest.raises(ConversationStoreError, match="directory"):
        ConversationStore(file_path / "conversations.sqlite3")


def test_migrates_existing_v1_database_to_tool_audit_v2(tmp_path: Path) -> None:
    database = tmp_path / "version-one.sqlite3"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE conversations (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            summary TEXT,
            summary_through_message_id INTEGER,
            revision INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL REFERENCES conversations(id)
                ON DELETE CASCADE,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        PRAGMA user_version = 1;
        """
    )
    connection.close()

    store = ConversationStore(database)
    version = store._connection.execute("PRAGMA user_version").fetchone()[0]

    assert version == 2
    store.close()
