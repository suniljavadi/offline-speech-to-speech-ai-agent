from __future__ import annotations

from pathlib import Path

import pytest

from app.memory.conversation_store import ConversationChangedError, ConversationStore
from app.memory.memory_manager import ConversationMemoryManager
from app.services.ollama_service import ChatMessage


def test_context_restoration_respects_summary_boundary_and_limits(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path / "context.sqlite3")
    manager = ConversationMemoryManager(
        store, max_context_messages=2, max_context_characters=200
    )
    conversation = manager.start_conversation()
    first = manager.append_message(conversation.id, "user", "old question")
    manager.append_message(conversation.id, "assistant", "old answer")
    manager.append_message(conversation.id, "user", "new question")
    store.save_summary(
        conversation.id,
        "Old exchange was about a project.",
        through_message_id=first.id + 1,
        expected_revision=3,
    )

    context = manager.load_context(conversation.id)

    assert context.summary == "Old exchange was about a project."
    assert context.history == (
        ChatMessage(
            "assistant",
            "Summary of earlier conversation:\nOld exchange was about a project.",
        ),
        ChatMessage("user", "new question"),
    )
    store.close()


def test_context_respects_character_budget_by_prioritizing_recent_messages(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path / "budget.sqlite3")
    manager = ConversationMemoryManager(
        store, max_context_messages=10, max_context_characters=13
    )
    conversation = manager.start_conversation()
    manager.append_message(conversation.id, "user", "old message")
    manager.append_message(conversation.id, "assistant", "very latest answer")

    context = manager.load_context(conversation.id)

    assert context.history == (ChatMessage("assistant", "latest answer"),)
    assert sum(len(message.content) for message in context.history) <= 13
    store.close()


def test_summarize_persists_locally_and_excludes_summarized_messages(
    tmp_path: Path,
) -> None:
    transcripts = []

    def summarize(transcript: str) -> str:
        transcripts.append(transcript)
        return "User greeted the assistant."

    store = ConversationStore(tmp_path / "summarize.sqlite3")
    manager = ConversationMemoryManager(store, summary_generator=summarize)
    conversation = manager.start_conversation()
    manager.append_message(conversation.id, "user", "Hello")
    manager.append_message(conversation.id, "assistant", "Hi")

    summarized = manager.summarize(conversation.id)
    manager.append_message(conversation.id, "user", "Follow-up")
    context = manager.load_context(conversation.id)

    assert transcripts == ["user: Hello\nassistant: Hi"]
    assert summarized.summary == "User greeted the assistant."
    assert [message.role for message in context.history] == [
        "assistant",
        "user",
    ]
    assert context.history[1].content == "Follow-up"
    store.close()


def test_summary_generation_is_required_and_empty_results_are_rejected(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path / "summary-errors.sqlite3")
    conversation = store.create_conversation()
    store.add_message(conversation.id, "user", "Hello")

    with pytest.raises(RuntimeError, match="No local"):
        ConversationMemoryManager(store).summarize(conversation.id)
    manager = ConversationMemoryManager(
        store, summary_generator=lambda _text: " "
    )
    with pytest.raises(ValueError, match="empty summary"):
        manager.summarize(conversation.id)
    store.close()


def test_summary_source_limit_prevents_unbounded_generation(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "summary-limit.sqlite3")
    conversation = store.create_conversation()
    store.add_message(conversation.id, "user", "Long source")
    manager = ConversationMemoryManager(
        store,
        summary_generator=lambda _text: "summary",
        max_summary_source_characters=3,
    )

    with pytest.raises(ValueError, match="source limit"):
        manager.summarize(conversation.id)
    store.close()


def test_summary_is_rejected_if_conversation_changes_during_generation(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path / "summary-race.sqlite3")
    conversation = store.create_conversation()
    store.add_message(conversation.id, "user", "First message")

    def summarize(_transcript: str) -> str:
        store.add_message(conversation.id, "user", "Concurrent message")
        return "Stale summary"

    manager = ConversationMemoryManager(store, summary_generator=summarize)

    with pytest.raises(ConversationChangedError, match="changed"):
        manager.summarize(conversation.id)
    assert store.get_conversation(conversation.id).conversation.summary is None
    store.close()


def test_delete_memory_controls_remove_one_or_all_sessions(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "delete-manager.sqlite3")
    manager = ConversationMemoryManager(store)
    one = manager.start_conversation()
    two = manager.start_conversation()
    manager.append_message(one.id, "user", "remove")
    manager.append_message(two.id, "user", "also remove")

    assert manager.delete_conversation(one.id) is True
    assert manager.delete_all_conversations() == 1
    assert store.list_conversations() == ()
    store.close()
