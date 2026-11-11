"""Conversation restoration, bounded LLM context, and local summarization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.memory.conversation_store import (
    Conversation,
    ConversationNotFoundError,
    ConversationSnapshot,
    ConversationStore,
    StoredMessage,
)
from app.services.ollama_service import ChatMessage


class SummaryGenerator(Protocol):
    """Generate a summary using a caller-selected local model."""

    def __call__(self, transcript: str) -> str:
        """Return a concise summary of the supplied conversation transcript."""
        ...


@dataclass(frozen=True)
class ConversationContext:
    """Summary plus bounded, recent history ready for a local chat request."""

    conversation: Conversation
    summary: str | None
    history: tuple[ChatMessage, ...]


class ConversationMemoryManager:
    """Manage persistent sessions without logging or exporting conversation data."""

    def __init__(
        self,
        store: ConversationStore,
        *,
        max_context_messages: int = 20,
        max_context_characters: int = 12000,
        summary_generator: SummaryGenerator | None = None,
        max_summary_source_characters: int = 50000,
    ) -> None:
        if max_context_messages < 1:
            raise ValueError("max_context_messages must be at least one.")
        if max_context_characters < 1:
            raise ValueError("max_context_characters must be at least one.")
        if max_summary_source_characters < 1:
            raise ValueError("max_summary_source_characters must be at least one.")
        self.store = store
        self.max_context_messages = max_context_messages
        self.max_context_characters = max_context_characters
        self.summary_generator = summary_generator
        self.max_summary_source_characters = max_summary_source_characters

    def start_conversation(self, conversation_id: str | None = None) -> Conversation:
        """Create a durable session or restore-by-id if that id already exists."""
        if conversation_id is None:
            return self.store.create_conversation()
        try:
            return self.store.get_conversation(
                conversation_id, message_limit=1
            ).conversation
        except ConversationNotFoundError:
            pass
        return self.store.create_conversation(conversation_id)

    def append_message(
        self, conversation_id: str, role: str, content: str
    ) -> StoredMessage:
        """Persist a single conversation message."""
        return self.store.add_message(conversation_id, role, content)

    def restore(
        self, conversation_id: str, message_limit: int | None = None
    ) -> ConversationSnapshot:
        """Restore conversation metadata and a chronological message window."""
        return self.store.get_conversation(conversation_id, message_limit)

    def load_context(self, conversation_id: str) -> ConversationContext:
        """Retrieve summary and recent messages within a character/message budget."""
        snapshot = self.store.get_conversation(conversation_id, message_limit=1)
        conversation = snapshot.conversation
        summary = conversation.summary
        summary_message: ChatMessage | None = None
        if summary:
            summary_prefix = "Summary of earlier conversation:\n"
            available = self.max_context_characters - len(summary_prefix)
            if available <= 0:
                raise ValueError(
                    "Context character limit is too small to include a summary."
                )
            truncated = summary[:available]
            if len(summary) > available and available > 1:
                truncated = truncated[:-1] + "…"
            summary_message = ChatMessage(
                "assistant", summary_prefix + truncated
            )

        used_characters = 0
        available_characters = (
            self.max_context_characters
            - (len(summary_message.content) if summary_message else 0)
        )
        after_id = conversation.summary_through_message_id or 0
        messages = self.store.get_messages_after(
            conversation_id,
            after_id,
            max(self.max_context_messages, 1),
        )

        selected: list[ChatMessage] = []
        for message in reversed(messages):
            remaining = available_characters - used_characters
            if remaining <= 0:
                break
            content = message.content
            if len(content) > remaining:
                if not selected:
                    content = content[-remaining:]
                    selected.append(ChatMessage(message.role, content))
                break
            selected.append(ChatMessage(message.role, content))
            used_characters += len(content)
        selected.reverse()

        return ConversationContext(
            conversation=conversation,
            summary=summary,
            history=tuple(
                ([summary_message] if summary_message is not None else [])
                + selected
            ),
        )

    def summarize(self, conversation_id: str) -> Conversation:
        """Summarize a stable conversation snapshot using an injected local model."""
        if self.summary_generator is None:
            raise RuntimeError(
                "No local conversation summary generator is configured."
            )
        snapshot = self.store.get_conversation(conversation_id)
        if not snapshot.messages:
            raise ValueError("An empty conversation cannot be summarized.")
        transcript = "\n".join(
            f"{message.role}: {message.content}" for message in snapshot.messages
        )
        if len(transcript) > self.max_summary_source_characters:
            raise ValueError(
                "Conversation exceeds the configured summary source limit."
            )
        summary = self.summary_generator(transcript)
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError("The local summary generator returned an empty summary.")
        last_message_id = snapshot.messages[-1].id
        return self.store.save_summary(
            conversation_id,
            summary,
            through_message_id=last_message_id,
            expected_revision=snapshot.conversation.revision,
        )

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete one conversation and all associated messages and summaries."""
        return self.store.delete_conversation(conversation_id)

    def delete_all_conversations(self) -> int:
        """Delete all locally persisted conversations."""
        return self.store.delete_all_conversations()
