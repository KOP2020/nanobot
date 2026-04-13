"""Conversation-level locks and message persistence."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

from app.storage import ConversationSessionStore


class ConversationRuntime:
    """Wrap persisted sessions with per-conversation serialization."""

    def __init__(self, sessions: ConversationSessionStore) -> None:
        self._sessions = sessions
        self._locks: dict[str, asyncio.Lock] = {}

    def lock_for(self, conversation_id: str) -> asyncio.Lock:
        """Return the lock for a given conversation."""
        return self._locks.setdefault(conversation_id, asyncio.Lock())

    def history(self, session_key: str, max_messages: int = 40) -> list[dict[str, Any]]:
        """Load persisted history."""
        return self._sessions.get_or_create(session_key).history(max_messages=max_messages)

    def append_messages(self, session_key: str, messages: list[dict[str, Any]]) -> None:
        """Append a new turn to the persisted session."""
        session = self._sessions.get_or_create(session_key)
        for message in messages:
            if message.get("role") == "assistant" and not message.get("content") and not message.get("tool_calls"):
                continue
            message.setdefault("timestamp", datetime.now(UTC).isoformat())
            session.messages.append(message)
        session.updated_at = datetime.now(UTC)
        self._sessions.save(session)
