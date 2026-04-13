"""JSON-backed repositories for the digital human demo."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from app.models import Conversation, DigitalHuman, TaskRecord

T = TypeVar("T", bound=BaseModel)


def ensure_dir(path: Path) -> Path:
    """Create a directory if missing."""
    path.mkdir(parents=True, exist_ok=True)
    return path


class JsonRepository(Generic[T]):
    """Persist one record per file."""

    def __init__(self, root: Path, model_cls: type[T]) -> None:
        self._root = ensure_dir(root)
        self._model_cls = model_cls

    def _path(self, item_id: str) -> Path:
        return self._root / f"{item_id}.json"

    def save(self, record: T) -> T:
        self._path(record.id).write_text(
            record.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return record

    def get(self, item_id: str) -> T | None:
        path = self._path(item_id)
        if not path.exists():
            return None
        return self._model_cls.model_validate_json(path.read_text(encoding="utf-8"))

    def list(self) -> list[T]:
        items: list[T] = []
        for path in sorted(self._root.glob("*.json")):
            items.append(self._model_cls.model_validate_json(path.read_text(encoding="utf-8")))
        return items


@dataclass
class ConversationSession:
    """Conversation history aligned to agent message boundaries."""

    key: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @staticmethod
    def _find_legal_start(messages: list[dict[str, Any]]) -> int:
        declared: set[str] = set()
        start = 0
        for idx, message in enumerate(messages):
            role = message.get("role")
            if role == "assistant":
                for tool_call in message.get("tool_calls") or []:
                    if isinstance(tool_call, dict) and tool_call.get("id"):
                        declared.add(str(tool_call["id"]))
            elif role == "tool":
                tool_call_id = message.get("tool_call_id")
                if tool_call_id and str(tool_call_id) not in declared:
                    start = idx + 1
                    declared.clear()
        return start

    def history(self, max_messages: int = 40) -> list[dict[str, Any]]:
        sliced = self.messages[-max_messages:] if max_messages > 0 else list(self.messages)
        start = self._find_legal_start(sliced)
        if start:
            sliced = sliced[start:]
        return sliced


class ConversationSessionStore:
    """Persist conversation sessions as JSONL files."""

    def __init__(self, root: Path) -> None:
        self._root = ensure_dir(root)
        self._cache: dict[str, ConversationSession] = {}

    def _path(self, key: str) -> Path:
        safe = key.replace("/", "_").replace(":", "_")
        return self._root / f"{safe}.jsonl"

    def get_or_create(self, key: str) -> ConversationSession:
        if key in self._cache:
            return self._cache[key]
        session = self._load(key) or ConversationSession(key=key)
        self._cache[key] = session
        return session

    def _load(self, key: str) -> ConversationSession | None:
        path = self._path(key)
        if not path.exists():
            return None
        messages: list[dict[str, Any]] = []
        created_at = datetime.now(UTC)
        updated_at = created_at
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if payload.get("_type") == "metadata":
                    created_at = datetime.fromisoformat(payload["created_at"])
                    updated_at = datetime.fromisoformat(payload["updated_at"])
                else:
                    messages.append(payload)
        return ConversationSession(
            key=key,
            messages=messages,
            created_at=created_at,
            updated_at=updated_at,
        )

    def save(self, session: ConversationSession) -> None:
        session.updated_at = datetime.now(UTC)
        path = self._path(session.key)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "_type": "metadata",
                        "key": session.key,
                        "created_at": session.created_at.isoformat(),
                        "updated_at": session.updated_at.isoformat(),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            for message in session.messages:
                handle.write(json.dumps(message, ensure_ascii=False) + "\n")


class AppRepositories:
    """All JSON-backed repositories used by the app."""

    def __init__(self, workspace: Path) -> None:
        root = ensure_dir(workspace / "app_state")
        self.digital_humans = JsonRepository(root / "digital_humans", DigitalHuman)
        self.conversations = JsonRepository(root / "conversations", Conversation)
        self.tasks = JsonRepository(root / "tasks", TaskRecord)
        self.sessions = ConversationSessionStore(root / "sessions")
