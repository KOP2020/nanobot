"""Domain models for the digital human demo."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    """Return an aware UTC timestamp."""
    return datetime.now(UTC)


class Publication(BaseModel):
    """Publication metadata used by the agent."""

    pub_id: str
    title: str
    abstract: str | None = None
    journal_or_venue: str | None = None
    publish_year: int | None = None
    citation_count: int | None = None
    pdf_url: str | None = None
    pdf_path: str | None = None
    authors: list[str] = Field(default_factory=list)


class ScholarProfile(BaseModel):
    """Scholar profile injected into context."""

    open_sid: str
    name: str
    institution: str | None = None
    title: str | None = None
    research_fields: list[str] = Field(default_factory=list)
    profile_summary: str | None = None
    keywords: list[str] = Field(default_factory=list)


class ScholarControlConfig(BaseModel):
    """Scholar-level answer controls."""

    featured_pub_ids: list[str] = Field(default_factory=list)
    allowed_topics: list[str] = Field(default_factory=list)
    restricted_topics: list[str] = Field(default_factory=list)
    disclosure_level: str = "public_summary"
    style_preference: str = "rigorous_academic"
    private_notes: dict[str, str] = Field(default_factory=dict)


class DigitalHuman(BaseModel):
    """Persisted digital human definition."""

    id: str = Field(default_factory=lambda: f"dh_{uuid4().hex[:12]}")
    open_sid: str
    profile: ScholarProfile
    control_config: ScholarControlConfig = Field(default_factory=ScholarControlConfig)
    publications: list[Publication] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class Conversation(BaseModel):
    """Persisted conversation metadata."""

    id: str = Field(default_factory=lambda: f"conv_{uuid4().hex[:12]}")
    digital_human_id: str
    end_user_id: str
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @property
    def session_key(self) -> str:
        """Stable key for conversation history isolation."""
        return f"digital-human:{self.digital_human_id}:conversation:{self.id}"


class TaskRecord(BaseModel):
    """Background task record, mainly for subagent work."""

    id: str = Field(default_factory=lambda: f"task_{uuid4().hex[:12]}")
    task_type: str
    status: str = "running"
    input: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class ChatTurnResponse(BaseModel):
    """HTTP response returned after a message turn."""

    conversation_id: str
    digital_human_id: str
    message: str
    task_ids: list[str] = Field(default_factory=list)


class CreateDigitalHumanRequest(BaseModel):
    """Request to create a digital human from an OpenSID."""

    open_sid: str = Field(min_length=1)


class CreateConversationRequest(BaseModel):
    """Request to open a new end-user conversation."""

    digital_human_id: str = Field(min_length=1)
    end_user_id: str = Field(min_length=1)


class SendMessageRequest(BaseModel):
    """Request to run one chat turn."""

    message: str = Field(min_length=1)

