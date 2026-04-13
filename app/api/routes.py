"""FastAPI routes for the digital human demo."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, Request

from app.core.runtime import RuntimeRequestContext
from app.models import (
    ChatTurnResponse,
    Conversation,
    CreateConversationRequest,
    CreateDigitalHumanRequest,
    DigitalHuman,
    SendMessageRequest,
)

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    """Health check."""
    return {"status": "ok"}


@router.post("/digital-humans", response_model=DigitalHuman)
async def create_digital_human(
    payload: CreateDigitalHumanRequest,
    request: Request,
) -> DigitalHuman:
    """Create a digital human from an OpenSID."""
    services = request.app.state.services
    digital_human = await services.scholar_client.load_digital_human(payload.open_sid)
    digital_human.updated_at = datetime.now(UTC)
    services.repositories.digital_humans.save(digital_human)
    return digital_human


@router.post("/conversations", response_model=Conversation)
async def create_conversation(
    payload: CreateConversationRequest,
    request: Request,
) -> Conversation:
    """Create a conversation for an existing digital human."""
    services = request.app.state.services
    digital_human = services.repositories.digital_humans.get(payload.digital_human_id)
    if digital_human is None:
        raise HTTPException(status_code=404, detail="Digital human not found.")
    conversation = Conversation(
        digital_human_id=digital_human.id,
        end_user_id=payload.end_user_id,
    )
    services.repositories.conversations.save(conversation)
    return conversation


@router.post("/conversations/{conversation_id}/messages", response_model=ChatTurnResponse)
async def send_message(
    conversation_id: str,
    payload: SendMessageRequest,
    request: Request,
) -> ChatTurnResponse:
    """Run one user turn."""
    services = request.app.state.services
    conversation = services.repositories.conversations.get(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    digital_human = services.repositories.digital_humans.get(conversation.digital_human_id)
    if digital_human is None:
        raise HTTPException(status_code=404, detail="Digital human not found.")
    return await services.runtime.run_turn(
        RuntimeRequestContext(
            digital_human=digital_human,
            conversation=conversation,
            user_message=payload.message,
        )
    )


@router.get("/tasks/{task_id}")
async def get_task(task_id: str, request: Request) -> dict:
    """Return the latest task status."""
    services = request.app.state.services
    task = services.repositories.tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found.")
    return task.model_dump(mode="json")
