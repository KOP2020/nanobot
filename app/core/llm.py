"""Minimal LLM abstractions for the digital human demo."""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass(slots=True)
class ToolCallRequest:
    """Structured tool call emitted by the model."""

    id: str
    name: str
    arguments: dict[str, Any]

    def to_openai_tool_call(self) -> dict[str, Any]:
        """Serialize to OpenAI-compatible tool-call payload."""
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False),
            },
        }


@dataclass(slots=True)
class LLMResponse:
    """Normalized response for chat generations."""

    content: str | None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


class LLMProvider(ABC):
    """Abstract interface for chat-capable providers."""

    _RETRY_DELAYS = (1, 2, 4)

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.1,
    ) -> LLMResponse:
        """Send one chat completion request."""

    async def chat_with_retry(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.1,
    ) -> LLMResponse:
        """Retry transient failures."""
        last: LLMResponse | None = None
        for idx, delay in enumerate(self._RETRY_DELAYS, start=1):
            try:
                return await self.chat(
                    messages,
                    tools=tools,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last = LLMResponse(content=f"Error calling LLM: {exc}", finish_reason="error")
                if idx == len(self._RETRY_DELAYS):
                    break
                await asyncio.sleep(delay)
        return last or LLMResponse(content="Error calling LLM.", finish_reason="error")

    @abstractmethod
    def get_default_model(self) -> str:
        """Return the configured default model."""


class OpenAICompatProvider(LLMProvider):
    """Small OpenAI-compatible provider with tool-call support."""

    def __init__(
        self,
        *,
        api_key: str,
        api_base: str,
        default_model: str,
        timeout_s: float = 120.0,
    ) -> None:
        self._api_key = api_key
        self._api_base = api_base.rstrip("/")
        self._default_model = default_model
        self._timeout_s = timeout_s

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.1,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": model or self._default_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            response = await client.post(
                f"{self._api_base}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()

        data = response.json()
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        raw_tool_calls = message.get("tool_calls") or []
        tool_calls: list[ToolCallRequest] = []
        for tool_call in raw_tool_calls:
            function = tool_call.get("function") or {}
            arguments = function.get("arguments") or "{}"
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            tool_calls.append(
                ToolCallRequest(
                    id=str(tool_call.get("id") or ""),
                    name=str(function.get("name") or ""),
                    arguments=arguments if isinstance(arguments, dict) else {},
                )
            )
        usage = data.get("usage") or {}
        return LLMResponse(
            content=message.get("content"),
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason") or "stop",
            usage={
                "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            },
        )

    def get_default_model(self) -> str:
        return self._default_model


class DemoEchoProvider(LLMProvider):
    """Fallback provider that keeps the app bootable without external credentials."""

    def __init__(self, model: str = "demo/echo") -> None:
        self._model = model

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.1,
    ) -> LLMResponse:
        del tools, model, max_tokens, temperature
        user_message = next(
            (msg.get("content") for msg in reversed(messages) if msg.get("role") == "user"),
            "",
        )
        return LLMResponse(
            content=(
                "Demo provider is active because no LLM credentials are configured. "
                f"Latest user message: {user_message}"
            )
        )

    def get_default_model(self) -> str:
        return self._model
