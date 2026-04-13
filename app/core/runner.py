"""Minimal tool-capable agent runner for the digital human demo."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.core.llm import LLMProvider, ToolCallRequest
from app.core.tools import ToolRegistry


def build_assistant_message(
    content: str | None,
    *,
    tool_calls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a normalized assistant message."""
    payload: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        payload["tool_calls"] = tool_calls
    return payload


@dataclass(slots=True)
class AgentRunSpec:
    """Configuration for one agent turn."""

    initial_messages: list[dict[str, Any]]
    tools: ToolRegistry
    model: str
    max_iterations: int
    temperature: float = 0.1
    max_tokens: int = 4096


@dataclass(slots=True)
class AgentRunResult:
    """Output of one agent turn."""

    final_content: str | None
    messages: list[dict[str, Any]]
    tools_used: list[str] = field(default_factory=list)
    stop_reason: str = "completed"
    error: str | None = None


class AgentRunner:
    """Small shared execution loop with tool support."""

    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    async def run(self, spec: AgentRunSpec) -> AgentRunResult:
        messages = list(spec.initial_messages)
        tools_used: list[str] = []
        final_content: str | None = None

        for _ in range(spec.max_iterations):
            response = await self._provider.chat_with_retry(
                messages,
                tools=spec.tools.get_definitions(),
                model=spec.model,
                max_tokens=spec.max_tokens,
                temperature=spec.temperature,
            )

            if response.finish_reason == "error":
                return AgentRunResult(
                    final_content=response.content,
                    messages=messages,
                    tools_used=tools_used,
                    stop_reason="error",
                    error=response.content,
                )

            if response.has_tool_calls:
                messages.append(
                    build_assistant_message(
                        response.content or "",
                        tool_calls=[call.to_openai_tool_call() for call in response.tool_calls],
                    )
                )
                tools_used.extend(call.name for call in response.tool_calls)
                for tool_call in response.tool_calls:
                    tool_result = await spec.tools.execute(tool_call.name, tool_call.arguments)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call.name,
                            "content": tool_result,
                        }
                    )
                continue

            final_content = response.content
            messages.append(build_assistant_message(final_content))
            return AgentRunResult(
                final_content=final_content,
                messages=messages,
                tools_used=tools_used,
            )

        return AgentRunResult(
            final_content=(
                f"I reached the maximum number of tool call iterations ({spec.max_iterations}) "
                "without completing the task."
            ),
            messages=messages,
            tools_used=tools_used,
            stop_reason="max_iterations",
        )
