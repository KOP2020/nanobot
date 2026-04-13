"""Minimal tool abstractions for the digital human demo."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Tool(ABC):
    """Abstract base class for tools."""

    _TYPE_MAP = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool name."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable tool description."""

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """JSON Schema for tool input."""

    @abstractmethod
    async def execute(self, **kwargs: Any) -> Any:
        """Execute the tool."""

    def cast_params(self, params: dict[str, Any]) -> dict[str, Any]:
        return params if isinstance(params, dict) else {}

    def validate_params(self, params: dict[str, Any]) -> list[str]:
        if not isinstance(params, dict):
            return [f"parameters must be an object, got {type(params).__name__}"]
        schema = self.parameters or {}
        required = schema.get("required", [])
        errors = [f"missing required {name}" for name in required if name not in params]
        properties = schema.get("properties", {})
        for key, expected in properties.items():
            if key not in params:
                continue
            kind = expected.get("type")
            if kind in self._TYPE_MAP and not isinstance(params[key], self._TYPE_MAP[kind]):
                errors.append(f"{key} should be {kind}")
        return errors

    def to_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """Registry for runtime tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get_definitions(self) -> list[dict[str, Any]]:
        return [tool.to_schema() for tool in self._tools.values()]

    async def execute(self, name: str, params: dict[str, Any]) -> Any:
        tool = self._tools.get(name)
        if tool is None:
            return f"Error: Tool '{name}' not found."
        params = tool.cast_params(params)
        errors = tool.validate_params(params)
        if errors:
            return f"Error: Invalid parameters for tool '{name}': " + "; ".join(errors)
        try:
            return await tool.execute(**params)
        except Exception as exc:
            return f"Error executing {name}: {exc}"
