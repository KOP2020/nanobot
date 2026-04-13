"""Configuration for the digital human demo server."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field


class ScholarApiSettings(BaseModel):
    """ScholarMate open-data settings."""

    endpoint: str = "https://open.scholarmate.com/scmopendata"
    openid: str = ""
    token: str = ""
    timeout_s: float = 30.0
    page_size: int = 100
    demo_mode: bool = True


class AppSettings(BaseModel):
    """Runtime settings for the digital human demo."""

    host: str = "127.0.0.1"
    port: int = 8000
    workspace: Path = Field(default_factory=lambda: Path("demo_workspace").resolve())
    model: str | None = None
    provider_api_base: str | None = None
    provider_api_key: str = ""
    provider_timeout_s: float = 120.0
    scholar_api: ScholarApiSettings = Field(default_factory=ScholarApiSettings)

    @classmethod
    def from_env(cls) -> "AppSettings":
        workspace = Path(
            os.environ.get("DIGITAL_HUMAN_WORKSPACE", "demo_workspace")
        ).expanduser().resolve()
        return cls(
            host=os.environ.get("DIGITAL_HUMAN_HOST", "127.0.0.1"),
            port=int(os.environ.get("DIGITAL_HUMAN_PORT", "8000")),
            workspace=workspace,
            model=os.environ.get("DIGITAL_HUMAN_MODEL") or None,
            provider_api_base=os.environ.get("DIGITAL_HUMAN_API_BASE") or None,
            provider_api_key=os.environ.get("DIGITAL_HUMAN_API_KEY", ""),
            provider_timeout_s=float(os.environ.get("DIGITAL_HUMAN_PROVIDER_TIMEOUT_S", "120")),
            scholar_api=ScholarApiSettings(
                endpoint=os.environ.get(
                    "SCHOLARMATE_OPEN_DATA_ENDPOINT",
                    "https://open.scholarmate.com/scmopendata",
                ),
                openid=os.environ.get("SCHOLARMATE_OPENID", ""),
                token=os.environ.get("SCHOLARMATE_TOKEN", ""),
                timeout_s=float(os.environ.get("SCHOLARMATE_TIMEOUT_S", "30")),
                page_size=int(os.environ.get("SCHOLARMATE_PAGE_SIZE", "100")),
                demo_mode=os.environ.get("SCHOLARMATE_DEMO_MODE", "true").lower()
                in {"1", "true", "yes"},
            ),
        )
