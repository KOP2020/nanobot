"""FastAPI entrypoint for the digital human demo."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import FastAPI

from app.api.routes import router
from app.config import AppSettings
from app.core.llm import DemoEchoProvider, OpenAICompatProvider
from app.core.runtime import ScholarAgentRuntime
from app.domain.scholar import ScholarDataClient
from app.storage import AppRepositories


@dataclass(slots=True)
class AppServices:
    """Services exposed through FastAPI app state."""

    repositories: AppRepositories
    scholar_client: ScholarDataClient
    runtime: ScholarAgentRuntime


def create_app(
    settings: AppSettings | None = None,
    *,
    runtime: ScholarAgentRuntime | None = None,
    repositories: AppRepositories | None = None,
    scholar_client: ScholarDataClient | None = None,
) -> FastAPI:
    """Create the FastAPI application."""
    settings = settings or AppSettings.from_env()
    repositories = repositories or AppRepositories(settings.workspace)
    scholar_client = scholar_client or ScholarDataClient(settings.scholar_api)
    runtime = runtime or _build_runtime(settings, repositories)

    app = FastAPI(title="Digital Human Demo", version="0.1.0")
    app.include_router(router, prefix="/api")
    app.state.settings = settings
    app.state.services = AppServices(
        repositories=repositories,
        scholar_client=scholar_client,
        runtime=runtime,
    )
    return app


def _build_runtime(settings: AppSettings, repositories: AppRepositories) -> ScholarAgentRuntime:
    """Build the scholar runtime from app-local provider settings."""
    provider = _build_provider(settings)
    return ScholarAgentRuntime(
        provider=provider,
        model=settings.model or provider.get_default_model(),
        repositories=repositories,
    )


def _build_provider(settings: AppSettings):
    """Create the runtime LLM provider."""
    if settings.provider_api_key and settings.provider_api_base:
        return OpenAICompatProvider(
            api_key=settings.provider_api_key,
            api_base=settings.provider_api_base,
            default_model=settings.model or "gpt-4o-mini",
            timeout_s=settings.provider_timeout_s,
        )
    return DemoEchoProvider(model=settings.model or "demo/echo")


def run() -> None:
    """Run the development server."""
    import uvicorn

    settings = AppSettings.from_env()
    uvicorn.run(
        "app.main:create_app",
        host=settings.host,
        port=settings.port,
        factory=True,
        reload=False,
    )
