"""Agent runtime service for scholar conversations."""

from __future__ import annotations

from dataclasses import dataclass

from app.core.context import ScholarContextBuilder
from app.core.llm import LLMProvider
from app.core.runner import AgentRunSpec, AgentRunner
from app.core.session import ConversationRuntime
from app.core.subagents import ScholarSubagentManager
from app.core.tools import Tool, ToolRegistry
from app.domain.retrieval import RetrievalService
from app.models import ChatTurnResponse, Conversation, DigitalHuman
from app.storage import AppRepositories


@dataclass(slots=True)
class RuntimeRequestContext:
    """Information needed to execute one turn."""

    digital_human: DigitalHuman
    conversation: Conversation
    user_message: str


class ScholarAgentRuntime:
    """Coordinate context building, tool execution, and session persistence."""

    def __init__(
        self,
        *,
        provider: LLMProvider,
        model: str,
        repositories: AppRepositories,
    ) -> None:
        self._provider = provider
        self._model = model
        self._repositories = repositories
        self._context_builder = ScholarContextBuilder()
        self._runner = AgentRunner(provider)
        self._conversation_runtime = ConversationRuntime(repositories.sessions)
        self._subagents = ScholarSubagentManager(provider, repositories.tasks, model)

    async def run_turn(self, request: RuntimeRequestContext) -> ChatTurnResponse:
        """Run a single user turn for a conversation."""
        conversation = request.conversation
        lock = self._conversation_runtime.lock_for(conversation.id)
        async with lock:
            retrieval = RetrievalService(request.digital_human)
            history = self._conversation_runtime.history(conversation.session_key)
            initial_messages = self._context_builder.build_messages(
                request.digital_human,
                conversation,
                history,
                request.user_message,
            )
            task_ids: list[str] = []
            registry = ToolRegistry()
            for tool in self._build_tools(request.digital_human, retrieval, task_ids):
                registry.register(tool)

            result = await self._runner.run(
                AgentRunSpec(
                    initial_messages=initial_messages,
                    tools=registry,
                    model=self._model,
                    max_iterations=12,
                )
            )
            final_message = result.final_content or "I have no grounded answer for that yet."
            new_messages = result.messages[len(initial_messages):]
            self._conversation_runtime.append_messages(
                conversation.session_key,
                [{"role": "user", "content": request.user_message}, *new_messages],
            )
            return ChatTurnResponse(
                conversation_id=conversation.id,
                digital_human_id=request.digital_human.id,
                message=final_message,
                task_ids=task_ids,
            )

    def _build_tools(
        self,
        digital_human: DigitalHuman,
        retrieval: RetrievalService,
        task_ids: list[str],
    ) -> list[Tool]:
        """Create scholar-specific tools for one turn."""

        runtime = self

        class FetchScholarProfileTool(Tool):
            @property
            def name(self) -> str:
                return "fetch_scholar_profile"

            @property
            def description(self) -> str:
                return "Return the scholar profile for the active digital human."

            @property
            def parameters(self) -> dict:
                return {
                    "type": "object",
                    "properties": {"open_sid": {"type": "string"}},
                    "required": ["open_sid"],
                }

            async def execute(self, **kwargs: str) -> str:
                if kwargs["open_sid"] != digital_human.open_sid:
                    return "Error: open_sid does not match the active digital human."
                profile = digital_human.profile
                return (
                    f"Scholar {profile.name}; institution={profile.institution or 'Unknown'}; "
                    f"title={profile.title or 'Unknown'}; "
                    f"fields={', '.join(profile.research_fields) or 'Unknown'}; "
                    f"summary={profile.profile_summary or 'None'}"
                )

        class FetchPublicationsTool(Tool):
            @property
            def name(self) -> str:
                return "fetch_publications"

            @property
            def description(self) -> str:
                return "Return the list of publications for the active digital human."

            @property
            def parameters(self) -> dict:
                return {
                    "type": "object",
                    "properties": {"open_sid": {"type": "string"}},
                    "required": ["open_sid"],
                }

            async def execute(self, **kwargs: str) -> str:
                if kwargs["open_sid"] != digital_human.open_sid:
                    return "Error: open_sid does not match the active digital human."
                return "\n".join(
                    f"[{publication.pub_id}] {publication.title} "
                    f"({publication.publish_year or 'Unknown'}, {publication.journal_or_venue or 'Unknown'})"
                    for publication in retrieval.list_publications()
                )

        class FetchPublicationAbstractTool(Tool):
            @property
            def name(self) -> str:
                return "fetch_publication_abstract"

            @property
            def description(self) -> str:
                return "Return the abstract for a publication when title-only matching is uncertain."

            @property
            def parameters(self) -> dict:
                return {
                    "type": "object",
                    "properties": {"pub_id": {"type": "string"}},
                    "required": ["pub_id"],
                }

            async def execute(self, **kwargs: str) -> str:
                publication = retrieval.get_publication(kwargs["pub_id"])
                if publication is None:
                    return "Error: publication not found."
                return publication.abstract or "No abstract available for this publication."

        class LoadPaperPdfTool(Tool):
            @property
            def name(self) -> str:
                return "load_paper_pdf"

            @property
            def description(self) -> str:
                return "Return the known PDF path or URL for a publication."

            @property
            def parameters(self) -> dict:
                return {
                    "type": "object",
                    "properties": {"pub_id": {"type": "string"}},
                    "required": ["pub_id"],
                }

            async def execute(self, **kwargs: str) -> str:
                publication = retrieval.get_publication(kwargs["pub_id"])
                if publication is None:
                    return "Error: publication not found."
                return publication.pdf_path or publication.pdf_url or "PDF is not available."

        class QueryGraphRelationsTool(Tool):
            @property
            def name(self) -> str:
                return "query_graph_relations"

            @property
            def description(self) -> str:
                return "Query precomputed graph relations for a scholar or publication."

            @property
            def parameters(self) -> dict:
                return {
                    "type": "object",
                    "properties": {
                        "entity_id": {"type": "string"},
                        "relation_type": {"type": "string"},
                    },
                    "required": ["entity_id"],
                }

            async def execute(self, **kwargs: str) -> str:
                relation_type = kwargs.get("relation_type") or "generic_relation"
                return (
                    f"Graph relations are not configured in this demo yet. "
                    f"Requested entity={kwargs['entity_id']}, relation_type={relation_type}."
                )

        class ReadPaperDeepTool(Tool):
            @property
            def name(self) -> str:
                return "read_paper_deep"

            @property
            def description(self) -> str:
                return "Run a focused subagent over one publication's metadata for deeper analysis."

            @property
            def parameters(self) -> dict:
                return {
                    "type": "object",
                    "properties": {
                        "pub_id": {"type": "string"},
                        "focus_question": {"type": "string"},
                    },
                    "required": ["pub_id", "focus_question"],
                }

            async def execute(self, **kwargs: str) -> str:
                publication = retrieval.get_publication(kwargs["pub_id"])
                if publication is None:
                    return "Error: publication not found."
                task = await runtime._subagents.read_paper(
                    scholar_name=digital_human.profile.name,
                    pub_id=publication.pub_id,
                    title=publication.title,
                    abstract=publication.abstract or "",
                    user_question=kwargs["focus_question"],
                )
                task_ids.append(task.id)
                return task.output.get("summary", "") if task.output else "No summary available."

        return [
            FetchScholarProfileTool(),
            FetchPublicationsTool(),
            FetchPublicationAbstractTool(),
            LoadPaperPdfTool(),
            QueryGraphRelationsTool(),
            ReadPaperDeepTool(),
        ]
