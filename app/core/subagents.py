"""Structured subagent execution for long-running scholar tasks."""

from __future__ import annotations

from datetime import UTC, datetime

from app.core.llm import LLMProvider
from app.core.runner import AgentRunSpec, AgentRunner
from app.core.tools import ToolRegistry
from app.models import TaskRecord
from app.storage import JsonRepository


class ScholarSubagentManager:
    """Run focused paper-reading tasks and persist their results."""

    def __init__(self, provider: LLMProvider, task_repo: JsonRepository[TaskRecord], model: str) -> None:
        self._provider = provider
        self._task_repo = task_repo
        self._model = model
        self._runner = AgentRunner(provider)

    async def read_paper(
        self,
        *,
        scholar_name: str,
        pub_id: str,
        title: str,
        abstract: str,
        user_question: str,
    ) -> TaskRecord:
        """Run a focused subagent turn over paper metadata."""
        task = TaskRecord(
            task_type="read_paper_deep",
            input={
                "scholar_name": scholar_name,
                "pub_id": pub_id,
                "title": title,
                "question": user_question,
            },
        )
        self._task_repo.save(task)

        system_prompt = (
            "You are a paper-reading subagent for a scholar digital human.\n"
            "Work only from the provided paper metadata.\n"
            "Return a concise grounded summary with 2-3 evidence-backed points and note uncertainty."
        )
        user_prompt = (
            f"Scholar: {scholar_name}\n"
            f"Publication: [{pub_id}] {title}\n"
            f"Abstract: {abstract or 'No abstract available.'}\n"
            f"Question: {user_question}"
        )
        result = await self._runner.run(
            AgentRunSpec(
                initial_messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                tools=ToolRegistry(),
                model=self._model,
                max_iterations=1,
            )
        )
        task.status = "completed" if result.stop_reason == "completed" else result.stop_reason
        task.output = {"summary": result.final_content or ""}
        task.updated_at = datetime.now(UTC)
        self._task_repo.save(task)
        return task
