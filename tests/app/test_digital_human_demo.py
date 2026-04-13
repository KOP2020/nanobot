from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient

from app.config import AppSettings, ScholarApiSettings
from app.core.llm import LLMProvider, LLMResponse, ToolCallRequest
from app.core.runtime import RuntimeRequestContext, ScholarAgentRuntime
from app.domain.scholar import ScholarDataClient
from app.main import create_app
from app.models import Conversation
from app.storage import AppRepositories


class FakeProvider(LLMProvider):
    def __init__(self, *, delay: float = 0.0) -> None:
        self.delay = delay
        self.active_calls = 0
        self.max_active_calls = 0

    async def chat(
        self,
        messages,
        *,
        tools=None,
        model=None,
        max_tokens=4096,
        temperature=0.7,
    ) -> LLMResponse:
        self.active_calls += 1
        self.max_active_calls = max(self.max_active_calls, self.active_calls)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)

            if not tools:
                return LLMResponse(content="Paper deep-read summary grounded in the abstract.")

            last = messages[-1]
            if last["role"] == "user":
                content = str(last["content"]).lower()
                if "abstract" in content or "uncertain" in content:
                    return LLMResponse(
                        content="",
                        tool_calls=[
                            ToolCallRequest(
                                id="tool-fetch-abstract",
                                name="fetch_publication_abstract",
                                arguments={"pub_id": "875-PUB-001"},
                            )
                        ],
                    )
                if "deep" in content:
                    return LLMResponse(
                        content="",
                        tool_calls=[
                            ToolCallRequest(
                                id="tool-read-paper",
                                name="read_paper_deep",
                                arguments={
                                    "pub_id": "875-PUB-001",
                                    "focus_question": "What is the paper's main contribution?",
                                },
                            )
                        ],
                    )
                return LLMResponse(content=f"Digital human answer: {last['content']}")

            if last["role"] == "tool":
                tool_name = last.get("name")
                if tool_name == "fetch_publication_abstract":
                    return LLMResponse(content=f"Abstract-based answer: {last['content']}")
                if tool_name == "read_paper_deep":
                    return LLMResponse(content=f"Deep-read answer: {last['content']}")
                return LLMResponse(content=f"Tool answer: {last['content']}")

            return LLMResponse(content="Fallback answer.")
        finally:
            self.active_calls -= 1

    def get_default_model(self) -> str:
        return "fake/model"


def build_test_app(tmp_path, *, provider: FakeProvider | None = None):
    provider = provider or FakeProvider()
    settings = AppSettings(
        workspace=tmp_path,
        scholar_api=ScholarApiSettings(demo_mode=True),
        model="fake/model",
    )
    repositories = AppRepositories(tmp_path)
    runtime = ScholarAgentRuntime(
        provider=provider,
        model="fake/model",
        repositories=repositories,
    )
    app = create_app(
        settings,
        runtime=runtime,
        repositories=repositories,
        scholar_client=ScholarDataClient(settings.scholar_api),
    )
    return app, repositories, runtime, provider


def test_create_digital_human_and_basic_conversation_flow(tmp_path) -> None:
    app, _, _, _ = build_test_app(tmp_path)

    with TestClient(app) as client:
        create_resp = client.post("/api/digital-humans", json={"open_sid": "875"})
        assert create_resp.status_code == 200
        digital_human = create_resp.json()
        assert digital_human["open_sid"] == "875"
        assert digital_human["profile"]["name"] == "Demo Scholar 875"

        conv_resp = client.post(
            "/api/conversations",
            json={"digital_human_id": digital_human["id"], "end_user_id": "user-1"},
        )
        assert conv_resp.status_code == 200
        conversation = conv_resp.json()

        message_resp = client.post(
            f"/api/conversations/{conversation['id']}/messages",
            json={"message": "请介绍一下你的研究方向"},
        )
        assert message_resp.status_code == 200
        body = message_resp.json()
        assert body["conversation_id"] == conversation["id"]
        assert "Digital human answer" in body["message"]


def test_message_turn_supports_abstract_fetch_and_subagent_task_status(tmp_path) -> None:
    app, _, _, _ = build_test_app(tmp_path)

    with TestClient(app) as client:
        digital_human = client.post("/api/digital-humans", json={"open_sid": "875"}).json()
        conversation = client.post(
            "/api/conversations",
            json={"digital_human_id": digital_human["id"], "end_user_id": "user-2"},
        ).json()

        abstract_resp = client.post(
            f"/api/conversations/{conversation['id']}/messages",
            json={"message": "This title is uncertain, please use the abstract."},
        )
        assert abstract_resp.status_code == 200
        assert "Abstract-based answer" in abstract_resp.json()["message"]

        deep_resp = client.post(
            f"/api/conversations/{conversation['id']}/messages",
            json={"message": "Please do a deep analysis of the first paper."},
        )
        assert deep_resp.status_code == 200
        deep_body = deep_resp.json()
        assert "Deep-read answer" in deep_body["message"]
        assert len(deep_body["task_ids"]) == 1

        task_resp = client.get(f"/api/tasks/{deep_body['task_ids'][0]}")
        assert task_resp.status_code == 200
        task = task_resp.json()
        assert task["status"] == "completed"
        assert "summary" in task["output"]


def test_conversation_runtime_serializes_same_session_and_persists_history(tmp_path) -> None:
    app, repositories, runtime, provider = build_test_app(tmp_path, provider=FakeProvider(delay=0.05))

    async def _run() -> None:
        digital_human = await app.state.services.scholar_client.load_digital_human("875")
        repositories.digital_humans.save(digital_human)
        conversation = Conversation(digital_human_id=digital_human.id, end_user_id="user-3")
        repositories.conversations.save(conversation)

        await asyncio.gather(
            runtime.run_turn(
                RuntimeRequestContext(
                    digital_human=digital_human,
                    conversation=conversation,
                    user_message="first message",
                )
            ),
            runtime.run_turn(
                RuntimeRequestContext(
                    digital_human=digital_human,
                    conversation=conversation,
                    user_message="second message",
                )
            ),
        )

        reloaded = AppRepositories(tmp_path)
        session = reloaded.sessions.get_or_create(conversation.session_key)
        contents = [msg.get("content") for msg in session.messages]
        assert "first message" in contents
        assert "second message" in contents

    asyncio.run(_run())

    assert provider.max_active_calls == 1
