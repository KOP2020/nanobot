"""ScholarMate-backed digital human provisioning."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

import httpx

from app.config import ScholarApiSettings
from app.domain.persona import default_control_config
from app.models import DigitalHuman, Publication, ScholarProfile


class ScholarDataClient:
    """Load scholar data from ScholarMate open-data APIs or a demo fallback."""

    def __init__(self, settings: ScholarApiSettings) -> None:
        self._settings = settings

    async def load_digital_human(self, open_sid: str) -> DigitalHuman:
        """Build a digital human snapshot from an OpenSID."""
        if self._settings.demo_mode or not (self._settings.openid and self._settings.token):
            return self._build_demo_human(open_sid)

        profile_payload = await self._request(open_sid, data_type="4", page_no=1)
        publications = await self._load_publications(open_sid)
        profile = self._parse_profile(open_sid, profile_payload)
        return DigitalHuman(
            open_sid=open_sid,
            profile=profile,
            control_config=default_control_config(profile),
            publications=publications,
        )

    async def _load_publications(self, open_sid: str) -> list[Publication]:
        publications: list[Publication] = []
        page_no = 1
        total_pages = 1
        while page_no <= total_pages:
            payload = await self._request(open_sid, data_type="3", page_no=page_no)
            publications.extend(self._parse_publications(payload))
            total_pages = int(payload.get("totalPages") or payload.get("total_pages") or 1)
            page_no += 1
        return publications

    async def _request(self, open_sid: str, *, data_type: str, page_no: int) -> dict[str, Any]:
        inner = {
            "meta": {
                "query_timestamp": str(int(time.time())),
                "query_id": uuid.uuid4().hex[:12],
            },
            "scholar_open_id": str(open_sid),
            "data_type": str(data_type),
            "pageSize": self._settings.page_size,
            "pageNo": page_no,
        }
        payload = {
            "openid": self._settings.openid,
            "token": self._settings.token,
            "data": json.dumps(inner, ensure_ascii=False),
        }
        async with httpx.AsyncClient(timeout=self._settings.timeout_s, verify=False) as client:
            response = await client.post(self._settings.endpoint, json=payload)
            response.raise_for_status()
        outer = response.json()
        result = outer.get("result", "{}")
        if isinstance(result, str):
            return json.loads(result)
        return result

    @staticmethod
    def _parse_profile(open_sid: str, payload: dict[str, Any]) -> ScholarProfile:
        research_fields = []
        for item in payload.get("kw_tf", []):
            keyword = item.get("keyword")
            if keyword and keyword not in research_fields:
                research_fields.append(str(keyword))
        if not research_fields and payload.get("keywords"):
            research_fields = [part.strip() for part in str(payload["keywords"]).split(",") if part.strip()]
        return ScholarProfile(
            open_sid=open_sid,
            name=str(payload.get("name") or f"Scholar {open_sid}"),
            institution=payload.get("institution"),
            title=payload.get("title"),
            research_fields=research_fields,
            profile_summary=payload.get("psnBrief") or payload.get("profile_summary"),
            keywords=research_fields,
        )

    @staticmethod
    def _parse_publications(payload: dict[str, Any]) -> list[Publication]:
        candidates = (
            payload.get("pub_info")
            or payload.get("publication_info")
            or payload.get("publications")
            or payload.get("data")
            or []
        )
        publications: list[Publication] = []
        for item in candidates:
            pub_id = (
                item.get("pub_id")
                or item.get("pubId")
                or item.get("publication_id")
                or item.get("id")
            )
            title = item.get("title") or item.get("pub_title")
            if not pub_id or not title:
                continue
            publications.append(
                Publication(
                    pub_id=str(pub_id),
                    title=str(title),
                    abstract=item.get("abstract"),
                    journal_or_venue=item.get("journal_or_venue")
                    or item.get("journalName")
                    or item.get("conference_name"),
                    publish_year=_coerce_int(item.get("publish_year") or item.get("pub_year")),
                    citation_count=_coerce_int(item.get("citation_count")),
                    pdf_url=item.get("pdf_url") or item.get("download_url"),
                    pdf_path=item.get("pdf_path"),
                    authors=_coerce_authors(item.get("authors")),
                )
            )
        return publications

    @staticmethod
    def _build_demo_human(open_sid: str) -> DigitalHuman:
        profile = ScholarProfile(
            open_sid=open_sid,
            name=f"Demo Scholar {open_sid}",
            institution="ScholarMate Demo Institute",
            title="Professor",
            research_fields=["scientific discovery", "knowledge graphs", "large language models"],
            profile_summary=(
                "This is a demo scholar snapshot generated locally because ScholarMate "
                "credentials are not configured."
            ),
            keywords=["scientific discovery", "knowledge graphs", "large language models"],
        )
        publications = [
            Publication(
                pub_id=f"{open_sid}-PUB-001",
                title="A Survey on Large Language Models for Scientific Discovery",
                abstract=(
                    "Reviews how large language models assist literature mining, hypothesis "
                    "generation, and scientific workflows."
                ),
                journal_or_venue="Nature Reviews",
                publish_year=2024,
            ),
            Publication(
                pub_id=f"{open_sid}-PUB-002",
                title="Multi-hop Reasoning over Heterogeneous Scholarly Graphs",
                abstract=(
                    "Introduces a graph reasoning framework for modeling cross-entity "
                    "relationships in scholarly networks."
                ),
                journal_or_venue="AAAI",
                publish_year=2023,
            ),
        ]
        return DigitalHuman(
            open_sid=open_sid,
            profile=profile,
            control_config=default_control_config(profile),
            publications=publications,
        )


def _coerce_authors(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item]
    if isinstance(value, str) and value.strip():
        return [part.strip() for part in value.split(",") if part.strip()]
    return []


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None and value != "" else None
    except (TypeError, ValueError):
        return None
