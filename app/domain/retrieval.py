"""Publication retrieval and cache helpers."""

from __future__ import annotations

from app.models import DigitalHuman, Publication


class RetrievalService:
    """Serve publication metadata from the digital human snapshot."""

    def __init__(self, digital_human: DigitalHuman) -> None:
        self._digital_human = digital_human
        self._pub_by_id = {publication.pub_id: publication for publication in digital_human.publications}

    def list_publications(self) -> list[Publication]:
        """Return all known publications."""
        return list(self._pub_by_id.values())

    def get_publication(self, pub_id: str) -> Publication | None:
        """Get a single publication."""
        return self._pub_by_id.get(pub_id)
