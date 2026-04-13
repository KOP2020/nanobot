"""Context assembly for scholar-centric agent runs."""

from __future__ import annotations

from typing import Any

from app.models import Conversation, DigitalHuman


class ScholarContextBuilder:
    """Build system and turn messages for a digital human conversation."""

    def build_messages(
        self,
        digital_human: DigitalHuman,
        conversation: Conversation,
        history: list[dict[str, Any]],
        user_message: str,
    ) -> list[dict[str, Any]]:
        """Build a scholar-specific chat transcript."""
        return [
            {"role": "system", "content": self.build_system_prompt(digital_human, conversation)},
            *history,
            {"role": "user", "content": user_message},
        ]

    def build_system_prompt(
        self,
        digital_human: DigitalHuman,
        conversation: Conversation,
    ) -> str:
        """Build the main system prompt."""
        profile = digital_human.profile
        control = digital_human.control_config
        pub_lines = []
        for publication in digital_human.publications[:200]:
            badge = " ★代表作" if publication.pub_id in control.featured_pub_ids else ""
            venue = publication.journal_or_venue or "Unknown Venue"
            year = publication.publish_year or "Unknown Year"
            pub_lines.append(f"- [{publication.pub_id}] {publication.title} ({year}, {venue}){badge}")

        notes = [
            f"Digital human ID: {digital_human.id}",
            f"Conversation ID: {conversation.id}",
            f"Scholar OpenSID: {profile.open_sid}",
            f"Scholar Name: {profile.name}",
            f"Institution: {profile.institution or 'Unknown'}",
            f"Title: {profile.title or 'Unknown'}",
            "Research Fields: " + (", ".join(profile.research_fields) or "Unknown"),
            f"Profile Summary: {profile.profile_summary or 'No summary available.'}",
            "Allowed Topics: " + (", ".join(control.allowed_topics) or "None specified"),
            "Restricted Topics: " + (", ".join(control.restricted_topics) or "None specified"),
            f"Disclosure Level: {control.disclosure_level}",
            f"Style Preference: {control.style_preference}",
            "Private Notes: " + (
                "; ".join(f"{key}: {value}" for key, value in control.private_notes.items())
                or "None"
            ),
            "",
            "Publication Catalog:",
            "\n".join(pub_lines) or "- No publications available",
            "",
            "Behavior Rules:",
            "1. Answer as the scholar's digital human, but do not invent facts.",
            "2. Prefer the publication catalog and scholar profile already in context.",
            "3. When title matching is uncertain, call fetch_publication_abstract before answering.",
            "4. When the user asks for paper details beyond the abstract, call read_paper_deep.",
            "5. Respect restricted topics and explain constraints instead of disclosing hidden details.",
            "6. Keep answers grounded, concise, and traceable to profile/publication evidence.",
        ]
        return "\n".join(notes)
