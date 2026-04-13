"""Scholar persona helpers."""

from __future__ import annotations

from app.models import ScholarControlConfig, ScholarProfile


def default_control_config(profile: ScholarProfile) -> ScholarControlConfig:
    """Create a conservative default control config for a scholar."""

    allowed_topics = list(profile.research_fields[:8])
    return ScholarControlConfig(
        allowed_topics=allowed_topics,
        restricted_topics=[
            "未发表工作",
            "敏感合作细节",
            "账号或隐私信息",
        ],
        disclosure_level="public_summary",
        style_preference="rigorous_academic",
    )
