"""Review an explicit source snapshot and bind the result to its identity."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from kd1_anime.domain.planning import ScenePlan
from kd1_anime.domain.review import ReviewResult
from kd1_anime.models import snapshot_value, validated_copy
from kd1_anime.run_store import sha256_text


@dataclass(frozen=True, slots=True)
class ReviewedCode:
    code_sha256: str
    result: ReviewResult


class ReviewService:
    @staticmethod
    def review(agent, code: str, plan: ScenePlan, options: Mapping[str, Any]) -> ReviewedCode:
        result = agent.review(
            code, validated_copy(plan, deep=True), **snapshot_value(dict(options))
        )
        if not isinstance(result, ReviewResult):
            raise TypeError("Reviewer returned an invalid structured result")
        return ReviewedCode(sha256_text(code), validated_copy(result, deep=True))
