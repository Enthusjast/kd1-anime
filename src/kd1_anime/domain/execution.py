"""Legal execution progression and durable external submission identities."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import ConfigDict, Field

from kd1_anime.models import ValidatedModel as BaseModel

RunPhase = Literal[
    "initializing", "planning", "executing", "merging", "evaluating", "finished", "failed"
]


def transition_run(previous: RunPhase, target: RunPhase, *, reopening: bool = False) -> RunPhase:
    allowed = {
        "initializing": {"planning", "executing", "merging", "evaluating", "failed", "finished"},
        "planning": {"executing", "failed", "finished"},
        "executing": {"planning", "merging", "failed", "finished"},
        "merging": {"executing", "evaluating", "failed", "finished"},
        "evaluating": {"planning", "executing", "merging", "failed", "finished"},
        "finished": set(),
        "failed": set(),
    }
    if previous == target or (reopening and previous == "failed"):
        return target
    if target not in allowed[previous]:
        raise ValueError(f"非法运行状态转移: {previous} -> {target}")
    return target


def validate_scene_transition(previous: str, target: str, *, revised: bool) -> None:
    if revised or previous == target or target == "failed":
        return
    allowed = {
        "pending": {
            "detailed",
            "plan_reviewing",
            "plan_reviewed",
            "technical_planning",
            "coded",
            "reviewed",
        },
        "detailed": {"plan_reviewing", "plan_reviewed", "technical_planning", "coded"},
        "plan_reviewing": {"plan_reviewed", "technical_planning", "coded", "reviewed"},
        "plan_reviewed": {"technical_planning", "coded", "reviewed"},
        "technical_planning": {"plan_reviewing", "plan_reviewed", "coded", "reviewed"},
        "coded": {"reviewed", "plan_reviewing", "technical_planning"},
        "reviewed": {"monitoring", "coded", "technical_planning", "plan_reviewing"},
        "monitoring": {"rendered", "coded", "reviewed", "technical_planning", "visual_evaluating"},
        "rendered": {"visual_evaluating", "visual_accepted"},
        "visual_evaluating": {"visual_accepted", "rendered"},
        "visual_accepted": {"visual_evaluating"},
        "failed": set(),
    }
    if target not in allowed.get(previous, set()):
        raise ValueError(f"非法场景状态转移: {previous} -> {target}")


class SubmissionAttempt(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    token: str = Field(default_factory=lambda: uuid4().hex[:12], pattern=r"^[0-9a-f]{12}$")
    backend: Literal["slurm", "local"]
    scene_revision: int = Field(ge=0)
    code_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["pending", "submitted", "cancelled", "unknown"] = "pending"
    job_id: str = Field(default="", pattern=r"^(?:\d+|local-[0-9a-f]{12})?$")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class StageAttempts(BaseModel):
    """Attempts for one logical input, retained across interruption and retry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    operation: Literal["generation", "technical"]
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    used: int = Field(ge=1, strict=True)


RETRY_LIMIT_NAMES = (
    "CODE_VALIDATION_ATTEMPTS",
    "MAX_TECHNICAL_SPEC_ATTEMPTS",
    "MAX_PLAN_REVIEW_ROUNDS",
    "MAX_PLAN_REPLAN_ATTEMPTS",
    "MAX_CONTINUITY_FIX_ROUNDS",
    "MAX_REVIEW_ROUNDS",
    "MAX_LOW_RISK_REVIEW_ROUNDS",
    "MAX_IDENTICAL_REVIEW_ATTEMPTS",
    "MAX_CODE_CANDIDATES_LOW",
    "MAX_CODE_CANDIDATES_MEDIUM",
    "MAX_CODE_CANDIDATES_HIGH",
    "MAX_FIX_ATTEMPTS",
    "MAX_INFRA_RETRIES",
    "MAX_STAGNANT_ATTEMPTS",
    "MAX_FIX_IDENTICAL_ERRORS",
    "MAX_EVAL_ROUNDS",
)
