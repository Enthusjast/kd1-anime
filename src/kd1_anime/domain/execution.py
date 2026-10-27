"""Legal execution progression and durable external submission identities."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum, auto
from typing import Literal
from uuid import uuid4

from pydantic import ConfigDict, Field

from kd1_anime.models import ValidatedModel as BaseModel

FSMState = Literal[
    "INIT",
    "PLANNING",
    "DETAILING",
    "PLAN_REVIEWING",
    "CODING",
    "REVIEWING",
    "DISPATCHING",
    "MONITORING",
    "FIXING",
    "VISUAL_EVALUATING",
    "MERGING",
    "EVALUATING",
    "DONE",
    "ERROR",
]

# 顶层状态机的可观察转移表。场景线程可以在同一检查点附近并发完成，
# 因此它只用于诊断和恢复校验，不作为拒绝并发检查点的硬闸门；真正的
# 终态一致性仍由 RunManifest.integrity_errors() 负责。
FSM_TRANSITIONS: dict[str, frozenset[str]] = {
    "INIT": frozenset({"PLANNING", "ERROR"}),
    "PLANNING": frozenset({"PLANNING", "DETAILING", "PLAN_REVIEWING", "CODING", "ERROR"}),
    "DETAILING": frozenset({"DETAILING", "PLAN_REVIEWING", "CODING", "REVIEWING", "ERROR"}),
    "PLAN_REVIEWING": frozenset({"PLAN_REVIEWING", "DETAILING", "CODING", "REVIEWING", "ERROR"}),
    "CODING": frozenset({"CODING", "REVIEWING", "DISPATCHING", "PLAN_REVIEWING", "ERROR"}),
    "REVIEWING": frozenset(
        {
            "REVIEWING",
            "CODING",
            "DISPATCHING",
            "FIXING",
            "DETAILING",
            "PLAN_REVIEWING",
            "ERROR",
        }
    ),
    "DISPATCHING": frozenset(
        {
            "DISPATCHING",
            "MONITORING",
            "MERGING",
            "DETAILING",
            "PLAN_REVIEWING",
            "CODING",
            "REVIEWING",
            "ERROR",
        }
    ),
    "MONITORING": frozenset(
        {
            "MONITORING",
            "FIXING",
            "DISPATCHING",
            "MERGING",
            "DETAILING",
            "PLAN_REVIEWING",
            "CODING",
            "REVIEWING",
            "ERROR",
        }
    ),
    "FIXING": frozenset(
        {
            "FIXING",
            "REVIEWING",
            "CODING",
            "MONITORING",
            "DISPATCHING",
            "DETAILING",
            "PLAN_REVIEWING",
            "ERROR",
        }
    ),
    "VISUAL_EVALUATING": frozenset(
        {
            "VISUAL_EVALUATING",
            "CODING",
            "DETAILING",
            "PLAN_REVIEWING",
            "REVIEWING",
            "MERGING",
            "ERROR",
        }
    ),
    "MERGING": frozenset(
        {"MERGING", "EVALUATING", "DETAILING", "PLAN_REVIEWING", "CODING", "ERROR"}
    ),
    "EVALUATING": frozenset(
        {"EVALUATING", "CODING", "DETAILING", "PLAN_REVIEWING", "MERGING", "DONE", "ERROR"}
    ),
    # DONE -> coding/reviewing 等是为“dry-run 失败后显式 resume”保留的恢复边，
    # 已完成正式运行仍会在 resume 入口先校验并直接返回，不会走这条边。
    "DONE": frozenset(
        {
            "DONE",
            "DETAILING",
            "CODING",
            "PLAN_REVIEWING",
            "REVIEWING",
            "DISPATCHING",
            "ERROR",
        }
    ),
    "ERROR": frozenset(
        {
            "ERROR",
            "INIT",
            "PLANNING",
            "DETAILING",
            "PLAN_REVIEWING",
            "CODING",
            "REVIEWING",
            "DISPATCHING",
            "MONITORING",
            "FIXING",
            "VISUAL_EVALUATING",
            "MERGING",
            "EVALUATING",
            "DONE",
        }
    ),
}


def is_valid_fsm_transition(previous_state: str, current_state: str) -> bool:
    """判断两个顶层检查点状态是否存在已知转移。"""

    if previous_state == current_state:
        return True
    return current_state in FSM_TRANSITIONS.get(previous_state, frozenset())


class State(Enum):
    INIT = auto()
    PLANNING = auto()
    DETAILING = auto()
    PLAN_REVIEWING = auto()
    CODING = auto()
    REVIEWING = auto()
    DISPATCHING = auto()
    MONITORING = auto()
    FIXING = auto()
    VISUAL_EVALUATING = auto()
    MERGING = auto()
    EVALUATING = auto()
    DONE = auto()
    ERROR = auto()


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


def validate_scene_transition(
    previous: str, target: str, *, revised: bool, reused: bool = False
) -> None:
    if revised or previous == target or target == "failed":
        return
    if previous == "reviewed" and target == "rendered" and reused:
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
