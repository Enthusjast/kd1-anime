"""Closed domain actions and persistence envelopes for the specialized runtime."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import (
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from kd1_anime.domain.dependencies import WorkIdentity
from kd1_anime.domain.diagnostics import Diagnostic
from kd1_anime.domain.execution import FSMState, SubmissionAttempt
from kd1_anime.models import ValidatedModel
from kd1_anime.security import redact_text, redact_value

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
RunId = Annotated[str, Field(pattern=r"^\d{8}-\d{6}-[0-9a-f]{8}$")]
Permission = Literal["read", "write", "exec", "cluster"]
ActionName = Literal[
    "propose_plan",
    "detail_scene",
    "publish_continuity",
    "revise_plan",
    "propose_technical_spec",
    "propose_code_candidate",
    "review_plan",
    "review_code",
    "inspect_candidate",
    "smoke_scene",
    "submit_render",
    "poll_render",
    "cancel_render",
    "propose_render_fix",
    "propose_visual_fix",
    "evaluate_visual",
    "evaluate_boundary",
    "merge_video",
    "evaluate_run",
    "reuse_scene",
    "retry_scene",
    "finish",
    "fail",
]
_HIDDEN = re.compile(
    r"<(?:think|thinking|analysis|reasoning|reasoning_content)\b[^>]*>.*?"
    r"(?:</(?:think|thinking|analysis|reasoning|reasoning_content)\s*>|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_HIDDEN_KEYS = {
    "reasoning_content",
    "chain_of_thought",
    "hidden_analysis",
    "analysis",
    "reasoning",
    "thinking",
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _is_submission_attempt(value: dict) -> bool:
    """Recognize only the complete persisted domain schema, never display aliases."""
    if value.keys() != SubmissionAttempt.model_fields.keys():
        return False
    try:
        SubmissionAttempt.model_validate(value)
    except ValidationError:
        return False
    return True


def sanitize_record(value, secrets: tuple[str, ...] = ()):
    """Persist visible JSON, retaining only validated submission tokens as public IDs.

    Complete SubmissionAttempt dumps must survive nested record roundtrips. Their
    token is an identity, not a credential; configured secrets still redact it.
    All other token fields keep the ordinary credential policy.
    """
    if isinstance(value, ValidatedModel):
        value = value.model_dump(mode="json")
    if isinstance(value, str):
        return redact_text(_HIDDEN.sub("<redacted>", value), secrets)
    if isinstance(value, dict):
        if any(str(key).lower().replace("-", "_") in _HIDDEN_KEYS for key in value):
            raise ValueError("Hidden reasoning cannot be persisted")
        submission = _is_submission_attempt(value)
        result = {}
        for key, item in value.items():
            safe_item = sanitize_record(item, secrets)
            if (
                not (submission and key == "token")
                and redact_value({key: None})[key] == "<redacted>"
            ):
                safe_item = "<redacted>"
            result[key] = safe_item
        return result
    if isinstance(value, (list, tuple)):
        return [sanitize_record(item, secrets) for item in value]
    if value is None or isinstance(value, (int, float, bool, datetime, WorkIdentity)):
        return value
    raise ValueError("Runtime records require typed models or JSON values")


class RuntimeRecord(ValidatedModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    @model_validator(mode="before")
    @classmethod
    def visible_only(cls, value):
        return sanitize_record(value) if isinstance(value, dict) else value

    @model_validator(mode="after")
    def aware_times(self):
        for name in type(self).model_fields:
            value = getattr(self, name)
            if isinstance(value, datetime) and value.utcoffset() is None:
                raise ValueError("Runtime timestamps require a timezone")
        return self


class ObjectRef(RuntimeRecord):
    path: str = Field(min_length=1)
    sha256: Sha256
    kind: Literal["plan", "code", "report", "frame", "video", "diff"]

    @field_validator("path")
    @classmethod
    def private_relative_path(cls, value):
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or "\\" in value or value in {".", ""}:
            raise ValueError("Object references must remain inside the private run")
        return value


class ActionBase(RuntimeRecord):
    run_id: RunId
    resource_id: str = Field(min_length=1)
    input_sha256: Sha256
    work_identity: WorkIdentity | None = None
    scene_id: int | None = Field(default=None, ge=1)


class ProposePlan(ActionBase):
    kind: Literal["propose_plan"] = "propose_plan"


class RevisePlan(ActionBase):
    kind: Literal["revise_plan"] = "revise_plan"
    feedback: str
    proposals: tuple[ObjectRef, ...] = ()


class DetailScene(ActionBase):
    kind: Literal["detail_scene"] = "detail_scene"
    scene_id: int = Field(ge=1)


class ProposeTechnicalSpec(ActionBase):
    kind: Literal["propose_technical_spec"] = "propose_technical_spec"
    scene_id: int = Field(ge=1)


class ProposeCodeCandidate(ActionBase):
    kind: Literal["propose_code_candidate"] = "propose_code_candidate"
    scene_id: int = Field(ge=1)


class ReviewPlan(ActionBase):
    kind: Literal["review_plan"] = "review_plan"


class ReviewCode(ActionBase):
    kind: Literal["review_code"] = "review_code"
    scene_id: int = Field(ge=1)


class DiffBinding(RuntimeRecord):
    """Exact proposed bytes and evidence, bound to the unchanged base scene."""

    work_identity: WorkIdentity
    base_source: ObjectRef
    candidate_source: ObjectRef
    inspection_report: ObjectRef
    review_report: ObjectRef
    context_sha256: Sha256

    def digest(self) -> str:
        from kd1_anime.domain.dependencies import digest

        return digest(self.model_dump(mode="json"))


class ContinuitySource(RuntimeRecord):
    scene_id: int = Field(ge=1)
    work_identity: WorkIdentity
    source: ObjectRef


class ContinuityBinding(RuntimeRecord):
    plan_identity: WorkIdentity
    scenes: tuple[ContinuitySource, ...] = Field(min_length=1)

    def digest(self) -> str:
        from kd1_anime.domain.dependencies import digest

        return digest(self.model_dump(mode="json"))


class PublishContinuity(ActionBase):
    kind: Literal["publish_continuity"] = "publish_continuity"
    publication: ContinuityBinding


class InspectCandidate(ActionBase):
    kind: Literal["inspect_candidate"] = "inspect_candidate"
    scene_id: int = Field(ge=1)
    candidate: ObjectRef
    proposal: DiffBinding | None = None


class SmokeScene(ActionBase):
    kind: Literal["smoke_scene"] = "smoke_scene"
    scene_id: int = Field(ge=1)


class SubmitRender(ActionBase):
    kind: Literal["submit_render"] = "submit_render"
    scene_id: int = Field(ge=1)
    backend: Literal["local", "slurm"]
    source: ObjectRef
    class_name: str = Field(min_length=1)


class PollRender(ActionBase):
    kind: Literal["poll_render"] = "poll_render"
    scene_id: int = Field(ge=1)
    job_id: str = Field(pattern=r"^(?:\d+|local-[0-9a-f]{12})$")


class CancelRender(ActionBase):
    kind: Literal["cancel_render"] = "cancel_render"
    scene_id: int = Field(ge=1)
    job_id: str = Field(pattern=r"^(?:\d+|local-[0-9a-f]{12})$")


class ProposeRenderFix(ActionBase):
    kind: Literal["propose_render_fix"] = "propose_render_fix"
    scene_id: int = Field(ge=1)
    diagnostic: Diagnostic


class ProposeVisualFix(ActionBase):
    kind: Literal["propose_visual_fix"] = "propose_visual_fix"
    scene_id: int = Field(ge=1)
    report: ObjectRef


class EvaluateVisual(ActionBase):
    kind: Literal["evaluate_visual"] = "evaluate_visual"
    video: ObjectRef
    scene_id: int = Field(ge=1)
    snapshot_sha256: Sha256 | None = None
    frames: tuple[ObjectRef, ...] = ()


class EvaluateBoundary(ActionBase):
    kind: Literal["evaluate_boundary"] = "evaluate_boundary"
    snapshot_sha256: Sha256


class MergeVideo(ActionBase):
    kind: Literal["merge_video"] = "merge_video"
    artifacts: tuple[ObjectRef, ...] = Field(min_length=1)


class EvaluateRun(ActionBase):
    kind: Literal["evaluate_run"] = "evaluate_run"
    video: ObjectRef


class ReuseScene(ActionBase):
    kind: Literal["reuse_scene"] = "reuse_scene"
    scene_id: int = Field(ge=1)
    base_run_id: RunId


class RetryScene(ActionBase):
    kind: Literal["retry_scene"] = "retry_scene"
    scene_id: int = Field(ge=1)


class Finish(ActionBase):
    kind: Literal["finish"] = "finish"
    output: ObjectRef | None = None


class Fail(ActionBase):
    kind: Literal["fail"] = "fail"
    diagnostic: Diagnostic


AgentAction = Annotated[
    ProposePlan
    | PublishContinuity
    | DetailScene
    | RevisePlan
    | ProposeTechnicalSpec
    | ProposeCodeCandidate
    | ReviewPlan
    | ReviewCode
    | InspectCandidate
    | SmokeScene
    | SubmitRender
    | PollRender
    | CancelRender
    | ProposeRenderFix
    | ProposeVisualFix
    | EvaluateVisual
    | EvaluateBoundary
    | MergeVideo
    | EvaluateRun
    | ReuseScene
    | RetryScene
    | Finish
    | Fail,
    Field(discriminator="kind"),
]
ACTION_ADAPTER = TypeAdapter(AgentAction)


class UsageReceipt(RuntimeRecord):
    request_id: str = Field(min_length=1)
    invocation_id: str | None = None
    prompt_sha256: Sha256 | None = None
    usage_source: Literal["provider_reported", "locally_estimated", "unknown"] = "unknown"
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    cost_status: Literal["reported", "estimated", "unknown"] = "unknown"
    cost: float | None = Field(default=None, ge=0)
    currency: Literal["USD"] = "USD"

    @model_validator(mode="after")
    def no_fabricated_unknown(self):
        if self.cost_status == "unknown" and self.cost is not None:
            raise ValueError("Unknown cost cannot have a numeric amount")
        if self.cost_status != "unknown" and self.cost is None:
            raise ValueError("Known cost requires an amount")
        if self.usage_source == "unknown" and any(
            item is not None
            for item in (self.prompt_tokens, self.completion_tokens, self.total_tokens)
        ):
            raise ValueError("Unknown usage cannot claim token counts")
        return self


class AgentMessage(RuntimeRecord):
    message_id: str = Field(default_factory=lambda: uuid4().hex)
    session_id: str = Field(min_length=1)
    run_id: RunId | None = None
    role: Literal["user", "assistant", "tool"]
    kind: Literal["goal", "clarification", "command", "action", "tool_result"]
    content: str = ""
    action: AgentAction | None = None
    references: tuple[ObjectRef, ...] = ()
    model: str | None = None
    usage: UsageReceipt | None = None
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def role_matches_content(self):
        expected = (
            "assistant"
            if self.kind == "action"
            else "tool"
            if self.kind == "tool_result"
            else "user"
        )
        if self.role != expected:
            raise ValueError("Message role does not match its visible content kind")
        if (self.action is not None) != (self.kind == "action"):
            raise ValueError("Only assistant action messages may contain a domain action")
        return self


class ApprovalGrant(RuntimeRecord):
    grant_id: str = Field(default_factory=lambda: uuid4().hex)
    action: ActionName
    resource_id: str = Field(min_length=1)
    input_sha256: Sha256
    session_id: str = Field(min_length=1)
    run_id: RunId
    work_identity: WorkIdentity | None = None
    issued_by: Literal["local_user"] = "local_user"
    issued_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime | None = None
    consumed_by_intent: str | None = None

    @model_validator(mode="after")
    def valid_expiration(self):
        if self.expires_at is not None and self.expires_at <= self.issued_at:
            raise ValueError("Grant expiration must follow issuance")
        return self


class PermissionScope(RuntimeRecord):
    permission: Permission
    resource_id: str = Field(min_length=1)


class ExecutionIntent(RuntimeRecord):
    intent_id: str = Field(default_factory=lambda: uuid4().hex)
    session_id: str = Field(min_length=1)
    action: AgentAction
    status: Literal["reserved", "dispatched", "completed", "unknown"] = "reserved"
    created_at: datetime = Field(default_factory=utc_now)


class ToolInvocation(RuntimeRecord):
    invocation_id: str = Field(default_factory=lambda: uuid4().hex)
    intent_id: str = Field(min_length=1)
    action: ActionName
    resource_id: str = Field(min_length=1)
    input_sha256: Sha256
    work_identity: WorkIdentity | None = None
    permission: Permission
    status: Literal["reserved", "running", "completed", "failed", "unknown", "cancelled"] = (
        "reserved"
    )
    attempt: int = Field(default=1, ge=1)
    external_job_id: str | None = None
    result_ref: ObjectRef | None = None
    error_diagnostic: Diagnostic | None = None
    created_at: datetime = Field(default_factory=utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None


EventName = Literal[
    "snapshot_committed",
    "action_started",
    "action_completed",
    "action_failed",
    "approval_requested",
    "progress",
    "stream_chunk",
    "diagnostic",
    "loop_stopped",
]


class EventPayload(RuntimeRecord):
    state: FSMState | None = None
    status: str | None = None
    scene_id: int | None = Field(default=None, ge=1)
    message: str = ""
    progress: float | None = Field(default=None, ge=0, le=1)
    action: ActionName | None = None
    diagnostics: tuple[Diagnostic, ...] = ()
    references: tuple[ObjectRef, ...] = ()
    usage: UsageReceipt | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class AgentEvent(RuntimeRecord):
    schema_version: Literal[3] = 3
    payload_version: Literal[1] = 1
    sequence: int = Field(ge=1, strict=True)
    event_id: str = Field(default_factory=lambda: uuid4().hex)
    timestamp: datetime = Field(default_factory=utc_now)
    stream_id: str = Field(min_length=1)
    run_id: RunId
    manifest_revision: int = Field(ge=0)
    scene_revision: int | None = Field(default=None, ge=0)
    durability: Literal["committed", "ephemeral"]
    name: EventName
    causation_id: str | None = None
    intent_id: str | None = None
    invocation_id: str | None = None
    payload: EventPayload = Field(default_factory=EventPayload)
