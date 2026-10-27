"""Manifest-owned specialized runtime state; sessions remain user-command authority."""

from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field, field_validator

from kd1_anime.agent_runtime.contracts import (
    ActionName,
    AgentAction,
    ExecutionIntent,
    ObjectRef,
    RuntimeRecord,
    Sha256,
    ToolInvocation,
    UsageReceipt,
)
from kd1_anime.domain.dependencies import WorkIdentity
from kd1_anime.domain.execution import SubmissionAttempt
from kd1_anime.models import ValidatedModel
from kd1_anime.prompt_identity import GenerationIdentity


class AgentStageAttempt(RuntimeRecord):
    operation: str = Field(min_length=1)
    resource_id: str = Field(min_length=1)
    input_sha256: Sha256
    used: int = Field(ge=1, strict=True)


class AgentSubmission(ValidatedModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scene_id: int = Field(ge=1)
    attempt: SubmissionAttempt


class AgentReviewState(RuntimeRecord):
    review_round: int = Field(default=0, ge=0)
    review_signature: str = ""
    identical_review_count: int = Field(default=0, ge=0)
    feedback: str = ""


class AgentPendingCandidate(RuntimeRecord):
    proposal: ObjectRef
    diff_sha256: Sha256


class AgentRunState(ValidatedModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    runtime: Literal["agent"] = "agent"
    session_id: str = Field(min_length=1)
    generation_identity: GenerationIdentity
    operation: Literal["generate", "plan", "render", "incremental"] = "generate"
    next_action: ActionName = "propose_plan"
    run_revision: int = Field(default=0, ge=0)
    run_input_sha256: Sha256
    step_count: int = Field(default=0, ge=0)
    planning_review: bool = True
    planning_preflight: bool = True
    smoke_requested: bool = False
    plan_sha256: Sha256 | None = None
    approved_plan_sha256: Sha256 | None = None
    approved_plan_identity: WorkIdentity | None = None
    reviewed_identities: dict[int, WorkIdentity] = Field(default_factory=dict)
    outline_identities: dict[int, WorkIdentity] = Field(default_factory=dict)
    scene_next_actions: dict[int, ActionName] = Field(default_factory=dict)
    scene_action_versions: dict[int, int] = Field(default_factory=dict)
    plan_review_complete: bool = False
    plan_repair_feedback: str = ""
    plan_repair_proposals: dict[int, ObjectRef] = Field(default_factory=dict)
    continuity_published: bool = False
    pending_candidates: dict[int, AgentPendingCandidate] = Field(default_factory=dict)
    scene_review_counters: dict[int, AgentReviewState] = Field(default_factory=dict)
    visual_repairs: dict[int, ObjectRef] = Field(default_factory=dict)
    boundary_input_sha256: Sha256 | None = None
    final_visual_report: ObjectRef | None = None
    final_evaluation_report: ObjectRef | None = None
    final_visual_status: Literal["skipped", "accepted", "rejected", "unknown"] = "skipped"
    pending_approval: AgentAction | None = None
    stop_reason: str = ""
    intents: tuple[ExecutionIntent, ...] = ()
    invocations: tuple[ToolInvocation, ...] = ()
    usage: tuple[UsageReceipt, ...] = ()
    attempts: tuple[AgentStageAttempt, ...] = ()
    submissions: tuple[AgentSubmission, ...] = ()
    committed_invocations: tuple[str, ...] = ()

    @field_validator("stop_reason", "plan_repair_feedback")
    @classmethod
    def visible_stop_reason(cls, value):
        from kd1_anime.agent_runtime.contracts import sanitize_record

        return sanitize_record(value)

    def integrity_errors(self, run_id: str, request_ids: set[str]) -> list[str]:
        errors = []
        intents = {item.intent_id: item for item in self.intents}
        invocations = {item.invocation_id: item for item in self.invocations}
        if len(intents) != len(self.intents) or len(invocations) != len(self.invocations):
            errors.append("Duplicate agent intent or invocation identity")
        for intent in self.intents:
            action = intent.action
            if action.kind == "inspect_candidate" and action.proposal is not None:
                valid_input = (
                    action.proposal.work_identity == action.work_identity
                    and action.proposal.digest() == action.input_sha256
                    and action.candidate == action.proposal.candidate_source
                )
            elif action.kind == "publish_continuity":
                valid_input = (
                    action.scene_id is None
                    and action.publication.plan_identity == action.work_identity
                    and action.publication.digest() == action.input_sha256
                )
            else:
                valid_input = (
                    action.work_identity is not None
                    and action.input_sha256 == action.work_identity.input_sha256
                )
            resource = f"run:{run_id}" + (
                f"/scene:{action.scene_id}" if action.scene_id is not None else ""
            )
            if (
                intent.session_id != self.session_id
                or action.run_id != run_id
                or action.resource_id != resource
                or action.work_identity is None
                or not valid_input
            ):
                errors.append("Agent intent input/session/resource binding mismatch")
        for invocation in self.invocations:
            intent = intents.get(invocation.intent_id)
            if intent is None or (
                invocation.action != intent.action.kind
                or invocation.resource_id != intent.action.resource_id
                or invocation.input_sha256 != intent.action.input_sha256
                or invocation.work_identity != intent.action.work_identity
            ):
                errors.append("Agent invocation does not match its durable intent")
        for identifier in self.committed_invocations:
            invocation = invocations.get(identifier)
            if invocation is None or invocation.status != "completed":
                errors.append("Committed agent result lacks completed invocation identity")
        if len(set(self.committed_invocations)) != len(self.committed_invocations):
            errors.append("Duplicate committed invocation identity")
        if len({item.request_id for item in self.usage}) != len(self.usage) or any(
            item.request_id not in request_ids for item in self.usage
        ):
            errors.append("Agent usage is not bound to unique reserved requests")
        attempt_keys = [
            (item.operation, item.resource_id, item.input_sha256) for item in self.attempts
        ]
        if len(set(attempt_keys)) != len(attempt_keys):
            errors.append("Duplicate agent stage attempt identity")
        if len({item.attempt.token for item in self.submissions}) != len(self.submissions):
            errors.append("Duplicate agent submission identity")
        return errors


def merge_transport_state(current: AgentRunState, proposed: AgentRunState) -> AgentRunState:
    """Preserve reserved transport facts while publishing a stage's typed outputs."""
    if (
        current.session_id != proposed.session_id
        or current.generation_identity != proposed.generation_identity
        or current.operation != proposed.operation
        or proposed.run_revision < current.run_revision
    ):
        raise ValueError("A runtime patch cannot replace its pinned identity")

    def merge_records(previous, updates, key, stable, ranks=None):
        merged = {key(item): item for item in previous}
        for update in updates:
            item = update
            old = merged.get(key(item))
            if old is not None:
                if any(getattr(old, name) != getattr(item, name) for name in stable):
                    raise ValueError("Conflicting durable runtime record identity")
                if ranks is None:
                    if old != item:
                        raise ValueError("Conflicting immutable runtime receipt")
                elif ranks[old.status] > ranks[item.status]:
                    continue
                if hasattr(old, "external_job_id") and old.external_job_id is not None:
                    if item.external_job_id not in {None, old.external_job_id}:
                        raise ValueError("Invocation cannot change its external job identity")
                    item = item.model_copy(update={"external_job_id": old.external_job_id})
            merged[key(item)] = item
        return tuple(merged.values())

    intents = merge_records(
        current.intents,
        proposed.intents,
        lambda item: item.intent_id,
        ("session_id", "action", "created_at"),
        {"reserved": 0, "dispatched": 1, "unknown": 2, "completed": 3},
    )
    invocations = merge_records(
        current.invocations,
        proposed.invocations,
        lambda item: item.invocation_id,
        (
            "intent_id",
            "action",
            "resource_id",
            "input_sha256",
            "work_identity",
            "permission",
            "created_at",
        ),
        {"reserved": 0, "running": 1, "unknown": 2, "failed": 2, "cancelled": 2, "completed": 3},
    )
    usage = merge_records(current.usage, proposed.usage, lambda item: item.request_id, ())
    attempts = {
        (item.operation, item.resource_id, item.input_sha256): item for item in current.attempts
    }
    for item in proposed.attempts:
        key = (item.operation, item.resource_id, item.input_sha256)
        if key not in attempts or item.used > attempts[key].used:
            attempts[key] = item
    submissions = {item.attempt.token: item for item in current.submissions}
    submission_ranks = {"pending": 0, "unknown": 1, "submitted": 2, "cancelled": 3}
    for item in proposed.submissions:
        old = submissions.get(item.attempt.token)
        if old is not None:
            if old.scene_id != item.scene_id or any(
                getattr(old.attempt, name) != getattr(item.attempt, name)
                for name in ("backend", "scene_revision", "code_sha256", "created_at")
            ):
                raise ValueError("Submission attempt cannot change its bound inputs")
            if old.attempt.job_id and item.attempt.job_id not in {"", old.attempt.job_id}:
                raise ValueError("Submission attempt cannot change its external job")
            if submission_ranks[old.attempt.status] > submission_ranks[item.attempt.status]:
                continue
        submissions[item.attempt.token] = item
    scene_next_actions = dict(current.scene_next_actions)
    scene_action_versions = dict(current.scene_action_versions)
    scene_maps = {
        name: dict(getattr(current, name))
        for name in ("reviewed_identities", "pending_candidates", "scene_review_counters")
    }
    for scene_id, action in proposed.scene_next_actions.items():
        version = proposed.scene_action_versions.get(scene_id, 0)
        if version >= scene_action_versions.get(scene_id, 0):
            scene_next_actions[scene_id] = action
            scene_action_versions[scene_id] = version
            for name, values in scene_maps.items():
                incoming = getattr(proposed, name)
                if scene_id in incoming:
                    values[scene_id] = incoming[scene_id]
                else:
                    values.pop(scene_id, None)
    return proposed.model_copy(
        update={
            "scene_next_actions": scene_next_actions,
            "scene_action_versions": scene_action_versions,
            **scene_maps,
            "intents": intents,
            "invocations": invocations,
            "usage": usage,
            "attempts": tuple(attempts.values()),
            "submissions": tuple(submissions.values()),
            "step_count": max(current.step_count, proposed.step_count),
            "committed_invocations": tuple(
                dict.fromkeys((*current.committed_invocations, *proposed.committed_invocations))
            ),
        }
    )
