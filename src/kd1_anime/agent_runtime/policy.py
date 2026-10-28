"""Pure, closed-domain authorization over committed snapshots, never live context."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Protocol, get_args

from pydantic import Field, model_validator

from kd1_anime.agent_runtime.contracts import (
    ACTION_ADAPTER,
    ActionName,
    AgentAction,
    ApprovalGrant,
    ObjectRef,
    Permission,
    PermissionScope,
    RunId,
    RuntimeRecord,
    Sha256,
    utc_now,
)
from kd1_anime.domain.dependencies import WorkIdentity
from kd1_anime.domain.diagnostics import Diagnostic
from kd1_anime.domain.execution import FSMState


class PolicyDependency(RuntimeRecord):
    scene_id: int = Field(ge=1)
    work_identity: WorkIdentity


class ScenePolicyState(RuntimeRecord):
    scene_id: int = Field(ge=1)
    phase: Literal[
        "pending",
        "detailed",
        "plan_reviewing",
        "plan_reviewed",
        "technical_planning",
        "coded",
        "reviewed",
        "monitoring",
        "rendered",
        "visual_evaluating",
        "visual_accepted",
        "failed",
    ]
    work_identity: WorkIdentity
    dependencies: tuple[PolicyDependency, ...] = ()
    plan_sha256: Sha256 | None = None
    approved_plan_sha256: Sha256 | None = None
    approved_plan_identity: WorkIdentity | None = None
    code_sha256: Sha256 | None = None
    approved_code_sha256: Sha256 | None = None
    approved_code_identity: WorkIdentity | None = None
    diff_sha256: Sha256 | None = None
    approved_diff_sha256: Sha256 | None = None
    approved_diff_identity: WorkIdentity | None = None
    external_job_id: str | None = None
    static_verified: bool = False
    source_ref: ObjectRef | None = None
    give_up: bool = False


class PolicySnapshot(RuntimeRecord):
    run_id: RunId
    revision: int = Field(default=0, ge=0)
    state: FSMState
    work_identity: WorkIdentity
    scenes: tuple[ScenePolicyState, ...] = ()
    plan_sha256: Sha256 | None = None
    approved_plan_sha256: Sha256 | None = None
    approved_plan_identity: WorkIdentity | None = None
    waiting_for_approval: bool = False
    waiting_for_job: bool = False
    plan_only: bool = False
    dry_run: bool = False
    recovery_epoch: int = Field(default=0, ge=0)
    allow_partial: bool = False
    now: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def unique_scenes(self):
        ids = [scene.scene_id for scene in self.scenes]
        if len(set(ids)) != len(ids):
            raise ValueError("Policy scenes must be unique")
        return self


class PolicyBudget(RuntimeRecord):
    steps_remaining: int = Field(default=100, ge=0)
    requests_remaining: int = Field(default=100, ge=0)
    cost_remaining: float | None = Field(default=None, ge=0)
    deadline: datetime | None = None


class ToolPolicySpec(RuntimeRecord):
    action: ActionName
    permission: Permission
    side_effect: Literal["none", "commit", "external"]
    timeout_seconds: float = Field(default=60, gt=0, le=86400)
    max_attempts: int = Field(default=1, ge=1, le=3)
    requires_approval: bool = False
    uses_model: bool = False

    @model_validator(mode="after")
    def retry_queries_only(self):
        if self.max_attempts != 1 and (
            self.side_effect != "none" or self.action not in {"poll_render", "inspect_candidate"}
        ):
            raise ValueError("Only declared domain queries may retry")
        if (
            self.action in {"submit_render", "cancel_render", "merge_video"}
            and self.side_effect != "external"
        ):
            raise ValueError("Execution tools must declare external side effects")
        return self


class PolicyDecision(RuntimeRecord):
    effect: Literal["allow", "ask", "deny"]
    reason: str
    diagnostic: Diagnostic | None = None
    grant_id: str | None = None
    allowed_tools: tuple[ActionName, ...] = ()


class PolicySession(Protocol):
    session_id: str
    run_ids: tuple[str, ...]
    permissions: tuple[PermissionScope, ...]
    grants: tuple[ApprovalGrant, ...]


RECOVERY_ACTIONS = frozenset(
    {
        "retry_scene",
        "cancel_render",
        "reconcile_submission",
        "reconcile_render",
        "resume_run",
        "reconcile_merge",
    }
)
_RECOVERY_PHASES = frozenset(get_args(FSMState)) - {"DONE"}

_PHASES: dict[str, frozenset[str]] = {
    "propose_plan": frozenset({"INIT", "PLANNING"}),
    "detail_scene": frozenset({"DETAILING"}),
    "publish_continuity": frozenset({"CODING", "REVIEWING"}),
    "revise_plan": frozenset(
        {
            "PLANNING",
            "DETAILING",
            "PLAN_REVIEWING",
            "CODING",
            "REVIEWING",
            "FIXING",
            "VISUAL_EVALUATING",
            "EVALUATING",
        }
    ),
    "propose_technical_spec": frozenset({"DETAILING", "PLAN_REVIEWING", "CODING", "FIXING"}),
    "propose_code_candidate": frozenset({"CODING", "REVIEWING", "FIXING"}),
    "review_plan": frozenset({"PLANNING", "DETAILING", "PLAN_REVIEWING"}),
    "review_code": frozenset({"CODING", "REVIEWING", "FIXING"}),
    "inspect_candidate": frozenset({"CODING", "REVIEWING", "FIXING"}),
    "smoke_scene": frozenset({"CODING", "REVIEWING", "FIXING"}),
    "submit_render": frozenset({"DISPATCHING"}),
    "poll_render": frozenset({"MONITORING"}),
    "cancel_render": _RECOVERY_PHASES,
    "reconcile_submission": _RECOVERY_PHASES,
    "reconcile_render": _RECOVERY_PHASES,
    "resume_run": _RECOVERY_PHASES,
    "reconcile_merge": _RECOVERY_PHASES,
    "propose_render_fix": frozenset({"FIXING"}),
    "propose_visual_fix": frozenset({"VISUAL_EVALUATING", "FIXING"}),
    "evaluate_visual": frozenset({"VISUAL_EVALUATING", "MONITORING"}),
    "evaluate_boundary": frozenset({"VISUAL_EVALUATING"}),
    "merge_video": frozenset({"MERGING"}),
    "evaluate_run": frozenset({"EVALUATING"}),
    "reuse_scene": frozenset({"CODING", "REVIEWING", "DISPATCHING"}),
    "retry_scene": _RECOVERY_PHASES,
    "finish": frozenset({"EVALUATING"}),
    "fail": frozenset(set()),
}
_SCENE_PHASES: dict[str, frozenset[str]] = {
    "detail_scene": frozenset({"pending"}),
    "propose_technical_spec": frozenset({"detailed", "plan_reviewed", "technical_planning"}),
    "propose_code_candidate": frozenset(
        {"plan_reviewed", "technical_planning", "coded", "reviewed"}
    ),
    "review_code": frozenset({"coded", "reviewed"}),
    "inspect_candidate": frozenset({"coded", "reviewed"}),
    "smoke_scene": frozenset({"coded", "reviewed"}),
    "submit_render": frozenset({"reviewed"}),
    "poll_render": frozenset({"monitoring"}),
    "cancel_render": frozenset(
        {"monitoring", "failed", "rendered", "visual_evaluating", "visual_accepted"}
    ),
    "propose_render_fix": frozenset({"failed", "monitoring", "coded", "reviewed"}),
    "propose_visual_fix": frozenset({"visual_evaluating", "rendered", "visual_accepted"}),
    "reuse_scene": frozenset({"pending", "detailed", "plan_reviewed", "reviewed"}),
}
_READ_ACTIONS = frozenset({"poll_render", "reconcile_submission", "reconcile_render"})


def required_permission(action: AgentAction) -> Permission:
    if action.kind == "reconcile_merge":
        return "exec"
    if action.kind == "submit_render":
        return "cluster" if action.backend == "slurm" else "exec"
    if action.kind == "cancel_render":
        return "exec" if action.job_id.startswith("local-") else "cluster"
    if action.kind == "smoke_scene":
        return "exec"
    if action.kind == "merge_video":
        return "exec"
    return "read" if action.kind in _READ_ACTIONS else "write"


def grant_is_current(grant: ApprovalGrant, session: PolicySession) -> bool:
    commands = getattr(session, "commands", None)
    if commands is None:
        return True

    def binding(action):
        return (
            action.kind,
            action.resource_id,
            action.input_sha256,
            action.run_id,
            action.work_identity,
        )

    expected = (
        grant.action,
        grant.resource_id,
        grant.input_sha256,
        grant.run_id,
        grant.work_identity,
    )
    source = next(
        (
            index
            for index, command in enumerate(commands)
            if command.kind == "approve"
            and command.role == "user"
            and command.created_at == grant.issued_at
            and command.action is not None
            and binding(command.action) == expected
        ),
        None,
    )
    if source is None:
        return False
    return not any(
        command.kind in {"goal", "clarification", "cancel"}
        or (
            command.kind == "reject"
            and command.action is not None
            and binding(command.action) == expected
        )
        for command in commands[source + 1 :]
    )


class Policy:
    def available_tools(
        self,
        snapshot: PolicySnapshot,
        session: PolicySession,
        budget: PolicyBudget,
        tools: tuple[ToolPolicySpec, ...],
    ) -> tuple[ActionName, ...]:
        """Conservative proposal vocabulary, not an execution authorization.

        The complete proposed action must still pass evaluate. In particular,
        a provider is not asked to propose an action blocked on user approval.
        """
        if (
            snapshot.state in {"DONE", "ERROR"}
            or snapshot.waiting_for_approval
            or snapshot.waiting_for_job
            or snapshot.run_id not in session.run_ids
            or budget.steps_remaining == 0
            or (budget.deadline is not None and snapshot.now >= budget.deadline)
        ):
            return ()
        root = f"run:{snapshot.run_id}"
        identities = {
            root: snapshot.work_identity,
            **{f"{root}/scene:{scene.scene_id}": scene.work_identity for scene in snapshot.scenes},
        }
        diff_inputs = {
            f"{root}/scene:{scene.scene_id}": scene.diff_sha256
            for scene in snapshot.scenes
            if scene.diff_sha256 is not None
        }
        resources = set(identities)
        available = []
        for tool in tools:
            if tool.action != "fail" and snapshot.state not in _PHASES[tool.action]:
                continue
            if tool.uses_model and (budget.requests_remaining == 0 or budget.cost_remaining == 0):
                continue
            if not any(
                scope.permission == tool.permission and scope.resource_id in resources
                for scope in session.permissions
            ):
                continue
            if tool.requires_approval and not any(
                grant.action == tool.action
                and grant.resource_id in resources
                and grant.run_id == snapshot.run_id
                and grant.session_id == session.session_id
                and grant.work_identity == identities[grant.resource_id]
                and grant.input_sha256
                == (
                    diff_inputs.get(grant.resource_id, identities[grant.resource_id].input_sha256)
                    if tool.action == "inspect_candidate"
                    else identities[grant.resource_id].input_sha256
                )
                and grant.issued_by == "local_user"
                and grant.consumed_by_intent is None
                and grant.issued_at <= snapshot.now
                and (grant.expires_at is None or snapshot.now < grant.expires_at)
                and grant_is_current(grant, session)
                for grant in session.grants
            ):
                continue
            available.append(tool.action)
        return tuple(sorted(available))

    def evaluate(
        self,
        snapshot: PolicySnapshot,
        action: AgentAction,
        tool: ToolPolicySpec,
        session: PolicySession,
        budget: PolicyBudget,
    ) -> PolicyDecision:
        def deny(reason: str) -> PolicyDecision:
            return PolicyDecision(
                effect="deny",
                reason=reason,
                diagnostic=Diagnostic(
                    code=f"policy.{reason}",
                    stage="review",
                    category="policy",
                    summary=reason,
                    run_id=snapshot.run_id,
                    scene_id=getattr(action, "scene_id", None),
                ),
            )

        try:
            action = ACTION_ADAPTER.validate_python(action.model_dump(mode="python"))
        except (ValueError, AttributeError, TypeError):
            return deny("unknown_action")
        resource = f"run:{action.run_id}"
        if action.scene_id is not None:
            resource += f"/scene:{action.scene_id}"
        if action.recovery_epoch != snapshot.recovery_epoch:
            return deny("stale_recovery_epoch")
        if action.run_id != snapshot.run_id or action.resource_id != resource:
            return deny("invalid_resource")
        if snapshot.run_id not in session.run_ids:
            return deny("run_not_linked")
        if tool.action != action.kind or tool.permission != required_permission(action):
            return deny("tool_declaration_mismatch")
        if (
            action.kind in {"submit_render", "cancel_render", "merge_video"}
            and tool.side_effect != "external"
        ):
            return deny("tool_declaration_mismatch")
        if snapshot.state == "DONE" or (
            snapshot.state == "ERROR" and action.kind not in RECOVERY_ACTIONS
        ):
            return deny("terminal_state")
        if action.kind != "fail" and snapshot.state not in _PHASES[action.kind]:
            return deny("invalid_phase")
        if (
            budget.steps_remaining == 0
            or (tool.uses_model and budget.requests_remaining == 0)
            or (tool.uses_model and budget.cost_remaining == 0)
            or (budget.deadline is not None and snapshot.now >= budget.deadline)
        ):
            return deny("budget_exhausted")
        scene = next((item for item in snapshot.scenes if item.scene_id == action.scene_id), None)
        if action.scene_id is not None and scene is None:
            return deny("unknown_scene")
        identity = scene.work_identity if scene else snapshot.work_identity
        diff = action.kind == "inspect_candidate" and action.proposal is not None
        if action.work_identity != identity:
            return deny("stale_work")
        if diff:
            if (
                action.proposal.work_identity != identity
                or action.proposal.digest() != action.input_sha256
                or action.input_sha256 != scene.diff_sha256
                or action.candidate != action.proposal.candidate_source
                or action.proposal.base_source.sha256 != scene.code_sha256
                or action.proposal.base_source.kind != "code"
                or action.proposal.candidate_source.kind != "code"
                or action.proposal.inspection_report.kind != "report"
                or action.proposal.review_report.kind != "report"
            ):
                return deny("diff_identity_mismatch")
        elif action.kind == "publish_continuity":
            sources = action.publication.scenes
            ordered = tuple(sorted(snapshot.scenes, key=lambda item: item.scene_id))
            if (
                action.scene_id is not None
                or action.publication.plan_identity != snapshot.work_identity
                or snapshot.plan_sha256 is None
                or snapshot.approved_plan_sha256 != snapshot.plan_sha256
                or snapshot.approved_plan_identity != snapshot.work_identity
                or action.input_sha256 != action.publication.digest()
                or len(sources) != len(ordered)
                or any(
                    bound.scene_id != committed.scene_id
                    or bound.work_identity != committed.work_identity
                    or bound.source != committed.source_ref
                    or bound.source.kind != "code"
                    or committed.phase not in {"reviewed", "rendered", "visual_accepted"}
                    or not committed.static_verified
                    or committed.approved_code_identity != committed.work_identity
                    or committed.approved_code_sha256 != committed.code_sha256
                    for bound, committed in zip(sources, ordered, strict=True)
                )
            ):
                return deny("continuity_inputs_not_reviewed")
        elif action.input_sha256 != identity.input_sha256:
            return deny("stale_work")
        if scene:
            if action.kind in _SCENE_PHASES and scene.phase not in _SCENE_PHASES[action.kind]:
                return deny("invalid_scene_phase")
            if action.kind in {
                "propose_code_candidate",
                "inspect_candidate",
                "review_code",
                "submit_render",
            }:
                for dependency in scene.dependencies:
                    source = next(
                        (item for item in snapshot.scenes if item.scene_id == dependency.scene_id),
                        None,
                    )
                    if (
                        source is None
                        or source.scene_id >= scene.scene_id
                        or source.work_identity != dependency.work_identity
                        or source.phase not in {"reviewed", "rendered", "visual_accepted"}
                    ):
                        return deny("stale_dependency")
            if (
                action.kind in {"poll_render", "cancel_render"}
                and action.job_id != scene.external_job_id
            ):
                return deny("job_identity_mismatch")
            if action.kind == "submit_render" and (
                scene.code_sha256 is None
                or scene.approved_code_sha256 != scene.code_sha256
                or action.source.sha256 != scene.code_sha256
                or action.source.kind != "code"
                or scene.approved_code_identity != identity
            ):
                return deny("code_not_approved")
            if (
                action.kind in {"propose_render_fix", "propose_visual_fix"}
                and scene.diff_sha256 is not None
                and (
                    scene.approved_diff_sha256 != scene.diff_sha256
                    or scene.approved_diff_identity != identity
                )
            ):
                return deny("diff_not_approved")
        if action.kind == "propose_code_candidate":
            plan = scene if scene and scene.plan_sha256 is not None else snapshot
            plan_identity = identity if plan is scene else snapshot.work_identity
            if (
                plan.plan_sha256 is None
                or plan.approved_plan_sha256 != plan.plan_sha256
                or plan.approved_plan_identity != plan_identity
            ):
                return deny("plan_not_approved")
        run_resource = f"run:{snapshot.run_id}"
        if not any(
            scope.permission == tool.permission and scope.resource_id in {resource, run_resource}
            for scope in session.permissions
        ):
            return deny("permission_denied")
        if action.kind in {"merge_video", "finish"}:
            valid_early_finish = (
                action.kind == "finish"
                and bool(snapshot.scenes)
                and (
                    (
                        snapshot.plan_only
                        and snapshot.plan_sha256 is not None
                        and all(item.phase == "plan_reviewed" for item in snapshot.scenes)
                    )
                    or (
                        snapshot.dry_run
                        and all(
                            item.phase == "reviewed"
                            and item.static_verified
                            and item.code_sha256 is not None
                            and item.approved_code_sha256 == item.code_sha256
                            and item.approved_code_identity == item.work_identity
                            for item in snapshot.scenes
                        )
                    )
                )
            )
            eligible = tuple(
                item
                for item in snapshot.scenes
                if not (snapshot.allow_partial and item.give_up and item.phase == "failed")
            )
            if not valid_early_finish and (
                not eligible or any(item.phase != "visual_accepted" for item in eligible)
            ):
                return deny("unfinished_scenes")
        needs_grant = tool.requires_approval or diff
        grant = (
            next(
                (
                    grant
                    for grant in session.grants
                    if (
                        grant.action == action.kind
                        and grant.resource_id == resource
                        and grant.input_sha256 == action.input_sha256
                        and grant.run_id == action.run_id
                        and grant.session_id == session.session_id
                        and grant.work_identity == identity
                        and grant.issued_by == "local_user"
                        and grant.consumed_by_intent is None
                        and grant.issued_at <= snapshot.now
                        and (grant.expires_at is None or snapshot.now < grant.expires_at)
                        and grant_is_current(grant, session)
                    )
                ),
                None,
            )
            if needs_grant
            else None
        )
        if needs_grant and grant is None:
            return PolicyDecision(effect="ask", reason="approval_required")
        return PolicyDecision(
            effect="allow",
            reason="approved" if grant else "allowed",
            grant_id=grant.grant_id if grant else None,
            allowed_tools=(action.kind,),
        )
