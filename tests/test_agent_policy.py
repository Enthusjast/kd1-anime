"""Pure domain permission, phase, identity and approval decisions."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from kd1_anime.agent_runtime.contracts import (
    ApprovalGrant,
    CancelRender,
    MergeVideo,
    PermissionScope,
    ProposeCodeCandidate,
    ProposePlan,
    SubmitRender,
)
from kd1_anime.agent_runtime.policy import (
    Policy,
    PolicyBudget,
    PolicyDependency,
    PolicySnapshot,
    ScenePolicyState,
    ToolPolicySpec,
)
from kd1_anime.domain.dependencies import WorkIdentity

RUN = "20261019-120000-1234abcd"
HASH = "a" * 64
NOW = datetime(2026, 10, 27, tzinfo=timezone.utc)
IDENTITY = WorkIdentity(1, HASH)


def snapshot(**updates):
    return PolicySnapshot(run_id=RUN, state="INIT", work_identity=IDENTITY, now=NOW, **updates)


def session(*permissions, grants=()):
    return SimpleNamespace(
        session_id="session-one", run_ids=(RUN,), permissions=tuple(permissions), grants=grants
    )


def plan():
    return ProposePlan(
        run_id=RUN, resource_id=f"run:{RUN}", input_sha256=HASH, work_identity=IDENTITY
    )


def render():
    return SubmitRender(
        run_id=RUN,
        resource_id=f"run:{RUN}/scene:1",
        input_sha256=HASH,
        work_identity=IDENTITY,
        scene_id=1,
        backend="slurm",
        source={"path": "scenes/1.py", "sha256": HASH, "kind": "code"},
        class_name="Scene",
    )


def test_phase_and_exact_resource_permissions_are_authoritative():
    policy = Policy()
    action = plan()
    tool = ToolPolicySpec(action="propose_plan", permission="write", side_effect="commit")
    assert policy.evaluate(snapshot(), action, tool, session(), PolicyBudget()).effect == "deny"
    wrong = PermissionScope(permission="write", resource_id=f"run:{RUN}/scene:1")
    assert (
        policy.evaluate(snapshot(), action, tool, session(wrong), PolicyBudget()).effect == "deny"
    )
    correct = PermissionScope(permission="write", resource_id=action.resource_id)
    assert (
        policy.evaluate(snapshot(), action, tool, session(correct), PolicyBudget()).effect
        == "allow"
    )
    assert (
        policy.evaluate(
            snapshot().model_copy(update={"state": "MONITORING"}),
            action,
            tool,
            session(correct),
            PolicyBudget(),
        ).reason
        == "invalid_phase"
    )
    assert (
        policy.evaluate(
            snapshot(), action, tool, session(correct), PolicyBudget(steps_remaining=0)
        ).reason
        == "budget_exhausted"
    )


def test_unknown_scopes_actions_and_underdeclared_permissions_fail_closed():
    with pytest.raises(ValidationError):
        PermissionScope(permission="admin", resource_id=f"run:{RUN}")
    with pytest.raises(ValidationError):
        ToolPolicySpec(action="shell", permission="exec", side_effect="external")
    result = Policy().evaluate(
        snapshot(),
        plan(),
        ToolPolicySpec(action="propose_plan", permission="read", side_effect="none"),
        session(PermissionScope(permission="read", resource_id=f"run:{RUN}")),
        PolicyBudget(),
    )
    assert result.effect == "deny"
    assert result.reason == "tool_declaration_mismatch"


def test_approval_binds_hash_revision_scope_and_expiration():
    action = render()
    scene = ScenePolicyState(
        scene_id=1,
        phase="reviewed",
        work_identity=IDENTITY,
        code_sha256=HASH,
        approved_code_sha256=HASH,
        approved_code_identity=IDENTITY,
    )
    snap = snapshot(scenes=(scene,)).model_copy(update={"state": "DISPATCHING"})
    tool = ToolPolicySpec(
        action="submit_render", permission="cluster", side_effect="external", requires_approval=True
    )
    scope = PermissionScope(permission="cluster", resource_id=action.resource_id)
    grant = ApprovalGrant(
        action=action.kind,
        resource_id=action.resource_id,
        input_sha256=HASH,
        run_id=RUN,
        session_id="session-one",
        work_identity=IDENTITY,
        issued_at=NOW - timedelta(hours=1),
        expires_at=NOW + timedelta(hours=1),
    )
    policy = Policy()
    assert policy.evaluate(snap, action, tool, session(scope), PolicyBudget()).effect == "ask"
    assert (
        policy.evaluate(
            snap, action, tool, session(scope, grants=(grant,)), PolicyBudget()
        ).grant_id
        == grant.grant_id
    )
    for updates in (
        {"input_sha256": "b" * 64},
        {"work_identity": WorkIdentity(2, HASH)},
        {"resource_id": f"run:{RUN}"},
        {"expires_at": NOW},
        {"consumed_by_intent": "old"},
        {"session_id": "other"},
        {"issued_at": NOW + timedelta(minutes=1)},
    ):
        assert (
            policy.evaluate(
                snap,
                action,
                tool,
                session(scope, grants=(grant.model_copy(update=updates),)),
                PolicyBudget(),
            ).effect
            == "ask"
        )
    stale_scene = scene.model_copy(update={"approved_code_identity": WorkIdentity(0, HASH)})
    assert (
        policy.evaluate(
            snap.model_copy(update={"scenes": (stale_scene,)}),
            action,
            tool,
            session(scope, grants=(grant,)),
            PolicyBudget(),
        ).reason
        == "code_not_approved"
    )


def test_scene_dependencies_and_work_revision_are_checked_before_grants():
    action = render()
    scene = ScenePolicyState(scene_id=1, phase="reviewed", work_identity=WorkIdentity(2, HASH))
    snap = snapshot(scenes=(scene,)).model_copy(update={"state": "DISPATCHING"})
    result = Policy().evaluate(
        snap,
        action,
        ToolPolicySpec(action="submit_render", permission="cluster", side_effect="external"),
        session(),
        PolicyBudget(),
    )
    assert result.reason == "stale_work"


def test_malicious_resource_cannot_be_authorized_even_by_an_identical_scope():
    action = plan().model_copy(update={"resource_id": "run:other/../../private"})
    scope = PermissionScope(permission="write", resource_id=action.resource_id)
    decision = Policy().evaluate(
        snapshot(),
        action,
        ToolPolicySpec(action="propose_plan", permission="write", side_effect="commit"),
        session(scope),
        PolicyBudget(),
    )
    assert decision.reason == "invalid_resource"


def test_coding_requires_exact_approved_plan_and_dependency_revision():
    source = ScenePolicyState(scene_id=1, phase="reviewed", work_identity=IDENTITY)
    consumer = ScenePolicyState(
        scene_id=2,
        phase="technical_planning",
        work_identity=IDENTITY,
        dependencies=(PolicyDependency(scene_id=1, work_identity=IDENTITY),),
        plan_sha256=HASH,
        approved_plan_sha256=HASH,
        approved_plan_identity=IDENTITY,
    )
    snap = snapshot(scenes=(source, consumer)).model_copy(update={"state": "CODING"})
    action = ProposeCodeCandidate(
        run_id=RUN,
        resource_id=f"run:{RUN}/scene:2",
        scene_id=2,
        input_sha256=HASH,
        work_identity=IDENTITY,
    )
    tool = ToolPolicySpec(action=action.kind, permission="write", side_effect="commit")
    user = session(PermissionScope(permission="write", resource_id=f"run:{RUN}"))
    policy = Policy()
    assert policy.evaluate(snap, action, tool, user, PolicyBudget()).effect == "allow"
    stale = source.model_copy(update={"work_identity": WorkIdentity(2, HASH)})
    assert (
        policy.evaluate(
            snap.model_copy(update={"scenes": (stale, consumer)}),
            action,
            tool,
            user,
            PolicyBudget(),
        ).reason
        == "stale_dependency"
    )
    unapproved = consumer.model_copy(update={"approved_plan_sha256": "b" * 64})
    assert (
        policy.evaluate(
            snap.model_copy(update={"scenes": (source, unapproved)}),
            action,
            tool,
            user,
            PolicyBudget(),
        ).reason
        == "plan_not_approved"
    )


def test_cancellation_needs_exact_current_job_permission_but_not_an_extra_grant():
    scene = ScenePolicyState(
        scene_id=1, phase="monitoring", work_identity=IDENTITY, external_job_id="123"
    )
    snap = snapshot(scenes=(scene,)).model_copy(update={"state": "MONITORING"})
    action = CancelRender(
        run_id=RUN,
        resource_id=f"run:{RUN}/scene:1",
        scene_id=1,
        input_sha256=HASH,
        work_identity=IDENTITY,
        job_id="123",
    )
    tool = ToolPolicySpec(action=action.kind, permission="cluster", side_effect="external")
    user = session(PermissionScope(permission="cluster", resource_id=action.resource_id))
    policy = Policy()
    assert policy.evaluate(snap, action, tool, user, PolicyBudget()).effect == "allow"
    assert (
        policy.evaluate(
            snap, action.model_copy(update={"job_id": "124"}), tool, user, PolicyBudget()
        ).reason
        == "job_identity_mismatch"
    )
    needs_approval = tool.model_copy(update={"requires_approval": True})
    submit_grant = ApprovalGrant(
        action="submit_render",
        resource_id=action.resource_id,
        run_id=RUN,
        input_sha256=HASH,
        work_identity=IDENTITY,
        session_id=user.session_id,
        issued_at=NOW - timedelta(minutes=1),
    )
    assert (
        policy.evaluate(
            snap,
            action,
            needs_approval,
            session(*user.permissions, grants=(submit_grant,)),
            PolicyBudget(),
        ).effect
        == "ask"
    )


def test_merge_requires_exec_scope_not_write_scope():
    snap = snapshot().model_copy(update={"state": "MERGING"})
    action = MergeVideo(
        run_id=RUN,
        resource_id=f"run:{RUN}",
        input_sha256=HASH,
        work_identity=IDENTITY,
        artifacts=({"path": "videos/1.mp4", "sha256": HASH, "kind": "video"},),
    )
    tool = ToolPolicySpec(action=action.kind, permission="exec", side_effect="external")
    policy = Policy()
    assert (
        policy.evaluate(
            snap,
            action,
            tool,
            session(PermissionScope(permission="write", resource_id=action.resource_id)),
            PolicyBudget(),
        ).effect
        == "deny"
    )
    assert (
        policy.evaluate(
            snap,
            action,
            tool,
            session(PermissionScope(permission="exec", resource_id=action.resource_id)),
            PolicyBudget(),
        ).effect
        == "allow"
    )


def test_later_user_command_invalidates_a_previously_matching_approval():
    action = plan()
    grant = ApprovalGrant(
        action=action.kind,
        resource_id=action.resource_id,
        run_id=RUN,
        input_sha256=HASH,
        work_identity=IDENTITY,
        session_id="session-one",
        issued_at=NOW - timedelta(minutes=1),
    )
    user = session(
        PermissionScope(permission="write", resource_id=action.resource_id), grants=(grant,)
    )
    approve = SimpleNamespace(
        kind="approve", role="user", action=action, created_at=grant.issued_at
    )
    user.commands = (approve,)
    tool = ToolPolicySpec(
        action=action.kind, permission="write", side_effect="commit", requires_approval=True
    )
    assert Policy().evaluate(snapshot(), action, tool, user, PolicyBudget()).effect == "allow"
    user.commands += (SimpleNamespace(kind="cancel", action=None),)
    assert Policy().evaluate(snapshot(), action, tool, user, PolicyBudget()).effect == "ask"
