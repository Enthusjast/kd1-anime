from dataclasses import replace

from test_agent_policy import HASH, IDENTITY, NOW, RUN, session, snapshot

from kd1_anime.agent_runtime.contracts import (
    ApprovalGrant,
    DiffBinding,
    InspectCandidate,
    ObjectRef,
    PermissionScope,
)
from kd1_anime.agent_runtime.policy import Policy, PolicyBudget, ScenePolicyState, ToolPolicySpec


def test_diff_grant_binds_candidate_evidence_and_unchanged_base_identity():
    source = ObjectRef(path="objects/base.py", sha256=HASH, kind="code")
    candidate = source.model_copy(update={"path": "objects/fix.py", "sha256": "b" * 64})
    report = ObjectRef(path="reports/inspection.json", sha256="c" * 64, kind="report")
    binding = DiffBinding(
        work_identity=IDENTITY,
        base_source=source,
        candidate_source=candidate,
        inspection_report=report,
        review_report=report,
        context_sha256="d" * 64,
    )
    action = InspectCandidate(
        run_id=RUN,
        resource_id=f"run:{RUN}/scene:1",
        scene_id=1,
        input_sha256=binding.digest(),
        work_identity=IDENTITY,
        candidate=candidate,
        proposal=binding,
    )
    scene = ScenePolicyState(
        scene_id=1,
        phase="coded",
        work_identity=IDENTITY,
        code_sha256=HASH,
        diff_sha256=binding.digest(),
    )
    snap = snapshot(scenes=(scene,)).model_copy(update={"state": "CODING"})
    tool = ToolPolicySpec(
        action="inspect_candidate",
        permission="write",
        side_effect="commit",
        requires_approval=False,
        uses_model=False,
    )
    scope = PermissionScope(permission="write", resource_id=action.resource_id)
    policy = Policy()
    ordinary = action.model_copy(
        update={"proposal": None, "candidate": source, "input_sha256": HASH}
    )
    assert policy.evaluate(snap, ordinary, tool, session(scope), PolicyBudget()).effect == "allow"
    assert action.input_sha256 != IDENTITY.input_sha256
    assert policy.evaluate(snap, action, tool, session(scope), PolicyBudget()).effect == "ask"
    grant = ApprovalGrant(
        action=action.kind,
        resource_id=action.resource_id,
        input_sha256=binding.digest(),
        run_id=RUN,
        session_id="session-one",
        work_identity=IDENTITY,
        issued_at=NOW,
    )
    assert (
        policy.evaluate(snap, action, tool, session(scope, grants=(grant,)), PolicyBudget()).effect
        == "allow"
    )
    for change in (
        {"candidate": source},
        {"proposal": binding.model_copy(update={"context_sha256": "e" * 64})},
    ):
        changed = action.model_copy(update=change)
        assert (
            policy.evaluate(
                snap, changed, tool, session(scope, grants=(grant,)), PolicyBudget()
            ).effect
            == "deny"
        )
    advanced = scene.model_copy(update={"work_identity": replace(IDENTITY, revision=2)})
    assert (
        policy.evaluate(
            snap.model_copy(update={"scenes": (advanced,)}),
            action,
            tool,
            session(scope, grants=(grant,)),
            PolicyBudget(),
        ).reason
        == "stale_work"
    )
