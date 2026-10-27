import pytest

from kd1_anime.agent_runtime.contracts import Finish, MergeVideo, ObjectRef, PermissionScope
from kd1_anime.agent_runtime.policy import (
    Policy,
    PolicyBudget,
    PolicySnapshot,
    ScenePolicyState,
    ToolPolicySpec,
)
from kd1_anime.agent_runtime.session import Session
from kd1_anime.domain.dependencies import WorkIdentity


@pytest.mark.parametrize(
    "plan_only,dry_run,phase,verified,allowed",
    [
        (True, False, "plan_reviewed", False, True),
        (True, False, "detailed", False, False),
        (False, True, "reviewed", True, True),
        (False, True, "coded", True, False),
        (False, True, "reviewed", False, False),
        (False, False, "reviewed", True, False),
    ],
)
def test_early_finish_requires_explicit_mode_and_validated_scene(
    plan_only, dry_run, phase, verified, allowed
):
    run = "20260101-000000-1234abcd"
    identity = WorkIdentity(1, "a" * 64)
    scope = f"run:{run}"
    scene = ScenePolicyState(
        scene_id=1,
        phase=phase,
        work_identity=identity,
        static_verified=verified,
        code_sha256="a" * 64,
        approved_code_sha256="a" * 64,
        approved_code_identity=identity,
    )
    snapshot = PolicySnapshot(
        run_id=run,
        state="EVALUATING",
        work_identity=identity,
        scenes=(scene,),
        plan_only=plan_only,
        dry_run=dry_run,
        plan_sha256="a" * 64,
    )
    session = Session(
        session_id="test",
        goal="pause",
        run_ids=(run,),
        permissions=(PermissionScope(permission="write", resource_id=scope),),
    )
    action = Finish(run_id=run, resource_id=scope, input_sha256="a" * 64, work_identity=identity)
    result = Policy().evaluate(
        snapshot,
        action,
        ToolPolicySpec(action="finish", permission="write", side_effect="commit"),
        session,
        PolicyBudget(),
    )
    assert (result.effect == "allow") is allowed


@pytest.mark.parametrize("kind,empty", [("finish", True), ("merge_video", False)])
def test_early_modes_never_authorize_empty_finish_or_unrendered_merge(kind, empty):
    run = "20260101-000000-1234abcd"
    identity = WorkIdentity(1, "a" * 64)
    scope = f"run:{run}"
    scenes = (
        ()
        if empty
        else (ScenePolicyState(scene_id=1, phase="plan_reviewed", work_identity=identity),)
    )
    snapshot = PolicySnapshot(
        run_id=run,
        state="EVALUATING" if kind == "finish" else "MERGING",
        work_identity=identity,
        scenes=scenes,
        plan_only=True,
        plan_sha256="a" * 64,
    )
    permission = "write" if kind == "finish" else "exec"
    session = Session(
        session_id="test",
        goal="pause",
        run_ids=(run,),
        permissions=(PermissionScope(permission=permission, resource_id=scope),),
    )
    options = dict(run_id=run, resource_id=scope, input_sha256="a" * 64, work_identity=identity)
    action = (
        Finish(**options)
        if kind == "finish"
        else MergeVideo(
            **options, artifacts=(ObjectRef(path="video.mp4", sha256="a" * 64, kind="video"),)
        )
    )
    result = Policy().evaluate(
        snapshot,
        action,
        ToolPolicySpec(
            action=kind,
            permission=permission,
            side_effect="commit" if kind == "finish" else "external",
        ),
        session,
        PolicyBudget(),
    )
    assert result.reason == "unfinished_scenes"
