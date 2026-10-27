import pytest
from test_agent_vertical_slice import agent_boundary as agent_boundary

from kd1_anime.agent_runtime.runtime import AgentRuntime
from kd1_anime.run_store import RunRepository


def test_continuity_publication_is_committed_before_render_and_replays_without_duplicate_action(
    agent_boundary,
):
    boundary, dependencies = agent_boundary

    def interrupt(manifest):
        if any(
            item.action == "publish_continuity" and item.status == "completed"
            for item in manifest.agent.invocations
        ):
            assert manifest.agent.continuity_published
            assert set(manifest.state_ledger.boundaries) == {1}
            assert manifest.scenes[1].reviewed
            assert not boundary.backend.sources
            raise OSError("after durable continuity publication")

    dependencies.after_commit = interrupt
    with pytest.raises(OSError, match="durable continuity publication"):
        AgentRuntime(dependencies=dependencies).generate("Show a pause", approve_plan=True)
    root = next((boundary.workspace / "runs").iterdir())
    before = RunRepository(boundary.workspace).load(root.name)
    invocation = next(
        item for item in before.agent.invocations if item.action == "publish_continuity"
    )
    intent = next(item for item in before.agent.intents if item.intent_id == invocation.intent_id)
    assert intent.action.publication.plan_identity == intent.action.work_identity
    assert intent.action.input_sha256 == intent.action.publication.digest()
    assert intent.action.input_sha256 != intent.action.work_identity.input_sha256
    assert intent.action.publication.scenes[0].source.sha256 == before.scenes[1].code_sha256
    dependencies.after_commit = None
    assert AgentRuntime(dependencies=dependencies).resume(root.name).is_file()
    after = RunRepository(boundary.workspace).load(root.name)
    assert (
        len([item for item in after.agent.invocations if item.action == "publish_continuity"]) == 1
    )
    assert len(boundary.backend.sources) == 1


def test_publication_requires_every_exact_ordered_reviewed_source():
    from test_agent_policy import HASH, IDENTITY, RUN, session, snapshot

    from kd1_anime.agent_runtime.contracts import (
        ContinuityBinding,
        ContinuitySource,
        ObjectRef,
        PermissionScope,
        PublishContinuity,
    )
    from kd1_anime.agent_runtime.policy import (
        Policy,
        PolicyBudget,
        ScenePolicyState,
        ToolPolicySpec,
    )
    from kd1_anime.domain.dependencies import WorkIdentity

    scenes = tuple(
        ScenePolicyState(
            scene_id=identifier,
            phase="reviewed",
            work_identity=WorkIdentity(identifier, HASH),
            source_ref=ObjectRef(path=f"scenes/{identifier}.py", sha256=HASH, kind="code"),
            code_sha256=HASH,
            approved_code_sha256=HASH,
            approved_code_identity=WorkIdentity(identifier, HASH),
            static_verified=True,
        )
        for identifier in (1, 2)
    )
    binding = ContinuityBinding(
        plan_identity=IDENTITY,
        scenes=tuple(
            ContinuitySource(
                scene_id=item.scene_id, work_identity=item.work_identity, source=item.source_ref
            )
            for item in scenes
        ),
    )
    action = PublishContinuity(
        run_id=RUN,
        resource_id=f"run:{RUN}",
        input_sha256=binding.digest(),
        work_identity=IDENTITY,
        publication=binding,
    )
    snap = snapshot(
        scenes=scenes, plan_sha256=HASH, approved_plan_sha256=HASH, approved_plan_identity=IDENTITY
    ).model_copy(update={"state": "CODING"})
    tool = ToolPolicySpec(
        action="publish_continuity", permission="write", side_effect="commit", uses_model=False
    )
    user = session(PermissionScope(permission="write", resource_id=action.resource_id))
    policy = Policy()
    assert policy.evaluate(snap, action, tool, user, PolicyBudget()).effect == "allow"
    reversed_binding = binding.model_copy(update={"scenes": tuple(reversed(binding.scenes))})
    reordered = action.model_copy(
        update={"publication": reversed_binding, "input_sha256": reversed_binding.digest()}
    )
    assert policy.evaluate(snap, reordered, tool, user, PolicyBudget()).effect == "deny"
    for update in (
        {"phase": "coded"},
        {"static_verified": False},
        {"work_identity": WorkIdentity(3, HASH)},
    ):
        changed = snap.model_copy(
            update={"scenes": (scenes[0], scenes[1].model_copy(update=update))}
        )
        assert policy.evaluate(changed, action, tool, user, PolicyBudget()).effect == "deny"
