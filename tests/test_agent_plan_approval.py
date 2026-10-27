import json

from test_agent_vertical_slice import agent_boundary as agent_boundary

from kd1_anime.agent_runtime.contracts import ACTION_ADAPTER
from kd1_anime.agent_runtime.runtime import AgentRuntime
from kd1_anime.agent_runtime.session import SessionStore, UserCommand
from kd1_anime.domain.planning import LessonSpec, PlanningDraft, TeachingGraph
from kd1_anime.run_store import RunRepository


def test_multiscene_plan_approval_is_one_exact_root_grant_without_repeating_review(agent_boundary):
    boundary, dependencies = agent_boundary
    original = dependencies.planner_factory

    class Planner(original):
        def plan_draft(self, prompt, **kwargs):
            draft = super().plan_draft(prompt, **kwargs)
            return PlanningDraft(
                items=[draft.items[0], draft.items[0].model_copy(update={"scene_id": 2})],
                lesson_spec=LessonSpec(topic=prompt),
                teaching_graph=TeachingGraph(),
            )

        def plan_detail(self, *args, **kwargs):
            return (
                super()
                .plan_detail(*args, **kwargs)
                .model_copy(update={"dependencies": [], "dependencies_complete": True})
            )

    dependencies.planner_factory = Planner
    assert AgentRuntime(dependencies=dependencies).generate("Two pauses", dry_run=True) is None
    root = next((boundary.workspace / "runs").iterdir())
    before = json.loads((root / "manifest.json").read_text())
    assert set(before["scenes"]) == {"1", "2"}
    assert all(scene["plan_reviewed"] for scene in before["scenes"].values())
    assert all(not scene["code_file"] for scene in before["scenes"].values())
    pending = ACTION_ADAPTER.validate_python(before["agent"]["pending_approval"])
    assert pending.resource_id == f"run:{root.name}" and pending.scene_id is None
    sessions = SessionStore(boundary.workspace / "sessions")
    session_id = before["agent"]["session_id"]
    grant = sessions.issue_grant(
        session_id, UserCommand(session_id=session_id, kind="approve", action=pending)
    )
    assert AgentRuntime(dependencies=dependencies).resume(root.name) is None
    after = RunRepository(boundary.workspace).load(root.name)
    assert after.status == "dry_run_complete"
    assert all(scene.reviewed for scene in after.scenes.values())
    old_review = [
        item for item in before["request_budget"]["reservations"] if item["stage"] == "plan_review"
    ]
    new_review = [
        item.model_dump(mode="json")
        for item in after.request_budget.reservations
        if item.stage == "plan_review"
    ]
    assert new_review == old_review
    assert sessions.load(session_id).grants[-1].grant_id == grant.grant_id
    assert sessions.load(session_id).grants[-1].consumed_by_intent is not None
    assert not boundary.backend.sources
