import json
import threading
from pathlib import Path

from test_agent_vertical_slice import agent_boundary as agent_boundary
from test_application import VIDEO

from kd1_anime.agent_runtime.runtime import AgentRuntime
from kd1_anime.config import settings
from kd1_anime.domain.planning import LessonSpec, PlanningDraft, TeachingGraph
from kd1_anime.run_store import RunRepository


def test_independent_multiscene_codegen_is_parallel_and_publishes_boundaries_before_render(
    agent_boundary, monkeypatch
):
    boundary, dependencies = agent_boundary
    original_planner = dependencies.planner_factory
    original_coder = dependencies.coder_factory
    participants = threading.Barrier(2)
    workers = []
    monkeypatch.setattr(settings, "LLM_PARALLEL_WORKERS", 2)

    class Planner(original_planner):
        def plan_draft(self, prompt, **kwargs):
            draft = super().plan_draft(prompt, **kwargs)
            first = draft.items[0]
            return PlanningDraft(
                items=[first, first.model_copy(update={"scene_id": 2, "title": "Second pause"})],
                lesson_spec=LessonSpec(topic=prompt),
                teaching_graph=TeachingGraph(),
            )

        def plan_detail(self, outline, outlines, prompt, **kwargs):
            return (
                super()
                .plan_detail(outline, outlines, prompt, **kwargs)
                .model_copy(update={"dependencies": [], "dependencies_complete": True})
            )

    class Coder(original_coder):
        def generate_code(self, plan, **kwargs):
            code = super().generate_code(plan, **kwargs)
            workers.append(threading.get_ident())
            participants.wait(timeout=5)
            return code

    class Merger:
        def merge(self, videos, output, **kwargs):
            assert len(videos) == 2
            assert all(Path(path).read_bytes() == VIDEO for path in videos)
            output = Path(output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"merged:" + VIDEO + b"|" + VIDEO)
            return output

    submit = boundary.backend.submit_scene

    def render_after_boundary_publication(*args, **kwargs):
        root = next((boundary.workspace / "runs").iterdir())
        committed = RunRepository(boundary.workspace).load(root.name)
        assert all(scene.reviewed for scene in committed.scenes.values())
        assert set(committed.state_ledger.boundaries) == {1, 2}
        return submit(*args, **kwargs)

    dependencies.planner_factory = Planner
    dependencies.coder_factory = Coder
    dependencies.merger_factory = Merger
    boundary.backend.submit_scene = render_after_boundary_publication
    output = AgentRuntime(dependencies=dependencies).generate("Show two pauses", approve_plan=True)
    assert output.read_bytes() == b"merged:" + VIDEO + b"|" + VIDEO
    assert len(set(workers)) == 2
    root = next((boundary.workspace / "runs").iterdir())
    raw = json.loads((root / "manifest.json").read_text())
    assert raw["schema_version"] == 11 and raw["status"] == "completed"
    assert set(raw["scenes"]) == {"1", "2"}
    assert len(boundary.backend.sources) == 2
    reservations = raw["request_budget"]["reservations"]
    assert len({item["request_id"] for item in reservations}) == len(reservations)
    assert {item["scene_id"] for item in reservations} >= {1, 2}


def test_detail_workers_have_distinct_intents_and_outline_identities(agent_boundary, monkeypatch):
    _, dependencies = agent_boundary
    original = dependencies.planner_factory
    participants = threading.Barrier(2)
    workers = []
    monkeypatch.setattr(settings, "LLM_PARALLEL_WORKERS", 2)

    class Planner(original):
        def plan_draft(self, prompt, **kwargs):
            draft = super().plan_draft(prompt, **kwargs)
            first = draft.items[0]
            return PlanningDraft(
                items=[first, first.model_copy(update={"scene_id": 2, "title": "Second pause"})],
                lesson_spec=LessonSpec(topic=prompt),
                teaching_graph=TeachingGraph(),
            )

        def plan_detail(self, outline, outlines, prompt, **kwargs):
            assert kwargs.get("stream") is False
            workers.append(threading.get_ident())
            participants.wait(timeout=5)
            return (
                super()
                .plan_detail(outline, outlines, prompt, **kwargs)
                .model_copy(update={"dependencies": [], "dependencies_complete": True})
            )

    dependencies.planner_factory = Planner
    runtime = AgentRuntime(dependencies=dependencies)
    assert len(runtime.plan_only("Show two pauses", approve_plan=True)) == 2
    manifest = runtime.manifest
    assert len(set(workers)) == 2
    intents = [item for item in manifest.agent.intents if item.action.kind == "detail_scene"]
    assert {item.action.scene_id for item in intents} == {1, 2}
    for intent in intents:
        assert (
            intent.action.work_identity == manifest.agent.outline_identities[intent.action.scene_id]
        )
        invocation = next(
            item for item in manifest.agent.invocations if item.intent_id == intent.intent_id
        )
        assert (
            invocation.status == "completed"
            and invocation.work_identity == intent.action.work_identity
        )
    reservations = {item.request_id: item for item in manifest.request_budget.reservations}
    for usage in manifest.agent.usage:
        if usage.invocation_id is None:
            continue
        invocation = next(
            item for item in manifest.agent.invocations if item.invocation_id == usage.invocation_id
        )
        scene_id = reservations[usage.request_id].scene_id
        if scene_id is not None:
            assert (
                invocation.resource_id.endswith(f"/scene:{scene_id}")
                or invocation.action == "review_plan"
            )


def test_dependency_revision_changed_during_codegen_rejects_result(agent_boundary):
    from kd1_anime.commits import CommitCommand, RunPatch, ScenePatch
    from kd1_anime.domain.dependencies import WorkIdentity

    boundary, dependencies = agent_boundary
    original_planner, original_coder = dependencies.planner_factory, dependencies.coder_factory

    class Planner(original_planner):
        def plan_draft(self, prompt, **kwargs):
            draft = super().plan_draft(prompt, **kwargs)
            first = draft.items[0]
            return PlanningDraft(
                items=[first, first.model_copy(update={"scene_id": 2, "title": "Dependent pause"})],
                lesson_spec=LessonSpec(topic=prompt),
                teaching_graph=TeachingGraph(),
            )

    class Coder(original_coder):
        def generate_code(self, plan, **kwargs):
            source = super().generate_code(plan, **kwargs)
            if plan.scene_id == 2:
                current = runtime.manifest
                producer = current.scenes[1]
                replacement = producer.model_copy(deep=True)
                replacement.failure_reason = "Concurrent dependency revision"
                runtime.committer.commit(
                    CommitCommand(
                        current.revision,
                        "dependency_race",
                        patch=RunPatch(
                            scenes=(
                                ScenePatch(
                                    scene_id=1,
                                    identity=WorkIdentity(producer.revision, producer.input_sha256),
                                    replacement=replacement,
                                ),
                            )
                        ),
                    )
                )
            return source

    dependencies.planner_factory, dependencies.coder_factory = Planner, Coder
    runtime = AgentRuntime(dependencies=dependencies)
    assert runtime.generate("Show dependent pauses", approve_plan=True) is None
    assert runtime.manifest.scenes[1].failure_reason == "Concurrent dependency revision"
    assert runtime.manifest.scenes[2].code_file == ""
    assert not boundary.backend.sources
    assert not runtime.manifest.agent.continuity_published
