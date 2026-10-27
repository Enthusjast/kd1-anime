from pathlib import Path

from test_agent_vertical_slice import agent_boundary as agent_boundary
from test_application import VIDEO

from kd1_anime.agent_runtime.runtime import AgentRuntime
from kd1_anime.domain.dependencies import SceneDependency
from kd1_anime.domain.planning import LessonSpec, PlanningDraft, TeachingGraph
from kd1_anime.run_store import RunRepository


def test_render_progress_does_not_invalidate_unchanged_teaching_dependency(agent_boundary):
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

        def plan_detail(self, outline, outlines, prompt, **kwargs):
            plan = super().plan_detail(outline, outlines, prompt, **kwargs)
            return plan.model_copy(
                update={
                    "dependencies_complete": True,
                    "dependencies": []
                    if outline.scene_id == 1
                    else [SceneDependency(source_scene_id=1, kind="teaching")],
                }
            )

    class Merger:
        def merge(self, videos, output, **kwargs):
            assert len(videos) == 2 and all(Path(path).read_bytes() == VIDEO for path in videos)
            output = Path(output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"two verified scenes")
            return output

    dependencies.planner_factory = Planner
    dependencies.merger_factory = Merger
    output = AgentRuntime(dependencies=dependencies).generate(
        "Two teaching steps", approve_plan=True
    )
    assert output.read_bytes() == b"two verified scenes"
    root = next((boundary.workspace / "runs").iterdir())
    manifest = RunRepository(boundary.workspace).load(root.name)
    assert manifest.status == "completed" and len(boundary.backend.sources) == 2
    assert any(
        binding.source_scene_id == 1 and binding.consumer_scene_id == 2
        for binding in manifest.dependency_graph.bindings
    )
    assert all(scene.reviewed and scene.rendered for scene in manifest.scenes.values())
