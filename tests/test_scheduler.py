"""Runtime scheduling, review barriers and isolated scene contexts."""

from threading import Event, Lock

import pytest
from test_agent_vertical_slice import agent_boundary as agent_boundary
from test_application import VIDEO

from kd1_anime.agent_runtime.runtime import AgentRuntime
from kd1_anime.application import Application
from kd1_anime.config import settings
from kd1_anime.domain.planning import SceneOutline
from kd1_anime.run_store import RunRepository


def test_single_scene_plan_review_and_inspection_precede_render(agent_boundary):
    boundary, dependencies = agent_boundary
    app = Application(runtime_factory=lambda: AgentRuntime(dependencies=dependencies))
    assert app.generate("Show a one-second pause", approve_plan=True) is not None
    root = next((boundary.workspace / "runs").iterdir())
    manifest = RunRepository(boundary.workspace).load_for_resume(root.name)
    actions = [item.action.kind for item in manifest.agent.intents]
    assert actions.index("review_plan") < actions.index("propose_code_candidate")
    assert actions.index("propose_technical_spec") < actions.index("propose_code_candidate")
    assert actions.index("inspect_candidate") < actions.index("review_code")
    assert actions.index("review_code") < actions.index("submit_render")
    assert actions.index("submit_render") < actions.index("merge_video")
    assert len(boundary.backend.sources) == 1


def test_rejected_plan_never_reaches_coder_or_render(agent_boundary):
    boundary, dependencies = agent_boundary
    boundary.rejecting.plan = True
    app = Application(runtime_factory=lambda: AgentRuntime(dependencies=dependencies))
    with pytest.raises(RuntimeError):
        app.generate("Show a one-second pause", approve_plan=True)
    root = next((boundary.workspace / "runs").iterdir())
    manifest = RunRepository(boundary.workspace).load_for_resume(root.name)
    assert manifest.status == "failed"
    assert "code" not in boundary.calls and "code_review" not in boundary.calls
    assert not boundary.backend.sources


def test_dry_run_never_submits_render(agent_boundary):
    boundary, dependencies = agent_boundary
    app = Application(runtime_factory=lambda: AgentRuntime(dependencies=dependencies))
    app.generate("Show a one-second pause", dry_run=True, approve_plan=True)
    root = next((boundary.workspace / "runs").iterdir())
    manifest = RunRepository(boundary.workspace).load_for_resume(root.name)
    assert manifest.status == "dry_run_complete"
    assert not boundary.backend.sources


def test_independent_scenes_finish_with_no_unrelated_inherited_context(agent_boundary, monkeypatch):
    boundary, dependencies = agent_boundary
    monkeypatch.setattr(settings, "LLM_PARALLEL_WORKERS", 2)
    original_planner = dependencies.planner_factory
    original_coder = dependencies.coder_factory
    calls = []
    calls_lock = Lock()
    both_started = Event()
    entered = []

    class Planner(original_planner):
        def plan_outline(self, prompt, **kwargs):
            first = super().plan_outline(prompt, **kwargs)[0]
            return [
                first,
                SceneOutline(
                    scene_id=2,
                    title="Second pause",
                    duration_seconds=1,
                    purpose="Show a second pause",
                    math_concept="Time",
                ),
            ]

        def plan_detail(self, outline, all_outlines, prompt, **kwargs):
            with calls_lock:
                entered.append(outline.scene_id)
                if len(entered) == 2:
                    both_started.set()
            assert both_started.wait(5), "independent detailing did not run concurrently"
            return (
                super()
                .plan_detail(outline, all_outlines, prompt, **kwargs)
                .model_copy(update={"dependencies_complete": True})
            )

    class Coder(original_coder):
        def generate_code(self, scene_plan, **kwargs):
            calls.append((scene_plan.scene_id, kwargs.get("inherited_elements_code", "")))
            return super().generate_code(scene_plan, **kwargs)

    class Merger:
        def merge(self, videos, output, **kwargs):
            assert len(videos) == 2
            assert all(path.read_bytes() == VIDEO for path in videos)
            output.write_bytes(b"merged:" + VIDEO + VIDEO)
            return output

    dependencies.planner_factory = Planner
    dependencies.coder_factory = Coder
    dependencies.merger_factory = Merger
    app = Application(runtime_factory=lambda: AgentRuntime(dependencies=dependencies))
    output = app.generate("Show two independent pauses", approve_plan=True)
    assert output.read_bytes() == b"merged:" + VIDEO + VIDEO
    root = next((boundary.workspace / "runs").iterdir())
    manifest = RunRepository(boundary.workspace).load_for_resume(root.name)
    assert sorted(entered) == [1, 2]
    assert sorted(calls) == [(1, ""), (2, "")]
    assert len(boundary.backend.sources) == 2
    assert all(scene.rendered and scene.reviewed for scene in manifest.scenes.values())
    actions = [item.action for item in manifest.agent.intents]
    detail_positions = [i for i, action in enumerate(actions) if action.kind == "detail_scene"]
    code_positions = [
        i for i, action in enumerate(actions) if action.kind == "propose_code_candidate"
    ]
    assert max(detail_positions) < min(code_positions)
