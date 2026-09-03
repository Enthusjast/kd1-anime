import json
import threading
from pathlib import Path

import pytest

import kd1_anime.orchestrator as module
from kd1_anime.agents.api_linter import lint_manim_api
from kd1_anime.agents.failure_router import classify_failure
from kd1_anime.agents.plan_reviewer import PlanReviewIssue, PlanReviewResult
from kd1_anime.agents.planner import (
    ContinuityBible,
    ExtractedElement,
    LessonSpec,
    MathClaim,
    PlanningDraft,
    SceneHandoff,
    SceneOutline,
    ScenePlan,
    TeachingGraph,
    TimelineEvent,
    VisualElementState,
)
from kd1_anime.agents.reviewer import ReviewFinding, ReviewResult
from kd1_anime.agents.technical_planner import TechnicalAnimation, TechnicalObject, TechnicalSpec
from kd1_anime.config import settings
from kd1_anime.eval.visual_eval import VisualAnalysisResult, VisualIssue
from kd1_anime.orchestrator import Orchestrator, PipelineContext, RunPaths, SceneState, State
from kd1_anime.rendering import SceneArtifact, VideoMetadata, sha256_file
from kd1_anime.run_store import RunManifest, StoredSceneState, sha256_text, write_manifest


def plan():
    return ScenePlan(
        scene_id=1,
        title="demo",
        duration_seconds=10,
        purpose="test",
        math_concept="circle",
        visual_design="dark",
        camera_movement="fixed",
        visual_flow=["show circle"],
        key_moments=["pause"],
        computation="radius=1",
    )


def test_failure_router_prioritizes_infrastructure_over_code_rewrite():
    route = classify_failure(
        "FileNotFoundError: No such file or directory: 'xelatex'",
        phase="render",
    )
    assert route.category == "infrastructure"
    assert route.handler == "infra_retry"
    assert classify_failure("render boom", phase="render").category == "render"


def test_failure_router_separates_plan_math_from_runtime_api_errors():
    assert classify_failure("数学断言不等价", phase="review").handler == "plan_review"
    assert (
        classify_failure("AttributeError: OpenGLCamera has no frame", phase="render").handler
        == "code_patch"
    )


def test_api_linter_rejects_deprecated_manim_api():
    result = lint_manim_api(
        "from manim import *\nclass Demo(Scene):\n"
        "    def construct(self): self.play(ShowCreation(Circle()))"
    )
    assert result.is_valid is False
    assert any("ShowCreation" in error for error in result.errors)


def test_api_linter_warns_about_unbounded_graph_and_updater():
    result = lint_manim_api(
        "from manim import *\nclass Demo(Scene):\n"
        "    def construct(self):\n"
        "        axes = Axes()\n"
        "        graph = axes.plot(lambda x: x**2)\n"
        "        dot = always_redraw(lambda: Dot())\n"
    )
    assert result.is_valid is True
    assert any("x_range" in warning for warning in result.warnings)
    assert any("clear_updaters" in warning for warning in result.warnings)


def test_continuity_context_mode_defaults_to_only_requested_exports(monkeypatch, tmp_path):
    previous_plan = plan()
    previous_plan = previous_plan.model_copy(
        update={
            "new_elements": [
                VisualElementState(element_id="kept", variable_name="kept"),
                VisualElementState(element_id="other", variable_name="other"),
            ]
        }
    )
    current_plan = plan().model_copy(
        update={
            "scene_id": 2,
            "inherited_elements": [VisualElementState(element_id="kept", variable_name="kept")],
        }
    )
    previous = SceneState(
        plan=previous_plan,
        exported_elements_code="kept = Circle()\n\nother = Square()",
        exported_elements=[
            ExtractedElement(element_id="kept", variable_name="kept", code="kept = Circle()"),
            ExtractedElement(element_id="other", variable_name="other", code="other = Square()"),
        ],
    )
    state = SceneState(plan=current_plan)
    ctx = PipelineContext(
        "prompt",
        paths=paths(tmp_path),
        scene_states={1: previous, 2: state},
    )
    orchestrator = Orchestrator()

    monkeypatch.setattr(settings, "CONTINUITY_CONTEXT_MODE", "minimal")
    orchestrator._prepare_inherited_context(ctx, 2, state)
    assert state.inherited_elements_code == "kept = Circle()"

    monkeypatch.setattr(settings, "CONTINUITY_CONTEXT_MODE", "full")
    orchestrator._prepare_inherited_context(ctx, 2, state)
    assert "other = Square()" in state.inherited_elements_code


def test_stateless_mode_does_not_inject_unrequested_legacy_exports(monkeypatch, tmp_path):
    previous = SceneState(
        plan=plan(),
        exported_elements_code="old = Circle()",
    )
    current = SceneState(plan=plan().model_copy(update={"scene_id": 2}))
    ctx = PipelineContext("prompt", paths=paths(tmp_path), scene_states={1: previous, 2: current})
    monkeypatch.setattr(settings, "CONTINUITY_CONTEXT_MODE", "stateless")

    Orchestrator()._prepare_inherited_context(ctx, 2, current)

    assert current.inherited_elements_code == ""


def paths(tmp_path: Path):
    root = tmp_path / "run"
    return RunPaths(
        "20260728-120000-1234abcd",
        root,
        root / "scenes",
        root / "logs",
        root / "videos",
        root / "out.mp4",
    )


def test_valid_review_after_rewrite_exits_review_loop(tmp_path):
    run_paths = paths(tmp_path)
    run_paths.scenes.mkdir(parents=True)
    ctx = PipelineContext("x", paths=run_paths)
    ctx.scene_states[1] = SceneState(
        plan=plan(),
        code="from manim import *\nclass Demo(Scene):\n    def construct(self): pass\n",
        class_name="Demo",
        review_round=1,
    )
    Orchestrator()._apply_review_result(ctx, 1, ctx.scene_states[1], ReviewResult(is_valid=True))
    assert ctx.scene_states[1].reviewed is True
    assert ctx.scene_states[1].review_round == 0


def test_review_warning_is_accepted_without_rewrite(monkeypatch, tmp_path):
    run_paths = paths(tmp_path)
    run_paths.scenes.mkdir(parents=True)
    state = SceneState(
        plan=plan(),
        code="from manim import *\nclass Demo(Scene):\n    def construct(self): pass\n",
        class_name="Demo",
        plan_ready=True,
    )
    ctx = PipelineContext("x", paths=run_paths, scene_states={1: state})
    events = []
    orchestrator = Orchestrator()
    orchestrator._callback = lambda event, data: events.append((event, data))
    monkeypatch.setattr(orchestrator, "_checkpoint", lambda *args, **kwargs: None)

    handled = orchestrator._apply_review_result(
        ctx,
        1,
        state,
        ReviewResult(is_valid=True, warnings=["[layout] 建议调整标题位置"]),
    )

    assert handled is True
    assert state.reviewed is True
    assert state.give_up is False
    assert state.rewrite_feedback == ""
    assert any(event == "scene_review_warning" for event, _ in events)
    assert any("标题位置" in warning for warning in ctx.continuity_warnings)


def test_coder_failure_uses_validated_safe_code_fallback(monkeypatch, tmp_path):
    run_paths = paths(tmp_path)
    run_paths.root.mkdir(parents=True)
    state = SceneState(plan=plan(), plan_ready=True)
    ctx = PipelineContext("prompt", paths=run_paths, scene_states={1: state})
    orchestrator = Orchestrator()
    monkeypatch.setattr(orchestrator, "_retrieve_rag", lambda *args, **kwargs: "")
    monkeypatch.setattr(module.settings, "CODEGEN_MODE", "python")
    monkeypatch.setattr(
        orchestrator,
        "_generate_validated_code",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("模拟 Coder 截断")),
    )
    monkeypatch.setattr(orchestrator, "_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(orchestrator, "_local_smoke_render", lambda *args, **kwargs: None)

    orchestrator._scene_code(ctx, 1, state)

    assert state.code.startswith("from manim import *")
    assert state.safe_fallback_used is True
    assert "最小安全代码降级" in state.safe_fallback_reason


def test_direct_render_skips_generation_barrier(monkeypatch, tmp_path):
    run_paths = paths(tmp_path)
    code = "from manim import *\nclass Demo(Scene):\n    def construct(self): self.wait()\n"
    state = SceneState(
        plan=plan(),
        code=code,
        class_name="Demo",
        plan_ready=True,
        reviewed=True,
    )
    ctx = PipelineContext(
        "direct",
        paths=run_paths,
        direct_render=True,
        scenes=[state.plan],
        scene_states={1: state},
    )
    orchestrator = Orchestrator()
    monkeypatch.setattr(
        orchestrator,
        "_ensure_technical_spec",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("direct render called LLM")),
    )

    orchestrator._run_code_review_barrier(ctx)


def test_checkpoint_and_events_are_private_and_redacted(monkeypatch, tmp_path):
    run_paths = paths(tmp_path)
    run_paths.scenes.mkdir(parents=True)
    code = "from manim import *\nclass Demo(Scene):\n    def construct(self): self.wait()\n"
    (run_paths.scenes / "scene_1.py").write_text(code, encoding="utf-8")
    ctx = PipelineContext(
        "prompt",
        paths=run_paths,
        scene_states={1: SceneState(plan=plan(), code=code, class_name="Demo")},
    )
    orchestrator = Orchestrator()
    orchestrator._ctx = ctx
    monkeypatch.setattr(settings, "LLM_API_KEY", "top-secret-api-key")

    orchestrator._checkpoint(ctx, State.CODING)
    orchestrator._emit("diagnostic", error="request failed with top-secret-api-key")

    event_path = run_paths.root / "events.jsonl"
    assert event_path.stat().st_mode & 0o777 == 0o600
    records = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines()]
    assert records[0]["event"] == "fsm_checkpoint"
    assert records[-1]["data"]["error"] == "request failed with <redacted>"
    assert "top-secret-api-key" not in event_path.read_text(encoding="utf-8")


def test_code_barrier_does_not_turn_downstream_missing_ledger_into_failure(monkeypatch, tmp_path):
    """上游技术合同失败时，下游应等待而不是伪造继承状态错误。"""

    run_paths = paths(tmp_path)
    run_paths.root.mkdir(parents=True)
    inherited = VisualElementState(
        element_id="previous_result",
        variable_name="previous_result",
        required=True,
    )
    first = plan().model_copy(
        update={
            "new_elements": [inherited],
            "handoff": [
                SceneHandoff(
                    element_id="previous_result",
                    variable_name="previous_result",
                    action="keep",
                )
            ],
        }
    )
    second = plan().model_copy(
        update={
            "scene_id": 2,
            "inherited_elements": [inherited],
            "new_elements": [],
        }
    )
    ctx = PipelineContext(
        "prompt",
        paths=run_paths,
        plan_review_status="passed",
        scene_states={
            1: SceneState(plan=first, plan_ready=True),
            2: SceneState(plan=second, plan_ready=True),
        },
    )
    orchestrator = Orchestrator()
    orchestrator._llm_sem = threading.Semaphore(1)
    events = []
    orchestrator._callback = lambda event, data: events.append((event, data))
    monkeypatch.setattr(orchestrator, "_checkpoint", lambda *args, **kwargs: None)

    def fail_first(current_ctx, scene_state):
        if scene_state.plan.scene_id == 1:
            raise RuntimeError("TechnicalSpec invalid")

    monkeypatch.setattr(orchestrator, "_ensure_technical_spec", fail_first)

    orchestrator._run_code_review_barrier(ctx)

    assert ctx.scene_states[1].failed is True
    assert ctx.scene_states[2].failed is False
    assert ctx.scene_states[2].give_up is False
    assert any(
        event == "scene_waiting_for_dependency" and data["scene_id"] == 2 for event, data in events
    )


def test_plan_review_replan_budget_stops_an_identical_plan_loop(monkeypatch, tmp_path):
    """重规划会重置单份计划的审查轮数，但不能重置总调用预算。"""

    run_paths = paths(tmp_path)
    run_paths.root.mkdir(parents=True)
    current_plan = plan()
    ctx = PipelineContext(
        "prompt",
        paths=run_paths,
        outlines=[
            SceneOutline(
                scene_id=1,
                title=current_plan.title,
                duration_seconds=current_plan.duration_seconds,
                purpose=current_plan.purpose,
                math_concept=current_plan.math_concept,
            )
        ],
        scene_states={1: SceneState(plan=current_plan, plan_ready=True)},
        plan_review_status="pending",
        continuity_bible=ContinuityBible(),
        continuity_review_round=1,
    )
    orchestrator = Orchestrator()
    orchestrator._llm_sem = threading.Semaphore(1)
    monkeypatch.setattr(orchestrator, "_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(orchestrator, "_run_plan_review_batch", lambda *args, **kwargs: {})
    monkeypatch.setattr(settings, "MAX_PLAN_REVIEW_ROUNDS", 2)
    monkeypatch.setattr(settings, "MAX_PLAN_REPLAN_ATTEMPTS", 2)

    class FakeReviewer:
        def review(self, *args, **kwargs):
            return PlanReviewResult(
                is_valid=False,
                severity="major",
                issues=[
                    PlanReviewIssue(
                        category="feasibility",
                        field="visual_flow",
                        message="方案不可实现",
                        fix_instruction="重新规划",
                    )
                ],
            )

    class FakePlanner:
        calls = 0

        def plan_detail(self, *args, **kwargs):
            self.calls += 1
            return current_plan

    fake_planner = FakePlanner()
    monkeypatch.setattr(module, "PlanReviewerAgent", FakeReviewer)
    monkeypatch.setattr(module, "PlannerAgent", lambda: fake_planner)

    orchestrator._run_plan_review_barrier(ctx)

    assert fake_planner.calls == 2
    assert ctx.scene_states[1].failed is True
    assert ctx.plan_review_status == "failed"
    assert "达到最大次数" in ctx.scene_states[1].failure_reason
    assert ctx.continuity_review_round == 1


def test_plan_review_replan_budget_uses_geometry_fallback(monkeypatch, tmp_path):
    """复杂几何重规划耗尽时应降级，而不是把场景直接判死。"""

    run_paths = paths(tmp_path)
    run_paths.root.mkdir(parents=True)
    geometry_plan = plan().model_copy(
        update={
            "visual_flow": ["切割碎片并无缝拼接到目标区域"],
            "new_elements": [
                VisualElementState(
                    element_id="piece_a",
                    variable_name="piece_a",
                    color_key="primary",
                )
            ],
        }
    )
    ctx = PipelineContext(
        "prompt",
        paths=run_paths,
        outlines=[
            SceneOutline(
                scene_id=1,
                title=geometry_plan.title,
                duration_seconds=geometry_plan.duration_seconds,
                purpose=geometry_plan.purpose,
                math_concept=geometry_plan.math_concept,
            )
        ],
        scene_states={1: SceneState(plan=geometry_plan, plan_ready=True)},
        plan_review_status="pending",
    )
    orchestrator = Orchestrator()
    orchestrator._llm_sem = threading.Semaphore(1)
    monkeypatch.setattr(orchestrator, "_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(orchestrator, "_run_plan_review_batch", lambda *args, **kwargs: {})
    monkeypatch.setattr(settings, "MAX_PLAN_REVIEW_ROUNDS", 2)
    monkeypatch.setattr(settings, "MAX_PLAN_REPLAN_ATTEMPTS", 2)
    monkeypatch.setattr(settings, "SAFE_FALLBACK_ENABLED", True)

    class FakeReviewer:
        def review(self, *args, **kwargs):
            return PlanReviewResult(is_valid=True, severity="info")

    class FakePlanner:
        calls = 0

        def plan_detail(self, *args, **kwargs):
            self.calls += 1
            return geometry_plan

    fake_planner = FakePlanner()
    monkeypatch.setattr(module, "PlanReviewerAgent", FakeReviewer)
    monkeypatch.setattr(module, "PlannerAgent", lambda: fake_planner)

    orchestrator._run_plan_review_barrier(ctx)

    state = ctx.scene_states[1]
    assert fake_planner.calls == 2
    assert state.safe_fallback_used is True
    assert state.failed is False
    assert state.give_up is False
    assert state.plan_reviewed is False
    assert ctx.continuity_rebuild_required is True


def test_resume_migrates_claims_out_of_transition_scene(monkeypatch, tmp_path):
    outlines = [
        SceneOutline(
            scene_id=1,
            title="建立",
            duration_seconds=10,
            purpose="建立基础",
            math_concept="基础",
            claim_ids=["claim_1"],
        ),
        SceneOutline(
            scene_id=2,
            title="章节过渡",
            duration_seconds=5,
            purpose="分隔并提示观众切换",
            math_concept="无",
            claim_ids=["claim_2"],
        ),
        SceneOutline(
            scene_id=3,
            title="推导",
            duration_seconds=10,
            purpose="展示结论",
            math_concept="结论",
            claim_ids=["claim_2"],
        ),
    ]
    states = {
        index: SceneState(
            plan=plan().model_copy(
                update={"scene_id": index, "claim_ids": list(outline.claim_ids)}
            ),
            plan_ready=True,
        )
        for index, outline in enumerate(outlines, 1)
    }
    ctx = PipelineContext(
        "prompt",
        paths=paths(tmp_path),
        outlines=outlines,
        teaching_graph=TeachingGraph(scene_claims={1: ["claim_1"], 2: ["claim_2"], 3: ["claim_2"]}),
        scene_states=states,
    )
    orchestrator = Orchestrator()
    monkeypatch.setattr(orchestrator, "_checkpoint", lambda *args, **kwargs: None)

    assert orchestrator._normalize_transition_claim_contracts(ctx) is True
    assert ctx.scene_states[2].plan.claim_ids == []
    assert ctx.scene_states[3].plan.claim_ids == ["claim_2"]
    assert ctx.teaching_graph.scene_claims == {1: ["claim_1"], 2: [], 3: ["claim_2"]}


def test_plan_compile_drops_new_handoff_without_next_scene_consumer(tmp_path):
    transition = VisualElementState(
        element_id="transition_title",
        variable_name="transition_title",
        required=True,
    )
    first_plan = plan().model_copy(
        update={
            "new_elements": [transition],
            "handoff": [
                SceneHandoff(
                    element_id="transition_title",
                    variable_name="transition_title",
                    action="keep",
                )
            ],
        }
    )
    second_plan = plan().model_copy(update={"scene_id": 2})
    ctx = PipelineContext(
        "prompt",
        paths=paths(tmp_path),
        scene_states={
            1: SceneState(plan=first_plan, plan_ready=True),
            2: SceneState(plan=second_plan, plan_ready=True),
        },
    )

    assert Orchestrator()._normalize_dangling_handoffs(ctx) is True
    assert ctx.scene_states[1].plan.handoff == []
    assert ctx.scene_states[1].plan.new_elements[0].required is False


def test_direct_render_flag_round_trips_in_manifest(tmp_path):
    run_paths = paths(tmp_path)
    code = "from manim import *\nclass Demo(Scene):\n    def construct(self): self.wait()\n"
    run_paths.scenes.mkdir(parents=True)
    (run_paths.scenes / "scene_1.py").write_text(code, encoding="utf-8")
    state = SceneState(
        plan=plan(),
        code=code,
        class_name="Demo",
        plan_ready=True,
        reviewed=True,
    )
    ctx = PipelineContext(
        "direct",
        paths=run_paths,
        direct_render=True,
        scenes=[state.plan],
        scene_states={1: state},
    )

    orchestrator = Orchestrator()
    orchestrator._checkpoint(ctx, State.DISPATCHING)

    manifest = RunManifest.model_validate_json(
        (run_paths.root / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest.direct_render is True
    restored = orchestrator._context_from_manifest(manifest, run_paths.root)
    assert restored.direct_render is True


def test_direct_render_wait_does_not_call_any_generation_agent(monkeypatch, tmp_path):
    from kd1_anime.cluster.slurm import SlurmJob
    from kd1_anime.config import settings
    from kd1_anime.rendering import VideoMetadata

    code = "from manim import *\nclass Demo(Scene):\n    def construct(self): self.wait()\n"

    class FakeSlurm:
        def submit_scene(self, scene_id, python_file, scene_class_name, **kwargs):
            run_root = python_file.parent.parent
            return SlurmJob(
                job_id="123",
                scene_id=scene_id,
                script_path=python_file.parent / "render.sh",
                log_out=run_root / "logs" / "scene.out",
                log_err=run_root / "logs" / "scene.err",
                media_dir=run_root / "videos" / f"scene_{scene_id}",
                scene_class_name=scene_class_name,
                submitted_at=1.0,
                code_sha256=kwargs["code_sha256"],
                render_profile=kwargs["render_profile"],
            )

        def poll_all_statuses(self, job_ids):
            return {job_id: "COMPLETED" for job_id in job_ids}

        def validate_completed_job(self, job):
            job.output_path = job.media_dir / f"{job.scene_class_name}.mp4"
            job.output_path.parent.mkdir(parents=True, exist_ok=True)
            job.output_path.write_bytes(b"video")
            job.output_metadata = VideoMetadata(
                size_bytes=5,
                duration_seconds=1,
                width=job.render_profile.pixel_width,
                height=job.render_profile.pixel_height,
                frame_rate=job.render_profile.frame_rate,
            )
            job.output_sha256 = sha256_file(job.output_path)
            return True

        def _forward_log(self, job, positions):
            return None

        def cancel_job(self, job_id):
            return True

    class FakeMerger:
        def merge(self, video_paths, output_path, **kwargs):
            output_path = Path(output_path)
            output_path.write_bytes(b"merged")
            return output_path

    monkeypatch.setattr(settings, "WORKSPACE_DIR", tmp_path / "workspace")
    monkeypatch.setattr(settings, "ENABLE_AUTO_EVAL", False)
    monkeypatch.setattr(
        Orchestrator,
        "_preflight_environment",
        staticmethod(lambda profile=None: None),
    )
    orchestrator = Orchestrator()
    orchestrator.slurm = FakeSlurm()
    orchestrator.merger = FakeMerger()
    monkeypatch.setattr(
        orchestrator,
        "_ensure_technical_spec",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("direct render must not call TechnicalPlanner")
        ),
    )

    _, final_video, _ = orchestrator.submit_existing_scene(code, "Demo", wait=True)

    assert final_video is not None
    assert final_video.read_bytes() == b"merged"


def test_resume_requeues_rendered_scene_when_video_was_deleted(monkeypatch, tmp_path):
    run_paths = paths(tmp_path)
    run_paths.root.mkdir(parents=True)
    profile = PipelineContext("prompt", paths=run_paths).render_profile
    code = "from manim import *\nclass Demo(Scene):\n    def construct(self): self.wait()\n"
    code_path = run_paths.scenes / "scene_1.py"
    code_path.parent.mkdir(parents=True)
    code_path.write_text(code, encoding="utf-8")
    artifact = SceneArtifact(
        origin="rendered",
        source_run_id=run_paths.run_id,
        job_id="123",
        scene_id=1,
        scene_class_name="Demo",
        code_sha256=sha256_text(code),
        render_profile_sha256=profile.digest(),
        video_path="videos/scene_1/Demo.mp4",
        video_sha256="a" * 64,
        metadata=VideoMetadata(
            size_bytes=1,
            duration_seconds=1,
            width=profile.pixel_width,
            height=profile.pixel_height,
            frame_rate=profile.frame_rate,
        ),
    )
    state = SceneState(
        plan=plan(),
        code=code,
        class_name="Demo",
        reviewed=True,
        rendered=True,
        artifact=artifact,
    )
    ctx = PipelineContext("prompt", paths=run_paths, scene_states={1: state})
    orchestrator = Orchestrator()
    monkeypatch.setattr(orchestrator, "_checkpoint", lambda *args, **kwargs: None)

    orchestrator._reconcile_rendered_artifacts(ctx)

    assert state.rendered is False
    assert state.artifact is None
    assert ctx.final_video is None
    assert "重新处理" in state.failure_reason


def test_local_smoke_status_is_persisted_and_failed_smoke_is_not_passed(monkeypatch, tmp_path):
    from kd1_anime.config import settings

    run_paths = paths(tmp_path)
    ctx = PipelineContext("prompt", paths=run_paths)
    state = SceneState(plan=plan())
    monkeypatch.setattr(settings, "LOCAL_SMOKE_RENDER_ENABLED", True)
    orchestrator = Orchestrator()
    monkeypatch.setattr(orchestrator, "_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(orchestrator, "_local_smoke_render_impl", lambda *args: None)

    orchestrator._local_smoke_render(ctx, state)

    assert state.local_smoke_status == "passed"

    def fail(*args):
        raise RuntimeError("smoke failed")

    monkeypatch.setattr(orchestrator, "_local_smoke_render_impl", fail)
    with pytest.raises(RuntimeError, match="smoke failed"):
        orchestrator._local_smoke_render(ctx, state)
    assert state.local_smoke_status == "failed"


def test_failed_scene_phase_does_not_remain_visual_evaluating():
    state = SceneState(plan=plan(), failed=True, visual_status="evaluating")

    assert Orchestrator._scene_phase(state) == "failed"


def test_scene_worker_clears_rendered_state_when_post_render_step_fails(monkeypatch, tmp_path):
    from kd1_anime.cluster.slurm import SlurmJob

    run_paths = paths(tmp_path)
    state = SceneState(
        plan=plan(),
        code="from manim import *\nclass Demo(Scene):\n    def construct(self): self.wait()\n",
        class_name="Demo",
        reviewed=True,
        slurm_job=SlurmJob(
            job_id="123",
            scene_id=1,
            script_path=run_paths.scenes / "render.sh",
            log_out=run_paths.logs / "out",
            log_err=run_paths.logs / "err",
            media_dir=run_paths.videos / "scene_1",
            scene_class_name="Demo",
            submitted_at=1.0,
        ),
    )
    ctx = PipelineContext("prompt", paths=run_paths, scene_states={1: state})
    orchestrator = Orchestrator()
    orchestrator._slot_lock = threading.Lock()
    orchestrator._in_flight = 1
    orchestrator._reserved_existing_scenes = {1}
    monkeypatch.setattr(orchestrator, "_checkpoint", lambda *args, **kwargs: None)

    def fail_after_marking_rendered(_ctx, current_state):
        current_state.rendered = True
        raise RuntimeError("post-render failure")

    monkeypatch.setattr(orchestrator, "_scene_wait_render", fail_after_marking_rendered)

    orchestrator._scene_worker(ctx, 1, state)

    assert state.failed is True
    assert state.rendered is False
    assert state.artifact is None


def test_disabled_rag_retrieval_is_recorded_without_network(tmp_path):
    orchestrator = Orchestrator()
    ctx = PipelineContext("prompt", paths=paths(tmp_path))

    context = orchestrator._retrieve_rag(
        ctx,
        "Manim Circle API",
        receipt_key="scene:1:code",
        stage="code",
    )

    assert context == ""
    assert ctx.rag_receipts["scene:1:code"].status == "disabled"


def test_visual_math_issue_routes_back_to_planner():
    result = VisualAnalysisResult(
        overall_analysis="公式不正确",
        mathematical_accuracy={"score": 2, "comprehensive_evaluation": "公式错误"},
        visual_relevance={"score": 4, "comprehensive_evaluation": "相关"},
        visual_quality={"score": 4, "comprehensive_evaluation": "清晰"},
        visual_consistency={"score": 4, "comprehensive_evaluation": "一致"},
        element_layout={"score": 4, "comprehensive_evaluation": "整齐"},
        issues=[
            VisualIssue(
                category="mathematics",
                severity="major",
                frame_ids=["F01"],
                evidence="画面等式前后不等价",
                recommendation="回到计划修正数学断言",
            )
        ],
    )

    assert Orchestrator._visual_repair_target(result) == "planner"


def test_visual_boundary_issue_uses_boundary_start_scene_for_repair():
    result = VisualAnalysisResult(
        overall_analysis="边界突变",
        mathematical_accuracy={"score": 4, "comprehensive_evaluation": "正确"},
        visual_relevance={"score": 4, "comprehensive_evaluation": "相关"},
        visual_quality={"score": 4, "comprehensive_evaluation": "清晰"},
        visual_consistency={"score": 2, "comprehensive_evaluation": "突变"},
        element_layout={"score": 4, "comprehensive_evaluation": "整齐"},
        issues=[
            VisualIssue(
                category="consistency",
                severity="major",
                repair_target="continuity",
                frame_ids=["F01", "F02"],
                boundary_ids=["B02"],
                evidence="Scene 2 开头丢失对象",
                recommendation="恢复边界交接",
            )
        ],
    )

    assert Orchestrator._visual_repair_target(result) == "continuity"


def test_visual_plan_feedback_survives_plan_compile(monkeypatch, tmp_path):
    run_paths = paths(tmp_path)
    run_paths.root.mkdir(parents=True)
    current_plan = plan()
    state = SceneState(plan=current_plan, plan_ready=True)
    ctx = PipelineContext(
        "prompt",
        paths=run_paths,
        outlines=[
            SceneOutline(
                scene_id=1,
                title="demo",
                duration_seconds=10,
                purpose="test",
                math_concept="circle",
            )
        ],
        scenes=[current_plan],
        scene_states={1: state},
    )
    orchestrator = Orchestrator()
    monkeypatch.setattr(orchestrator, "_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(orchestrator, "_request_continuity_rebuild", lambda *args, **kwargs: None)

    orchestrator._schedule_visual_plan_repair(
        ctx,
        1,
        state,
        "画面中的数学关系不成立",
        "planner",
    )
    orchestrator._compile_scene_plans(ctx)

    assert any(issue.field == "visual_evaluation" for issue in ctx.plan_compile_issues[1])


def test_plan_only_runs_new_teaching_contract_path(monkeypatch, tmp_path):
    outline = SceneOutline(
        scene_id=1,
        title="公式",
        duration_seconds=10,
        purpose="展示公式",
        math_concept="恒等式",
        claim_ids=["claim_1"],
    )
    plan = ScenePlan(
        **outline.model_dump(),
        visual_design="固定画面",
        camera_movement="固定",
        visual_flow=["展示公式"],
        key_moments=["结论定格"],
        computation="a=a",
        opening_state=["画面为空"],
        closing_state=["公式保留"],
        transition_in="从空画面建立公式",
        transition_out="保留公式并收束",
        timeline=[
            {
                "event_id": "show_formula",
                "start_seconds": 0,
                "end_seconds": 10,
                "action": "展示公式",
                "math_claim_ids": ["claim_1"],
            }
        ],
        math_claims=[
            MathClaim(
                claim_id="claim_1",
                statement="a=a",
                expression_before="a",
                expression_after="a",
                relation="equivalent",
            )
        ],
    )

    class FakePlanner:
        def plan_draft(self, prompt, **kwargs):
            return PlanningDraft(
                lesson_spec=LessonSpec(
                    topic="公式",
                    claims=[MathClaim(claim_id="claim_1", statement="a=a", relation="definition")],
                ),
                teaching_graph=TeachingGraph(
                    claim_order=["claim_1"],
                    scene_claims={1: ["claim_1"]},
                ),
                items=[outline],
            )

        def plan_continuity_bible(self, prompt, outlines, **kwargs):
            return ContinuityBible()

        def plan_detail(self, *args, **kwargs):
            return plan

    class FakePlanReviewer:
        def review(self, *args, **kwargs):
            from kd1_anime.agents.plan_reviewer import PlanReviewResult

            return PlanReviewResult(is_valid=True, severity="info", summary="通过")

    class FakeContinuityReviewer:
        def review(self, *args, **kwargs):
            from kd1_anime.agents.continuity import ContinuityReviewResult

            return ContinuityReviewResult(is_valid=True, summary="通过")

    monkeypatch.setattr(settings, "WORKSPACE_DIR", tmp_path)
    monkeypatch.setattr(module, "PlannerAgent", FakePlanner)
    monkeypatch.setattr(module, "PlanReviewerAgent", FakePlanReviewer)
    monkeypatch.setattr(module, "ContinuityReviewerAgent", FakeContinuityReviewer)

    scenes = Orchestrator().plan_only("解释公式", preflight=False)

    assert [scene.scene_id for scene in scenes] == [1]
    assert scenes[0].claim_ids == ["claim_1"]


def test_continuity_replan_snapshot_stays_within_prompt_section_budget(tmp_path):
    run_paths = paths(tmp_path)
    long_text = "连续性状态 " + "x" * 19_000
    states = {}
    for scene_id in (1, 2, 3):
        scene_plan = plan().model_copy(
            update={
                "scene_id": scene_id,
                "visual_design": long_text,
                "computation": long_text,
                "visual_flow": [long_text[:2_000]] * 10,
            }
        )
        states[scene_id] = SceneState(plan=scene_plan, plan_ready=True)
    ctx = PipelineContext("prompt", paths=run_paths, scene_states=states)

    snapshot = Orchestrator._continuity_plan_context(ctx, 2)

    assert len(snapshot) <= 22_000


def test_resume_reopens_plan_review_when_new_compiler_finds_old_plan_error(tmp_path):
    run_paths = paths(tmp_path)
    bad_plan = plan().model_copy(
        update={
            "new_elements": [
                VisualElementState(element_id="formula", variable_name="formula", required=True)
            ],
            "handoff": [
                SceneHandoff(element_id="formula", variable_name="formula", action="remove")
            ],
        }
    )
    ctx = PipelineContext(
        "prompt",
        paths=run_paths,
        outlines=[
            SceneOutline(
                scene_id=1,
                title="公式",
                duration_seconds=10,
                purpose="展示",
                math_concept="公式",
            )
        ],
        scene_states={
            1: SceneState(
                plan=bad_plan,
                plan_ready=True,
                plan_reviewed=True,
            )
        },
        plan_review_status="passed",
    )
    Orchestrator()._compile_scene_plans(ctx)

    assert ctx.plan_review_status == "pending"
    assert ctx.scene_states[1].plan_reviewed is False


def test_scheduler_stops_when_checkpoint_persistence_fails(monkeypatch, tmp_path):
    run_paths = paths(tmp_path)
    ctx = PipelineContext(
        "x",
        paths=run_paths,
        dry_run=True,
        scene_states={
            1: SceneState(
                plan=plan(),
                code="from manim import *\nclass Demo(Scene):\n    def construct(self): pass\n",
                class_name="Demo",
                plan_ready=True,
                reviewed=True,
            )
        },
    )
    monkeypatch.setattr(
        module,
        "write_manifest",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )

    with pytest.raises(RuntimeError, match="持久化失败"):
        Orchestrator()._run_scheduler(ctx)


def test_external_cancel_is_checkpointed_as_interrupted(monkeypatch, tmp_path):
    orch = Orchestrator()
    orch._cancel_requested.set()
    ctx = PipelineContext(
        "x",
        paths=paths(tmp_path),
        dry_run=True,
        scene_states={1: SceneState(plan=plan(), plan_ready=True)},
    )
    captured = {}
    monkeypatch.setattr(orch, "_run_scheduler", lambda current: None)
    monkeypatch.setattr(
        module,
        "write_manifest",
        lambda path, manifest: captured.setdefault("manifest", manifest),
    )

    with pytest.raises(KeyboardInterrupt):
        orch._execute(ctx, State.INIT)

    assert captured["manifest"].status == "interrupted"


def test_resume_snapshot_emits_state_for_existing_scenes(monkeypatch, tmp_path):
    """resume 后快照应为已渲染/进行中的场景补发事件, 避免仪表盘显示"未开始"。"""
    events = []
    orch = Orchestrator()
    orch._callback = lambda event, data: events.append((event, data))
    ctx = PipelineContext("x", paths=paths(tmp_path))

    def make_state(sid, *, rendered=False, reviewed=False, coded=False):
        p = plan()
        p.scene_id = sid
        p.title = f"场景{sid}"
        return SceneState(
            plan=p,
            code=(
                "from manim import *\nclass Demo(Scene):\n    def construct(self): pass\n"
                if coded
                else ""
            ),
            class_name="Demo",
            reviewed=reviewed,
            rendered=rendered,
            plan_ready=True,
        )

    ctx.scene_states = {
        1: make_state(1, rendered=True, reviewed=True, coded=True),  # 已完成
        2: make_state(2, reviewed=True, coded=True),  # 已审查、有代码
        3: make_state(3),  # 仅分镜完成
    }

    orch._emit_scene_snapshot(ctx)

    by_scene: dict[int, list[str]] = {}
    for event, data in events:
        by_scene.setdefault(data["scene_id"], []).append(event)

    # 已渲染 → 补完整流水线 + scene_rendered (仪表盘标绿)
    assert by_scene[1] == ["scene_detailed", "scene_coded", "scene_review_pass", "scene_rendered"]
    # 已编码+已审查 → 按顺序补 detailed → coded → review_pass
    assert by_scene[2] == ["scene_detailed", "scene_coded", "scene_review_pass"]
    # 仅分镜完成 → 只补 scene_detailed
    assert by_scene[3] == ["scene_detailed"]


def test_resume_error_retries_all_give_up_scenes(monkeypatch, tmp_path):
    """ERROR 清单即使所有场景都已放弃，也应能被 resume 重置并重试。"""
    from kd1_anime.config import settings

    run_id = "20260728-120000-1234abcd"
    workspace = tmp_path / "workspace"
    root = workspace / "runs" / run_id
    root.mkdir(parents=True)
    manifest = RunManifest(
        run_id=run_id,
        status="failed",
        state="ERROR",
        user_prompt="prompt",
        output_path=str((root / "output.mp4").resolve()),
        scenes={
            1: StoredSceneState(
                plan=plan(),
                class_name="Demo",
                give_up=True,
                failed=True,
                failure_reason="render failed",
            )
        },
    )
    write_manifest(root / "manifest.json", manifest)
    monkeypatch.setattr(settings, "WORKSPACE_DIR", workspace)
    captured: dict[str, State] = {}

    def fake_execute(self, context, state):
        captured["state"] = state
        assert context.scene_states[1].give_up is False
        assert context.scene_states[1].failed is False
        return None

    monkeypatch.setattr(Orchestrator, "_execute", fake_execute)

    assert Orchestrator().resume(run_id) is None
    assert captured["state"] is State.CODING


def test_resume_retries_incomplete_dry_run_instead_of_returning(monkeypatch, tmp_path):
    """旧版本把含失败场景的 dry-run 标为完成，resume 仍应重新进入编码屏障。"""
    from kd1_anime.config import settings

    run_id = "20260728-120000-1234abcd"
    workspace = tmp_path / "workspace"
    root = workspace / "runs" / run_id
    root.mkdir(parents=True)
    manifest = RunManifest(
        run_id=run_id,
        status="dry_run_complete",
        state="DONE",
        dry_run=True,
        user_prompt="prompt",
        output_path=str((root / "output.mp4").resolve()),
        scenes={
            1: StoredSceneState(
                plan=plan(),
                plan_ready=True,
                give_up=True,
                failure_reason="审查未通过",
            )
        },
    )
    write_manifest(root / "manifest.json", manifest)
    monkeypatch.setattr(settings, "WORKSPACE_DIR", workspace)
    captured = {}

    def fake_execute(self, context, state):
        captured["context"] = context
        captured["state"] = state
        return None

    monkeypatch.setattr(Orchestrator, "_execute", fake_execute)

    assert Orchestrator().resume(run_id) is None
    assert captured["state"] is State.CODING
    assert captured["context"].scene_states[1].give_up is False


def test_resume_resets_failed_scene_from_monitoring_snapshot(monkeypatch, tmp_path):
    """最后检查点停在 MONITORING 时，resume 也不能跳过 failed 场景。"""
    from kd1_anime.config import settings

    run_id = "20260728-120000-1234abcd"
    workspace = tmp_path / "workspace"
    root = workspace / "runs" / run_id
    (root / "scenes").mkdir(parents=True)
    code = "from manim import *\nclass Demo(Scene):\n    def construct(self): pass\n"
    code_path = root / "scenes" / "scene_1.py"
    code_path.write_text(code, encoding="utf-8")
    manifest = RunManifest(
        run_id=run_id,
        status="failed",
        state="MONITORING",
        user_prompt="prompt",
        output_path=str((root / "output.mp4").resolve()),
        scenes={
            1: StoredSceneState(
                plan=plan(),
                code_file="scenes/scene_1.py",
                code_sha256=sha256_text(code),
                class_name="Demo",
                plan_ready=True,
                reviewed=True,
                failed=True,
                failure_reason="previous monitor failure",
            )
        },
    )
    write_manifest(root / "manifest.json", manifest)
    monkeypatch.setattr(settings, "WORKSPACE_DIR", workspace)
    captured = {}

    def fake_execute(self, context, state):
        captured["context"] = context
        captured["state"] = state
        return None

    monkeypatch.setattr(Orchestrator, "_execute", fake_execute)

    assert Orchestrator().resume(run_id) is None
    assert captured["state"] is State.MONITORING
    assert captured["context"].scene_states[1].failed is False
    assert captured["context"].scene_states[1].failure_reason == ""


def test_resume_rechecks_continuity_warning_once(monkeypatch, tmp_path):
    """连续性 warning 的恢复只重新开启一次修正预算。"""
    from kd1_anime.config import settings

    run_id = "20260728-120000-1234abcd"
    workspace = tmp_path / "workspace"
    root = workspace / "runs" / run_id
    root.mkdir(parents=True)
    manifest = RunManifest(
        run_id=run_id,
        status="running",
        state="CODING",
        user_prompt="prompt",
        output_path=str((root / "output.mp4").resolve()),
        continuity_bible=ContinuityBible(),
        continuity_review_status="warning",
        continuity_review_round=3,
        continuity_warnings=["达到最大连续性修正轮数"],
        scenes={1: StoredSceneState(plan=plan(), plan_ready=True)},
    )
    write_manifest(root / "manifest.json", manifest)
    monkeypatch.setattr(settings, "WORKSPACE_DIR", workspace)
    captured = {}

    def fake_execute(self, context, state):
        captured["context"] = context
        return None

    monkeypatch.setattr(Orchestrator, "_execute", fake_execute)

    assert Orchestrator().resume(run_id) is None
    assert captured["context"].continuity_review_status == "pending"
    assert captured["context"].continuity_review_round == 0
    assert captured["context"].continuity_resume_recheck_used is True


def test_resume_does_not_recheck_continuity_warning_twice(monkeypatch, tmp_path):
    """已经用过恢复重检查机会时，resume 应直接沿用 warning 计划。"""
    run_id = "20260728-120000-1234abcd"
    workspace = tmp_path / "workspace"
    root = workspace / "runs" / run_id
    root.mkdir(parents=True)
    manifest = RunManifest(
        run_id=run_id,
        status="running",
        state="CODING",
        user_prompt="prompt",
        output_path=str((root / "output.mp4").resolve()),
        continuity_bible=ContinuityBible(),
        continuity_review_status="warning",
        continuity_review_round=3,
        continuity_resume_recheck_used=True,
        continuity_warnings=["达到最大连续性修正轮数"],
        scenes={1: StoredSceneState(plan=plan(), plan_ready=True)},
    )
    write_manifest(root / "manifest.json", manifest)
    monkeypatch.setattr(settings, "WORKSPACE_DIR", workspace)
    captured = {}

    def fake_execute(self, context, state):
        captured["context"] = context
        return None

    monkeypatch.setattr(Orchestrator, "_execute", fake_execute)

    assert Orchestrator().resume(run_id) is None
    assert captured["context"].continuity_review_status == "warning"
    assert captured["context"].continuity_review_round == 3
    assert captured["context"].continuity_resume_recheck_used is True


def test_context_derives_pending_plan_review_for_legacy_incomplete_run(tmp_path):
    run_paths = paths(tmp_path)
    run_paths.root.mkdir(parents=True)
    manifest = RunManifest(
        run_id=run_paths.run_id,
        status="failed",
        state="REVIEWING",
        user_prompt="prompt",
        output_path=str(run_paths.output.resolve()),
        scenes={
            1: StoredSceneState(
                plan=plan(),
                plan_ready=True,
                reviewed=False,
            )
        },
    )
    raw = manifest.model_dump(mode="json")
    raw.pop("plan_review_status", None)
    (run_paths.root / "manifest.json").write_text(
        json.dumps(raw, ensure_ascii=False), encoding="utf-8"
    )

    loaded = RunManifest.model_validate_json((run_paths.root / "manifest.json").read_text())
    context = Orchestrator._context_from_manifest(loaded, run_paths.root)

    assert context.plan_review_status == "pending"


def test_resume_does_not_block_code_recovery_on_stale_plan_failure(monkeypatch, tmp_path):
    from kd1_anime.config import settings

    run_id = "20260728-120000-1234abcd"
    workspace = tmp_path / "workspace"
    root = workspace / "runs" / run_id
    root.mkdir(parents=True)
    manifest = RunManifest(
        run_id=run_id,
        status="failed",
        state="REVIEWING",
        user_prompt="prompt",
        output_path=str(root / "output.mp4"),
        plan_review_status="failed",
        scenes={
            1: StoredSceneState(
                plan=plan(),
                plan_ready=True,
                plan_reviewed=True,
                failed=True,
                failure_reason="代码审查失败",
            )
        },
    )
    write_manifest(root / "manifest.json", manifest)
    monkeypatch.setattr(settings, "WORKSPACE_DIR", workspace)
    captured = {}

    def fake_execute(self, context, state):
        captured["context"] = context
        return None

    monkeypatch.setattr(Orchestrator, "_execute", fake_execute)

    assert Orchestrator().resume(run_id) is None
    assert captured["context"].plan_review_status == "passed"


def test_run_paths_are_unique(monkeypatch, tmp_path):
    from kd1_anime.config import settings

    monkeypatch.setattr(settings, "WORKSPACE_DIR", tmp_path)
    first = RunPaths.create()
    second = RunPaths.create()
    assert first.root != second.root


def test_local_smoke_render_is_skipped_for_dry_run(monkeypatch, tmp_path):
    run_paths = paths(tmp_path)
    source = run_paths.scenes / "scene_1.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("from manim import *\nclass Demo(Scene):\n    def construct(self): pass\n")
    ctx = PipelineContext("x", paths=run_paths, dry_run=True)
    state = SceneState(plan=plan(), class_name="Demo")
    orchestrator = Orchestrator()
    called = False

    def unexpected_run(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(module.settings, "LOCAL_SMOKE_RENDER_ENABLED", True)
    monkeypatch.setattr(module.subprocess, "run", unexpected_run)

    orchestrator._local_smoke_render(ctx, state)

    assert called is False


def test_explicit_smoke_override_enables_dry_run_canary(tmp_path):
    ctx = PipelineContext(
        "x",
        paths=paths(tmp_path),
        dry_run=True,
        local_smoke_enabled=True,
    )

    assert Orchestrator._local_smoke_enabled(ctx) is True


def test_local_smoke_render_checks_output_and_failure(monkeypatch, tmp_path):
    run_paths = paths(tmp_path)
    source = run_paths.scenes / "scene_1.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("from manim import *\nclass Demo(Scene):\n    def construct(self): pass\n")
    ctx = PipelineContext("x", paths=run_paths, dry_run=False)
    state = SceneState(plan=plan(), class_name="Demo")
    orchestrator = Orchestrator()
    monkeypatch.setattr(module.settings, "LOCAL_SMOKE_RENDER_ENABLED", True)
    monkeypatch.setattr(module.settings, "LOCAL_SMOKE_RENDER_MODE", "video")

    def successful_run(command, **kwargs):
        media_index = command.index("--media_dir") + 1
        media_dir = Path(command[media_index])
        output = media_dir / "nested" / "Demo.mp4"
        output.parent.mkdir(parents=True)
        output.write_bytes(b"smoke")
        return module.subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(module, "_run_limited_process", successful_run)
    orchestrator._local_smoke_render(ctx, state)

    def failed_run(command, **kwargs):
        return module.subprocess.CompletedProcess(command, 1, "", "render boom")

    monkeypatch.setattr(module, "_run_limited_process", failed_run)
    with pytest.raises(RuntimeError, match="Smoke Render 失败"):
        orchestrator._local_smoke_render(ctx, state)


def test_local_frame_canary_checks_last_frame(monkeypatch, tmp_path):
    run_paths = paths(tmp_path)
    source = run_paths.scenes / "scene_1.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("from manim import *\nclass Demo(Scene):\n    def construct(self): pass\n")
    ctx = PipelineContext("x", paths=run_paths, dry_run=False)
    state = SceneState(plan=plan(), class_name="Demo")
    orchestrator = Orchestrator()
    monkeypatch.setattr(module.settings, "LOCAL_SMOKE_RENDER_ENABLED", True)
    monkeypatch.setattr(module.settings, "LOCAL_SMOKE_RENDER_MODE", "frame")
    captured = {}

    def successful_run(command, **kwargs):
        captured["command"] = command
        media_dir = Path(command[command.index("--media_dir") + 1])
        output = media_dir / "nested" / "Demo.png"
        output.parent.mkdir(parents=True)
        output.write_bytes(b"frame")
        return module.subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(module, "_run_limited_process", successful_run)
    orchestrator._local_smoke_render(ctx, state)

    assert "--format" in captured["command"]
    assert "png" in captured["command"]
    assert "--save_last_frame" in captured["command"]


def test_minor_review_applies_unique_fix_before_round_limit(monkeypatch, tmp_path):
    monkeypatch.setattr(module.settings, "MAX_REVIEW_ROUNDS", 1)
    run_paths = paths(tmp_path)
    run_paths.scenes.mkdir(parents=True)
    ctx = PipelineContext("x", paths=run_paths)
    ctx.scene_states[1] = SceneState(
        plan=plan(),
        code="from manim import *\nclass Demo(Scene):\n    def construct(self): pass\n",
        class_name="Demo",
        review_round=1,
    )

    Orchestrator()._apply_review_result(
        ctx,
        1,
        ctx.scene_states[1],
        ReviewResult(
            is_valid=False,
            severity="minor",
            fixes=[{"find": "pass", "replace": "self.wait(1)", "reason": "demo"}],
        ),
    )
    assert ctx.scene_states[1].give_up is False
    assert ctx.scene_states[1].reviewed is False
    assert "self.wait(1)" in ctx.scene_states[1].code
    assert ctx.scene_states[1].review_round == 0


def test_major_review_with_verified_local_fix_stays_in_code_review(monkeypatch, tmp_path):
    run_paths = paths(tmp_path)
    run_paths.scenes.mkdir(parents=True)
    state = SceneState(
        plan=plan(),
        code="from manim import *\nclass Demo(Scene):\n    def construct(self): pass\n",
        class_name="Demo",
        plan_ready=True,
    )
    ctx = PipelineContext("x", paths=run_paths, scene_states={1: state})
    monkeypatch.setattr(Orchestrator, "_checkpoint", lambda *args, **kwargs: None)

    result = ReviewResult(
        is_valid=False,
        severity="major",
        feedback="代码局部实现需要修正",
        fixes=[
            {
                "find": "pass",
                "replace": "self.wait(1)",
                "reason": "补充场景停留",
            }
        ],
    )

    Orchestrator()._apply_review_result(ctx, 1, state, result)

    assert "self.wait(1)" in state.code
    assert state.reviewed is False
    assert state.rewrite_feedback == ""
    assert state.give_up is False


def test_code_level_math_finding_is_sent_back_to_coder_not_planner(monkeypatch, tmp_path):
    run_paths = paths(tmp_path)
    run_paths.scenes.mkdir(parents=True)
    code = "from manim import *\nclass Demo(Scene):\n    def construct(self): self.wait()\n"
    state = SceneState(
        plan=plan(),
        code=code,
        class_name="Demo",
        plan_ready=True,
    )
    ctx = PipelineContext("x", paths=run_paths, scene_states={1: state})
    monkeypatch.setattr(Orchestrator, "_checkpoint", lambda *args, **kwargs: None)

    result = ReviewResult(
        is_valid=False,
        severity="major",
        feedback="代码中的线性变换 API 需要修正",
        findings=[
            ReviewFinding(
                category="math",
                severity="major",
                line_start=3,
                line_end=3,
                evidence="self.wait()",
                why="代码实现使用了错误的变换 API",
                repair="改用 apply_matrix，修复当前代码",
            )
        ],
    )

    Orchestrator()._apply_review_result(ctx, 1, state, result)

    assert state.rewrite_feedback
    assert not ctx.plan_compile_issues
    assert state.give_up is False


def test_review_exhaustion_switches_high_risk_geometry_to_safe_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(module.settings, "MAX_REVIEW_ROUNDS", 1)
    monkeypatch.setattr(module.settings, "SAFE_FALLBACK_ENABLED", True)
    run_paths = paths(tmp_path)
    run_paths.scenes.mkdir(parents=True)
    geometry_plan = plan().model_copy(
        update={
            "visual_flow": ["切割碎片并无缝拼接到目标区域"],
            "new_elements": [
                VisualElementState(
                    element_id="piece_a",
                    variable_name="piece_a",
                    color_key="primary",
                )
            ],
        }
    )
    state = SceneState(
        plan=geometry_plan,
        code="from manim import *\nclass Demo(Scene):\n    def construct(self): pass\n",
        class_name="Demo",
        plan_ready=True,
    )
    ctx = PipelineContext(
        "x",
        paths=run_paths,
        continuity_bible=ContinuityBible(),
        scene_states={1: state},
    )
    events = []
    orchestrator = Orchestrator()
    orchestrator._callback = lambda event, data: events.append((event, data))

    orchestrator._apply_review_result(
        ctx,
        1,
        state,
        ReviewResult(
            is_valid=False,
            severity="major",
            feedback="碎片几何关系无法验证，目标区域覆盖不正确",
        ),
    )

    assert state.safe_fallback_used is True
    assert state.reviewed is False
    assert state.give_up is False
    assert state.review_round == 0
    assert state.code == ""
    assert "保守教学方案" in state.rewrite_feedback
    assert any(event == "scene_safe_fallback" for event, _ in events)


def test_identical_review_feedback_stops_repeated_rewrites(monkeypatch, tmp_path):
    monkeypatch.setattr(module.settings, "MAX_IDENTICAL_REVIEW_ATTEMPTS", 2)
    run_paths = paths(tmp_path)
    run_paths.scenes.mkdir(parents=True)
    state = SceneState(
        plan=plan(),
        code="from manim import *\nclass Demo(Scene):\n    def construct(self): pass\n",
        class_name="Demo",
        plan_ready=True,
    )
    ctx = PipelineContext("x", paths=run_paths, scene_states={1: state})
    result = ReviewResult(
        is_valid=False,
        severity="major",
        feedback="缺少一个明确的动画步骤",
    )

    Orchestrator()._apply_review_result(ctx, 1, state, result)
    assert state.give_up is False
    # 模拟 Coder 原样返回同一份代码、Reviewer 原样返回同一份反馈。
    state.rewrite_feedback = ""
    Orchestrator()._apply_review_result(ctx, 1, state, result)

    assert state.give_up is True
    assert "相同代码和审查反馈" in state.failure_reason


def test_safe_fallback_is_not_repeated_after_resume(monkeypatch, tmp_path):
    monkeypatch.setattr(module.settings, "MAX_REVIEW_ROUNDS", 1)
    monkeypatch.setattr(module.settings, "MAX_IDENTICAL_REVIEW_ATTEMPTS", 2)
    run_paths = paths(tmp_path)
    run_paths.scenes.mkdir(parents=True)
    geometry_plan = plan().model_copy(update={"visual_flow": ["切割碎片并无缝拼接到目标区域"]})
    state = SceneState(
        plan=geometry_plan,
        code="from manim import *\nclass Demo(Scene):\n    def construct(self): pass\n",
        class_name="Demo",
        plan_ready=True,
        safe_fallback_used=True,
    )
    ctx = PipelineContext("x", paths=run_paths, scene_states={1: state})

    Orchestrator()._apply_review_result(
        ctx,
        1,
        state,
        ReviewResult(is_valid=False, severity="major", feedback="几何方案错误"),
    )

    assert state.give_up is True
    assert state.safe_fallback_used is True
    assert "保守教学方案" not in state.failure_reason


def test_scene_review_rejects_invalid_export_before_llm(monkeypatch, tmp_path):
    run_paths = paths(tmp_path)
    run_paths.scenes.mkdir(parents=True)
    code = """
from manim import *
class Demo(Scene):
    def construct(self):
        # KD1_CONTINUITY_EXPORT_BEGIN
        circle = Circle()
        # KD1_CONTINUITY_EXPORT_END
        # KD1_CONTINUITY_EXPORT_BEGIN
        square = Square()
        # KD1_CONTINUITY_EXPORT_END
"""
    state = SceneState(plan=plan(), code=code, class_name="Demo", plan_ready=True)
    ctx = PipelineContext("x", paths=run_paths, scene_states={1: state})
    called = False

    def reviewer_should_not_run(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("invalid export should be rejected deterministically")

    monkeypatch.setattr(module.ReviewerAgent, "review", reviewer_should_not_run)
    orchestrator = Orchestrator()
    orchestrator._scene_review(ctx, 1, state)

    assert called is False
    assert state.reviewed is False
    assert "连续性导出区无效" in state.rewrite_feedback


def test_code_generation_validates_continuity_contract_before_code_review(monkeypatch):
    from kd1_anime.agents.validator import CodeValidationResult

    scene_plan = plan().model_copy(
        update={"new_elements": [VisualElementState(element_id="formula", variable_name="formula")]}
    )
    invalid = """from manim import *
class Demo(Scene):
    def construct(self):
        # KD1_CONTINUITY_EXPORT_BEGIN
        # element_id: formula
        formula = Circle()
        # element_id: formula
        other = Square()
        # KD1_CONTINUITY_EXPORT_END
"""
    valid = """from manim import *
class Demo(Scene):
    def construct(self):
        # KD1_CONTINUITY_EXPORT_BEGIN
        # element_id: formula
        formula = Circle()
        # KD1_CONTINUITY_EXPORT_END
"""

    class FakeCoder:
        def __init__(self):
            self.calls = []

        def generate_code(self, scene_plan, feedback="", **kwargs):
            self.calls.append(feedback)
            return invalid if len(self.calls) == 1 else valid

    coder = FakeCoder()
    monkeypatch.setattr(module, "CoderAgent", lambda: coder)
    monkeypatch.setattr(
        Orchestrator,
        "_validate",
        staticmethod(lambda code, **kwargs: CodeValidationResult(True, scene_classes=["Demo"])),
    )

    generated, class_name = Orchestrator()._generate_validated_code(
        scene_plan,
        stream=False,
    )

    assert generated == valid
    assert class_name == "Demo"
    assert len(coder.calls) == 2
    assert "连续性导出合同" in coder.calls[1]
    assert "第 2/3 次修复" in coder.calls[1]


def test_code_generation_explains_single_export_definition_on_lifecycle_failure(monkeypatch):
    from kd1_anime.agents.validator import CodeValidationResult

    scene_plan = plan().model_copy(
        update={
            "new_elements": [
                VisualElementState(
                    element_id="formula",
                    variable_name="formula",
                    required=True,
                )
            ]
        }
    )
    technical_spec = TechnicalSpec(
        scene_id=1,
        objects=[
            TechnicalObject(
                element_id="formula",
                variable_name="formula",
                constructor="Circle",
                lifecycle=["define", "create", "keep"],
                exported=True,
            )
        ],
        animations=[
            TechnicalAnimation(
                event_id="show_formula",
                start_seconds=0,
                end_seconds=1,
                operation="create",
                target_element_ids=["formula"],
                create_element_ids=["formula"],
            )
        ],
        export_element_ids=["formula"],
    )
    invalid = """from manim import *
class Demo(Scene):
    def construct(self):
        formula = Circle()
        self.play(Create(formula))
        # KD1_CONTINUITY_EXPORT_BEGIN
        # element_id: formula
        formula = Circle()
        # KD1_CONTINUITY_EXPORT_END
"""
    valid = """from manim import *
class Demo(Scene):
    def construct(self):
        # KD1_CONTINUITY_EXPORT_BEGIN
        # element_id: formula
        formula = Circle()
        # KD1_CONTINUITY_EXPORT_END
        self.play(Create(formula))
"""

    class FakeCoder:
        def __init__(self):
            self.calls = []

        def generate_code(self, scene_plan, feedback="", **kwargs):
            self.calls.append(feedback)
            return invalid if len(self.calls) == 1 else valid

    coder = FakeCoder()
    monkeypatch.setattr(module, "CoderAgent", lambda: coder)
    monkeypatch.setattr(
        Orchestrator,
        "_validate",
        staticmethod(lambda code, **kwargs: CodeValidationResult(True, scene_classes=["Demo"])),
    )

    generated, class_name = Orchestrator()._generate_validated_code(
        scene_plan,
        technical_spec=technical_spec,
        stream=False,
    )

    assert generated == valid
    assert class_name == "Demo"
    assert len(coder.calls) == 2
    assert "生命周期修复规则" in coder.calls[1]
    assert "导出区只能有一个" in coder.calls[1]


def test_code_generation_marks_an_unchanged_invalid_candidate(monkeypatch):
    from kd1_anime.agents.validator import CodeValidationResult

    scene_plan = plan().model_copy(
        update={
            "new_elements": [
                VisualElementState(
                    element_id="formula",
                    variable_name="formula",
                    required=True,
                )
            ]
        }
    )
    invalid = """from manim import *
class Demo(Scene):
    def construct(self):
        # KD1_CONTINUITY_EXPORT_BEGIN
        # element_id: formula
        formula = Circle()
        # element_id: formula
        duplicate = Square()
        # KD1_CONTINUITY_EXPORT_END
"""

    class FakeCoder:
        def __init__(self):
            self.calls = []

        def generate_code(self, scene_plan, feedback="", **kwargs):
            self.calls.append(feedback)
            return invalid

    coder = FakeCoder()
    monkeypatch.setattr(module, "CoderAgent", lambda: coder)
    monkeypatch.setattr(settings, "CODE_VALIDATION_ATTEMPTS", 3)
    monkeypatch.setattr(
        Orchestrator,
        "_validate",
        staticmethod(lambda code, **kwargs: CodeValidationResult(True, scene_classes=["Demo"])),
    )

    with pytest.raises(module.ValidationError):
        Orchestrator()._generate_validated_code(scene_plan, stream=False)

    assert len(coder.calls) == 3
    assert "完全相同" in coder.calls[2]


def test_state_ledger_keeps_removed_element_as_historical_tombstone(tmp_path):
    run_paths = paths(tmp_path)
    run_paths.root.mkdir(parents=True)
    formula = VisualElementState(
        element_id="formula",
        variable_name="formula",
        semantic_state="核心公式",
        required=True,
    )
    first_plan = plan().model_copy(update={"new_elements": [formula]})
    first_code = "from manim import *\nclass Demo(Scene):\n    def construct(self): pass\n"
    first_state = SceneState(
        plan=first_plan,
        code=first_code,
        exported_elements_code="formula = Circle()",
        exported_elements=[
            ExtractedElement(
                element_id="formula",
                variable_name="formula",
                code="formula = Circle()",
            )
        ],
    )
    ctx = PipelineContext("prompt", paths=run_paths)
    orchestrator = Orchestrator()

    orchestrator._update_state_ledger(ctx, first_state)

    second_plan = plan().model_copy(
        update={
            "scene_id": 2,
            "inherited_elements": [formula],
            "elements_to_remove": [formula],
        }
    )
    second_state = SceneState(plan=second_plan)

    orchestrator._update_state_ledger(ctx, second_state)

    historical = next(item for item in ctx.state_ledger.elements if item.element_id == "formula")
    assert historical.active is False
    assert historical.required_next is False
    assert "formula" in ctx.state_ledger.boundaries[1].closing_element_ids
    assert "formula" in ctx.state_ledger.boundaries[2].opening_element_ids
    assert "formula" not in ctx.state_ledger.boundaries[2].closing_element_ids


def test_removing_reexported_scene_keeps_ids_used_by_older_boundaries(tmp_path):
    run_paths = paths(tmp_path)
    run_paths.root.mkdir(parents=True)
    axes = VisualElementState(
        element_id="axes",
        variable_name="axes",
        semantic_state="坐标轴",
        required=True,
    )
    first_plan = plan().model_copy(update={"new_elements": [axes]})
    first_state = SceneState(
        plan=first_plan,
        code="from manim import *\nclass Demo(Scene):\n    def construct(self): pass\n",
        exported_elements_code="axes = ThreeDAxes()",
        exported_elements=[
            ExtractedElement(element_id="axes", variable_name="axes", code="axes = ThreeDAxes()")
        ],
    )
    ctx = PipelineContext("prompt", paths=run_paths)
    orchestrator = Orchestrator()
    orchestrator._update_state_ledger(ctx, first_state)

    second_plan = plan().model_copy(update={"scene_id": 2, "inherited_elements": [axes]})
    second_state = SceneState(
        plan=second_plan,
        code=first_state.code,
        exported_elements_code="axes = ThreeDAxes()",
        exported_elements=[
            ExtractedElement(element_id="axes", variable_name="axes", code="axes = ThreeDAxes()")
        ],
    )
    orchestrator._update_state_ledger(ctx, second_state)

    orchestrator._remove_element_manifest_scene(ctx, 2)

    historical = next(item for item in ctx.state_ledger.elements if item.element_id == "axes")
    assert historical.active is False
    assert "axes" in ctx.state_ledger.boundaries[1].closing_element_ids
    assert ctx.state_ledger.current_scene_id == 1


def test_code_generation_does_not_treat_unmarked_internal_objects_as_exports(monkeypatch):
    from kd1_anime.agents.validator import CodeValidationResult

    scene_plan = plan().model_copy(
        update={
            "new_elements": [
                VisualElementState(
                    element_id="temporary_formula",
                    variable_name="temporary_formula",
                    required=False,
                )
            ]
        }
    )
    code = """from manim import *
class Demo(Scene):
    def construct(self):
        tex_template = TexTemplate(tex_compiler="xelatex", output_format=".xdv")
        temporary_formula = MathTex(r"x^2", tex_template=tex_template)
        self.play(Write(temporary_formula))
"""

    class FakeCoder:
        calls = 0

        def generate_code(self, scene_plan, feedback="", **kwargs):
            self.calls += 1
            return code

    coder = FakeCoder()
    monkeypatch.setattr(module, "CoderAgent", lambda: coder)
    monkeypatch.setattr(
        Orchestrator,
        "_validate",
        staticmethod(lambda value, **kwargs: CodeValidationResult(True, scene_classes=["Demo"])),
    )

    generated, class_name = Orchestrator()._generate_validated_code(scene_plan, stream=False)

    assert generated == code
    assert class_name == "Demo"
    assert coder.calls == 1


def test_plan_review_repairs_explicit_handoff_ids_across_neighboring_scenes(monkeypatch, tmp_path):
    p1 = plan().model_copy(
        update={
            "scene_id": 1,
            "new_elements": [
                VisualElementState(
                    element_id="function_label",
                    variable_name="function_label",
                    required=False,
                )
            ],
        }
    )
    p2 = plan().model_copy(
        update={
            "scene_id": 2,
            "inherited_elements": [],
            "new_elements": [
                VisualElementState(
                    element_id="tangent_plane_surface",
                    variable_name="tangent_plane_surface",
                    required=False,
                )
            ],
        }
    )
    p3 = plan().model_copy(update={"scene_id": 3, "new_elements": []})
    ctx = PipelineContext(
        "prompt",
        paths=paths(tmp_path),
        scene_states={
            1: SceneState(plan=p1, plan_ready=True),
            2: SceneState(plan=p2, plan_ready=True),
            3: SceneState(plan=p3, plan_ready=True),
        },
    )
    orchestrator = Orchestrator()
    monkeypatch.setattr(orchestrator, "_checkpoint", lambda *args, **kwargs: None)
    issues = [
        PlanReviewIssue(
            category="contract",
            field="handoff",
            message=(
                "tangent_plane_surface 未列入 handoff，应传递给场景3；"
                "function_label 需要从场景1继承并传递给场景3。"
            ),
            fix_instruction="将两个元素标记为 required，并补充 handoff。",
        )
    ]

    repairs = orchestrator._repair_plan_handoff_issues(
        ctx,
        2,
        ctx.scene_states[2],
        issues,
    )

    assert repairs
    assert any(
        item.element_id == "function_label" and item.required
        for item in ctx.scene_states[1].plan.new_elements
    )
    assert any(
        item.element_id == "function_label" and item.action == "create"
        for item in ctx.scene_states[1].plan.handoff
    )
    assert any(
        item.element_id == "tangent_plane_surface" and item.required
        for item in ctx.scene_states[2].plan.new_elements
    )
    assert {item.element_id for item in ctx.scene_states[3].plan.inherited_elements} >= {
        "function_label",
        "tangent_plane_surface",
    }


def test_plan_review_does_not_resurrect_elements_from_explicit_full_exit(monkeypatch, tmp_path):
    previous_element = VisualElementState(
        element_id="temporary_grid",
        variable_name="temporary_grid",
        required=False,
    )
    previous = plan().model_copy(
        update={
            "scene_id": 1,
            "new_elements": [previous_element],
            "closing_state": ["所有元素整体淡出，场景结束"],
        }
    )
    current = plan().model_copy(
        update={
            "scene_id": 2,
            "opening_state": ["接管 temporary_grid"],
            "inherited_elements": [],
        }
    )
    ctx = PipelineContext(
        "prompt",
        paths=paths(tmp_path),
        scene_states={
            1: SceneState(plan=previous, plan_ready=True),
            2: SceneState(plan=current, plan_ready=True),
        },
    )
    orchestrator = Orchestrator()
    monkeypatch.setattr(orchestrator, "_checkpoint", lambda *args, **kwargs: None)

    repairs = orchestrator._repair_plan_handoff_issues(
        ctx,
        2,
        ctx.scene_states[2],
        [
            PlanReviewIssue(
                category="contract",
                field="inherited_elements",
                message="temporary_grid 应从场景1继承到场景2",
                fix_instruction="将 temporary_grid 加入 inherited_elements。",
            )
        ],
    )

    assert repairs == []
    assert ctx.scene_states[1].plan.new_elements[0].required is False
    assert ctx.scene_states[2].plan.inherited_elements == []


def test_mixed_plan_issues_do_not_trigger_handoff_repair_loop(monkeypatch, tmp_path):
    previous = plan().model_copy(update={"scene_id": 1})
    current = plan().model_copy(
        update={
            "scene_id": 2,
            "inherited_elements": [],
            "new_elements": [
                VisualElementState(
                    element_id="grid_sheared",
                    variable_name="grid_sheared",
                    required=False,
                )
            ],
        }
    )
    ctx = PipelineContext(
        "prompt",
        paths=paths(tmp_path),
        scene_states={
            1: SceneState(plan=previous, plan_ready=True),
            2: SceneState(plan=current, plan_ready=True),
        },
    )
    orchestrator = Orchestrator()
    monkeypatch.setattr(orchestrator, "_checkpoint", lambda *args, **kwargs: None)

    repairs = orchestrator._repair_plan_handoff_issues(
        ctx,
        2,
        ctx.scene_states[2],
        [
            PlanReviewIssue(
                category="geometry",
                field="geometry_specs[grid_sheared].vertices",
                message="几何顶点超出安全范围",
                fix_instruction="缩小网格或改用保守表达",
            ),
            PlanReviewIssue(
                category="contract",
                field="handoff",
                message="grid_sheared 未列入 handoff",
                fix_instruction="将 grid_sheared 标记为 required 并补充 handoff",
            ),
        ],
    )

    assert repairs == []
    assert ctx.scene_states[1].plan == previous
    assert ctx.scene_states[2].plan == current


def test_compile_scene_plans_removes_extra_detail_claims(monkeypatch, tmp_path):
    scene_plan = plan().model_copy(
        update={
            "claim_ids": ["claim_1"],
            "math_claims": [
                MathClaim(claim_id="claim_1", statement="x=x"),
                MathClaim(claim_id="claim_1_extra", statement="x+1=x+1"),
            ],
            "timeline": [
                TimelineEvent(
                    event_id="show",
                    start_seconds=0,
                    end_seconds=10,
                    action="展示",
                    math_claim_ids=["claim_1", "claim_1_extra"],
                )
            ],
        }
    )
    ctx = PipelineContext(
        "prompt",
        paths=paths(tmp_path),
        continuity_bible=ContinuityBible(),
        scene_states={1: SceneState(plan=scene_plan, plan_ready=True)},
    )
    orchestrator = Orchestrator()
    monkeypatch.setattr(orchestrator, "_checkpoint", lambda *args, **kwargs: None)

    assert orchestrator._normalize_scene_claim_contracts(ctx) is True
    assert [claim.claim_id for claim in ctx.scene_states[1].plan.math_claims] == ["claim_1"]
    assert ctx.scene_states[1].plan.timeline[0].math_claim_ids == ["claim_1"]


def test_normalize_scene_claim_contracts_restores_lesson_claim_and_timeline_evidence(
    monkeypatch, tmp_path
):
    scene_plan = plan().model_copy(
        update={
            "claim_ids": ["claim_1"],
            "math_claims": [],
            "timeline": [
                TimelineEvent(
                    event_id="show_formula",
                    start_seconds=0,
                    end_seconds=10,
                    action="展示函数公式",
                )
            ],
        }
    )
    lesson_claim = MathClaim(
        claim_id="claim_1",
        statement="函数公式 z=x^2+y^2",
        expression_before="z=x^2+y^2",
        expression_after="z=x^2+y^2",
    )
    ctx = PipelineContext(
        "prompt",
        paths=paths(tmp_path),
        lesson_spec=LessonSpec(claims=[lesson_claim]),
        scene_states={1: SceneState(plan=scene_plan, plan_ready=True)},
    )
    orchestrator = Orchestrator()
    monkeypatch.setattr(orchestrator, "_checkpoint", lambda *args, **kwargs: None)

    assert orchestrator._normalize_scene_claim_contracts(ctx) is True
    repaired = ctx.scene_states[1].plan
    assert [claim.claim_id for claim in repaired.math_claims] == ["claim_1"]
    assert repaired.timeline[0].math_claim_ids == ["claim_1"]


def test_plan_review_revisits_neighbors_after_mechanical_handoff_repair(monkeypatch, tmp_path):
    boundary = {
        "opening_state": ["对象进入画面"],
        "closing_state": ["对象保留到场景结束"],
        "transition_in": "对象从上一状态接入",
        "transition_out": "对象交给下一场景",
    }
    p1 = plan().model_copy(
        update={
            "scene_id": 1,
            **boundary,
            "new_elements": [
                VisualElementState(
                    element_id="function_label",
                    variable_name="function_label",
                    required=False,
                )
            ],
        }
    )
    p2 = plan().model_copy(
        update={
            "scene_id": 2,
            **boundary,
            "new_elements": [
                VisualElementState(
                    element_id="tangent_plane_surface",
                    variable_name="tangent_plane_surface",
                    required=False,
                )
            ],
        }
    )
    p3 = plan().model_copy(update={"scene_id": 3, **boundary, "new_elements": []})
    outlines = [
        SceneOutline(
            scene_id=scene_id,
            title=f"Scene {scene_id}",
            duration_seconds=10,
            purpose="test",
            math_concept="test",
        )
        for scene_id in (1, 2, 3)
    ]
    ctx = PipelineContext(
        "prompt",
        paths=paths(tmp_path),
        outlines=outlines,
        continuity_bible=ContinuityBible(),
        plan_review_status="pending",
        continuity_review_status="passed",
        scene_states={
            1: SceneState(plan=p1, plan_ready=True),
            2: SceneState(plan=p2, plan_ready=True),
            3: SceneState(plan=p3, plan_ready=True),
        },
    )
    orchestrator = Orchestrator()
    orchestrator._llm_sem = threading.Semaphore(1)
    monkeypatch.setattr(orchestrator, "_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(orchestrator, "_run_plan_review_batch", lambda *args, **kwargs: {})

    class Reviewer:
        def __init__(self):
            self.calls = []

        def review(self, current_plan, **kwargs):
            self.calls.append(current_plan.scene_id)
            if current_plan.scene_id == 2 and self.calls.count(2) == 1:
                return PlanReviewResult(
                    is_valid=False,
                    severity="major",
                    issues=[
                        {
                            "category": "contract",
                            "field": "handoff",
                            "confidence": "high",
                            "evidence_type": "contract",
                            "evidence": "tangent_plane_surface 未列入 handoff",
                            "message": (
                                "tangent_plane_surface 未列入 handoff，应传递给场景3；"
                                "function_label 需要从场景1继承并传递给场景3。"
                            ),
                            "fix_instruction": "将两个元素标记为 required，并补充 handoff。",
                        }
                    ],
                )
            return PlanReviewResult(is_valid=True, severity="info")

    reviewer = Reviewer()
    monkeypatch.setattr(module, "PlanReviewerAgent", lambda: reviewer)

    orchestrator._run_plan_review_barrier(ctx)

    assert reviewer.calls.count(1) == 2
    assert reviewer.calls.count(2) == 2
    assert reviewer.calls.count(3) == 1
    assert ctx.plan_review_status == "passed"
    assert all(state.plan_reviewed for state in ctx.scene_states.values())


def test_dry_run_with_failed_scene_is_not_marked_complete(monkeypatch, tmp_path):
    run_paths = paths(tmp_path)
    ctx = PipelineContext(
        "x",
        paths=run_paths,
        dry_run=True,
        scene_states={
            1: SceneState(
                plan=plan(),
                plan_ready=True,
                give_up=True,
                failure_reason="review failed",
            )
        },
    )
    monkeypatch.setattr(Orchestrator, "_run_scheduler", lambda self, current: None)

    with pytest.raises(RuntimeError, match="Dry-run 未完成"):
        Orchestrator()._execute(ctx, State.CODING)

    manifest = RunManifest.model_validate(
        json.loads((run_paths.root / "manifest.json").read_text(encoding="utf-8"))
    )
    assert manifest.status == "failed"
    assert manifest.state == "ERROR"


def test_dry_run_repeats_after_continuity_rebuild(monkeypatch, tmp_path):
    run_paths = paths(tmp_path)
    state = SceneState(plan=plan(), plan_ready=True)
    ctx = PipelineContext(
        "x",
        paths=run_paths,
        dry_run=True,
        scene_states={1: state},
    )
    orchestrator = Orchestrator()
    calls = 0

    def scheduler(current):
        nonlocal calls
        calls += 1
        if calls == 1:
            current.continuity_rebuild_required = True
        else:
            current.scene_states[1].reviewed = True

    monkeypatch.setattr(orchestrator, "_run_scheduler", scheduler)

    assert orchestrator._execute(ctx, State.CODING) is None
    assert calls == 2


def test_run_directories_and_prompt_are_private(monkeypatch, tmp_path):
    from kd1_anime.config import settings

    monkeypatch.setattr(settings, "WORKSPACE_DIR", tmp_path)
    ctx = PipelineContext("private prompt", dry_run=True)

    state = Orchestrator()._handle_init(ctx)

    assert state is State.PLANNING
    assert ctx.paths.root.stat().st_mode & 0o777 == 0o700
    assert (ctx.paths.root / "prompt.md").stat().st_mode & 0o777 == 0o600


def test_infrastructure_error_does_not_invoke_auto_fixer(monkeypatch, tmp_path):
    import time

    from kd1_anime.cluster.slurm import SlurmJob

    run_paths = paths(tmp_path)
    run_paths.scenes.mkdir(parents=True)
    state = SceneState(
        plan=plan(),
        code="from manim import *\nclass Demo(Scene):\n    def construct(self): self.wait()\n",
        class_name="Demo",
    )
    state.slurm_job = SlurmJob(
        job_id="123",
        scene_id=1,
        script_path=run_paths.scenes / "render.sh",
        log_out=run_paths.logs / "out",
        log_err=run_paths.logs / "err",
        media_dir=run_paths.videos / "scene_1",
        scene_class_name="Demo",
        submitted_at=time.time(),
        status="FAILED",
    )
    ctx = PipelineContext("x", paths=run_paths, scene_states={1: state})
    orchestrator = Orchestrator()
    monkeypatch.setattr(
        orchestrator.slurm,
        "get_error_log",
        lambda **kwargs: "Conda: command not found",
    )

    class FakeFixer:
        @staticmethod
        def is_infrastructure_error(error_log):
            return True

        def fix(self, *args):
            raise AssertionError("must not be called")

    monkeypatch.setattr(module, "AutoFixerAgent", FakeFixer)

    orchestrator._scene_fix(ctx, 1, state)
    assert state.give_up is True
    assert "环境或 Slurm" in state.failure_reason


def test_dispatch_respects_max_in_flight(monkeypatch, tmp_path):
    import time

    from kd1_anime.agents.validator import CodeValidationResult
    from kd1_anime.cluster.slurm import SlurmJob
    from kd1_anime.config import settings

    run_paths = paths(tmp_path)
    for directory in (run_paths.scenes, run_paths.logs, run_paths.videos):
        directory.mkdir(parents=True)
    ctx = PipelineContext("x", paths=run_paths)
    code = "from manim import *\nclass Demo(Scene):\n    def construct(self): self.wait()\n"
    for scene_id in range(1, 4):
        scene_plan = plan().model_copy(update={"scene_id": scene_id})
        ctx.scene_states[scene_id] = SceneState(
            plan=scene_plan,
            code=code,
            class_name="Demo",
        )
        (run_paths.scenes / f"scene_{scene_id}.py").write_text(code, encoding="utf-8")

    submitted: list[int] = []

    def fake_submit(scene_id, python_file, scene_class_name, **kwargs):
        submitted.append(scene_id)
        return SlurmJob(
            job_id=str(100 + scene_id),
            scene_id=scene_id,
            script_path=run_paths.scenes / f"render_{scene_id}.sh",
            log_out=run_paths.logs / f"scene_{scene_id}.out",
            log_err=run_paths.logs / f"scene_{scene_id}.err",
            media_dir=run_paths.videos / f"scene_{scene_id}",
            scene_class_name=scene_class_name,
            submitted_at=time.time(),
        )

    orchestrator = Orchestrator()
    monkeypatch.setattr(settings, "SLURM_MAX_IN_FLIGHT", 2)
    monkeypatch.setattr(
        orchestrator,
        "_validate",
        lambda value, **kwargs: CodeValidationResult(True, scene_classes=["Demo"]),
    )
    monkeypatch.setattr(orchestrator.slurm, "submit_scene", fake_submit)

    next_state = orchestrator._handle_dispatching(ctx)

    assert next_state is State.MONITORING
    assert submitted == [1, 2]
    assert ctx.scene_states[3].slurm_job is None


def test_dispatch_rejects_code_changed_on_disk(monkeypatch, tmp_path):
    run_paths = paths(tmp_path)
    for directory in (run_paths.scenes, run_paths.logs, run_paths.videos):
        directory.mkdir(parents=True)
    expected = "from manim import *\nclass Demo(Scene):\n    def construct(self): self.wait()\n"
    (run_paths.scenes / "scene_1.py").write_text(expected + "# tampered\n", encoding="utf-8")
    scene_state = SceneState(plan=plan(), code=expected, class_name="Demo")
    ctx = PipelineContext("x", paths=run_paths, scene_states={1: scene_state})
    orchestrator = Orchestrator()
    monkeypatch.setattr(
        orchestrator.slurm,
        "submit_scene",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not submit")),
    )

    next_state = orchestrator._handle_dispatching(ctx)

    assert next_state is State.MERGING
    assert scene_state.failed is True
    assert "一致性" in scene_state.failure_reason


def test_merging_reuses_only_checkpointed_run_local_output(monkeypatch, tmp_path):
    from kd1_anime.config import settings

    run_paths = paths(tmp_path)
    for directory in (run_paths.scenes, run_paths.logs, run_paths.videos):
        directory.mkdir(parents=True)
    run_paths.output.write_bytes(b"finished video")
    code = "from manim import *\nclass Demo(Scene):\n    def construct(self): self.wait()\n"
    (run_paths.scenes / "scene_1.py").write_text(code, encoding="utf-8")
    video = run_paths.videos / "scene_1" / "Demo.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"scene video")
    profile = PipelineContext("x").render_profile
    artifact = SceneArtifact(
        origin="rendered",
        source_run_id=run_paths.run_id,
        job_id="123",
        scene_id=1,
        scene_class_name="Demo",
        code_sha256=sha256_text(code),
        render_profile_sha256=profile.digest(),
        video_path=video.relative_to(run_paths.root).as_posix(),
        video_sha256=sha256_file(video),
        metadata=VideoMetadata(
            size_bytes=video.stat().st_size,
            duration_seconds=1,
            width=profile.pixel_width,
            height=profile.pixel_height,
            frame_rate=profile.frame_rate,
        ),
    )
    scene_state = SceneState(
        plan=plan(),
        code=code,
        class_name="Demo",
        artifact=artifact,
        rendered=True,
    )
    ctx = PipelineContext(
        "x",
        paths=run_paths,
        scene_states={1: scene_state},
        final_video=run_paths.output.resolve(),
        final_video_sha256=sha256_file(run_paths.output),
        render_profile=profile,
    )
    orchestrator = Orchestrator()
    monkeypatch.setattr(settings, "OVERWRITE_OUTPUT", False)
    monkeypatch.setattr(
        orchestrator.merger,
        "merge",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must reuse output")),
    )

    orchestrator._merge(ctx)
    assert ctx.final_video == run_paths.output.resolve()
    assert ctx.final_video_sha256 == sha256_file(run_paths.output)


def test_failed_run_local_remerge_preserves_previous_output(monkeypatch, tmp_path):
    """重新拼接失败时, run 内上一次输出不能因预删文件而丢失。"""
    run_paths = paths(tmp_path)
    for directory in (run_paths.scenes, run_paths.logs, run_paths.videos):
        directory.mkdir(parents=True)
    run_paths.output.write_bytes(b"previous output")
    code = "from manim import *\nclass Demo(Scene):\n    def construct(self): self.wait()\n"
    (run_paths.scenes / "scene_1.py").write_text(code, encoding="utf-8")
    video = run_paths.videos / "scene_1" / "Demo.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"scene video")
    profile = PipelineContext("x").render_profile
    artifact = SceneArtifact(
        origin="rendered",
        source_run_id=run_paths.run_id,
        job_id="123",
        scene_id=1,
        scene_class_name="Demo",
        code_sha256=sha256_text(code),
        render_profile_sha256=profile.digest(),
        video_path=video.relative_to(run_paths.root).as_posix(),
        video_sha256=sha256_file(video),
        metadata=VideoMetadata(
            size_bytes=video.stat().st_size,
            duration_seconds=1,
            width=profile.pixel_width,
            height=profile.pixel_height,
            frame_rate=profile.frame_rate,
        ),
    )
    ctx = PipelineContext(
        "x",
        paths=run_paths,
        scene_states={
            1: SceneState(
                plan=plan(),
                code=code,
                class_name="Demo",
                artifact=artifact,
                rendered=True,
            )
        },
        render_profile=profile,
    )
    orchestrator = Orchestrator()

    def fail_merge(*args, **kwargs):
        assert kwargs["replace_existing"] is True
        raise RuntimeError("ffmpeg failed")

    monkeypatch.setattr(orchestrator.merger, "merge", fail_merge)

    with pytest.raises(RuntimeError, match="ffmpeg failed"):
        orchestrator._merge(ctx)

    assert run_paths.output.read_bytes() == b"previous output"


def test_eval_remerge_may_replace_only_matching_checkpointed_external_output(monkeypatch, tmp_path):
    from kd1_anime.config import settings

    run_paths = paths(tmp_path)
    external_output = tmp_path / "published.mp4"
    run_paths = RunPaths(
        run_paths.run_id,
        run_paths.root,
        run_paths.scenes,
        run_paths.logs,
        run_paths.videos,
        external_output,
    )
    for directory in (run_paths.scenes, run_paths.logs, run_paths.videos):
        directory.mkdir(parents=True)
    code = "from manim import *\nclass Demo(Scene):\n    def construct(self): self.wait()\n"
    (run_paths.scenes / "scene_1.py").write_text(code, encoding="utf-8")
    video = run_paths.videos / "scene_1" / "Demo.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"scene video")
    external_output.write_bytes(b"old merged video")
    profile = PipelineContext("x").render_profile
    artifact = SceneArtifact(
        origin="rendered",
        source_run_id=run_paths.run_id,
        job_id="123",
        scene_id=1,
        scene_class_name="Demo",
        code_sha256=sha256_text(code),
        render_profile_sha256=profile.digest(),
        video_path=video.relative_to(run_paths.root).as_posix(),
        video_sha256=sha256_file(video),
        metadata=VideoMetadata(
            size_bytes=video.stat().st_size,
            duration_seconds=1,
            width=profile.pixel_width,
            height=profile.pixel_height,
            frame_rate=profile.frame_rate,
        ),
    )
    ctx = PipelineContext(
        "x",
        paths=run_paths,
        scene_states={
            1: SceneState(
                plan=plan(),
                code=code,
                class_name="Demo",
                artifact=artifact,
                rendered=True,
            )
        },
        final_video=external_output,
        final_video_sha256=sha256_file(external_output),
        render_profile=profile,
        eval_round=1,
    )
    orchestrator = Orchestrator()
    monkeypatch.setattr(settings, "OVERWRITE_OUTPUT", False)

    def fake_merge(video_paths, output_path, *, replace_existing=False, render_profile=None):
        assert video_paths == [video]
        assert output_path == external_output
        assert replace_existing is True
        assert render_profile == profile
        external_output.write_bytes(b"improved merged video")
        return external_output

    monkeypatch.setattr(orchestrator.merger, "merge", fake_merge)

    orchestrator._merge(ctx)

    assert external_output.read_bytes() == b"improved merged video"
    assert ctx.final_video_sha256 == sha256_file(external_output)


def test_merge_rejects_rendered_scene_without_artifact(tmp_path):
    run_paths = paths(tmp_path)
    state = SceneState(
        plan=plan(),
        code="from manim import *\nclass Demo(Scene):\n    def construct(self): pass\n",
        class_name="Demo",
        rendered=True,
    )
    ctx = PipelineContext("x", paths=run_paths, scene_states={1: state})

    with pytest.raises(RuntimeError, match="缺少产物凭据"):
        Orchestrator()._merge(ctx)


def test_execute_failure_preserves_latest_checkpointed_stage(monkeypatch, tmp_path):
    from kd1_anime.run_store import RunManifest

    run_paths = paths(tmp_path)
    run_paths.root.mkdir(parents=True)
    ctx = PipelineContext(
        "x",
        paths=run_paths,
        scene_states={1: SceneState(plan=plan(), plan_ready=True, reviewed=True)},
    )
    orchestrator = Orchestrator()
    monkeypatch.setattr(orchestrator, "_run_scheduler", lambda context: None)

    def fail_during_merge(context):
        orchestrator._checkpoint(context, State.MERGING)
        raise RuntimeError("merge failed")

    monkeypatch.setattr(orchestrator, "_merge", fail_during_merge)

    with pytest.raises(RuntimeError, match="merge failed"):
        orchestrator._execute(ctx, State.INIT)

    manifest = RunManifest.model_validate_json(
        (run_paths.root / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest.status == "failed"
    assert manifest.state == "MERGING"


def test_eval_improvement_state_and_round_are_checkpointed(monkeypatch, tmp_path):
    from kd1_anime.config import settings
    from kd1_anime.eval.metrics import EvalMetric, EvalResult, QualityScore
    from kd1_anime.run_store import RunManifest

    run_paths = paths(tmp_path)
    for directory in (run_paths.root, run_paths.scenes, run_paths.logs, run_paths.videos):
        directory.mkdir(parents=True, exist_ok=True)
    code = "from manim import *\nclass Demo(Scene):\n    def construct(self): self.wait()\n"
    (run_paths.scenes / "scene_1.py").write_text(code, encoding="utf-8")
    run_paths.output.write_bytes(b"merged")
    state = SceneState(
        plan=plan(),
        code=code,
        class_name="Demo",
        plan_ready=True,
        reviewed=True,
        rendered=True,
        failed=True,
        give_up=True,
        failure_reason="previous failure",
    )
    ctx = PipelineContext(
        "prompt",
        paths=run_paths,
        scene_states={1: state},
        final_video=run_paths.output,
    )

    low = EvalResult(run_id=run_paths.run_id)
    low.add_score(QualityScore(EvalMetric.CODE_STYLE, 1))

    class FakeEvaluator:
        def __init__(self, **kwargs):
            pass

        def evaluate_code(self, source):
            return low

        def evaluate_run(self, *args, **kwargs):
            return low

    monkeypatch.setattr("kd1_anime.eval.Evaluator", FakeEvaluator)
    monkeypatch.setattr(settings, "ENABLE_AUTO_EVAL", True)
    monkeypatch.setattr(settings, "MAX_EVAL_ROUNDS", 2)
    monkeypatch.setattr(settings, "EVAL_THRESHOLD", 3.5)

    should_improve = Orchestrator()._eval(ctx)

    assert should_improve is True
    assert ctx.eval_round == 1
    assert state.code == ""
    assert state.rendered is False
    assert state.failed is False
    assert state.give_up is False
    assert state.failure_reason == ""
    manifest = RunManifest.model_validate_json(
        (run_paths.root / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest.state == "EVALUATING"
    assert manifest.eval_round == 1
    restored = Orchestrator._context_from_manifest(manifest, run_paths.root)
    assert restored.eval_round == 1


def test_auto_eval_does_not_repeat_visual_calls_or_depend_on_visual_endpoint(monkeypatch, tmp_path):
    from kd1_anime.config import settings
    from kd1_anime.eval.metrics import EvalMetric, EvalResult, QualityScore

    run_paths = paths(tmp_path)
    run_paths.root.mkdir(parents=True)
    run_paths.output.write_bytes(b"merged")
    state = SceneState(
        plan=plan(),
        code="from manim import *\nclass Demo(Scene):\n    def construct(self): self.wait()\n",
        class_name="Demo",
        plan_ready=True,
        reviewed=True,
        rendered=True,
    )
    ctx = PipelineContext(
        "prompt",
        paths=run_paths,
        scene_states={1: state},
        final_video=run_paths.output,
    )
    low_code = EvalResult(run_id="code")
    low_code.add_score(QualityScore(EvalMetric.CODE_STYLE, 1))
    run_result = EvalResult(run_id=run_paths.run_id)
    run_result.add_score(QualityScore(EvalMetric.CODE_STYLE, 1))
    run_result.add_error("visual", "vision endpoint unavailable")
    calls = {}

    class FakeEvaluator:
        def __init__(self, **kwargs):
            calls["init"] = kwargs

        def evaluate_code(self, source):
            return low_code

        def evaluate_run(self, *args, **kwargs):
            calls["run"] = kwargs
            return run_result

    monkeypatch.setattr("kd1_anime.eval.Evaluator", FakeEvaluator)
    monkeypatch.setattr(settings, "ENABLE_AUTO_EVAL", True)
    monkeypatch.setattr(settings, "ENABLE_VISUAL_EVAL", True)
    monkeypatch.setattr(settings, "MAX_EVAL_ROUNDS", 2)
    monkeypatch.setattr(settings, "EVAL_THRESHOLD", 3.5)

    assert Orchestrator()._eval(ctx) is True
    assert calls["init"]["enable_visual_eval"] is False
    assert calls["run"]["enable_visual"] is False
    assert ctx.eval_round == 1
    assert state.code == ""
    assert state.rendered is False


def _make_visual_eval_context(tmp_path):
    from kd1_anime.run_store import VisualEvalProfile

    run_paths = paths(tmp_path)
    for directory in (run_paths.root, run_paths.scenes, run_paths.logs, run_paths.videos):
        directory.mkdir(parents=True, exist_ok=True)
    code = "from manim import *\nclass Demo(Scene):\n    def construct(self): self.wait()\n"
    (run_paths.scenes / "scene_1.py").write_text(code, encoding="utf-8")
    video = run_paths.videos / "scene_1" / "Demo.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"scene-video")
    profile = PipelineContext("x").render_profile
    artifact = SceneArtifact(
        origin="rendered",
        source_run_id=run_paths.run_id,
        job_id="123",
        scene_id=1,
        scene_class_name="Demo",
        code_sha256=sha256_text(code),
        render_profile_sha256=profile.digest(),
        video_path=video.relative_to(run_paths.root).as_posix(),
        video_sha256=sha256_file(video),
        metadata=VideoMetadata(
            size_bytes=video.stat().st_size,
            duration_seconds=1,
            width=profile.pixel_width,
            height=profile.pixel_height,
            frame_rate=profile.frame_rate,
        ),
    )
    state = SceneState(
        plan=plan(),
        code=code,
        class_name="Demo",
        plan_ready=True,
        reviewed=True,
        rendered=True,
        artifact=artifact,
        visual_status="pending",
    )
    ctx = PipelineContext(
        "prompt",
        paths=run_paths,
        scene_states={1: state},
        render_profile=profile,
        visual_eval_profile=VisualEvalProfile(
            enabled=True,
            model="vision-model",
            frame_count=2,
            threshold=3.5,
            max_fix_attempts=2,
        ),
    )
    return ctx, state


def test_visual_gate_records_endpoint_failure_as_unknown_without_rewrite(monkeypatch, tmp_path):
    ctx, state = _make_visual_eval_context(tmp_path)
    events = []

    class BrokenEvaluator:
        def __init__(self, **kwargs):
            pass

        def evaluate_scene_video(self, *args, **kwargs):
            raise RuntimeError("vision endpoint unavailable")

    monkeypatch.setattr("kd1_anime.eval.Evaluator", BrokenEvaluator)
    orchestrator = Orchestrator()
    orchestrator._callback = lambda event, data: events.append((event, data))

    assert orchestrator._visual_gate(ctx) is False
    assert state.rendered is True
    assert state.visual_status == "unknown"
    assert state.visual_artifact_sha256 == state.artifact.video_sha256
    assert state.visual_report_file
    assert (ctx.paths.root / state.visual_report_file).is_file()
    assert state.visual_fix_attempts == 0
    assert any(event == "scene_visual_unknown" for event, _ in events)


def test_visual_gate_low_score_schedules_bounded_coder_rewrite(monkeypatch, tmp_path):
    from kd1_anime.eval.visual_eval import VisualAnalysisResult

    ctx, state = _make_visual_eval_context(tmp_path)
    dimension = {"score": 2, "comprehensive_evaluation": "元素重叠"}
    result = VisualAnalysisResult(
        overall_analysis="布局需要修复",
        mathematical_accuracy={"score": 4, "comprehensive_evaluation": "数学正确"},
        visual_relevance=dimension,
        visual_quality=dimension,
        visual_consistency=dimension,
        element_layout=dimension,
        issues=[],
    )

    class LowScoreEvaluator:
        def __init__(self, **kwargs):
            pass

        def evaluate_scene_video(self, *args, **kwargs):
            return result, []

    monkeypatch.setattr("kd1_anime.eval.Evaluator", LowScoreEvaluator)

    assert Orchestrator()._visual_gate(ctx) is True
    assert state.visual_status == "needs_fix"
    assert state.visual_fix_attempts == 1
    assert state.rendered is False
    assert state.artifact is None
    assert state.reviewed is False
    assert "Visual Evaluation Feedback" in state.rewrite_feedback
    assert state.visual_best_candidate is not None
    assert state.visual_best_candidate.artifact.video_sha256
    manifest = RunManifest.model_validate_json(
        (ctx.paths.root / "manifest.json").read_text(encoding="utf-8")
    )
    restored = Orchestrator._context_from_manifest(manifest, ctx.paths.root)
    restored_candidate = restored.scene_states[1].visual_best_candidate
    assert restored.scene_states[1].visual_status == "needs_fix"
    assert restored_candidate is not None
    assert restored_candidate.code == state.visual_best_candidate.code
    assert (
        restored_candidate.artifact.video_sha256
        == state.visual_best_candidate.artifact.video_sha256
    )
    stored_candidate = manifest.scenes[1].visual_best_candidate
    assert stored_candidate is not None
    (ctx.paths.root / stored_candidate.code_file).write_text("# tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="最佳视觉候选代码哈希"):
        Orchestrator._context_from_manifest(manifest, ctx.paths.root)


def test_merge_rejects_visual_receipt_for_a_different_video(tmp_path):
    ctx, state = _make_visual_eval_context(tmp_path)
    state.visual_status = "passed"
    state.visual_artifact_sha256 = "0" * 64

    with pytest.raises(RuntimeError, match="视觉评估记录不属于当前视频"):
        Orchestrator()._merge(ctx)


def test_visual_rebuild_preserves_and_reuses_compatible_passed_downstream_candidate(
    monkeypatch, tmp_path
):
    from kd1_anime.eval.visual_eval import VisualAnalysisResult

    ctx, first = _make_visual_eval_context(tmp_path)
    code2 = "from manim import *\nclass Demo2(Scene):\n    def construct(self): self.wait()\n"
    (ctx.paths.scenes / "scene_2.py").write_text(code2, encoding="utf-8")
    video2 = ctx.paths.videos / "scene_2" / "Demo2.mp4"
    video2.parent.mkdir(parents=True)
    video2.write_bytes(b"scene-video-2")
    plan2 = plan().model_copy(update={"scene_id": 2, "title": "second"})
    artifact2 = SceneArtifact(
        origin="rendered",
        source_run_id=ctx.paths.run_id,
        job_id="124",
        scene_id=2,
        scene_class_name="Demo2",
        code_sha256=sha256_text(code2),
        render_profile_sha256=ctx.render_profile.digest(),
        video_path=video2.relative_to(ctx.paths.root).as_posix(),
        video_sha256=sha256_file(video2),
        metadata=VideoMetadata(
            size_bytes=video2.stat().st_size,
            duration_seconds=1,
            width=ctx.render_profile.pixel_width,
            height=ctx.render_profile.pixel_height,
            frame_rate=ctx.render_profile.frame_rate,
        ),
    )
    second = SceneState(
        plan=plan2,
        code=code2,
        class_name="Demo2",
        plan_ready=True,
        reviewed=True,
        rendered=True,
        artifact=artifact2,
        visual_status="pending",
    )
    ctx.scene_states[2] = second
    low_dimension = {"score": 2, "comprehensive_evaluation": "需修复"}
    high_dimension = {"score": 5, "comprehensive_evaluation": "清晰"}

    def analysis(dimension):
        return VisualAnalysisResult(
            overall_analysis="assessment",
            mathematical_accuracy=dimension,
            visual_relevance=dimension,
            visual_quality=dimension,
            visual_consistency=dimension,
            element_layout=dimension,
            issues=[],
        )

    class MixedEvaluator:
        def __init__(self, **kwargs):
            pass

        def evaluate_scene_video(self, video_path, **kwargs):
            return (
                analysis(low_dimension if "scene_1" in str(video_path) else high_dimension),
                [],
            )

    monkeypatch.setattr("kd1_anime.eval.Evaluator", MixedEvaluator)
    orchestrator = Orchestrator()

    assert orchestrator._visual_gate(ctx) is True
    assert second.code == ""
    assert second.visual_best_candidate is not None
    assert second.visual_best_candidate.passed is True

    # 模拟 Scene 1 的视觉重写完成但导出合同未变化；Scene 2 的候选绑定同一
    # 继承上下文，应直接恢复而不是再次调用 Coder 或 Slurm。
    first_candidate = first.visual_best_candidate
    assert first_candidate is not None
    first.code = first_candidate.code
    first.class_name = first_candidate.class_name
    first.artifact = first_candidate.artifact
    first.rendered = True
    first.reviewed = True
    first.rewrite_feedback = ""
    first.exported_elements_code = ""
    first.visual_status = "warning"
    first.visual_artifact_sha256 = first_candidate.artifact.video_sha256
    monkeypatch.setattr(orchestrator, "_refresh_scene_export", lambda state: None)

    class FakeTechnicalPlanner:
        def plan(self, scene_plan, *, renderer=None, **kwargs):
            return TechnicalSpec(
                scene_id=scene_plan.scene_id,
                renderer=renderer or "cairo",
            )

    monkeypatch.setattr(
        module,
        "TechnicalPlannerAgent",
        FakeTechnicalPlanner,
    )
    orchestrator._stop_event.clear()

    orchestrator._run_code_review_barrier(ctx)

    assert second.code == code2
    assert second.rendered is True
    assert second.visual_status == "passed"
    assert second.visual_artifact_sha256 == artifact2.video_sha256


def test_auto_eval_invalidates_downstream_continuity_dependents(monkeypatch, tmp_path):
    from kd1_anime.config import settings
    from kd1_anime.eval.metrics import EvalMetric, EvalResult, QualityScore

    run_paths = paths(tmp_path)
    for directory in (run_paths.root, run_paths.scenes, run_paths.logs, run_paths.videos):
        directory.mkdir(parents=True, exist_ok=True)
    states = {}
    for scene_id in (1, 2, 3):
        source = (
            "from manim import *\n"
            f"class Scene{scene_id}(Scene):\n"
            "    def construct(self): self.wait()\n"
        )
        (run_paths.scenes / f"scene_{scene_id}.py").write_text(source, encoding="utf-8")
        scene_plan = plan().model_copy(update={"scene_id": scene_id})
        states[scene_id] = SceneState(
            plan=scene_plan,
            code=source,
            class_name=f"Scene{scene_id}",
            plan_ready=True,
            reviewed=True,
            rendered=True,
        )
    run_paths.output.write_bytes(b"merged")
    ctx = PipelineContext(
        "prompt",
        paths=run_paths,
        scene_states=states,
        final_video=run_paths.output,
    )

    low = EvalResult(run_id="low")
    low.add_score(QualityScore(EvalMetric.CODE_STYLE, 1))
    high = EvalResult(run_id="high")
    high.add_score(QualityScore(EvalMetric.CODE_STYLE, 5))

    class FakeEvaluator:
        def __init__(self, **kwargs):
            pass

        def evaluate_code(self, source):
            return low if "Scene1" in source else high

        def evaluate_run(self, *args, **kwargs):
            return low

    monkeypatch.setattr("kd1_anime.eval.Evaluator", FakeEvaluator)
    monkeypatch.setattr(settings, "ENABLE_AUTO_EVAL", True)
    monkeypatch.setattr(settings, "MAX_EVAL_ROUNDS", 2)
    monkeypatch.setattr(settings, "EVAL_THRESHOLD", 3.5)

    assert Orchestrator()._eval(ctx) is True
    assert ctx.final_video is None
    assert [state.code for state in states.values()] == ["", "", ""]
    assert [state.rendered for state in states.values()] == [False, False, False]
