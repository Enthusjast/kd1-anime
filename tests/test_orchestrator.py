import json
from pathlib import Path

import pytest

import kd1_anime.orchestrator as module
from kd1_anime.agents.planner import ContinuityBible, ScenePlan, VisualElementState
from kd1_anime.agents.reviewer import ReviewResult
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


def test_resume_reopens_continuity_warning_for_unfinished_scenes(monkeypatch, tmp_path):
    """带已知连续性 warning 的中断运行应在 resume 时重新开启修正预算。"""
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


def test_run_paths_are_unique(monkeypatch, tmp_path):
    from kd1_anime.config import settings

    monkeypatch.setattr(settings, "WORKSPACE_DIR", tmp_path)
    first = RunPaths.create()
    second = RunPaths.create()
    assert first.root != second.root


def test_minor_review_is_bounded_by_max_review_rounds(monkeypatch, tmp_path):
    monkeypatch.setattr(module.settings, "MAX_REVIEW_ROUNDS", 2)
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
    assert ctx.scene_states[1].give_up is True
    assert ctx.scene_states[1].review_round == 2


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
