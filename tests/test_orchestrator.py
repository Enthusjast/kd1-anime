from pathlib import Path

import pytest

import kd1_anime.orchestrator as module
from kd1_anime.agents.planner import ScenePlan
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
    manifest = RunManifest.model_validate_json(
        (run_paths.root / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest.state == "EVALUATING"
    assert manifest.eval_round == 1
    restored = Orchestrator._context_from_manifest(manifest, run_paths.root)
    assert restored.eval_round == 1


def test_auto_eval_does_not_decide_when_requested_visual_metrics_are_unknown(monkeypatch, tmp_path):
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

    class FakeEvaluator:
        def __init__(self, **kwargs):
            pass

        def evaluate_code(self, source):
            return low_code

        def evaluate_run(self, *args, **kwargs):
            return run_result

    monkeypatch.setattr("kd1_anime.eval.Evaluator", FakeEvaluator)
    monkeypatch.setattr(settings, "ENABLE_AUTO_EVAL", True)
    monkeypatch.setattr(settings, "ENABLE_VISUAL_EVAL", True)
    monkeypatch.setattr(settings, "MAX_EVAL_ROUNDS", 2)
    monkeypatch.setattr(settings, "EVAL_THRESHOLD", 3.5)

    assert Orchestrator()._eval(ctx) is False
    assert ctx.eval_round == 0
    assert state.code
    assert state.rendered is True
