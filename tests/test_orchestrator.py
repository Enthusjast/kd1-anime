from pathlib import Path

import kd1_anime.orchestrator as module
from kd1_anime.agents.planner import ScenePlan
from kd1_anime.agents.reviewer import ReviewResult
from kd1_anime.orchestrator import Orchestrator, PipelineContext, RunPaths, SceneState, State


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


def test_valid_review_after_rewrite_exits_review_loop(monkeypatch, tmp_path):
    class FakeReviewer:
        def review(self, code, scene_plan):
            return ReviewResult(is_valid=True)

    monkeypatch.setattr(module, "ReviewerAgent", FakeReviewer)
    run_paths = paths(tmp_path)
    run_paths.scenes.mkdir(parents=True)
    ctx = PipelineContext("x", paths=run_paths)
    ctx.scene_states[1] = SceneState(
        plan=plan(),
        code="from manim import *\nclass Demo(Scene):\n    def construct(self): pass\n",
        class_name="Demo",
        review_round=1,
    )
    state = Orchestrator()._handle_reviewing(ctx)
    assert state is State.DISPATCHING
    assert ctx.scene_states[1].review_round == 0


def test_run_paths_are_unique(monkeypatch, tmp_path):
    from kd1_anime.config import settings

    monkeypatch.setattr(settings, "WORKSPACE_DIR", tmp_path)
    first = RunPaths.create()
    second = RunPaths.create()
    assert first.root != second.root


def test_minor_review_is_bounded_by_max_review_rounds(monkeypatch, tmp_path):
    class FakeReviewer:
        def review(self, code, scene_plan):
            return ReviewResult(
                is_valid=False,
                severity="minor",
                fixes=[
                    {
                        "find": "pass",
                        "replace": "self.wait(1)",
                        "reason": "demo",
                    }
                ],
            )

    monkeypatch.setattr(module, "ReviewerAgent", FakeReviewer)
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

    state = Orchestrator()._handle_reviewing(ctx)

    assert state is State.DISPATCHING
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
    monkeypatch.setattr(
        orchestrator.auto_fixer,
        "fix",
        lambda *args: (_ for _ in ()).throw(AssertionError("must not be called")),
    )

    next_state = orchestrator._handle_fixing(ctx)

    assert next_state is State.MERGING
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
        lambda value: CodeValidationResult(True, scene_classes=["Demo"]),
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


def test_merging_recovers_run_local_atomic_output(monkeypatch, tmp_path):
    import time

    from kd1_anime.cluster.slurm import SlurmJob
    from kd1_anime.config import settings

    run_paths = paths(tmp_path)
    for directory in (run_paths.scenes, run_paths.logs, run_paths.videos):
        directory.mkdir(parents=True)
    run_paths.output.write_bytes(b"finished video")
    code = "from manim import *\nclass Demo(Scene):\n    def construct(self): self.wait()\n"
    (run_paths.scenes / "scene_1.py").write_text(code, encoding="utf-8")
    job = SlurmJob(
        job_id="123",
        scene_id=1,
        script_path=run_paths.scenes / "render_1.sh",
        log_out=run_paths.logs / "scene_1.out",
        log_err=run_paths.logs / "scene_1.err",
        media_dir=run_paths.videos / "scene_1",
        scene_class_name="Demo",
        submitted_at=time.time(),
        status="COMPLETED",
    )
    scene_state = SceneState(
        plan=plan(),
        code=code,
        class_name="Demo",
        slurm_job=job,
        rendered=True,
    )
    ctx = PipelineContext("x", paths=run_paths, scene_states={1: scene_state})
    orchestrator = Orchestrator()
    monkeypatch.setattr(settings, "OVERWRITE_OUTPUT", False)
    monkeypatch.setattr(
        orchestrator.merger,
        "merge_jobs",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must reuse output")),
    )

    next_state = orchestrator._handle_merging(ctx)

    assert next_state is State.DONE
    assert ctx.final_video == run_paths.output.resolve()
