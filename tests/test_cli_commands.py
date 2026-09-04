import pytest
import typer
from typer.testing import CliRunner

from kd1_anime.agents.planner import ScenePlan
from kd1_anime.cli import _manifest_requires_generation_apis, _print_comparison, app
from kd1_anime.config import settings
from kd1_anime.eval.metrics import ComparisonResult, EvalResult
from kd1_anime.run_store import RunManifest, StoredSceneState, StoredSlurmJob, write_manifest


def test_cli_registers_all_public_commands():
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0, result.output
    for command in ("generate", "resume", "batch", "doctor", "evaluate", "test-llm", "rag"):
        assert command in result.output


def test_rag_status_is_read_only():
    result = CliRunner().invoke(app, ["rag", "status"])

    assert result.exit_code == 0, result.output
    assert "状态: disabled" in result.output


def test_default_startup_checks_llm_before_opening_chat(monkeypatch):
    calls = []

    monkeypatch.setattr(
        "kd1_anime.cli._ensure_generation_apis",
        lambda *, dry_run: calls.append(("healthcheck", dry_run)),
    )
    monkeypatch.setattr(
        "kd1_anime.cli._start_chat",
        lambda *, dry_run: calls.append(("chat", dry_run)),
    )

    result = CliRunner().invoke(app, [])

    assert result.exit_code == 0, result.output
    assert calls == [("healthcheck", False), ("chat", False)]


def test_generation_startup_checks_rag_services_when_enabled(monkeypatch):
    calls = []

    monkeypatch.setattr("kd1_anime.cli._ensure_llm_api_available", lambda: calls.append("llm"))
    monkeypatch.setattr(settings, "RAG_ENABLED", True)

    class FakeRagService:
        def probe(self):
            calls.append("rag")

    monkeypatch.setattr("kd1_anime.rag.service.RagService", FakeRagService)

    from kd1_anime.cli import _ensure_generation_apis

    _ensure_generation_apis(dry_run=True)

    assert calls == ["llm", "rag"]


def test_generation_startup_stops_when_rag_service_is_unavailable(monkeypatch):
    monkeypatch.setattr("kd1_anime.cli._ensure_llm_api_available", lambda: None)
    monkeypatch.setattr(settings, "RAG_ENABLED", True)

    class FakeRagService:
        def probe(self):
            raise RuntimeError("embedding offline")

    monkeypatch.setattr("kd1_anime.rag.service.RagService", FakeRagService)

    from kd1_anime.cli import _ensure_generation_apis

    with pytest.raises(typer.Exit):
        _ensure_generation_apis(dry_run=True)


def test_print_comparison_handles_unknown_scores():
    comparison = ComparisonResult(
        baseline_run_id="baseline",
        current_run_id="current",
        baseline_result=EvalResult(run_id="baseline"),
        current_result=EvalResult(run_id="current"),
    )

    _print_comparison(comparison)

    assert comparison.score_diff is None


def test_generate_resume_does_not_require_a_dummy_prompt(monkeypatch, tmp_path):
    from kd1_anime.config import settings
    from kd1_anime.run_store import RunManifest, write_manifest

    run_id = "20260728-120000-1234abcd"
    root = tmp_path / "runs" / run_id
    root.mkdir(parents=True)
    output = root / "output_final.mp4"
    output.write_bytes(b"video")
    write_manifest(
        root / "manifest.json",
        RunManifest(
            run_id=run_id,
            status="completed",
            state="DONE",
            user_prompt="prompt",
            output_path=str(output.resolve()),
            final_video=str(output.resolve()),
        ),
    )
    monkeypatch.setattr(settings, "WORKSPACE_DIR", tmp_path)
    monkeypatch.setattr(settings, "LLM_API_KEY", "")
    monkeypatch.setattr(settings, "LLM_MODEL", "")

    result = CliRunner().invoke(app, ["generate", "--resume", run_id])

    assert result.exit_code == 0, result.output
    assert "成功" in result.output
    assert "output_final.mp4" in result.output


def test_global_dry_run_is_forwarded_to_generate(monkeypatch, tmp_path):
    import kd1_anime.orchestrator as orchestrator_module

    monkeypatch.setattr("kd1_anime.cli._ensure_llm_api_available", lambda: None)
    calls = []

    class FakeOrchestrator:
        def __init__(self):
            pass

        def run(self, prompt, *, dry_run=False):
            calls.append((prompt, dry_run))
            return None

    monkeypatch.setattr(orchestrator_module, "Orchestrator", FakeOrchestrator)
    monkeypatch.setattr(settings, "LLM_API_KEY", "key")
    monkeypatch.setattr(settings, "LLM_MODEL", "model")

    result = CliRunner().invoke(app, ["--dry-run", "generate", "demo"])

    assert result.exit_code == 0, result.output
    assert calls == [("demo", True)]


def test_generate_exits_before_pipeline_when_healthcheck_fails(monkeypatch):
    monkeypatch.setattr(settings, "LLM_API_KEY", "key")
    monkeypatch.setattr(settings, "LLM_MODEL", "model")

    def fail_healthcheck():
        raise typer.Exit(1)

    monkeypatch.setattr("kd1_anime.cli._ensure_llm_api_available", fail_healthcheck)

    result = CliRunner().invoke(app, ["generate", "demo", "--dry-run"])

    assert result.exit_code == 1


def test_evaluate_image_enables_visual_evaluation_by_default(monkeypatch, tmp_path):
    image = tmp_path / "frame.png"
    image.write_bytes(b"png")
    captured = {}

    class FakeEvaluator:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def evaluate_visual(self, image_path, description):
            return EvalResult(run_id="visual-test")

    monkeypatch.setattr(
        "kd1_anime.cli._ensure_visual_llm_api_available",
        lambda **kwargs: True,
    )
    monkeypatch.setattr("kd1_anime.eval.Evaluator", FakeEvaluator)
    result = CliRunner().invoke(app, ["evaluate", "--image", str(image)])

    assert result.exit_code == 0, result.output
    assert captured["enable_visual_eval"] is True


def test_evaluate_preserves_empty_code_as_an_explicit_target(monkeypatch):
    captured = {}

    class FakeEvaluator:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def evaluate_code(self, code):
            captured["code"] = code
            return EvalResult(run_id="empty-code")

    monkeypatch.setattr("kd1_anime.eval.Evaluator", FakeEvaluator)
    monkeypatch.setattr(settings, "ENABLE_VISUAL_EVAL", False)

    result = CliRunner().invoke(app, ["evaluate", "--code", ""])

    assert result.exit_code == 0, result.output
    assert captured["code"] == ""


def test_evaluate_scene_id_uses_visual_scene_artifact_path(monkeypatch):
    captured = {}

    class FakeEvaluator:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def evaluate_run_scene(self, run_id, scene_id, *, description=""):
            captured["call"] = (run_id, scene_id, description)
            return EvalResult(run_id=f"{run_id}:scene:{scene_id}")

    monkeypatch.setattr(
        "kd1_anime.cli._ensure_visual_llm_api_available",
        lambda **kwargs: True,
    )
    monkeypatch.setattr("kd1_anime.eval.Evaluator", FakeEvaluator)

    run_id = "20260728-120000-1234abcd"
    result = CliRunner().invoke(
        app,
        ["evaluate", run_id, "--scene-id", "2", "--desc", "平方差"],
    )

    assert result.exit_code == 0, result.output
    assert captured["init"]["enable_visual_eval"] is True
    assert captured["call"] == (run_id, 2, "平方差")


def test_generate_does_not_clear_configured_overwrite_setting(monkeypatch):
    import kd1_anime.orchestrator as orchestrator_module

    monkeypatch.setattr("kd1_anime.cli._ensure_llm_api_available", lambda: None)

    class FakeOrchestrator:
        def __init__(self):
            pass

        def run(self, prompt, *, dry_run=False):
            return None

    monkeypatch.setattr(orchestrator_module, "Orchestrator", FakeOrchestrator)
    monkeypatch.setattr(settings, "LLM_API_KEY", "key")
    monkeypatch.setattr(settings, "LLM_MODEL", "model")
    monkeypatch.setattr(settings, "OVERWRITE_OUTPUT", True)

    result = CliRunner().invoke(app, ["generate", "demo", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert settings.OVERWRITE_OUTPUT is True


def test_global_dry_run_is_forwarded_to_batch(monkeypatch, tmp_path):
    from kd1_anime.batch import BatchProcessor

    monkeypatch.setattr("kd1_anime.cli._ensure_llm_api_available", lambda: None)
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("demo\n", encoding="utf-8")
    monkeypatch.setattr(settings, "LLM_API_KEY", "key")
    monkeypatch.setattr(settings, "LLM_MODEL", "model")
    calls = []

    def fake_execute(self):
        calls.append(self.config.dry_run)
        return []

    monkeypatch.setattr(BatchProcessor, "execute_all", fake_execute)

    result = CliRunner().invoke(app, ["--dry-run", "batch", str(prompts)])

    assert result.exit_code == 0, result.output
    assert calls == [True]


def test_global_dry_run_skips_direct_render_submission(tmp_path):
    scene = tmp_path / "scene.py"
    scene.write_text(
        "from manim import *\n"
        "class Demo(Scene):\n"
        "    def construct(self):\n"
        "        self.add(Square())\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, ["--dry-run", "render", str(scene)])

    assert result.exit_code == 0, result.output
    assert "不提交" in result.output


def test_resume_preflight_includes_auto_fix_after_monitoring():
    scene = ScenePlan(
        scene_id=1,
        title="scene",
        duration_seconds=10,
        purpose="test",
        math_concept="circle",
        visual_design="dark",
        camera_movement="fixed",
        visual_flow=["show"],
        key_moments=["pause"],
        computation="radius=1",
    )
    manifest = RunManifest(
        run_id="20260728-120000-1234abcd",
        status="running",
        state="MONITORING",
        auto_fix=True,
        user_prompt="prompt",
        output_path="/tmp/output.mp4",
        scenes={
            1: StoredSceneState(
                plan=scene,
                plan_ready=True,
                reviewed=True,
                slurm_job=StoredSlurmJob(
                    job_id="123",
                    scene_id=1,
                    script_path="scenes/render_1.sh",
                    log_out="logs/scene_1.out",
                    log_err="logs/scene_1.err",
                    media_dir="videos/scene_1",
                    scene_class_name="Scene1",
                    submitted_at=1,
                    status="RUNNING",
                ),
            )
        },
    )

    assert _manifest_requires_generation_apis(manifest) is True


def test_generate_resume_uses_manifest_dry_run_for_completion_message(monkeypatch, tmp_path):
    import kd1_anime.orchestrator as orchestrator_module

    run_id = "20260728-120000-1234abcd"
    root = tmp_path / "runs" / run_id
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
                plan=ScenePlan(
                    scene_id=1,
                    title="scene",
                    duration_seconds=10,
                    purpose="test",
                    math_concept="circle",
                    visual_design="dark",
                    camera_movement="fixed",
                    visual_flow=["show"],
                    key_moments=["pause"],
                    computation="radius=1",
                ),
                plan_ready=True,
                plan_reviewed=True,
                reviewed=True,
            )
        },
    )
    write_manifest(root / "manifest.json", manifest)
    monkeypatch.setattr(settings, "WORKSPACE_DIR", tmp_path)

    class FakeOrchestrator:
        def resume(self, requested_run_id, *, interactive=False):
            assert requested_run_id == run_id
            return None

    monkeypatch.setattr(orchestrator_module, "Orchestrator", FakeOrchestrator)

    result = CliRunner().invoke(app, ["generate", "--resume", run_id])

    assert result.exit_code == 0, result.output
    assert "Dry-run 完成" in result.output


def _running_manifest(tmp_path):
    from datetime import datetime, timedelta, timezone

    import kd1_anime.run_store as run_store

    run_id = "20260728-120000-1234abcd"
    root = tmp_path / "runs" / run_id
    root.mkdir(parents=True)
    plan = ScenePlan(
        scene_id=1,
        title="scene",
        duration_seconds=10,
        purpose="test",
        math_concept="circle",
        visual_design="dark",
        camera_movement="fixed",
        visual_flow=["show"],
        key_moments=["pause"],
        computation="radius=1",
    )
    old = datetime.now(timezone.utc) - timedelta(days=60)
    manifest = RunManifest(
        run_id=run_id,
        status="running",
        state="MONITORING",
        user_prompt="prompt",
        output_path=str((root / "output.mp4").resolve()),
        created_at=old,
        updated_at=old,
        scenes={
            1: StoredSceneState(
                plan=plan,
                plan_ready=True,
                reviewed=True,
                slurm_job=StoredSlurmJob(
                    job_id="123",
                    scene_id=1,
                    script_path="scenes/render_1.sh",
                    log_out="logs/scene_1.out",
                    log_err="logs/scene_1.err",
                    media_dir="videos/scene_1",
                    scene_class_name="Scene1",
                    submitted_at=1,
                    status="RUNNING",
                ),
            )
        },
    )
    return root, manifest, old, run_store


def test_clean_include_running_cancels_jobs_before_deleting(monkeypatch, tmp_path):
    root, manifest, old, run_store = _running_manifest(tmp_path)
    monkeypatch.setattr(run_store, "utc_now", lambda: old)
    write_manifest(root / "manifest.json", manifest)
    monkeypatch.setattr(settings, "WORKSPACE_DIR", tmp_path)
    cancelled = []

    from kd1_anime.cluster.slurm import SlurmDispatcher

    monkeypatch.setattr(
        SlurmDispatcher,
        "cancel_job",
        lambda self, job_id: cancelled.append(job_id) or True,
    )
    monkeypatch.setattr(
        SlurmDispatcher,
        "poll_all_statuses",
        lambda self, job_ids: {job_id: "RUNNING" for job_id in job_ids},
    )

    result = CliRunner().invoke(
        app,
        ["clean", "--older-than", "30d", "--yes", "--include-running"],
    )

    assert result.exit_code == 0, result.output
    assert cancelled == ["123"]
    assert not root.exists()


def test_clean_include_running_keeps_run_when_job_cancellation_fails(monkeypatch, tmp_path):
    root, manifest, old, run_store = _running_manifest(tmp_path)
    monkeypatch.setattr(run_store, "utc_now", lambda: old)
    write_manifest(root / "manifest.json", manifest)
    monkeypatch.setattr(settings, "WORKSPACE_DIR", tmp_path)

    from kd1_anime.cluster.slurm import SlurmDispatcher

    monkeypatch.setattr(SlurmDispatcher, "cancel_job", lambda self, job_id: False)
    monkeypatch.setattr(
        SlurmDispatcher,
        "poll_all_statuses",
        lambda self, job_ids: {job_id: "RUNNING" for job_id in job_ids},
    )

    result = CliRunner().invoke(
        app,
        ["clean", "--older-than", "30d", "--yes", "--include-running"],
    )

    assert result.exit_code == 0, result.output
    assert root.exists()
    assert "拒绝删除" in result.output


def test_clean_cancels_job_in_failed_run_too(monkeypatch, tmp_path):
    root, manifest, old, run_store = _running_manifest(tmp_path)
    manifest.status = "failed"
    monkeypatch.setattr(run_store, "utc_now", lambda: old)
    write_manifest(root / "manifest.json", manifest)
    monkeypatch.setattr(settings, "WORKSPACE_DIR", tmp_path)
    cancelled = []

    from kd1_anime.cluster.slurm import SlurmDispatcher

    monkeypatch.setattr(
        SlurmDispatcher,
        "cancel_job",
        lambda self, job_id: cancelled.append(job_id) or True,
    )
    monkeypatch.setattr(
        SlurmDispatcher,
        "poll_all_statuses",
        lambda self, job_ids: {job_id: "RUNNING" for job_id in job_ids},
    )

    result = CliRunner().invoke(app, ["clean", "--older-than", "30d", "--yes"])

    assert result.exit_code == 0, result.output
    assert cancelled == ["123"]
    assert not root.exists()


def test_clean_keeps_run_when_job_status_cannot_be_confirmed(monkeypatch, tmp_path):
    root, manifest, old, run_store = _running_manifest(tmp_path)
    monkeypatch.setattr(run_store, "utc_now", lambda: old)
    write_manifest(root / "manifest.json", manifest)
    monkeypatch.setattr(settings, "WORKSPACE_DIR", tmp_path)

    from kd1_anime.cluster.slurm import SlurmDispatcher

    monkeypatch.setattr(
        SlurmDispatcher,
        "poll_all_statuses",
        lambda self, job_ids: {job_id: "UNKNOWN" for job_id in job_ids},
    )

    result = CliRunner().invoke(
        app,
        ["clean", "--older-than", "30d", "--yes", "--include-running"],
    )

    assert result.exit_code == 0, result.output
    assert root.exists()
    assert "无法确认" in result.output


def test_plan_command_reviews_generated_plans(monkeypatch):
    from kd1_anime.agents.plan_reviewer import PlanReviewResult
    from kd1_anime.agents.planner import ContinuityBible, SceneOutline

    outline = SceneOutline(
        scene_id=1,
        title="圆",
        duration_seconds=10,
        purpose="展示圆",
        math_concept="圆的面积",
    )
    planned = ScenePlan(
        scene_id=1,
        title=outline.title,
        duration_seconds=outline.duration_seconds,
        purpose=outline.purpose,
        math_concept=outline.math_concept,
        visual_design="统一背景",
        camera_movement="固定机位",
        visual_flow=["显示圆"],
        key_moments=["停顿"],
        computation="半径=1",
        opening_state=["画面为空"],
        closing_state=["圆和面积结论保留"],
        transition_in="从空画面淡入",
        transition_out="保留圆和结论交给下一场景",
    )
    calls = []

    class FakePlanner:
        def plan_outline(self, prompt, **kwargs):
            return [outline]

        def plan_continuity_bible(self, prompt, outlines, **kwargs):
            return ContinuityBible()

        def plan_detail(self, outline, outlines, prompt, **kwargs):
            return planned

    class FakePlanReviewer:
        def review(self, plan, **kwargs):
            calls.append((plan.scene_id, kwargs["all_plans"]))
            return PlanReviewResult(
                is_valid=True,
                severity="info",
                summary="通过",
                issues=[],
            )

    monkeypatch.setattr("kd1_anime.cli._ensure_generation_apis", lambda **kwargs: None)
    monkeypatch.setattr("kd1_anime.agents.planner.PlannerAgent", FakePlanner)
    monkeypatch.setattr("kd1_anime.agents.plan_reviewer.PlanReviewerAgent", FakePlanReviewer)
    result = CliRunner().invoke(app, ["plan", "解释圆"])

    assert result.exit_code == 0, result.output
    assert "已完成计划审查" in result.output
    assert calls == [(1, [planned])]
