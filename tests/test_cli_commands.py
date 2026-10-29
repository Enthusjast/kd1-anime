import json

import pytest
import typer
from test_application import application_boundary as application_boundary
from typer.testing import CliRunner

from kd1_anime.agents.planner import ScenePlan
from kd1_anime.cli import _manifest_requires_generation_apis, _print_comparison, app
from kd1_anime.config import settings
from kd1_anime.domain.planning import LessonSpec, PlanningDraft, TeachingGraph
from kd1_anime.eval.metrics import ComparisonResult, EvalResult
from kd1_anime.run_store import StoredSlurmJob, write_manifest


def test_cli_registers_all_public_commands():
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0, result.output
    for command in (
        "generate",
        "resume",
        "batch",
        "doctor",
        "evaluate",
        "test-llm",
        "stats",
        "rag",
    ):
        assert command in result.output


def test_version_uses_source_package_version():
    result = CliRunner().invoke(app, ["version"])

    assert result.exit_code == 0, result.output
    assert "kd1-anime v0.4.0" in result.output


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


def test_generate_resume_does_not_require_a_dummy_prompt(application_boundary, monkeypatch):
    boundary = application_boundary
    output = boundary.workspace / "output_final.mp4"
    boundary.app.generate("prompt", output_path=output, approve_plan=True)
    run_id = next((boundary.workspace / "runs").iterdir()).name
    monkeypatch.setattr("kd1_anime.application._agent_runtime", boundary.runtime_factory)
    monkeypatch.setattr(settings, "LLM_API_KEY", "")
    monkeypatch.setattr(settings, "LLM_MODEL", "")

    result = CliRunner().invoke(app, ["generate", "--resume", run_id])

    assert result.exit_code == 0, result.output
    assert "成功" in result.output
    assert output.read_bytes() == b"merged:deterministic-backend-video"


@pytest.mark.parametrize("version", range(4, 11))
@pytest.mark.parametrize("entry", ["generate_resume", "resume", "retry", "incremental"])
def test_execution_commands_reject_read_only_versions_before_side_effects(
    monkeypatch, tmp_path, version, entry
):
    run_id = "20260728-120000-1234abcd"
    root = tmp_path / "runs" / run_id
    root.mkdir(parents=True)
    manifest_path = root / "manifest.json"
    original = json.dumps({"schema_version": version, "run_id": run_id}).encode()
    manifest_path.write_bytes(original)
    monkeypatch.setattr(settings, "WORKSPACE_DIR", tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("Read-only commands must reject before probes, runtime or run locks")

    monkeypatch.setattr("kd1_anime.cli._ensure_generation_apis", forbidden)
    monkeypatch.setattr("kd1_anime.cli._ensure_llm_api_available", forbidden)
    monkeypatch.setattr("kd1_anime.cli._ensure_rag_apis_available", forbidden)
    monkeypatch.setattr("kd1_anime.application._agent_runtime", forbidden)
    monkeypatch.setattr("kd1_anime.run_store.lock_run", forbidden)
    commands = {
        "generate_resume": ["generate", "--resume", run_id],
        "resume": ["resume", run_id],
        "retry": ["retry", run_id, "--scene-id", "1"],
        "incremental": ["generate", "Show a pause", "--incremental", run_id],
    }
    before = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    result = CliRunner().invoke(app, commands[entry])
    assert result.exit_code == 1, result.output
    assert "只读" in result.output
    after = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert manifest_path.read_bytes() == original


def test_global_dry_run_reports_completion_without_video(monkeypatch):
    import kd1_anime.application as application_module

    monkeypatch.setattr("kd1_anime.cli._ensure_llm_api_available", lambda: None)
    calls = []

    class FakeApplication:
        def generate(self, prompt, *, dry_run=False, callback=None):
            calls.append((prompt, dry_run))
            return None

    monkeypatch.setattr(application_module, "Application", FakeApplication)
    monkeypatch.setattr(settings, "LLM_API_KEY", "key")
    monkeypatch.setattr(settings, "LLM_MODEL", "model")

    result = CliRunner().invoke(app, ["--dry-run", "generate", "demo"])

    assert result.exit_code == 0, result.output
    assert calls == [("demo", True)]
    assert "Dry-run 完成" in result.output
    assert "输出文件" not in result.output


def test_generate_strict_flag_overrides_configured_generation_mode(monkeypatch):
    import kd1_anime.application as application_module

    calls = []

    class FakeApplication:
        def generate(self, prompt, *, dry_run=False, callback=None):
            calls.append((prompt, dry_run, settings.GENERATION_MODE))
            return None

    monkeypatch.setattr(application_module, "Application", FakeApplication)
    monkeypatch.setattr("kd1_anime.cli._ensure_llm_api_available", lambda: None)
    monkeypatch.setattr(settings, "LLM_API_KEY", "key")
    monkeypatch.setattr(settings, "LLM_MODEL", "model")
    monkeypatch.setattr(settings, "GENERATION_MODE", "relaxed")

    result = CliRunner().invoke(app, ["generate", "demo", "--dry-run", "--strict"])

    assert result.exit_code == 0, result.output
    assert calls == [("demo", True, "strict")]


def test_generate_exits_before_pipeline_when_healthcheck_fails(monkeypatch):
    monkeypatch.setattr(settings, "LLM_API_KEY", "key")
    monkeypatch.setattr(settings, "LLM_MODEL", "model")

    def fail_healthcheck():
        raise typer.Exit(1)

    monkeypatch.setattr("kd1_anime.cli._ensure_llm_api_available", fail_healthcheck)

    result = CliRunner().invoke(app, ["generate", "demo", "--dry-run"])

    assert result.exit_code == 1


@pytest.mark.parametrize("command", ["generate", "plan"])
@pytest.mark.parametrize(
    ("failure", "exit_code", "message"),
    [
        (RuntimeError("endpoint [offline]"), 1, "endpoint [offline]"),
        (KeyboardInterrupt(), 130, "用户中断"),
    ],
)
def test_generation_commands_report_failure_and_cancellation(
    monkeypatch, command, failure, exit_code, message
):
    class FailingApplication:
        def generate(self, prompt, **kwargs):
            raise failure

        def plan_only(self, prompt, **kwargs):
            raise failure

    monkeypatch.setattr("kd1_anime.application.Application", FailingApplication)
    monkeypatch.setattr("kd1_anime.cli._ensure_llm_api_available", lambda: None)
    monkeypatch.setattr("kd1_anime.cli._ensure_generation_apis", lambda **kwargs: None)

    result = CliRunner().invoke(app, [command, "demo"])

    assert result.exit_code == exit_code, result.output
    assert message in result.output
    assert "成功" not in result.output
    assert "场景规划结果" not in result.output


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


def test_evaluate_scene_id_uses_visual_scene_artifact_path(application_boundary, monkeypatch):
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

    boundary = application_boundary
    boundary.app.generate("Show a one-second pause", approve_plan=True)
    run_id = next((boundary.workspace / "runs").iterdir()).name
    result = CliRunner().invoke(
        app,
        ["evaluate", run_id, "--scene-id", "1", "--desc", "平方差"],
    )

    assert result.exit_code == 0, result.output
    assert captured["init"]["enable_visual_eval"] is True
    assert captured["call"] == (run_id, 1, "平方差")


def test_generate_does_not_clear_configured_overwrite_setting(monkeypatch):
    import kd1_anime.application as application_module

    monkeypatch.setattr("kd1_anime.cli._ensure_llm_api_available", lambda: None)

    class FakeApplication:
        def generate(self, prompt, *, dry_run=False, callback=None):
            return None

    monkeypatch.setattr(application_module, "Application", FakeApplication)
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


def test_resume_preflight_includes_auto_fix_after_monitoring(application_boundary):
    _, manifest, _, _ = _running_manifest(application_boundary)
    manifest.auto_fix = True

    assert _manifest_requires_generation_apis(manifest) is True


def test_generate_resume_uses_manifest_dry_run_for_completion_message(
    application_boundary, monkeypatch
):
    boundary = application_boundary
    boundary.app.generate("prompt", dry_run=True, approve_plan=True)
    run_id = next((boundary.workspace / "runs").iterdir()).name
    monkeypatch.setattr("kd1_anime.application._agent_runtime", boundary.runtime_factory)

    result = CliRunner().invoke(app, ["generate", "--resume", run_id])

    assert result.exit_code == 0, result.output
    assert "Dry-run 完成" in result.output


def _running_manifest(boundary):
    from datetime import datetime, timedelta, timezone

    import kd1_anime.run_store as run_store
    from kd1_anime.run_store import RunRepository

    boundary.app.plan_only("prompt")
    root = next((boundary.workspace / "runs").iterdir())
    manifest = RunRepository(boundary.workspace).load(root.name)
    old = datetime.now(timezone.utc) - timedelta(days=60)
    manifest.status = "running"
    manifest.state = "MONITORING"
    manifest.created_at = manifest.updated_at = old
    manifest.scenes[1].reviewed = True
    manifest.scenes[1].slurm_job = StoredSlurmJob(
        job_id="123",
        scene_id=1,
        script_path="scenes/render_1.sh",
        log_out="logs/scene_1.out",
        log_err="logs/scene_1.err",
        media_dir="videos/scene_1",
        scene_class_name="Scene1",
        submitted_at=1,
        status="RUNNING",
    )
    return root, manifest, old, run_store


def test_clean_include_running_cancels_jobs_before_deleting(
    application_boundary, monkeypatch, tmp_path
):
    root, manifest, old, run_store = _running_manifest(application_boundary)
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


def test_clean_include_running_keeps_run_when_job_cancellation_fails(
    application_boundary, monkeypatch, tmp_path
):
    root, manifest, old, run_store = _running_manifest(application_boundary)
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


def test_clean_cancels_job_in_failed_run_too(application_boundary, monkeypatch, tmp_path):
    root, manifest, old, run_store = _running_manifest(application_boundary)
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


def test_clean_keeps_run_when_job_status_cannot_be_confirmed(
    application_boundary, monkeypatch, tmp_path
):
    root, manifest, old, run_store = _running_manifest(application_boundary)
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


def test_plan_command_reviews_generated_plans(application_boundary, monkeypatch):
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
        def plan_draft(self, prompt, **kwargs):
            return PlanningDraft(
                items=self.plan_outline(prompt, **kwargs),
                lesson_spec=LessonSpec(topic=prompt),
                teaching_graph=TeachingGraph(),
            )

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

    class FakeContinuityReviewer:
        def review(self, *args, **kwargs):
            from kd1_anime.agents.continuity import ContinuityReviewResult

            return ContinuityReviewResult(
                is_valid=True,
                severity="info",
                summary="通过",
                issues=[],
            )

    monkeypatch.setattr("kd1_anime.cli._ensure_generation_apis", lambda **kwargs: None)
    dependencies = application_boundary.dependencies
    dependencies.planner_factory = FakePlanner
    dependencies.plan_reviewer_factory = FakePlanReviewer
    dependencies.continuity_reviewer_factory = FakeContinuityReviewer
    monkeypatch.setattr(
        "kd1_anime.application._agent_runtime", application_boundary.runtime_factory
    )
    result = CliRunner().invoke(app, ["plan", "解释圆"])

    assert result.exit_code == 0, result.output
    assert "已完成计划审查" in result.output
    assert calls == [(1, [planned])]


def test_plan_command_exports_structured_plan(application_boundary, monkeypatch, tmp_path):
    from kd1_anime.agents.planner import ContinuityBible, SceneOutline
    from kd1_anime.run_store import RunRepository

    outline = SceneOutline(
        scene_id=1,
        title="圆",
        duration_seconds=10,
        purpose="展示圆",
        math_concept="圆的面积",
    )
    plan = ScenePlan(
        scene_id=1,
        title="圆",
        duration_seconds=10,
        purpose="展示圆",
        math_concept="圆的面积",
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

    class FakePlanner:
        def plan_draft(self, prompt, **kwargs):
            return PlanningDraft(
                items=self.plan_outline(prompt, **kwargs),
                lesson_spec=LessonSpec(topic=prompt),
                teaching_graph=TeachingGraph(),
            )

        def plan_outline(self, prompt, **kwargs):
            return [outline]

        def plan_continuity_bible(self, prompt, outlines, **kwargs):
            return ContinuityBible()

        def plan_detail(self, outline, outlines, prompt, **kwargs):
            return plan

    monkeypatch.setattr("kd1_anime.cli._ensure_generation_apis", lambda **kwargs: None)
    application_boundary.dependencies.planner_factory = FakePlanner
    monkeypatch.setattr(
        "kd1_anime.application._agent_runtime", application_boundary.runtime_factory
    )
    output = tmp_path / "plan.json"

    result = CliRunner().invoke(
        app,
        ["plan", "解释圆", "--no-review", "--output", str(output)],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["items"][0]["title"] == "圆"
    assert payload["user_prompt"] == "解释圆"
    manifest = RunRepository(settings.WORKSPACE_DIR).load(payload["run_id"])
    assert payload["lesson_spec"] == manifest.lesson_spec.model_dump(mode="json")
    assert payload["teaching_graph"] == manifest.teaching_graph.model_dump(mode="json")
    assert payload["continuity_bible"] == manifest.continuity_bible.model_dump(mode="json")
    assert payload["items"] == [
        scene.plan.model_dump(mode="json") for scene in manifest.scenes.values()
    ]
    assert "预览模式，未执行计划审查" in result.output
