import pytest
import typer
from typer.testing import CliRunner

from kd1_anime.cli import _print_comparison, app
from kd1_anime.config import settings
from kd1_anime.eval.metrics import ComparisonResult, EvalResult


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
