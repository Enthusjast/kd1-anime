from typer.testing import CliRunner

from kd1_anime.cli import _print_comparison, app
from kd1_anime.config import settings
from kd1_anime.eval.metrics import ComparisonResult, EvalResult


def test_cli_registers_all_public_commands():
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0, result.output
    for command in ("generate", "resume", "batch", "doctor", "evaluate", "test-llm"):
        assert command in result.output


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


def test_generate_does_not_clear_configured_overwrite_setting(monkeypatch):
    import kd1_anime.orchestrator as orchestrator_module

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
