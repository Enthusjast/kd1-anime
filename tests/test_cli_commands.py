from typer.testing import CliRunner

from kd1_anime.cli import _print_comparison, app
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
