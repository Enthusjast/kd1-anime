import pytest
from test_agent_vertical_slice import agent_boundary as agent_boundary
from typer.testing import CliRunner

from kd1_anime.agent_runtime.runtime import AgentRuntime
from kd1_anime.cli import app
from kd1_anime.run_store import RunRepository


@pytest.mark.parametrize("approve", [False, True])
def test_batch_requires_explicit_plan_approval_and_never_reads_worker_stdin(
    agent_boundary, monkeypatch, tmp_path, approve
):
    boundary, dependencies = agent_boundary
    monkeypatch.setattr(
        "kd1_anime.application._agent_runtime",
        lambda **kwargs: AgentRuntime(dependencies=dependencies, **kwargs),
    )
    monkeypatch.setattr(
        "builtins.input", lambda *a, **k: (_ for _ in ()).throw(AssertionError("worker read stdin"))
    )
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("Show a pause\n", encoding="utf-8")
    args = ["batch", str(prompts), "--dry-run"]
    if approve:
        args.append("--approve-plan")
    result = CliRunner().invoke(app, args)
    root = next((boundary.workspace / "runs").iterdir())
    manifest = RunRepository(boundary.workspace).load(root.name)
    if approve:
        assert result.exit_code == 0, result.output
        assert manifest.status == "dry_run_complete" and manifest.agent.pending_approval is None
    else:
        assert result.exit_code != 0
        assert manifest.agent.pending_approval is not None and manifest.status == "running"
        resumed = CliRunner().invoke(app, ["resume", root.name, "--approve-plan"])
        assert resumed.exit_code == 0, resumed.output
        completed = RunRepository(boundary.workspace).load(root.name)
        assert completed.status == "dry_run_complete" and completed.agent.pending_approval is None
    assert not boundary.backend.sources
