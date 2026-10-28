import pytest
from test_agent_vertical_slice import agent_boundary as agent_boundary

from kd1_anime.agent_runtime.runtime import AgentRuntime
from kd1_anime.application import Application
from kd1_anime.run_store import RunRepository


def test_incremental_reuses_current_verified_artifact_and_resumes_without_base(
    agent_boundary, monkeypatch
):
    boundary, dependencies = agent_boundary
    monkeypatch.setattr(
        "kd1_anime.application._agent_runtime", lambda: AgentRuntime(dependencies=dependencies)
    )
    app = Application()
    base_video = app.generate("Show a pause", approve_plan=True)
    repository = RunRepository(boundary.workspace)
    base_id = next((boundary.workspace / "runs").iterdir()).name
    base_before = repository.manifest_path(base_id).read_bytes()
    output = app.run_incremental("Show a pause", base_id, approve_plan=True)
    runs = [entry for entry in (boundary.workspace / "runs").iterdir() if entry.name != base_id]
    assert len(runs) == 1
    manifest = repository.load(runs[0].name)
    assert base_video.is_file() and output.is_file() and manifest.status == "completed"
    assert manifest.agent.session_id != repository.load(base_id).agent.session_id
    assert manifest.request_budget.reservations is not None
    assert manifest.scenes[1].artifact.origin == "reused"
    assert manifest.scenes[1].slurm_job is None and manifest.scenes[1].artifact.job_id is None
    assert manifest.agent.reuse_receipts[1].sha256
    assert len(boundary.backend.sources) == 1
    assert repository.manifest_path(base_id).read_bytes() == base_before
    import shutil

    shutil.rmtree(repository.run_root(base_id))
    assert app.resume(manifest.run_id) == output
    assert len(boundary.backend.sources) == 1


def test_cli_incremental_forwards_plan_approval_and_output(agent_boundary, monkeypatch, tmp_path):
    from typer.testing import CliRunner

    from kd1_anime.cli import app as cli

    boundary, dependencies = agent_boundary
    monkeypatch.setattr(
        "kd1_anime.application._agent_runtime", lambda: AgentRuntime(dependencies=dependencies)
    )
    runner = CliRunner()
    assert runner.invoke(cli, ["generate", "Show a pause", "--approve-plan"]).exit_code == 0
    base = next((boundary.workspace / "runs").iterdir()).name
    target = tmp_path / "custom.mp4"
    result = runner.invoke(
        cli,
        [
            "generate",
            "Show a pause",
            "--incremental",
            base,
            "--approve-plan",
            "--output",
            str(target),
        ],
    )
    assert result.exit_code == 0, result.output
    assert target.is_file() and len(boundary.backend.sources) == 1
    new_run = next(path for path in (boundary.workspace / "runs").iterdir() if path.name != base)
    manifest = RunRepository(boundary.workspace).load(new_run.name)
    assert manifest.status == "completed" and manifest.scenes[1].artifact.origin == "reused"


def test_explicit_retry_preserves_budget_and_requires_new_render(agent_boundary, monkeypatch):
    boundary, dependencies = agent_boundary
    runtime = AgentRuntime(dependencies=dependencies)
    interrupted = []

    def stop_after_review(manifest):
        if manifest.state == "DISPATCHING" and not interrupted:
            interrupted.append(True)
            raise OSError("interrupt after reviewed code")

    dependencies.after_commit = stop_after_review
    with pytest.raises(OSError):
        runtime.generate("Show a pause", approve_plan=True)
    assert interrupted
    root = next((boundary.workspace / "runs").iterdir())
    repo = RunRepository(boundary.workspace)
    before = repo.load(root.name)
    reservations = {item.request_id for item in before.request_budget.reservations}
    dependencies.after_commit = None
    monkeypatch.setattr(
        "kd1_anime.application._agent_runtime", lambda: AgentRuntime(dependencies=dependencies)
    )
    output = Application().retry(root.name, 1)
    after = repo.load(root.name)
    assert output is not None and output.is_file() and after.status == "completed"
    assert reservations <= {item.request_id for item in after.request_budget.reservations}
    assert after.request_budget.run_limit == before.request_budget.run_limit
    assert after.scenes[1].revision > before.scenes[1].revision
    assert len(boundary.backend.sources) == 1
