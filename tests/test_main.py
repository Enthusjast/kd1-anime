from test_agent_vertical_slice import agent_boundary as agent_boundary
from test_application import SOURCE
from typer.testing import CliRunner

from kd1_anime.agent_runtime.runtime import AgentRuntime
from kd1_anime.cli import app
from kd1_anime.rendering import sha256_file
from kd1_anime.run_store import RunRepository


def test_render_copies_source_into_private_run_directory(agent_boundary, monkeypatch, tmp_path):
    boundary, dependencies = agent_boundary
    source = tmp_path / "external" / "scene.py"
    source.parent.mkdir()
    source.write_text(SOURCE, encoding="utf-8")
    monkeypatch.setattr(
        "kd1_anime.application._agent_runtime", lambda: AgentRuntime(dependencies=dependencies)
    )
    result = CliRunner().invoke(app, ["render", str(source), "--class", "Pause"])
    assert result.exit_code == 0, result.output
    (copied,) = boundary.backend.sources
    assert copied != source and copied.parent.name == "scene_1"
    assert copied.parent.parent.name == "revisions" and copied.stem == sha256_file(source)
    assert copied.read_bytes() == source.read_bytes()
    assert copied.stat().st_mode & 0o777 == 0o600
    manifest = RunRepository(boundary.workspace).load(copied.parents[3].name)
    assert manifest.schema_version == 11 and manifest.direct_render
    assert manifest.state == "MONITORING" and manifest.status == "running"
    assert manifest.auto_fix is False and not manifest.request_budget.reservations
    assert manifest.scenes[1].slurm_job.job_id in boundary.backend.jobs
    assert manifest.run_id in result.output
