import json

import pytest
from test_agent_vertical_slice import agent_boundary as agent_boundary
from test_application import SOURCE, VIDEO

from kd1_anime.agent_runtime.runtime import AgentRuntime
from kd1_anime.run_store import RunRepository, restore_run_path


def test_direct_render_copies_private_source_and_never_calls_models(agent_boundary):
    boundary, dependencies = agent_boundary
    runtime = AgentRuntime(dependencies=dependencies)
    job, output, run_id = runtime.submit_existing_scene(SOURCE, "Pause", wait=True)
    manifest = RunRepository(boundary.workspace).load(run_id)
    assert manifest.schema_version == 11 and manifest.direct_render
    assert manifest.status == "completed" and job.job_id == manifest.scenes[1].slurm_job.job_id
    assert output.read_bytes() == b"merged:" + VIDEO
    assert not manifest.request_budget.reservations and not boundary.calls
    source = restore_run_path(runtime.root, manifest.scenes[1].code_file)
    assert source.read_text() == SOURCE and source.stat().st_mode & 0o777 == 0o600
    assert AgentRuntime(dependencies=dependencies).resume(run_id) == output
    assert len(boundary.backend.sources) == 1


def test_direct_render_no_wait_persists_exact_job_before_return(agent_boundary):
    boundary, dependencies = agent_boundary
    runtime = AgentRuntime(dependencies=dependencies)
    job, output, run_id = runtime.submit_existing_scene(SOURCE, "Pause", wait=False)
    raw = json.loads((runtime.root / "manifest.json").read_text())
    assert output is None and raw["scenes"]["1"]["slurm_job"]["job_id"] == job.job_id
    assert raw["agent"]["submissions"][0]["attempt"]["job_id"] == job.job_id
    completed = AgentRuntime(dependencies=dependencies).resume(run_id)
    assert completed.read_bytes() == b"merged:" + VIDEO
    assert not boundary.calls and len(boundary.backend.sources) == 1


def test_invalid_direct_source_never_submits(agent_boundary):
    boundary, dependencies = agent_boundary
    with pytest.raises(ValueError):
        AgentRuntime(dependencies=dependencies).submit_existing_scene(
            SOURCE + "\nimport subprocess\n", "Pause", wait=True
        )
    assert not boundary.backend.sources and not boundary.calls
