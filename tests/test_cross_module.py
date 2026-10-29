"""Cross persistence, backend and recovery regressions through immutable contracts."""

import ast
from pathlib import Path

import pytest
from test_agent_vertical_slice import agent_boundary as agent_boundary

from kd1_anime.agent_runtime.runtime import AgentRuntime
from kd1_anime.application import Application
from kd1_anime.config import settings
from kd1_anime.domain import state as state_module
from kd1_anime.pipeline.contracts import SceneInputs
from kd1_anime.run_store import RunRepository, restore_render_job, store_render_job


def scene_inputs(manifest, root, scene_id=1):
    """Build an explicit worker snapshot from a durable public repository result."""
    return SceneInputs(
        run_id=manifest.run_id,
        root=root,
        config=manifest.run_config,
        render_profile=manifest.render_profile,
        execution_spec=manifest.execution_spec,
        scene=manifest.scenes[scene_id].model_copy(deep=True),
        plans=tuple(item.plan for _, item in sorted(manifest.scenes.items())),
        lesson_spec=manifest.lesson_spec,
        teaching_graph=manifest.teaching_graph,
        continuity_bible=manifest.continuity_bible,
        element_manifest=manifest.element_manifest,
        state_ledger=manifest.state_ledger,
        visual_eval_profile=manifest.visual_eval_profile,
    )


def local_boundary(agent_boundary, monkeypatch):
    boundary, dependencies = agent_boundary
    monkeypatch.setattr(settings, "RENDER_BACKEND", "local")
    boundary.backend.name = "local"
    submit = boundary.backend.submit_scene

    def local_submit(*args, **kwargs):
        job = submit(*args, **kwargs)
        job.backend = "local"
        job.job_id = "local-abcdef123456"
        return job

    monkeypatch.setattr(boundary.backend, "submit_scene", local_submit)
    return boundary, dependencies


def test_local_backend_identity_round_trips_through_manifest(agent_boundary, monkeypatch):
    boundary, dependencies = local_boundary(agent_boundary, monkeypatch)
    app = Application(runtime_factory=lambda: AgentRuntime(dependencies=dependencies))
    output = app.generate("Show a one-second pause", approve_plan=True)
    assert output is not None
    root = next((boundary.workspace / "runs").iterdir())
    manifest = RunRepository(boundary.workspace).load_for_resume(root.name)
    assert manifest.backend == "local"
    stored = manifest.scenes[1].slurm_job
    assert stored.backend == "local" and stored.job_id == "local-abcdef123456"
    restored = restore_render_job(stored, root)
    assert restored.backend == "local"
    assert store_render_job(restored, root) == stored
    assert manifest.scenes[1].artifact.backend == "local"


def test_local_resume_preserves_unconfirmed_job_and_blocks_resubmission(
    agent_boundary, monkeypatch
):
    boundary, dependencies = local_boundary(agent_boundary, monkeypatch)
    interrupted = []

    def stop_after_submission(manifest):
        if not interrupted and manifest.scenes and manifest.scenes[1].slurm_job is not None:
            interrupted.append(True)
            raise OSError("client interrupted after durable submission")

    dependencies.after_commit = stop_after_submission
    app = Application(runtime_factory=lambda: AgentRuntime(dependencies=dependencies))
    with pytest.raises(OSError, match="durable submission"):
        app.generate("Show a one-second pause", approve_plan=True)
    dependencies.after_commit = None
    root = next((boundary.workspace / "runs").iterdir())
    before = RunRepository(boundary.workspace).load_for_resume(root.name)
    assert before.scenes[1].slurm_job.backend == "local"
    assert app.resume(root.name) is None
    saved = RunRepository(boundary.workspace).load_for_resume(root.name)
    assert saved.scenes[1].slurm_job.job_id == before.scenes[1].slurm_job.job_id
    assert not saved.scenes[1].rendered
    assert len(boundary.backend.sources) == 1
    assert not saved.final_video


def test_domain_has_no_storage_imports():
    domain_root = Path(state_module.__file__).parent
    forbidden = {"run_store", "kd1_anime.run_store"}
    for source in domain_root.rglob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.module not in forbidden, source
            elif isinstance(node, ast.Import):
                assert not any(alias.name in forbidden for alias in node.names), source
