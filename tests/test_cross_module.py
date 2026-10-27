"""跨持久化、后端和恢复边界的回归测试。"""

from __future__ import annotations

import ast
import hashlib
import time
from dataclasses import fields
from pathlib import Path

from kd1_anime.agents.planner import ScenePlan
from kd1_anime.cluster.slurm import SlurmJob
from kd1_anime.config import settings
from kd1_anime.domain import state as state_module
from kd1_anime.domain.scene_fields import SCENE_FIELDS, SCENE_GROUPS
from kd1_anime.domain.state import (
    CodeRevisionState,
    ExecutionState,
    LifecycleState,
    PlanningState,
    VerificationState,
)
from kd1_anime.orchestrator import Orchestrator, PipelineContext, RunPaths, SceneState, State
from kd1_anime.rendering import RenderProfile
from kd1_anime.run_store import RunManifest, RunRepository, StoredSceneState, write_manifest

RUN_ID = "20260905-220000-1234abcd"
CODE = "from manim import *\nclass Demo(Scene):\n    def construct(self): self.wait()\n"


def make_plan() -> ScenePlan:
    return ScenePlan(
        scene_id=1,
        title="本地恢复",
        duration_seconds=1,
        purpose="验证恢复",
        math_concept="等待",
        visual_design="简洁",
        camera_movement="固定",
        visual_flow=["等待"],
        key_moments=["等待"],
        computation="无",
    )


def test_local_backend_identity_round_trips_through_manifest(tmp_path: Path):
    workspace = tmp_path / "workspace"
    root = workspace / "runs" / RUN_ID
    scenes = root / "scenes"
    scenes.mkdir(parents=True)
    code_path = scenes / "scene_1.py"
    code_path.write_text(CODE, encoding="utf-8")
    profile = RenderProfile.current()
    job = SlurmJob(
        job_id="local-abcdef123456",
        scene_id=1,
        script_path=code_path,
        log_out=root / "logs" / "scene.out",
        log_err=root / "logs" / "scene.err",
        media_dir=root / "videos" / "scene_1" / "attempt_abcdef123456",
        scene_class_name="Demo",
        submitted_at=time.time(),
        code_sha256=hashlib.sha256(CODE.encode()).hexdigest(),
        render_profile=profile,
        backend="local",
    )
    context = PipelineContext(
        "prompt",
        paths=RunPaths(
            RUN_ID,
            root,
            scenes,
            root / "logs",
            root / "videos",
            root / "output.mp4",
        ),
        backend="local",
        scene_states={
            1: SceneState(
                plan=make_plan(),
                code=CODE,
                class_name="Demo",
                plan_ready=True,
                reviewed=True,
                slurm_job=job,
            )
        },
    )
    orchestrator = Orchestrator()
    orchestrator._checkpoint(context, State.MONITORING)

    manifest = RunRepository(workspace).load(RUN_ID)
    assert manifest.backend == "local"
    assert manifest.scenes[1].slurm_job is not None
    assert manifest.scenes[1].slurm_job.backend == "local"
    restored = Orchestrator._context_from_manifest(manifest, root)
    assert restored.backend == "local"
    assert restored.scene_states[1].slurm_job is not None
    assert restored.scene_states[1].slurm_job.backend == "local"


def test_local_resume_preserves_unconfirmed_job_and_blocks_resubmission(
    monkeypatch, tmp_path: Path
):
    workspace = tmp_path / "workspace"
    root = workspace / "runs" / RUN_ID
    scenes = root / "scenes"
    scenes.mkdir(parents=True)
    (scenes / "scene_1.py").write_text(CODE, encoding="utf-8")
    from kd1_anime.objects import ObjectStore

    reference = ObjectStore(root).source(1, CODE)
    code_hash = hashlib.sha256(CODE.encode()).hexdigest()
    job = {
        "job_id": "local-abcdef123456",
        "scene_id": 1,
        "script_path": "scenes/scene_1.py",
        "log_out": "logs/scene.out",
        "log_err": "logs/scene.err",
        "media_dir": "videos/scene_1/attempt_abcdef123456",
        "scene_class_name": "Demo",
        "submitted_at": time.time(),
        "code_sha256": code_hash,
        "render_profile": RenderProfile.current().model_dump(mode="json"),
        "status": "RUNNING",
        "backend": "local",
    }
    manifest = RunManifest(
        run_id=RUN_ID,
        state="MONITORING",
        status="running",
        user_prompt="prompt",
        output_path=str((root / "output.mp4").resolve()),
        backend="local",
        scenes={
            1: StoredSceneState(
                plan=make_plan(),
                code_file=reference,
                code_sha256=code_hash,
                class_name="Demo",
                plan_ready=True,
                plan_reviewed=True,
                reviewed=True,
                slurm_job=job,
            )
        },
    )
    manifest.scenes[1].slurm_job.execution_sha256 = manifest.execution_spec.digest()
    write_manifest(root / "manifest.json", manifest)
    monkeypatch.setattr(settings, "WORKSPACE_DIR", workspace)
    captured = {}

    def fake_execute(self, context, state):
        captured["context"] = context
        captured["state"] = state
        return None

    monkeypatch.setattr(Orchestrator, "_execute", fake_execute)
    assert Orchestrator().resume(RUN_ID) is None
    assert captured["context"].backend == "local"
    restored = captured["context"].scene_states[1]
    assert restored.slurm_job.job_id == job["job_id"]
    assert restored.failed and restored.give_up
    assert "禁止自动重提" in restored.failure_reason
    assert captured["state"] is State.MONITORING


def test_scene_metadata_covers_each_runtime_field_once():
    groups = {
        "planning": PlanningState,
        "code_revision": CodeRevisionState,
        "execution": ExecutionState,
        "verification": VerificationState,
        "lifecycle": LifecycleState,
    }
    assert tuple(groups) == SCENE_GROUPS
    expected = {item.name: group for group, model in groups.items() for item in fields(model)}
    assert len(expected) == sum(len(fields(model)) for model in groups.values())
    assert {name: metadata.domain_group for name, metadata in SCENE_FIELDS.items()} == expected
    scene = SceneState(plan=make_plan())
    for name, metadata in SCENE_FIELDS.items():
        assert getattr(scene, name) is getattr(getattr(scene, metadata.domain_group), name)
    scene.review_round = 2
    assert scene.code_revision.review_round == 2


def test_scene_state_rejects_unknown_compatibility_fields():
    import pytest

    with pytest.raises(TypeError, match="Unknown scene state field"):
        SceneState(plan=make_plan(), invented_field=True)
    scene = SceneState(plan=make_plan())
    with pytest.raises(AttributeError, match="invented_field"):
        scene.invented_field = True
    with pytest.raises(AttributeError, match="invented_field"):
        _ = scene.invented_field


def test_domain_has_no_storage_imports_or_incremental_manifest():
    domain_root = Path(state_module.__file__).parent
    forbidden = {"run_store", "kd1_anime.run_store"}
    for source in domain_root.rglob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.module not in forbidden, source
            elif isinstance(node, ast.Import):
                assert not any(alias.name in forbidden for alias in node.names), source
    assert "base_manifest" not in {item.name for item in fields(PipelineContext)}
