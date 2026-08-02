from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from typer.testing import CliRunner

import kd1_anime.run_store as store_module
from kd1_anime.agents.planner import ScenePlan
from kd1_anime.cli import app
from kd1_anime.config import settings
from kd1_anime.orchestrator import Orchestrator, PipelineContext, RunPaths, SceneState, State
from kd1_anime.run_store import RunManifest, RunRepository, write_manifest

RUN_ID = "20260728-120000-1234abcd"


def make_plan(scene_id: int = 1) -> ScenePlan:
    return ScenePlan(
        scene_id=scene_id,
        title=f"scene {scene_id}",
        duration_seconds=10,
        purpose="test",
        math_concept="circle",
        visual_design="dark",
        camera_movement="fixed",
        visual_flow=["show"],
        key_moments=["pause"],
        computation="radius=1",
    )


def make_paths(workspace: Path) -> RunPaths:
    root = workspace / "runs" / RUN_ID
    return RunPaths(
        RUN_ID,
        root,
        root / "scenes",
        root / "logs",
        root / "videos",
        root / "output_final.mp4",
    )


def test_checkpoint_round_trip_rejects_tampered_code(tmp_path):
    workspace = tmp_path / "workspace"
    paths = make_paths(workspace)
    paths.scenes.mkdir(parents=True)
    code = "from manim import *\nclass Demo(Scene):\n    def construct(self): self.wait()\n"
    code_path = paths.scenes / "scene_1.py"
    code_path.write_text(code, encoding="utf-8")
    ctx = PipelineContext(
        "prompt",
        paths=paths,
        scene_states={1: SceneState(plan=make_plan(), code=code, class_name="Demo")},
    )
    orchestrator = Orchestrator()

    orchestrator._checkpoint(ctx, State.CODING)

    manifest_path = paths.root / "manifest.json"
    assert manifest_path.stat().st_mode & 0o777 == 0o600
    manifest = RunRepository(workspace).load(RUN_ID)
    restored = orchestrator._context_from_manifest(manifest, paths.root)
    assert restored.scene_states[1].code == code

    code_path.write_text(code + "# changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="哈希"):
        orchestrator._context_from_manifest(manifest, paths.root)


def test_repository_rejects_symlink_run_directory(tmp_path):
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    outside.mkdir()
    runs = workspace / "runs"
    runs.mkdir(parents=True)
    (runs / RUN_ID).symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="真实目录"):
        RunRepository(workspace).load(RUN_ID)


def test_status_and_clean_remove_only_run_directory(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    root = workspace / "runs" / RUN_ID
    root.mkdir(parents=True)
    outside_output = tmp_path / "keep.mp4"
    outside_output.write_bytes(b"video")
    old = datetime.now(timezone.utc) - timedelta(days=60)
    manifest = RunManifest(
        run_id=RUN_ID,
        created_at=old,
        updated_at=old,
        status="completed",
        state="DONE",
        user_prompt="prompt",
        output_path=str(outside_output),
        final_video=str(outside_output),
    )
    monkeypatch.setattr(store_module, "utc_now", lambda: old)
    write_manifest(root / "manifest.json", manifest)
    monkeypatch.setattr(settings, "WORKSPACE_DIR", workspace)

    status_result = CliRunner().invoke(app, ["status", RUN_ID])
    clean_result = CliRunner().invoke(app, ["clean", "--older-than", "30d", "--yes"])

    assert status_result.exit_code == 0, status_result.output
    assert RUN_ID in status_result.output
    assert clean_result.exit_code == 0, clean_result.output
    assert not root.exists()
    assert outside_output.read_bytes() == b"video"


def test_repository_lists_most_recently_updated_run_first(tmp_path):
    workspace = tmp_path / "workspace"
    runs = workspace / "runs"
    now = datetime.now(timezone.utc)
    resumed_id = RUN_ID
    newer_created_id = "20260729-120000-deadbeef"
    manifests = [
        RunManifest(
            run_id=resumed_id,
            created_at=now - timedelta(days=10),
            updated_at=now,
            user_prompt="resumed",
            output_path=str(tmp_path / "resumed.mp4"),
        ),
        RunManifest(
            run_id=newer_created_id,
            created_at=now - timedelta(days=1),
            updated_at=now - timedelta(days=1),
            user_prompt="newer",
            output_path=str(tmp_path / "newer.mp4"),
        ),
    ]
    for manifest in manifests:
        root = runs / manifest.run_id
        root.mkdir(parents=True)
        (root / "manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")

    ordered = RunRepository(workspace).list()

    assert [manifest.run_id for manifest in ordered] == [resumed_id, newer_created_id]
