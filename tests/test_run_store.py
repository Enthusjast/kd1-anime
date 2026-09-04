import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from typer.testing import CliRunner

import kd1_anime.run_store as store_module
from kd1_anime.agents.planner import ContinuityBible, ScenePlan
from kd1_anime.agents.technical_planner import TechnicalObject, TechnicalSpec
from kd1_anime.cli import app
from kd1_anime.config import settings
from kd1_anime.orchestrator import Orchestrator, PipelineContext, RunPaths, SceneState, State
from kd1_anime.rendering import sha256_file
from kd1_anime.run_store import (
    RunManifest,
    RunRepository,
    StoredSceneState,
    StoredSlurmJob,
    is_valid_fsm_transition,
    lock_run,
    restore_run_path,
    sha256_text,
    write_manifest,
)

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


def test_checkpoint_round_trip_persists_continuity_bible(tmp_path):
    workspace = tmp_path / "workspace"
    paths = make_paths(workspace)
    paths.scenes.mkdir(parents=True)
    bible = ContinuityBible(
        background="#101010",
        palette=["蓝色=输入"],
        persistent_elements=["公式"],
        transition_rules=["保留公式"],
    )
    ctx = PipelineContext(
        "prompt",
        paths=paths,
        continuity_bible=bible,
        plan_review_status="passed",
        continuity_review_status="pending",
        continuity_review_round=1,
        continuity_resume_recheck_used=True,
        continuity_warnings=["warning"],
        scene_states={
            1: SceneState(
                plan=make_plan(),
                safe_fallback_used=True,
                safe_fallback_reason="几何方案无法验证",
                failure_category="review",
                plan_ready=True,
                plan_reviewed=True,
                plan_review_round=0,
            )
        },
    )
    orchestrator = Orchestrator()

    orchestrator._checkpoint(ctx, State.DETAILING)

    manifest = RunRepository(workspace).load(RUN_ID)
    restored = orchestrator._context_from_manifest(manifest, paths.root)
    assert restored.continuity_bible == bible
    assert restored.continuity_review_status == "pending"
    assert restored.continuity_review_round == 1
    assert restored.continuity_resume_recheck_used is True
    assert restored.continuity_warnings == ["warning"]
    assert restored.scene_states[1].safe_fallback_used is True
    assert restored.scene_states[1].safe_fallback_reason == "几何方案无法验证"
    assert restored.scene_states[1].failure_category == "review"
    assert restored.scene_states[1].plan_reviewed is True


def test_checkpoint_round_trip_persists_technical_spec_identity(tmp_path):
    workspace = tmp_path / "workspace"
    paths = make_paths(workspace)
    paths.scenes.mkdir(parents=True)
    spec = TechnicalSpec(
        scene_id=1,
        objects=[TechnicalObject(element_id="formula", variable_name="formula", exported=True)],
        export_element_ids=["formula"],
    )
    state = SceneState(
        plan=make_plan(),
        plan_ready=True,
        plan_reviewed=True,
        technical_spec=spec,
        technical_spec_sha256=sha256_text(spec.model_dump_json()),
        technical_input_sha256="a" * 64,
        technical_status="passed",
    )
    ctx = PipelineContext(
        "prompt",
        paths=paths,
        plan_review_status="passed",
        scene_states={1: state},
    )

    Orchestrator()._checkpoint(ctx, State.REVIEWING)

    manifest = RunRepository(workspace).load(RUN_ID)
    restored = Orchestrator._context_from_manifest(manifest, paths.root)
    restored_spec = restored.scene_states[1].technical_spec
    assert restored_spec == spec
    assert restored.scene_states[1].technical_status == "passed"
    assert manifest.integrity_errors() == []


def test_manifest_integrity_rejects_passed_plan_review_with_pending_scene():
    manifest = RunManifest(
        run_id=RUN_ID,
        user_prompt="prompt",
        output_path="/tmp/output.mp4",
        plan_review_status="passed",
        scenes={1: StoredSceneState(plan=make_plan(), plan_ready=True)},
    )

    assert any("未通过计划审查" in error for error in manifest.integrity_errors())


def test_manifest_validate_for_resume_rejects_semantic_corruption():
    manifest = RunManifest(
        run_id=RUN_ID,
        user_prompt="prompt",
        output_path="/tmp/output.mp4",
        plan_review_status="passed",
        scenes={1: StoredSceneState(plan=make_plan(), plan_ready=True)},
    )

    with pytest.raises(ValueError, match="完整性校验失败"):
        manifest.validate_for_resume()


@pytest.mark.parametrize(
    ("previous", "current", "expected"),
    [
        ("INIT", "PLANNING", True),
        ("REVIEWING", "FIXING", True),
        ("DONE", "CODING", True),  # incomplete dry-run resume edge
        ("DONE", "PLANNING", False),
        ("UNKNOWN", "CODING", False),
    ],
)
def test_fsm_transition_table_is_explicit(previous, current, expected):
    assert is_valid_fsm_transition(previous, current) is expected


@pytest.mark.parametrize(
    ("status", "state", "message"),
    [
        ("completed", "MONITORING", "completed.*FSM"),
        ("dry_run_complete", "CODING", "dry_run_complete.*FSM"),
        ("failed", "DONE", "不能与 FSM 终态 DONE"),
    ],
)
def test_manifest_integrity_rejects_status_state_mismatch(status, state, message):
    manifest = RunManifest(
        run_id=RUN_ID,
        status=status,
        state=state,
        user_prompt="prompt",
        output_path="/tmp/output.mp4",
        final_video="/tmp/output.mp4" if status == "completed" else None,
    )

    assert any(re.search(message, error) for error in manifest.integrity_errors())


def test_restore_run_path_rejects_internal_symlink(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    target = root / "target.txt"
    target.write_text("ok", encoding="utf-8")
    (root / "link.txt").symlink_to(target)

    with pytest.raises(ValueError, match="符号链接"):
        restore_run_path(root, "link.txt")


def test_manifest_integrity_rejects_unverified_rendered_artifact():
    profile = PipelineContext("prompt", paths=make_paths(Path("/tmp"))).render_profile
    code = "from manim import *\nclass Demo(Scene):\n    def construct(self): self.wait()\n"
    scene = StoredSceneState(
        plan=make_plan(),
        code_file="scenes/scene_1.py",
        code_sha256=sha256_text(code),
        class_name="Demo",
        rendered=True,
        artifact={
            "origin": "rendered",
            "source_run_id": RUN_ID,
            "job_id": "123",
            "scene_id": 1,
            "scene_class_name": "Demo",
            "code_sha256": sha256_text(code),
            "render_profile_sha256": profile.digest(),
            "video_path": "videos/scene_1/Demo.mp4",
            "video_sha256": "a" * 64,
            "metadata": {
                "size_bytes": 1,
                "duration_seconds": 1,
                "width": profile.pixel_width,
                "height": profile.pixel_height,
                "frame_rate": profile.frame_rate,
            },
            "verified": False,
        },
    )
    manifest = RunManifest(
        run_id=RUN_ID,
        user_prompt="prompt",
        output_path="/tmp/output.mp4",
        scenes={1: scene},
    )

    assert any("artifact 未验证" in error for error in manifest.integrity_errors())


def test_resume_completed_run_rejects_tampered_final_video(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    paths = make_paths(workspace)
    paths.root.mkdir(parents=True)
    paths.output.write_bytes(b"original")
    manifest = RunManifest(
        run_id=RUN_ID,
        status="completed",
        state="DONE",
        user_prompt="prompt",
        output_path=str(paths.output.resolve()),
        final_video=str(paths.output.resolve()),
        final_video_sha256=sha256_file(paths.output),
    )
    write_manifest(paths.root / "manifest.json", manifest)
    paths.output.write_bytes(b"tampered")
    monkeypatch.setattr(settings, "WORKSPACE_DIR", workspace)

    with pytest.raises(RuntimeError, match="哈希"):
        Orchestrator().resume(RUN_ID)


def test_context_rejects_manifest_final_video_different_from_output(tmp_path):
    workspace = tmp_path / "workspace"
    paths = make_paths(workspace)
    paths.root.mkdir(parents=True)
    output = paths.root / "output.mp4"
    other = paths.root / "other.mp4"
    manifest = RunManifest(
        run_id=RUN_ID,
        status="failed",
        state="MERGING",
        user_prompt="prompt",
        output_path=str(output.resolve()),
        final_video=str(other.resolve()),
    )

    with pytest.raises(ValueError, match="final_video"):
        Orchestrator._context_from_manifest(manifest, paths.root)


def test_restore_rejects_slurm_job_with_different_code_identity(tmp_path):
    workspace = tmp_path / "workspace"
    paths = make_paths(workspace)
    paths.scenes.mkdir(parents=True)
    code = "from manim import *\nclass Demo(Scene):\n    def construct(self): self.wait()\n"
    (paths.scenes / "scene_1.py").write_text(code, encoding="utf-8")
    ctx = PipelineContext(
        "prompt",
        paths=paths,
        scene_states={1: SceneState(plan=make_plan(), code=code, class_name="Demo")},
    )
    orchestrator = Orchestrator()
    orchestrator._checkpoint(ctx, State.CODING)
    manifest = RunRepository(workspace).load(RUN_ID)
    stored = manifest.scenes[1]
    stored.slurm_job = StoredSlurmJob(
        job_id="123",
        scene_id=1,
        script_path="scenes/render_1.sh",
        log_out="logs/scene_1.out",
        log_err="logs/scene_1.err",
        media_dir="videos/scene_1",
        scene_class_name="Demo",
        submitted_at=1,
        code_sha256="0" * 64,
        status="PENDING",
    )

    with pytest.raises(ValueError, match="Job 代码哈希"):
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


def test_lock_run_tightens_legacy_directory_permissions(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    root.chmod(0o755)

    with lock_run(root):
        assert root.stat().st_mode & 0o777 == 0o700
        assert (root / ".run.lock").stat().st_mode & 0o777 == 0o600


def test_lock_run_rejects_symlink(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "run"
    link.symlink_to(target, target_is_directory=True)

    with pytest.raises(RuntimeError, match="符号链接"), lock_run(link):
        pass


def test_lock_run_rejects_symlink_lock_file(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    outside = tmp_path / "outside.lock"
    outside.write_bytes(b"")
    (root / ".run.lock").symlink_to(outside)

    with pytest.raises(RuntimeError, match="运行锁"), lock_run(root):
        pass

    assert outside.read_bytes() == b""


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


def test_repository_list_ignores_non_object_manifest(tmp_path):
    workspace = tmp_path / "workspace"
    root = workspace / "runs" / RUN_ID
    root.mkdir(parents=True)
    (root / "manifest.json").write_text("[]", encoding="utf-8")

    assert RunRepository(workspace).list() == []
    with pytest.raises(ValueError, match="JSON 对象"):
        RunRepository(workspace).load(RUN_ID)


def test_repository_rejects_malformed_legacy_scene_mapping(tmp_path):
    workspace = tmp_path / "workspace"
    root = workspace / "runs" / RUN_ID
    root.mkdir(parents=True)
    (root / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "run_id": RUN_ID, "scenes": []}),
        encoding="utf-8",
    )

    assert RunRepository(workspace).list() == []
    with pytest.raises(ValueError, match="不支持旧版"):
        RunRepository(workspace).load(RUN_ID)


def test_repository_rejects_previous_manifest_schema_with_actionable_message(tmp_path):
    workspace = tmp_path / "workspace"
    root = workspace / "runs" / RUN_ID
    root.mkdir(parents=True)
    (root / "manifest.json").write_text(
        json.dumps({"schema_version": 3, "run_id": RUN_ID}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"schema_version=3.*重新生成"):
        RunRepository(workspace).load(RUN_ID)


def test_current_manifest_uses_v6_schema_and_merge_profile():
    assert (
        RunManifest(run_id=RUN_ID, user_prompt="test", output_path="/tmp/out.mp4").schema_version
        == 6
    )
    assert (
        RunManifest(
            run_id=RUN_ID,
            user_prompt="test",
            output_path="/tmp/out.mp4",
        ).merge_profile.video_codec
        == "libx264"
    )


def test_v4_manifest_is_readable_but_read_only(tmp_path):
    workspace = tmp_path / "workspace"
    root = workspace / "runs" / RUN_ID
    root.mkdir(parents=True)
    raw = RunManifest(
        run_id=RUN_ID,
        user_prompt="legacy",
        output_path=str((root / "output.mp4").resolve()),
    ).model_dump(mode="json")
    raw["schema_version"] = 4
    path = root / "manifest.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    loaded = RunRepository(workspace).load(RUN_ID)

    assert loaded.schema_version == 4
    with pytest.raises(ValueError, match="仅支持只读查看"):
        loaded.validate_for_resume()
    with pytest.raises(ValueError, match="只允许写入 v6"):
        write_manifest(path, loaded)


def test_v5_manifest_must_persist_teaching_contract_fields(tmp_path):
    workspace = tmp_path / "workspace"
    root = workspace / "runs" / RUN_ID
    root.mkdir(parents=True)
    raw = RunManifest(
        run_id=RUN_ID,
        user_prompt="prompt",
        output_path=str((root / "output.mp4").resolve()),
    ).model_dump(mode="json")
    raw["schema_version"] = 5
    for field_name in ("lesson_spec", "teaching_graph", "state_ledger"):
        raw.pop(field_name)
    (root / "manifest.json").write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="v5 manifest 缺少必需字段"):
        RunRepository(workspace).load(RUN_ID)


def _write_v1_manifest(workspace: Path, *, reused_job: bool = False) -> Path:
    root = workspace / "runs" / RUN_ID
    root.mkdir(parents=True, exist_ok=True)
    scene = StoredSceneState(
        plan=make_plan(),
        code_file="scenes/scene_1.py",
        code_sha256="a" * 64,
        class_name="Demo",
        reviewed=True,
        rendered=True,
        slurm_job=StoredSlurmJob(
            job_id="123",
            scene_id=1,
            script_path="scenes/render_1.sh",
            log_out="logs/scene_1.out",
            log_err="logs/scene_1.err",
            media_dir="videos/scene_1",
            scene_class_name="Demo",
            submitted_at=1,
            status="COMPLETED",
        ),
    )
    raw = RunManifest(
        run_id=RUN_ID,
        user_prompt="legacy",
        output_path=str(root / "output.mp4"),
        scenes={1: scene},
    ).model_dump(mode="json")
    raw["schema_version"] = 1
    raw.pop("revision")
    raw.pop("render_profile")
    legacy_scene = raw["scenes"]["1"]
    legacy_scene.pop("artifact")
    legacy_scene.pop("phase")
    legacy_job = legacy_scene["slurm_job"]
    for key in (
        "code_sha256",
        "render_profile",
        "output_path",
        "output_metadata",
        "elapsed_seconds",
    ):
        legacy_job.pop(key)
    if reused_job:
        legacy_job["job_id"] = "reused-1"
    path = root / "manifest.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


def test_v1_reused_job_is_migrated_to_safe_rerender(tmp_path):
    workspace = tmp_path / "workspace"
    _write_v1_manifest(workspace, reused_job=True)

    with pytest.raises(ValueError, match="不支持旧版"):
        RunRepository(workspace).load(RUN_ID)


def test_v1_rendered_video_is_rejected_without_unsafe_migration(tmp_path):
    workspace = tmp_path / "workspace"
    _write_v1_manifest(workspace)

    with pytest.raises(ValueError, match="不支持旧版"):
        RunRepository(workspace).load(RUN_ID)


def test_v1_unverifiable_video_is_forced_to_rerender(tmp_path):
    workspace = tmp_path / "workspace"
    _write_v1_manifest(workspace)

    with pytest.raises(ValueError, match="不支持旧版"):
        RunRepository(workspace).load(RUN_ID)


def test_v1_rendered_scene_without_job_is_forced_to_rerender(tmp_path):
    workspace = tmp_path / "workspace"
    path = _write_v1_manifest(workspace)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["scenes"]["1"]["slurm_job"] = None
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="不支持旧版"):
        RunRepository(workspace).load(RUN_ID)


def test_checkpoint_revision_is_monotonic_under_threads(tmp_path):
    workspace = tmp_path / "workspace"
    paths = make_paths(workspace)
    paths.scenes.mkdir(parents=True)
    code = "from manim import *\nclass Demo(Scene):\n    def construct(self): self.wait()\n"
    (paths.scenes / "scene_1.py").write_text(code, encoding="utf-8")
    ctx = PipelineContext(
        "prompt",
        paths=paths,
        scene_states={1: SceneState(plan=make_plan(), code=code, class_name="Demo")},
    )
    orchestrator = Orchestrator()

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(orchestrator._checkpoint, ctx, State.CODING) for _ in range(20)]
        for future in futures:
            future.result()

    manifest = RunRepository(workspace).load(RUN_ID)
    assert manifest.revision == 20
    assert json.loads((paths.root / "manifest.json").read_text())["revision"] == 20


def test_manifest_rejects_unknown_fsm_state_and_scene_phase():
    with pytest.raises(ValueError):
        RunManifest(
            run_id=RUN_ID,
            user_prompt="prompt",
            output_path="/tmp/output.mp4",
            state="NOT_A_STATE",
        )
    with pytest.raises(ValueError):
        StoredSceneState(plan=make_plan(), phase="not-a-phase")

    scene = StoredSceneState(plan=make_plan())
    with pytest.raises(ValueError):
        scene.phase = "not-a-phase"
