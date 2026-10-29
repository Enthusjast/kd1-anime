import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from typer.testing import CliRunner

import kd1_anime.run_store as store_module
from kd1_anime.agent_runtime.state import AgentRunState
from kd1_anime.agents.planner import ContinuityBible, ScenePlan
from kd1_anime.agents.technical_planner import TechnicalObject, TechnicalSpec
from kd1_anime.cli import app
from kd1_anime.commits import CommitCommand, RunCommitter, RunPatch
from kd1_anime.config import settings
from kd1_anime.domain.candidates import CodeCandidate
from kd1_anime.domain.dependencies import StaleWorkError
from kd1_anime.domain.execution import is_valid_fsm_transition
from kd1_anime.domain.runtime import VisualEvalProfile
from kd1_anime.domain.scene_fields import SCENE_FIELDS, SCENE_STORAGE_FIELDS
from kd1_anime.objects import ObjectStore
from kd1_anime.pipeline.contracts import FinalizationInputs, FinalizationOutput
from kd1_anime.pipeline.finalization import FinalizationPipeline
from kd1_anime.prompt_identity import capture_generation_identity
from kd1_anime.rendering import RenderProfile, sha256_file
from kd1_anime.run_store import (
    RunManifest,
    RunRepository,
    StoredSceneState,
    StoredSlurmJob,
    lock_run,
    restore_run_path,
    scene_field_group,
    scene_snapshot,
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


def make_manifest(**values) -> RunManifest:
    run_id = values.get("run_id", RUN_ID)
    return RunManifest(
        agent=AgentRunState(
            session_id=f"session-{run_id}",
            generation_identity=capture_generation_identity(),
            run_input_sha256="a" * 64,
        ),
        **values,
    )


def test_manifest_round_trip_preserves_immutable_code_identity(tmp_path):
    workspace = tmp_path / "workspace"
    root = workspace / "runs" / RUN_ID
    objects = ObjectStore(root)
    code = "from manim import *\nclass Demo(Scene):\n    def construct(self): self.wait()\n"
    code_file = objects.source(1, code)
    manifest = make_manifest(
        run_id=RUN_ID,
        user_prompt="prompt",
        output_path=str(root / "output.mp4"),
        scenes={
            1: StoredSceneState(
                plan=make_plan(),
                code_file=code_file,
                code_sha256=sha256_text(code),
                class_name="Demo",
            )
        },
    )
    manifest_path = root / "manifest.json"
    write_manifest(manifest_path, manifest)
    assert manifest_path.stat().st_mode & 0o777 == 0o600
    restored = RunRepository(workspace).load_for_resume(RUN_ID)
    stored = restored.scenes[1]
    assert stored.code_file == code_file
    assert sha256_file(root / stored.code_file) == stored.code_sha256

    (root / "scenes" / "scene_1.py").write_text("# display changed\n", encoding="utf-8")
    assert (root / stored.code_file).read_text(encoding="utf-8") == code
    (root / stored.code_file).write_text(code + "# changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="identity mismatch"):
        objects.source(1, code)


def test_manifest_round_trip_persists_continuity_bible(tmp_path):
    workspace = tmp_path / "workspace"
    root = workspace / "runs" / RUN_ID
    bible = ContinuityBible(
        background="#101010",
        palette=["蓝色=输入"],
        persistent_elements=["公式"],
        transition_rules=["保留公式"],
    )
    manifest = make_manifest(
        run_id=RUN_ID,
        user_prompt="prompt",
        output_path=str(root / "output.mp4"),
        continuity_bible=bible,
        plan_review_status="passed",
        continuity_review_status="pending",
        continuity_review_round=1,
        continuity_resume_recheck_used=True,
        continuity_warnings=["warning"],
        scenes={
            1: StoredSceneState(
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
    write_manifest(root / "manifest.json", manifest)

    restored = RunRepository(workspace).load_for_resume(RUN_ID)
    assert restored.continuity_bible == bible
    assert restored.continuity_review_status == "pending"
    assert restored.continuity_review_round == 1
    assert restored.continuity_resume_recheck_used is True
    assert restored.continuity_warnings == ["warning"]
    assert restored.scenes[1].safe_fallback_used is True
    assert restored.scenes[1].safe_fallback_reason == "几何方案无法验证"
    assert restored.scenes[1].failure_category == "review"
    assert restored.scenes[1].plan_reviewed is True


def test_manifest_round_trip_persists_technical_spec_identity(tmp_path):
    workspace = tmp_path / "workspace"
    root = workspace / "runs" / RUN_ID
    spec = TechnicalSpec(
        scene_id=1,
        objects=[TechnicalObject(element_id="formula", variable_name="formula", exported=True)],
        export_element_ids=["formula"],
    )
    state = StoredSceneState(
        plan=make_plan(),
        plan_ready=True,
        plan_reviewed=True,
        technical_spec=spec,
        technical_spec_sha256=sha256_text(spec.model_dump_json()),
        technical_input_sha256="a" * 64,
        technical_status="passed",
    )
    manifest = make_manifest(
        run_id=RUN_ID,
        user_prompt="prompt",
        output_path=str(root / "output.mp4"),
        plan_review_status="passed",
        scenes={1: state},
    )
    write_manifest(root / "manifest.json", manifest)

    restored = RunRepository(workspace).load_for_resume(RUN_ID)
    restored_spec = restored.scenes[1].technical_spec
    assert restored_spec == spec
    assert restored.scenes[1].technical_status == "passed"
    assert restored.integrity_errors() == []


def test_legacy_technical_spec_is_readable_but_cannot_resume():
    legacy = StoredSceneState.model_validate(
        {
            "plan": make_plan(),
            "technical_spec": {
                "scene_id": 1,
                "renderer": "cairo",
                "objects": [],
                "animations": [],
                "export_element_ids": [],
                "removed_element_ids": [],
            },
        }
    )

    assert legacy.technical_spec is None
    assert legacy.technical_contract_stale is True
    manifest = make_manifest(
        run_id=RUN_ID,
        user_prompt="prompt",
        output_path="/tmp/output.mp4",
        scenes={1: legacy},
    )
    with pytest.raises(ValueError, match="旧版 TechnicalSpec"):
        manifest.validate_for_resume()


def test_manifest_integrity_rejects_passed_plan_review_with_pending_scene():
    manifest = make_manifest(
        run_id=RUN_ID,
        user_prompt="prompt",
        output_path="/tmp/output.mp4",
        plan_review_status="passed",
        scenes={1: StoredSceneState(plan=make_plan(), plan_ready=True)},
    )

    assert any("未通过计划审查" in error for error in manifest.integrity_errors())


def test_manifest_validate_for_resume_rejects_semantic_corruption():
    manifest = make_manifest(
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
    manifest = make_manifest(
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
    profile = RenderProfile.current()
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
    manifest = make_manifest(
        run_id=RUN_ID,
        user_prompt="prompt",
        output_path="/tmp/output.mp4",
        scenes={1: scene},
    )

    assert any("artifact 未验证" in error for error in manifest.integrity_errors())


@pytest.mark.parametrize("tamper", ["content", "output_identity"])
def test_completed_manifest_rejects_tampered_final_video(tmp_path, tamper):
    workspace = tmp_path / "workspace"
    root = workspace / "runs" / RUN_ID
    root.mkdir(parents=True)
    output = root / "output.mp4"
    output.write_bytes(b"original")
    manifest = make_manifest(
        run_id=RUN_ID,
        status="completed",
        state="DONE",
        user_prompt="prompt",
        output_path=str(output.resolve()),
        final_video=str(output.resolve()),
        final_video_sha256=sha256_file(output),
    )
    write_manifest(root / "manifest.json", manifest)
    if tamper == "content":
        output.write_bytes(b"tampered")

    restored = RunRepository(workspace).load_for_resume(RUN_ID)
    result = FinalizationOutput(
        video={
            "path": "output.mp4",
            "sha256": restored.final_video_sha256,
            "kind": "video",
        },
        output_path=str(root / "other.mp4")
        if tamper == "output_identity"
        else restored.final_video,
    )
    inputs = FinalizationInputs(
        restored.run_id,
        root,
        restored.run_config,
        restored.render_profile,
        restored.merge_profile,
        (),
        Path(restored.output_path),
    )
    with pytest.raises(ValueError, match="hash or output identity mismatch"):
        FinalizationPipeline._validate_video(inputs, result)


def test_restore_rejects_slurm_job_with_different_code_identity(tmp_path):
    root = tmp_path / "runs" / RUN_ID
    stored = StoredSceneState(
        plan=make_plan(),
        code_file="scenes/scene_1.py",
        code_sha256="a" * 64,
        class_name="Demo",
    )
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

    manifest = make_manifest(
        run_id=RUN_ID,
        user_prompt="prompt",
        output_path=str(root / "output.mp4"),
        scenes={1: stored},
    )
    with pytest.raises(ValueError, match="Job 代码哈希"):
        manifest.validate_for_resume()


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
    manifest = make_manifest(
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
        make_manifest(
            run_id=resumed_id,
            created_at=now - timedelta(days=10),
            updated_at=now,
            user_prompt="resumed",
            output_path=str(tmp_path / "resumed.mp4"),
        ),
        make_manifest(
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


def test_current_manifest_uses_v11_schema_and_merge_profile():
    manifest = make_manifest(run_id=RUN_ID, user_prompt="test", output_path="/tmp/out.mp4")
    assert manifest.schema_version == 11
    assert manifest.merge_profile.video_codec == "libx264"


def test_v7_manifest_remains_read_only_without_version_upgrade(tmp_path):
    workspace = tmp_path / "workspace"
    root = workspace / "runs" / RUN_ID
    root.mkdir(parents=True)
    raw = make_manifest(
        run_id=RUN_ID,
        user_prompt="legacy v7",
        output_path=str((root / "output.mp4").resolve()),
    ).model_dump(mode="json")
    raw["schema_version"] = 7
    raw.pop("generation_mode")
    raw.pop("agent")
    (root / "manifest.json").write_text(json.dumps(raw), encoding="utf-8")

    repository = RunRepository(workspace)
    before = (root / "manifest.json").read_bytes()
    loaded = repository.load(RUN_ID)
    assert loaded.schema_version == 7
    with pytest.raises(ValueError, match="只读"):
        repository.load_for_resume(RUN_ID)
    assert (root / "manifest.json").read_bytes() == before


def test_v4_manifest_is_readable_but_read_only(tmp_path):
    workspace = tmp_path / "workspace"
    root = workspace / "runs" / RUN_ID
    root.mkdir(parents=True)
    raw = make_manifest(
        run_id=RUN_ID,
        user_prompt="legacy",
        output_path=str((root / "output.mp4").resolve()),
    ).model_dump(mode="json")
    raw["schema_version"] = 4
    raw.pop("agent")
    path = root / "manifest.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    loaded = RunRepository(workspace).load(RUN_ID)

    assert loaded.schema_version == 4
    with pytest.raises(ValueError, match="仅支持只读查看"):
        loaded.validate_for_resume()
    with pytest.raises(ValueError):
        write_manifest(path, loaded)


def test_v5_manifest_must_persist_teaching_contract_fields(tmp_path):
    workspace = tmp_path / "workspace"
    root = workspace / "runs" / RUN_ID
    root.mkdir(parents=True)
    raw = make_manifest(
        run_id=RUN_ID,
        user_prompt="prompt",
        output_path=str((root / "output.mp4").resolve()),
    ).model_dump(mode="json")
    raw["schema_version"] = 5
    raw.pop("agent")
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
    raw = make_manifest(
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


def test_commit_revision_is_monotonic_under_threads(tmp_path):
    workspace = tmp_path / "workspace"
    root = workspace / "runs" / RUN_ID
    manifest = make_manifest(
        run_id=RUN_ID,
        user_prompt="prompt",
        output_path=str(root / "output.mp4"),
    )
    write_manifest(root / "manifest.json", manifest)
    committer = RunCommitter(root, manifest)

    def commit():
        while True:
            current = committer.current
            command = CommitCommand(
                current.revision,
                "INIT",
                patch=RunPatch(error="checkpoint"),
            )
            try:
                committer.commit(command)
                return
            except StaleWorkError:
                continue

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(commit) for _ in range(20)]
        for future in futures:
            future.result()

    restored = RunRepository(workspace).load_for_resume(RUN_ID)
    assert restored.revision == 20
    assert json.loads((root / "manifest.json").read_text())["revision"] == 20


def test_manifest_rejects_unknown_fsm_state_and_scene_phase():
    with pytest.raises(ValueError):
        make_manifest(
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


def test_scene_storage_metadata_is_complete_and_explicit():
    stored_fields = set(StoredSceneState.model_fields)
    assert set(SCENE_STORAGE_FIELDS) == stored_fields
    assert sum(len(metadata.stored_names) for metadata in SCENE_FIELDS.values()) == len(
        stored_fields
    )
    assert SCENE_FIELDS["code"].stored_names == ("code_file", "code_sha256")
    for alias in ("code_file", "code_sha256"):
        assert SCENE_STORAGE_FIELDS[alias] is SCENE_FIELDS["code"]
        assert SCENE_STORAGE_FIELDS[alias].domain_group == "code_revision"
        assert scene_field_group(alias) == "code"
    for unknown in ("invented", "plan_invented", "visual_invented", "last_invented"):
        with pytest.raises(KeyError):
            scene_field_group(unknown)


def test_scene_snapshot_preserves_v10_groups_and_exact_flat_projection():
    scene = StoredSceneState(
        plan=make_plan(),
        code_file="objects/source.py",
        code_sha256="a" * 64,
        class_name="Demo",
        plan_review_feedback="plan feedback",
        identical_plan_review_count=2,
        last_error_fp="render failure",
        visual_feedback="visual feedback",
        failure_reason="failure",
    )
    snapshot = scene_snapshot(scene).model_dump(mode="json")
    assert tuple(snapshot) == ("planning", "code", "execution", "verification", "lifecycle")
    assert snapshot["planning"]["plan_review_feedback"] == "plan feedback"
    assert snapshot["code"]["identical_plan_review_count"] == 2
    assert snapshot["execution"]["last_error_fp"] == "render failure"
    assert snapshot["verification"]["visual_feedback"] == "visual feedback"
    assert snapshot["lifecycle"]["failure_reason"] == "failure"
    flattened = {name: value for group in snapshot.values() for name, value in group.items()}
    assert len(flattened) == sum(len(group) for group in snapshot.values())
    assert flattened == scene.model_dump(mode="json")
    assert StoredSceneState.model_validate(flattened) == scene


def test_extracted_domain_schemas_round_trip_v11_with_one_scene_projection(tmp_path):
    candidate_payload = {
        "code_file": "objects/source.py",
        "code_sha256": "a" * 64,
        "class_name": "Demo",
        "verification": "validated",
        "inherited_elements_sha256": "",
        "exported_elements_code": "",
        "exported_elements": [],
        "artifact": None,
        "visual_score": None,
    }
    profile_payload = {
        "enabled": True,
        "model": "vision-model",
        "frame_count": 6,
        "threshold": 3.5,
        "max_fix_attempts": 2,
        "repair_enabled": True,
        "evaluator_version": "1",
    }
    candidate = CodeCandidate.model_validate(candidate_payload)
    profile = VisualEvalProfile.model_validate(profile_payload)
    assert candidate.model_dump(mode="json") == candidate_payload
    assert profile.model_dump(mode="json") == profile_payload
    assert profile.digest() == sha256_text(json.dumps(profile_payload, separators=(",", ":")))
    workspace = tmp_path / "workspace"
    root = workspace / "runs" / RUN_ID
    manifest = make_manifest(
        run_id=RUN_ID,
        user_prompt="prompt",
        output_path=str(root / "output.mp4"),
        visual_eval_profile=profile,
        scenes={1: StoredSceneState(plan=make_plan(), candidates=[candidate])},
    )
    manifest_path = root / "manifest.json"
    write_manifest(manifest_path, manifest)
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert raw["schema_version"] == 11
    assert raw["visual_eval_profile"] == profile_payload
    assert raw["scenes"]["1"]["candidates"] == [candidate_payload]
    assert "snapshot" not in raw["scenes"]["1"]
    restored = RunRepository(tmp_path / "workspace").load(RUN_ID)
    assert restored.model_dump(mode="json") == raw
    assert restored.integrity_errors() == []
    assert isinstance(restored.scenes[1].candidates[0], CodeCandidate)
    assert isinstance(restored.visual_eval_profile, VisualEvalProfile)
    assert not hasattr(restored.scenes[1], "snapshot")


@pytest.mark.parametrize("version", range(4, 11))
def test_legacy_manifests_are_exactly_read_only_before_resume_side_effects(
    tmp_path, monkeypatch, version
):
    workspace = tmp_path / "workspace"
    root = workspace / "runs" / RUN_ID
    root.mkdir(parents=True)
    root.chmod(0o755)
    scene = StoredSceneState(plan=make_plan(), revision=3, input_sha256="b" * 64)
    raw = make_manifest(
        run_id=RUN_ID,
        user_prompt=f"legacy v{version}",
        output_path=str(root / "output.mp4"),
        scenes={1: scene},
    ).model_dump(mode="json")
    raw["schema_version"] = version
    raw.pop("agent")
    raw["approve_plan"] = True
    raw["plan_approved"] = True
    raw["scenes"]["1"]["snapshot"] = scene_snapshot(scene).model_dump(mode="json")
    path = root / "manifest.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    before = path.read_bytes()
    repository = RunRepository(workspace)

    loaded = repository.load(RUN_ID)
    assert loaded.schema_version == version
    assert loaded.agent is None
    assert loaded.model_dump(mode="json") == raw
    assert json.loads(loaded.model_dump_json()) == raw
    assert [item.run_id for item in repository.list()] == [RUN_ID]
    timestamp = loaded.updated_at

    def forbidden(*args, **kwargs):
        pytest.fail("Read-only rejection must precede migration, mutation, and writes")

    monkeypatch.setattr(store_module, "migrate_manifest_data", forbidden)
    monkeypatch.setattr(store_module, "atomic_write_text", forbidden)
    monkeypatch.setattr(store_module, "utc_now", forbidden)
    with pytest.raises(ValueError, match=rf"schema_version={version}.*只读"):
        repository.load_for_resume(RUN_ID)
    with pytest.raises(ValueError, match="只读"):
        loaded.validate_for_resume()
    with pytest.raises(ValueError, match="只读"):
        RunCommitter(root, loaded)
    with pytest.raises(ValueError, match="只读"):
        write_manifest(path, loaded)
    assert loaded.updated_at == timestamp
    assert loaded.scenes[1].revision == 3
    assert path.read_bytes() == before
    assert root.stat().st_mode & 0o777 == 0o755
    assert not (root / ".run.lock").exists()


def test_v11_historical_snapshot_is_validated_then_removed_without_identity_changes(tmp_path):
    root = tmp_path / "runs" / RUN_ID
    root.mkdir(parents=True)
    scene = StoredSceneState(
        plan=make_plan(),
        revision=7,
        input_sha256="b" * 64,
        code_file="scenes/revisions/scene_1/source.py",
        code_sha256="c" * 64,
    )
    manifest = make_manifest(
        run_id=RUN_ID,
        user_prompt="prompt",
        output_path=str(root / "output.mp4"),
        revision=9,
        scenes={1: scene},
    )
    raw = manifest.model_dump(mode="json")
    raw["scenes"]["1"]["snapshot"] = scene_snapshot(scene).model_dump(mode="json")
    path = root / "manifest.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    loaded = RunRepository(tmp_path).load_for_resume(RUN_ID)
    assert loaded.revision == 9
    assert loaded.agent == manifest.agent
    assert loaded.scenes[1] == scene
    assert loaded.execution_spec == manifest.execution_spec
    assert "snapshot" not in loaded.scenes[1].model_dump(mode="json")
    write_manifest(path, loaded)
    canonical = json.loads(path.read_text(encoding="utf-8"))
    assert canonical["scenes"]["1"] == scene.model_dump(mode="json")
    assert canonical["agent"] == raw["agent"]
    assert canonical["revision"] == raw["revision"]


def test_v11_historical_snapshot_mismatch_is_rejected_without_rewriting(tmp_path):
    root = tmp_path / "runs" / RUN_ID
    root.mkdir(parents=True)
    scene = StoredSceneState(plan=make_plan(), input_sha256="b" * 64)
    raw = make_manifest(
        run_id=RUN_ID,
        user_prompt="prompt",
        output_path=str(root / "output.mp4"),
        scenes={1: scene},
    ).model_dump(mode="json")
    raw["scenes"]["1"]["snapshot"] = scene_snapshot(scene).model_dump(mode="json")
    raw["scenes"]["1"]["snapshot"]["code"]["code_sha256"] = "d" * 64
    path = root / "manifest.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    before = path.read_bytes()

    with pytest.raises(ValueError, match="嵌套状态与兼容投影不一致"):
        RunRepository(tmp_path).load_for_resume(RUN_ID)
    assert path.read_bytes() == before
    assert not (root / ".run.lock").exists()


def test_v11_default_requires_agent_identity_and_session_state():
    with pytest.raises(ValueError, match="v11 requires complete agent"):
        RunManifest(run_id=RUN_ID, user_prompt="prompt", output_path="output.mp4")
