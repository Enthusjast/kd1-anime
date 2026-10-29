"""Independent verification receipts remain bound to exact source and media."""

import pytest
from test_agent_vertical_slice import agent_boundary as agent_boundary
from test_runtime_reliability import completed_run

from kd1_anime.application import Application
from kd1_anime.run_store import StoredSceneState
from kd1_anime.stats import summarize_manifest
from kd1_anime.verification import ExecutionVerification, StaticVerification, VisualVerification


def test_static_execution_visual_receipts_are_independent():
    assert StaticVerification().status == "not_run"
    assert ExecutionVerification().status == "not_run"
    assert VisualVerification().status == "not_run"


def test_public_run_persists_separate_verification_conclusions(agent_boundary):
    _, manifest = completed_run(agent_boundary)
    scene = manifest.scenes[1]
    assert scene.static_verification.status == "passed"
    assert scene.static_verification.code_sha256 == scene.code_sha256
    assert scene.execution_verification.status == "passed"
    assert scene.execution_verification.scope == "formal_video"
    assert scene.execution_verification.code_sha256 == scene.code_sha256
    assert scene.execution_verification.artifact_sha256 == scene.artifact.video_sha256
    assert scene.visual_verification.status == "not_run"
    assert scene.visual_verification.score is None
    assert manifest.integrity_errors() == []


def test_stored_receipts_round_trip_without_collapsing_unknown_into_success(agent_boundary):
    _, manifest = completed_run(agent_boundary)
    scene = manifest.scenes[1].model_copy(deep=True)
    scene.execution_verification = ExecutionVerification(status="unknown", scope="formal_video")
    scene.visual_verification = VisualVerification(status="unknown")
    restored = StoredSceneState.model_validate_json(scene.model_dump_json())
    assert restored.static_verification.status == "passed"
    assert restored.execution_verification.status == "unknown"
    assert restored.visual_verification.status == "unknown"
    assert restored.visual_verification.score is None


def test_integrity_catches_stale_static_receipt(agent_boundary):
    _, manifest = completed_run(agent_boundary)
    manifest.scenes[1].static_verification.code_sha256 = "a" * 64
    assert any("静态验证代码哈希不一致" in error for error in manifest.integrity_errors())


@pytest.mark.parametrize("receipt", ["static", "execution", "visual"])
def test_code_change_cannot_retain_old_verification_success(agent_boundary, receipt):
    _, manifest = completed_run(agent_boundary)
    scene = manifest.scenes[1]
    scene.code_sha256 = "b" * 64
    if receipt == "static":
        marker = "静态验证代码哈希不一致"
    elif receipt == "execution":
        marker = "执行验证代码哈希不一致"
    else:
        scene.visual_verification = VisualVerification(
            status="passed",
            code_sha256="a" * 64,
            artifact_sha256="c" * 64,
            score=4,
        )
        marker = "视觉验证视频哈希不一致"
    assert any(marker in error for error in manifest.integrity_errors())


def test_resume_rejects_orphaned_formal_receipt_before_external_actions(agent_boundary):
    root, manifest = completed_run(agent_boundary)
    scene = manifest.scenes[1]
    scene.rendered = False
    scene.artifact = None
    path = root / "manifest.json"
    path.write_text(manifest.model_dump_json())
    before = path.read_bytes()

    def forbidden():
        pytest.fail("invalid verification receipt reached executable runtime")

    with pytest.raises(ValueError, match="正式执行验证通过但没有渲染产物"):
        Application(runtime_factory=forbidden).resume(root.name)
    assert path.read_bytes() == before


def test_visual_unknown_is_not_counted_as_verified_or_accepted(agent_boundary):
    _, manifest = completed_run(agent_boundary)
    scene = manifest.scenes[1]
    scene.visual_verification = VisualVerification(status="unknown")
    scene.visual_status = "unknown"
    report = summarize_manifest(manifest)
    assert report["scenes"]["visual_unknown"] == 1
    assert report["scenes"]["visual_verified"] == 0
    assert report["scenes"]["visual_accepted"] == 0
