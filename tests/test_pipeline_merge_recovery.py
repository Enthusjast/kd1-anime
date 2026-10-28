"""Unknown merge outcomes are recovered from exact inputs, not public-file probes.

Real VideoMerger locking/atomic publication runs against deterministic media I/O;
no media executable, generated Python, model request, or scheduler command runs.
"""

import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_application import VIDEO
from test_pipeline_incremental import context as reuse_context
from test_pipeline_incremental import incremental as incremental
from test_pipeline_render import final_inputs, rendered
from test_pipeline_render import stage as stage

import kd1_anime.media.merger as merger_module
import kd1_anime.pipeline.finalization as finalization_module
from kd1_anime.domain.runtime import RunConfig, VisualEvalProfile
from kd1_anime.media.merger import VideoMerger
from kd1_anime.pipeline.finalization import FinalizationPipeline
from kd1_anime.pipeline.incremental import IncrementalPipeline
from kd1_anime.rendering import VideoMetadata, sha256_file

MERGED = b"merged:" + VIDEO


@pytest.fixture
def recovery(stage, monkeypatch):
    inputs = final_inputs(rendered(stage))
    calls = SimpleNamespace(merges=[], commands=[], probes=[], verified=True)

    class Merger(VideoMerger):
        def merge(self, videos, output, **kwargs):
            calls.merges.append((tuple(videos), Path(output), kwargs))
            assert kwargs["render_profile"] == inputs.render_profile
            assert kwargs["merge_profile"] == inputs.merge_profile
            assert merger_module.settings.MERGE_VIDEO_CRF == inputs.config.resolve().MERGE_VIDEO_CRF
            return super().merge(videos, output, **kwargs)

        @staticmethod
        def _run_ffmpeg(command, output, label):
            sources = [Path(command[index + 1]) for index, arg in enumerate(command) if arg == "-i"]
            calls.commands.append(tuple(sources))
            output.write_bytes(b"merged:" + b",".join(path.read_bytes() for path in sources))
            return True

    def probe(path, profile):
        calls.probes.append(Path(path))
        assert profile == inputs.render_profile
        if not calls.verified:
            raise ValueError("recomputed candidate failed media verification")
        return VideoMetadata(
            size_bytes=path.stat().st_size,
            duration_seconds=1,
            width=profile.pixel_width,
            height=profile.pixel_height,
            frame_rate=profile.frame_rate,
        )

    monkeypatch.setattr(merger_module.shutil, "which", lambda executable: "/fixture/" + executable)
    monkeypatch.setattr(merger_module, "verify_video", probe)
    return inputs, stage[2], FinalizationPipeline(Merger()), calls


def external_output(inputs):
    return inputs.root.parent / (inputs.root.name + "-external") / "movie.mp4"


@pytest.mark.parametrize("moment", ["before", "after"])
@pytest.mark.parametrize("external", [False, True])
def test_lost_merge_result_recomputes_before_or_after_public_replace(
    recovery, monkeypatch, moment, external
):
    inputs, context, pipeline, calls = recovery
    if external:
        inputs = replace(inputs, output_path=external_output(inputs))
    requested = inputs.output_path
    original_replace = Path.replace

    def lost_replace(source, destination):
        if Path(destination) == requested:
            if moment == "after":
                original_replace(source, destination)
            raise OSError("merge result checkpoint lost")
        return original_replace(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", lost_replace)
        with pytest.raises(OSError, match="checkpoint lost"):
            pipeline.merge(inputs, context)
    assert requested.exists() is (moment == "after")
    inode = requested.stat().st_ino if requested.exists() else None
    output = pipeline.reconcile_merge(inputs, context)
    assert requested.read_bytes() == MERGED
    assert output.output_path == str(requested.resolve())
    assert sha256_file(requested) == output.video.sha256
    private = inputs.root / output.video.path
    assert private.read_bytes() == MERGED
    assert private.stat().st_ino != requested.stat().st_ino
    assert private.stat().st_mode & 0o777 == 0o600
    assert requested.stat().st_mode & 0o777 == 0o600
    if inode is not None:
        assert requested.stat().st_ino == inode
    assert len(calls.merges) == 2 and len(calls.probes) == 2
    candidate = calls.merges[-1][1]
    assert candidate.is_relative_to(inputs.root)
    assert candidate.parent.name.startswith(".merge-recovery-")
    assert not candidate.parent.exists()
    assert calls.merges[-1][2]["replace_existing"] is False
    assert all(path != requested for path in calls.probes)
    assert calls.commands == [(inputs.root / inputs.scenes[0].artifact.video_path,)] * 2


def test_identical_unreferenced_output_is_adopted_only_after_recomputation(recovery):
    inputs, context, pipeline, calls = recovery
    inputs.output_path.write_bytes(MERGED)
    inode = inputs.output_path.stat().st_ino
    output = pipeline.reconcile_merge(inputs, context)
    assert inputs.existing_output_sha256 == ""
    assert inputs.output_path.stat().st_ino == inode
    assert len(calls.commands) == 1 and len(calls.probes) == 1
    assert (inputs.root / output.video.path).read_bytes() == MERGED
    inputs.output_path.write_bytes(b"public path remains mutable")
    assert (inputs.root / output.video.path).read_bytes() == MERGED


def test_reconciliation_never_discovers_media_or_uses_live_merge_policy(recovery, monkeypatch):
    inputs, context, pipeline, calls = recovery
    pinned_crf = inputs.config.resolve().MERGE_VIDEO_CRF
    monkeypatch.setattr(merger_module.settings, "MERGE_VIDEO_CRF", (pinned_crf + 1) % 52)

    def reject_scan(*args, **kwargs):
        pytest.fail("Recovery must use exact recorded scene artifacts, never scan media")

    monkeypatch.setattr(Path, "glob", reject_scan)
    monkeypatch.setattr(Path, "rglob", reject_scan)
    output = pipeline.reconcile_merge(inputs, context)
    assert len(calls.merges) == 1
    assert calls.merges[0][2]["merge_profile"] == inputs.merge_profile
    assert sha256_file(inputs.output_path) == output.video.sha256


@pytest.mark.parametrize("external", [False, True])
def test_unrelated_output_fails_closed_even_with_force_or_run_local_path(recovery, external):
    inputs, context, pipeline, calls = recovery
    values = json.loads(inputs.config.values_json)
    values["OVERWRITE_OUTPUT"] = True
    inputs = replace(inputs, config=RunConfig(values_json=json.dumps(values)))
    if external:
        inputs = replace(inputs, output_path=external_output(inputs))
    inputs.output_path.parent.mkdir(parents=True, exist_ok=True)
    unrelated = b"unrelated but probe-compatible movie"
    inputs.output_path.write_bytes(unrelated)
    before = inputs.output_path.stat()
    with pytest.raises(ValueError, match="differs from the verified merge candidate"):
        pipeline.reconcile_merge(inputs, context)
    after = inputs.output_path.stat()
    assert inputs.output_path.read_bytes() == unrelated
    assert (after.st_ino, after.st_mtime_ns, after.st_mode) == (
        before.st_ino,
        before.st_mtime_ns,
        before.st_mode,
    )
    assert len(calls.probes) == 1 and inputs.output_path not in calls.probes


def test_previously_authorized_external_output_can_be_atomically_replaced(recovery):
    inputs, context, pipeline, _ = recovery
    requested = external_output(inputs)
    requested.parent.mkdir()
    requested.write_bytes(b"previously authorized same-run result")
    inode = requested.stat().st_ino
    inputs = replace(inputs, output_path=requested, existing_output_sha256=sha256_file(requested))
    output = pipeline.reconcile_merge(inputs, context)
    assert requested.read_bytes() == MERGED and requested.stat().st_ino != inode
    assert output.output_path == str(requested)
    assert (inputs.root / output.video.path).read_bytes() == MERGED


def test_authorized_hash_is_rechecked_under_the_existing_output_lock(recovery, monkeypatch):
    inputs, context, pipeline, _ = recovery
    requested = inputs.output_path
    requested.write_bytes(b"previous same-run output")
    inputs = replace(inputs, existing_output_sha256=sha256_file(requested))
    original_copy = finalization_module.shutil.copyfileobj
    intercepted = []

    def racing_copy(source, destination, **kwargs):
        original_copy(source, destination, **kwargs)
        if Path(source.name).parent == inputs.root / "artifacts/videos":
            with pytest.raises(RuntimeError), VideoMerger._output_lock(requested):
                pytest.fail("Reconciliation must hold the standard output lock")
            requested.write_bytes(b"unrelated racing replacement")
            intercepted.append(True)

    monkeypatch.setattr(finalization_module.shutil, "copyfileobj", racing_copy)
    with pytest.raises(ValueError, match="authorized final output changed"):
        pipeline.reconcile_merge(inputs, context)
    assert intercepted == [True]
    assert requested.read_bytes() == b"unrelated racing replacement"


@pytest.mark.parametrize("writer_bytes", [b"unrelated racing writer", MERGED])
def test_absent_output_publication_never_clobbers_a_racing_writer(
    recovery, monkeypatch, writer_bytes
):
    inputs, context, pipeline, _ = recovery
    original_link = os.link
    raced = []

    def racing_link(source, destination, **kwargs):
        if Path(destination) == inputs.output_path:
            assert Path(source).read_bytes() == MERGED
            assert Path(source).stat().st_mode & 0o777 == 0o600
            inputs.output_path.write_bytes(writer_bytes)
            raced.append(inputs.output_path.stat().st_ino)
        return original_link(source, destination, **kwargs)

    monkeypatch.setattr(finalization_module.os, "link", racing_link)
    if writer_bytes == MERGED:
        output = pipeline.reconcile_merge(inputs, context)
        assert sha256_file(inputs.output_path) == output.video.sha256
    else:
        with pytest.raises(ValueError, match="appeared with unrelated bytes"):
            pipeline.reconcile_merge(inputs, context)
    assert inputs.output_path.read_bytes() == writer_bytes
    assert inputs.output_path.stat().st_ino == raced[0]


@pytest.mark.parametrize(
    "mutation", ["source", "video", "job_hash", "metadata", "environment", "profile"]
)
def test_exact_input_tampering_refuses_even_an_identical_public_video(recovery, mutation):
    inputs, context, pipeline, calls = recovery
    inputs.output_path.write_bytes(MERGED)
    scene = inputs.scenes[0].model_copy(deep=True)
    if mutation == "source":
        (inputs.root / scene.code_file).write_text("tampered source")
    elif mutation == "video":
        (inputs.root / scene.artifact.video_path).write_bytes(b"tampered artifact")
    elif mutation == "job_hash":
        scene.slurm_job.output_sha256 = "0" * 64
    elif mutation == "metadata":
        scene.artifact.metadata = scene.artifact.metadata.model_copy(update={"size_bytes": 1})
    elif mutation == "environment":
        scene.artifact.environment_fingerprint = {"execution_sha256": "0" * 64}
    else:
        inputs = replace(
            inputs, render_profile=inputs.render_profile.model_copy(update={"frame_rate": 30})
        )
    inputs = replace(inputs, scenes=(scene,))
    with pytest.raises(ValueError):
        pipeline.reconcile_merge(inputs, context)
    assert inputs.output_path.read_bytes() == MERGED
    assert not calls.merges and not calls.probes


def test_artifact_tampering_during_recompute_blocks_publication(recovery):
    inputs, context, pipeline, _ = recovery
    merge = pipeline.merger.merge

    def tampering_merge(*args, **kwargs):
        result = merge(*args, **kwargs)
        (inputs.root / inputs.scenes[0].artifact.video_path).write_bytes(b"tampered after reading")
        return result

    pipeline.merger.merge = tampering_merge
    with pytest.raises(ValueError):
        pipeline.reconcile_merge(inputs, context)
    assert not inputs.output_path.exists()


def test_failed_candidate_verification_cannot_adopt_identical_public_bytes(recovery):
    inputs, context, pipeline, calls = recovery
    inputs.output_path.write_bytes(MERGED)
    calls.verified = False
    with pytest.raises(RuntimeError, match="FFmpeg"):
        pipeline.reconcile_merge(inputs, context)
    assert inputs.output_path.read_bytes() == MERGED
    assert not (inputs.root / "artifacts/videos").exists()


def test_existing_standard_output_lock_blocks_reconciliation_publication(recovery):
    inputs, context, pipeline, _ = recovery
    with VideoMerger._output_lock(inputs.output_path), pytest.raises(RuntimeError):
        pipeline.reconcile_merge(inputs, context)
    assert not inputs.output_path.exists()


@pytest.mark.parametrize("moment", ["before", "after"])
def test_reconciliation_lost_result_around_authorized_replace_is_recoverable(
    recovery, monkeypatch, moment
):
    inputs, context, pipeline, _ = recovery
    inputs.output_path.write_bytes(b"old authorized output")
    inputs = replace(inputs, existing_output_sha256=sha256_file(inputs.output_path))
    original_replace = os.replace

    def lost_replace(source, destination, **kwargs):
        if Path(destination) == inputs.output_path:
            if moment == "after":
                original_replace(source, destination, **kwargs)
            raise OSError("reconciliation checkpoint lost")
        return original_replace(source, destination, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(finalization_module.os, "replace", lost_replace)
        with pytest.raises(OSError, match="checkpoint lost"):
            pipeline.reconcile_merge(inputs, context)
    assert inputs.output_path.read_bytes() == (
        MERGED if moment == "after" else b"old authorized output"
    )
    result = pipeline.reconcile_merge(inputs, context)
    assert inputs.output_path.read_bytes() == MERGED
    assert sha256_file(inputs.output_path) == result.video.sha256


def test_publish_copy_fsync_failure_leaves_no_half_output(recovery, monkeypatch):
    inputs, context, pipeline, _ = recovery
    original_fsync = os.fsync

    def failed_fsync(descriptor):
        path = Path(os.readlink(f"/proc/self/fd/{descriptor}"))
        if path.parent == inputs.output_path.parent and path.name.startswith(".merge-recovery-"):
            raise OSError("publication copy fsync failed")
        return original_fsync(descriptor)

    monkeypatch.setattr(finalization_module.os, "fsync", failed_fsync)
    with pytest.raises(OSError, match="fsync failed"):
        pipeline.reconcile_merge(inputs, context)
    assert not inputs.output_path.exists()
    assert not list(inputs.output_path.parent.glob(".merge-recovery-*"))


def test_reused_scene_requires_current_typed_provenance_for_reconciliation(incremental):
    result = IncrementalPipeline().reuse(incremental, reuse_context())
    assert result.eligible
    scene_inputs = replace(incremental.current, scene=result.scene)
    inputs = replace(
        final_inputs(scene_inputs), visual_eval_profile=VisualEvalProfile(enabled=False)
    )
    calls = []

    class Merger:
        def merge(self, videos, output, **kwargs):
            assert videos == [inputs.root / result.video.path]
            assert kwargs["render_profile"] == inputs.render_profile
            assert kwargs["merge_profile"] == inputs.merge_profile
            calls.append(tuple(videos))
            output.write_bytes(b"merged:" + videos[0].read_bytes())
            return output

    pipeline = FinalizationPipeline(Merger())
    with pytest.raises(ValueError):
        pipeline.reconcile_merge(inputs, reuse_context())
    assert not calls and not inputs.output_path.exists()
    inputs = replace(inputs, reuse_receipts={1: result.provenance})
    output = pipeline.reconcile_merge(inputs, reuse_context())
    assert len(calls) == 1
    assert (
        inputs.output_path.read_bytes()
        == b"merged:" + (inputs.root / result.video.path).read_bytes()
    )
    assert sha256_file(inputs.output_path) == output.video.sha256
    receipt = inputs.root / result.provenance.path
    receipt.write_text("{}")
    with pytest.raises(ValueError):
        pipeline.reconcile_merge(inputs, reuse_context())
    assert len(calls) == 1
