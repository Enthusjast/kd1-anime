"""Stage boundaries use real immutable files and fake backend/media transports."""

import json
import time
from dataclasses import replace
from pathlib import Path

import pytest
from test_application import SOURCE, VIDEO, ArtifactBackend, ArtifactMerger

from kd1_anime.agent_runtime.tools import ToolContext
from kd1_anime.config import settings
from kd1_anime.domain.execution import SubmissionAttempt
from kd1_anime.domain.planning import LessonSpec, ScenePlan, TeachingGraph
from kd1_anime.domain.runtime import ExecutionSpec, RunConfig
from kd1_anime.objects import ObjectStore
from kd1_anime.pipeline.contracts import FinalizationInputs, SceneInputs
from kd1_anime.pipeline.finalization import FinalizationPipeline
from kd1_anime.pipeline.render import RenderPipeline
from kd1_anime.rendering import MergeProfile, RenderProfile, sha256_file
from kd1_anime.run_store import StoredSceneState, sha256_text


@pytest.fixture
def stage(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "SLURM_REQUIRE_CONTAINER", False)
    monkeypatch.setattr(settings, "SLURM_CONTAINER_IMAGE", None)
    monkeypatch.setattr(settings, "ENABLE_VISUAL_EVAL", False)
    monkeypatch.setattr(settings, "ENABLE_AUTO_EVAL", False)
    profile = RenderProfile(
        renderer="cairo",
        quality="l",
        pixel_width=854,
        pixel_height=480,
        frame_rate=15,
        manim_version="fixture manim",
        ffmpeg_version="fixture ffmpeg",
        xelatex_version="fixture xelatex",
    )
    plan = ScenePlan(
        scene_id=1,
        title="Pause",
        duration_seconds=1,
        purpose="Pause",
        math_concept="Time",
        visual_design="Empty frame",
        camera_movement="Fixed",
        visual_flow=["Wait"],
        key_moments=["Pause"],
        computation="None",
    )
    source = ObjectStore(tmp_path).source(1, SOURCE)
    scene = StoredSceneState(
        plan=plan,
        revision=7,
        code_file=source,
        code_sha256=sha256_text(SOURCE),
        class_name="Pause",
        reviewed=True,
    )
    inputs = SceneInputs(
        "20261027-120000-aabbccdd",
        tmp_path,
        RunConfig.capture(),
        profile,
        ExecutionSpec.capture(profile),
        scene,
        (plan,),
        LessonSpec(),
        TeachingGraph(),
        None,
    )
    attempt = SubmissionAttempt(
        backend="slurm", scene_revision=scene.revision, code_sha256=scene.code_sha256
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "agent": {
                    "submissions": [{"scene_id": 1, "attempt": attempt.model_dump(mode="json")}]
                }
            }
        )
    )
    external_ids = []
    context = ToolContext(
        deadline=time.monotonic() + 60,
        cancelled=lambda: False,
        record_external_id=external_ids.append,
    )
    backend = ArtifactBackend()
    pipeline = RenderPipeline(backend, preflight=lambda profile, backend: None, control=None)
    return inputs, attempt, context, external_ids, backend, pipeline


def rendered(stage):
    inputs, attempt, context, _, _, pipeline = stage
    submitted = pipeline.submit(inputs, context, attempt)
    inputs = replace(
        inputs,
        scene=inputs.scene.model_copy(update={"slurm_job": submitted.job, "submission": attempt}),
    )
    output = pipeline.poll(inputs, context)
    assert output.status == "rendered"
    return replace(
        inputs,
        scene=inputs.scene.model_copy(
            update={"slurm_job": output.job, "artifact": output.artifact, "rendered": True}
        ),
    )


def final_inputs(inputs, output=None):
    return FinalizationInputs(
        inputs.run_id,
        inputs.root,
        inputs.config,
        inputs.render_profile,
        MergeProfile.current(),
        (inputs.scene,),
        output or inputs.root / "final.mp4",
    )


def test_submit_records_external_id_and_never_duplicates_token(stage):
    inputs, attempt, context, external_ids, backend, pipeline = stage
    result = pipeline.submit(inputs, context, attempt)
    assert result.status == "pending" and external_ids == [result.job.job_id]
    assert result.job.code_sha256 == inputs.scene.code_sha256
    assert result.job.execution_sha256 == inputs.execution_spec.digest()
    assert result.job.media_dir.endswith("attempt_" + attempt.token)
    with pytest.raises(ValueError):
        pipeline.submit(inputs, context, attempt)
    assert len(backend.sources) == 1


def test_submission_requires_exact_durable_reservation(stage):
    inputs, attempt, context, _, backend, pipeline = stage
    (inputs.root / "manifest.json").write_text('{"agent":{"submissions":[]}}')
    with pytest.raises(ValueError):
        pipeline.submit(inputs, context, attempt)
    assert not backend.sources


@pytest.mark.parametrize("mutation", ["hash", "path", "class", "capability"])
def test_submit_checks_source_without_mutating_it(stage, mutation):
    inputs, attempt, context, _, backend, pipeline = stage
    original = inputs.root / inputs.scene.code_file
    before = original.read_bytes()
    if mutation == "hash":
        inputs = replace(inputs, scene=inputs.scene.model_copy(update={"code_sha256": "0" * 64}))
    elif mutation == "path":
        alias = inputs.root / "mutable.py"
        alias.write_bytes(before)
        inputs = replace(inputs, scene=inputs.scene.model_copy(update={"code_file": "mutable.py"}))
    elif mutation == "class":
        inputs = replace(inputs, scene=inputs.scene.model_copy(update={"class_name": "Missing"}))
    else:
        from kd1_anime.agents.capability import build_capability_contract

        contract = build_capability_contract(inputs.scene.plan, renderer="opengl")
        inputs = replace(
            inputs, scene=inputs.scene.model_copy(update={"capability_contract": contract})
        )
    with pytest.raises(ValueError):
        pipeline.submit(inputs, context, attempt)
    assert original.read_bytes() == before and not backend.sources


@pytest.mark.parametrize("cancelled", [True, False])
def test_external_identity_persistence_failure_cancels_and_blocks_resubmit(stage, cancelled):
    inputs, attempt, _, _, backend, pipeline = stage
    cancellations = []

    def persist(_):
        raise OSError("durable identity unavailable")

    backend.cancel_job = lambda job_id: cancellations.append(job_id) or cancelled
    context = ToolContext(
        deadline=time.monotonic() + 60, cancelled=lambda: False, record_external_id=persist
    )
    with pytest.raises((OSError, RuntimeError)):
        pipeline.submit(inputs, context, attempt)
    assert cancellations == ["1"]
    with pytest.raises(ValueError):
        pipeline.submit(inputs, context, attempt)
    assert len(backend.sources) == 1


def test_wrong_returned_job_identity_is_cancelled_after_recording(stage):
    inputs, attempt, context, external_ids, backend, pipeline = stage
    original_submit = backend.submit_scene
    cancelled = []

    def submit(*args, **kwargs):
        job = original_submit(*args, **kwargs)
        job.code_sha256 = "0" * 64
        return job

    backend.submit_scene = submit
    backend.cancel_job = lambda job_id: cancelled.append(job_id) or True
    with pytest.raises(ValueError):
        pipeline.submit(inputs, context, attempt)
    assert external_ids == ["1"] and cancelled == ["1"]


def test_poll_records_only_exact_current_job_artifact(stage):
    inputs = rendered(stage)
    artifact = inputs.scene.artifact
    assert artifact.video_sha256 == sha256_text(VIDEO.decode())
    assert artifact.job_id == inputs.scene.slurm_job.job_id
    assert artifact.metadata == inputs.scene.slurm_job.output_metadata
    assert artifact.environment_fingerprint == inputs.scene.slurm_job.environment_fingerprint
    assert artifact.execution_sha256 == inputs.execution_spec.digest()


def test_poll_rejects_environment_mismatch(stage):
    inputs, attempt, context, _, backend, pipeline = stage
    submitted = pipeline.submit(inputs, context, attempt)
    validate = backend.validate_completed_job

    def wrong_environment(job):
        validate(job)
        job.environment_fingerprint["execution_sha256"] = "0" * 64
        return True

    backend.validate_completed_job = wrong_environment
    output = pipeline.poll(
        replace(inputs, scene=inputs.scene.model_copy(update={"slurm_job": submitted.job})), context
    )
    assert output.status == "failed" and output.artifact is None


def test_unknown_poll_is_single_observation_then_bounded_cancel(stage, monkeypatch):
    from kd1_anime.cluster.jobs import PollObservation

    inputs, attempt, context, _, backend, pipeline = stage
    submitted = pipeline.submit(inputs, context, attempt)
    inputs = replace(inputs, scene=inputs.scene.model_copy(update={"slurm_job": submitted.job}))
    observations = []
    backend.poll = lambda job_ids: observations.append(job_ids) or PollObservation({}, {})
    now = [time.time()]
    monkeypatch.setattr("kd1_anime.cluster.monitoring.time.time", lambda: now[0])
    # Capture the explicit limits as run inputs, rather than changing ambient settings.
    values = json.loads(inputs.config.values_json)
    values.update(MONITOR_UNKNOWN_TIMEOUT=1, MONITOR_MAX_UNKNOWN=2)
    inputs = replace(inputs, config=RunConfig(values_json=json.dumps(values)))
    first = pipeline.poll(inputs, context)
    assert first.status == "pending" and len(observations) == 1
    now[0] += 2
    second = pipeline.poll(
        replace(inputs, scene=inputs.scene.model_copy(update={"slurm_job": first.job})), context
    )
    assert second.status == "failed" and len(observations) == 2
    assert second.job.cancelled


def test_merge_keeps_private_immutable_video_for_external_output(stage, tmp_path):
    inputs = rendered(stage)
    requested = tmp_path.parent / (tmp_path.name + "-external.mp4")
    final = final_inputs(inputs, requested)
    pipeline = FinalizationPipeline(ArtifactMerger())
    output = pipeline.merge(final, stage[2])
    assert Path(output.output_path) == requested
    assert not Path(output.video.path).is_absolute()
    private = inputs.root / output.video.path
    assert private.read_bytes() == requested.read_bytes() == b"merged:" + VIDEO
    assert output.video.sha256 == sha256_file(private)
    requested.write_bytes(b"changed external output")
    with pytest.raises(ValueError):
        pipeline.evaluate(final, output, stage[2])


def test_merge_rejects_artifact_not_matching_current_job(stage):
    inputs = rendered(stage)
    artifact = inputs.scene.artifact.model_copy(update={"job_id": "999"})
    inputs = replace(inputs, scene=inputs.scene.model_copy(update={"artifact": artifact}))
    with pytest.raises(ValueError):
        FinalizationPipeline(ArtifactMerger()).merge(final_inputs(inputs), stage[2])


def test_merge_does_not_replace_existing_final_on_atomic_merger_failure(stage, monkeypatch):
    from kd1_anime.media.merger import VideoMerger

    inputs = rendered(stage)
    final = final_inputs(inputs)
    final.output_path.write_bytes(b"previous final")
    merger = VideoMerger()
    monkeypatch.setattr("kd1_anime.media.merger.shutil.which", lambda _: "/fixture/ffmpeg")
    monkeypatch.setattr(merger, "_run_ffmpeg", lambda *args, **kwargs: False)
    values = json.loads(final.config.values_json)
    values["OVERWRITE_OUTPUT"] = True
    final = replace(final, config=RunConfig(values_json=json.dumps(values)))
    with pytest.raises(RuntimeError):
        FinalizationPipeline(merger).merge(final, stage[2])
    assert final.output_path.read_bytes() == b"previous final"


def test_enabled_visual_evaluation_stops_explicitly(stage):
    inputs = rendered(stage)
    final = final_inputs(inputs)
    pipeline = FinalizationPipeline(ArtifactMerger())
    output = pipeline.merge(final, stage[2])
    values = json.loads(final.config.values_json)
    values["ENABLE_VISUAL_EVAL"] = True
    final = replace(final, config=RunConfig(values_json=json.dumps(values)))
    with pytest.raises(NotImplementedError):
        pipeline.evaluate(final, output, stage[2])


def test_tampered_immutable_source_is_rejected_without_repair(stage):
    inputs, attempt, context, _, backend, pipeline = stage
    source = inputs.root / inputs.scene.code_file
    source.write_text(SOURCE + "# changed identity\n")
    before = source.read_bytes()
    with pytest.raises(ValueError):
        pipeline.submit(inputs, context, attempt)
    assert source.read_bytes() == before and not backend.sources


def test_source_symlink_is_never_adopted(stage):
    inputs, attempt, context, _, backend, pipeline = stage
    source = inputs.root / inputs.scene.code_file
    alias = inputs.root / "source-copy.py"
    alias.write_bytes(source.read_bytes())
    source.unlink()
    source.symlink_to(alias)
    with pytest.raises(ValueError):
        pipeline.submit(inputs, context, attempt)
    assert not backend.sources


def test_poll_cancellation_failure_is_unknown_not_retryable(stage, monkeypatch):
    from kd1_anime.cluster.jobs import PollObservation

    inputs, attempt, context, _, backend, pipeline = stage
    submitted = pipeline.submit(inputs, context, attempt)
    values = json.loads(inputs.config.values_json)
    values.update(MONITOR_QUEUE_TIMEOUT=1, MONITOR_TIMEOUT=None)
    inputs = replace(
        inputs,
        config=RunConfig(values_json=json.dumps(values)),
        scene=inputs.scene.model_copy(update={"slurm_job": submitted.job}),
    )
    backend.poll = lambda job_ids: PollObservation({"1": "PENDING"}, {})
    backend.cancel_job = lambda job_id: False
    monkeypatch.setattr(
        "kd1_anime.cluster.monitoring.time.time", lambda: submitted.job.submitted_at + 2
    )
    result = pipeline.poll(inputs, context)
    assert result.status == "unknown" and result.job.status == "CANCEL_FAILED"
    assert len(backend.sources) == 1 and not result.job.cancelled


def test_gone_poll_uses_only_current_job_artifact(stage):
    from kd1_anime.cluster.jobs import PollObservation

    inputs, attempt, context, _, backend, pipeline = stage
    submitted = pipeline.submit(inputs, context, attempt)
    inputs = replace(inputs, scene=inputs.scene.model_copy(update={"slurm_job": submitted.job}))
    backend.poll = lambda job_ids: PollObservation({"1": "GONE"}, {})
    backend.classify_gone = lambda job: "COMPLETED" if backend.validate_completed_job(job) else None
    output = pipeline.poll(inputs, context)
    assert output.status == "rendered" and output.artifact.job_id == "1"


def test_evaluate_rejects_changed_private_object_even_when_external_is_intact(stage):
    inputs = rendered(stage)
    final = final_inputs(inputs)
    pipeline = FinalizationPipeline(ArtifactMerger())
    output = pipeline.merge(final, stage[2])
    (inputs.root / output.video.path).write_bytes(b"corrupt immutable object")
    with pytest.raises(ValueError):
        pipeline.evaluate(final, output, stage[2])
