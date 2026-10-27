"""Single-observation render stages bound to immutable scene and execution inputs."""

from __future__ import annotations

import json
from pathlib import Path

from kd1_anime.agent_runtime.state import AgentSubmission
from kd1_anime.agent_runtime.tools import ToolContext
from kd1_anime.agents.capability import build_capability_contract, validate_capability_contract
from kd1_anime.candidate_acceptor import CandidateAcceptor
from kd1_anime.cluster.jobs import RenderJob
from kd1_anime.cluster.monitoring import JobMonitor
from kd1_anime.domain.execution import SubmissionAttempt
from kd1_anime.pipeline.contracts import RenderingOutput, SceneInputs
from kd1_anime.rendering import SceneArtifact, sha256_file
from kd1_anime.run_store import restore_render_job, restore_run_path, store_render_job
from kd1_anime.services.rendering import RenderingService


def _private_path(root: Path, path: Path) -> str:
    relative = path.relative_to(root.resolve()).as_posix()
    if restore_run_path(root, relative) != path:
        raise ValueError("Render path identity differs from the private run")
    return relative


def validate_artifact(root, run_id, scene, profile) -> Path:
    """Bind an exact artifact to the current job; never discover a different video."""
    artifact, stored = scene.artifact, scene.slurm_job
    if artifact is None or stored is None or not scene.rendered or not artifact.verified:
        raise ValueError("Scene has no verified current-job artifact identity")
    if (
        artifact.origin != "rendered"
        or artifact.source_run_id != run_id
        or artifact.scene_id != scene.plan.scene_id
        or artifact.job_id != stored.job_id
        or artifact.backend != stored.backend
        or stored.scene_id != scene.plan.scene_id
        or stored.status != "COMPLETED"
        or stored.cancelled
        or artifact.code_sha256 != scene.code_sha256
        or stored.code_sha256 != scene.code_sha256
        or artifact.scene_class_name != scene.class_name
        or stored.scene_class_name != scene.class_name
        or stored.render_profile != profile
        or artifact.render_profile_sha256 != profile.digest()
        or not stored.execution_sha256
        or artifact.execution_sha256 != stored.execution_sha256
        or artifact.video_path != stored.output_path
        or artifact.video_sha256 != stored.output_sha256
        or artifact.metadata != stored.output_metadata
        or artifact.environment_fingerprint != stored.environment_fingerprint
        or artifact.environment_warning != stored.environment_warning
    ):
        raise ValueError("Scene artifact identity differs from the current render job")
    job = restore_render_job(stored, root)
    path = _exact_output(root, job)
    if sha256_file(path) != artifact.video_sha256:
        raise ValueError("Scene artifact hash differs from the current render job")
    return path


def _exact_output(root: Path, job: RenderJob) -> Path:
    if job.output_path is None or job.output_metadata is None or not job.output_sha256:
        raise ValueError("Render job lacks exact output metadata and hash")
    path = job.output_path
    _private_path(root, path)
    _private_path(root, job.media_dir)
    if (
        not path.is_relative_to(job.media_dir)
        or path.name != f"{job.scene_class_name}.mp4"
        or "partial_movie_files" in path.parts
        or "__smoke__" in path.parts
        or not path.is_file()
        or path.stat().st_size != job.output_metadata.size_bytes
        or sha256_file(path) != job.output_sha256
        or job.output_metadata.width != job.render_profile.pixel_width
        or job.output_metadata.height != job.render_profile.pixel_height
        or abs(job.output_metadata.frame_rate - job.render_profile.frame_rate) > 0.1
    ):
        raise ValueError("Render output identity, hash, metadata, or profile mismatch")
    return path


class RenderPipeline:
    def __init__(self, backend, *, preflight=None, control=None):
        self.backend = backend
        self.preflight = preflight
        self.control = control
        self._claimed: set[tuple[str, str]] = set()
        self._monitors: dict[tuple[str, str], JobMonitor] = {}

    def _checkpoint(self, context):
        context.checkpoint()
        if self.control is not None:
            self.control.check()

    def _bind(self, inputs):
        inputs.execution_spec.verify_image()
        if inputs.execution_spec.for_profile(inputs.render_profile) != inputs.execution_spec:
            raise ValueError("Render profile differs from the pinned execution identity")
        self.backend.execution_spec = inputs.execution_spec

    def _check_job(self, inputs, job, *, attempt=None):
        scene = inputs.scene
        if (
            job.scene_id != scene.plan.scene_id
            or job.backend != self.backend.name
            or job.code_sha256 != scene.code_sha256
            or job.execution_sha256 != inputs.execution_spec.digest()
            or job.scene_class_name != scene.class_name
            or job.render_profile != inputs.render_profile
            or job.resource_profile != scene.resource_profile
        ):
            raise ValueError("Render job identity differs from the immutable scene inputs")
        for path in (job.script_path, job.log_out, job.log_err, job.media_dir):
            _private_path(inputs.root, path)
        attempt = attempt or scene.submission
        if attempt is not None:
            expected = (
                inputs.root.resolve()
                / "videos"
                / f"scene_{scene.plan.scene_id}"
                / f"attempt_{attempt.token}"
            )
            if (
                job.media_dir != expected
                or attempt.backend != job.backend
                or attempt.code_sha256 != job.code_sha256
                or (attempt.job_id and attempt.job_id != job.job_id)
            ):
                raise ValueError("Render job identity differs from the reserved attempt")

    def _reserve(self, inputs, attempt):
        scene = inputs.scene
        key = (str(inputs.root.resolve()), attempt.token)
        if key in self._claimed:
            raise ValueError("Submission attempt has already been dispatched")
        if (
            attempt.status != "pending"
            or attempt.job_id
            or attempt.backend != self.backend.name
            or attempt.scene_revision != scene.revision
            or attempt.code_sha256 != scene.code_sha256
            or scene.slurm_job is not None
        ):
            raise ValueError("Submission attempt does not match the immutable scene inputs")
        manifest = restore_run_path(inputs.root, "manifest.json")
        try:
            raw = json.loads(manifest.read_text(encoding="utf-8"))
            reservations = [
                AgentSubmission.model_validate(item) for item in raw["agent"]["submissions"]
            ]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError("Submission attempt requires a durable persisted reservation") from exc
        matching = [item for item in reservations if item.attempt.token == attempt.token]
        if (
            len(matching) != 1
            or matching[0].scene_id != scene.plan.scene_id
            or matching[0].attempt != attempt
        ):
            raise ValueError("Submission attempt requires one exact persisted reservation")
        return key

    def submit(
        self, inputs: SceneInputs, context: ToolContext, attempt: SubmissionAttempt
    ) -> RenderingOutput:
        self._checkpoint(context)
        key = self._reserve(inputs, attempt)
        scene = inputs.scene
        expected = f"scenes/revisions/scene_{scene.plan.scene_id}/{scene.code_sha256}.py"
        if scene.code_file != expected:
            raise ValueError("Render source must be an immutable ObjectStore source")
        source = restore_run_path(inputs.root, scene.code_file)
        if not source.is_file() or sha256_file(source) != scene.code_sha256:
            raise ValueError("Render source hash differs from the immutable scene inputs")
        code = source.read_text(encoding="utf-8")
        inspection = CandidateAcceptor().inspect_candidate(
            code,
            scene.plan,
            technical_spec=scene.technical_spec,
            renderer=inputs.render_profile.renderer,
            manim_version=inputs.render_profile.manim_version,
            expected_class_name=scene.class_name,
            allow_repair=False,
            continuity_mode="none" if inputs.direct_render else "contract",
        )
        if not inspection.accepted or inspection.code_sha256 != scene.code_sha256:
            raise ValueError("Read-only render source inspection rejected: " + inspection.feedback)
        settings = inputs.config.resolve()
        contract = build_capability_contract(
            scene.plan,
            scene.technical_spec,
            renderer=inputs.render_profile.renderer,
            render_profile=inputs.render_profile,
            gpu_type=settings.SLURM_GPU_TYPE,
        )
        if scene.capability_contract is not None and scene.capability_contract != contract:
            raise ValueError("Render capability contract differs from immutable planning inputs")
        capability = validate_capability_contract(
            contract,
            render_profile=inputs.render_profile,
            gpu_type=settings.SLURM_GPU_TYPE,
            code=code,
        )
        if contract.scene_id != scene.plan.scene_id or not capability.is_valid:
            raise ValueError("Render capability contract rejected: " + "; ".join(capability.errors))
        with inputs.config.activate():
            self._bind(inputs)
            if self.preflight is not None:
                self.preflight(inputs.render_profile, self.backend)
            else:
                RenderingService(self.backend, self.backend.name).preflight(inputs.render_profile)
            self._checkpoint(context)
            # Claim before dispatch: exceptions/unknown side effects are never generic retries.
            self._claimed.add(key)
            job = self.backend.submit_scene(
                scene.plan.scene_id,
                source,
                scene.class_name,
                scenes_dir=inputs.root.resolve() / "scenes",
                logs_dir=inputs.root.resolve() / "logs",
                videos_dir=inputs.root.resolve() / "videos",
                code_sha256=scene.code_sha256,
                render_profile=inputs.render_profile,
                resource_profile=scene.resource_profile,
                attempt_token=attempt.token,
            )
            try:
                context.record_external_id(job.job_id)
                self._check_job(inputs, job, attempt=attempt)
                stored = store_render_job(job, inputs.root)
            except Exception:
                try:
                    cancelled = self.backend.cancel_job(job.job_id)
                except Exception:
                    cancelled = False
                if not cancelled:
                    raise RuntimeError(
                        "Render identity persistence failed and cancellation failed; resubmit forbidden"
                    ) from None
                raise
        return RenderingOutput(status="pending", job=stored)

    def poll(self, inputs: SceneInputs, context: ToolContext) -> RenderingOutput:
        self._checkpoint(context)
        if inputs.scene.slurm_job is None:
            raise ValueError("Polling requires a durable current render job")
        job = restore_render_job(inputs.scene.slurm_job, inputs.root)
        self._check_job(inputs, job)
        with inputs.config.activate():
            self._bind(inputs)
            key = (str(inputs.root.resolve()), job.job_id)
            monitor = self._monitors.get(key)
            if monitor is None:
                monitor = JobMonitor(self.backend)
                self._monitors[key] = monitor
                monitor.add_job(job)
                if job.status in {"UNKNOWN", "GONE"}:
                    monitor.indeterminate_since[job.job_id] = job.submitted_at
            else:
                monitor.jobs[job.job_id] = job
                if job.job_id in monitor.pending:
                    monitor.pending[job.job_id] = job
            monitor.poll_once()
            job = monitor.jobs[job.job_id]
            success = monitor.results.get(job.job_id)
            artifact = None
            if success:
                try:
                    self._check_job(inputs, job)
                    output = _exact_output(inputs.root, job)
                    if not inputs.execution_spec.matches_environment(
                        job.environment_fingerprint
                    ) or (
                        inputs.render_profile.renderer == "opengl"
                        and job.environment_fingerprint.get("pyopengl_platform")
                        != inputs.render_profile.opengl_platform
                    ):
                        raise ValueError(
                            "Render environment differs from the immutable execution identity"
                        )
                    artifact = SceneArtifact(
                        execution_sha256=job.execution_sha256,
                        origin="rendered",
                        source_run_id=inputs.run_id,
                        job_id=job.job_id,
                        backend=job.backend,
                        scene_id=job.scene_id,
                        scene_class_name=job.scene_class_name,
                        code_sha256=job.code_sha256,
                        render_profile_sha256=job.render_profile.digest(),
                        video_path=_private_path(inputs.root, output),
                        video_sha256=job.output_sha256,
                        metadata=job.output_metadata,
                        environment_fingerprint=dict(job.environment_fingerprint),
                        environment_warning=job.environment_warning,
                    )
                except (OSError, ValueError) as exc:
                    job.status = "FAILED"
                    job.failure_reason = str(exc)
                    success = False
            status = "rendered" if success else "pending" if success is None else "failed"
            if job.status == "CANCEL_FAILED":
                status = "unknown"
            return RenderingOutput(
                status=status, job=store_render_job(job, inputs.root), artifact=artifact
            )

    def cancel(self, inputs: SceneInputs, context: ToolContext) -> RenderingOutput:
        self._checkpoint(context)
        if inputs.scene.slurm_job is None:
            raise ValueError("Cancellation requires a durable current render job")
        job = restore_render_job(inputs.scene.slurm_job, inputs.root)
        self._check_job(inputs, job)
        with inputs.config.activate():
            self._bind(inputs)
            try:
                cancelled = self.backend.cancel_job(job.job_id)
            except Exception:
                cancelled = False
        job.cancelled = cancelled
        job.status = "CANCELLED" if cancelled else "CANCEL_FAILED"
        job.failure_reason = (
            "Cancellation requested" if cancelled else "Cancellation failed; resubmit forbidden"
        )
        return RenderingOutput(
            status="cancelled" if cancelled else "unknown", job=store_render_job(job, inputs.root)
        )
