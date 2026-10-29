"""Fail-closed recovery observations and pure resume/retry decisions.

Only the parent publishes replacements. Scheduler queries require an already
persisted reconciliation invocation; cancellation is always a separate tool.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from pydantic import Field

from kd1_anime.agent_runtime.contracts import RuntimeRecord
from kd1_anime.agent_runtime.tools import ToolContext
from kd1_anime.cluster.status import FAILURE_STATES, TERMINAL_STATES
from kd1_anime.domain.dependencies import WorkIdentity
from kd1_anime.domain.execution import SubmissionAttempt
from kd1_anime.pipeline.contracts import SceneInputs
from kd1_anime.pipeline.render import _exact_output, _private_path, validate_artifact
from kd1_anime.rag.models import RagRuntimeProfile
from kd1_anime.rendering import SceneArtifact, sha256_file
from kd1_anime.run_store import (
    RunManifest,
    StoredSceneState,
    StoredSlurmJob,
    restore_render_job,
    restore_run_path,
    store_render_job,
)
from kd1_anime.services.dependencies import DependencyService


class RecoveryOutput(RuntimeRecord):
    status: Literal["unchanged", "adopted", "pending", "rendered", "restart", "failed", "blocked"]
    job: StoredSlurmJob | None = None
    artifact: SceneArtifact | None = None
    submission: SubmissionAttempt | None = None
    invalidate_artifact: bool = False
    reason: str = Field(default="", max_length=50_000)


class RecoveryCancellation(RuntimeRecord):
    scene_id: int = Field(ge=1)
    job_id: str
    identity: WorkIdentity


class RecoveryScenePlan(RuntimeRecord):
    scene_id: int = Field(ge=1)
    identity: WorkIdentity
    replacement: StoredSceneState
    next_action: Literal[
        "detail_scene",
        "propose_technical_spec",
        "propose_code_candidate",
        "review_code",
        "submit_render",
        "poll_render",
        "evaluate_visual",
    ]


class RecoveryPlan(RuntimeRecord):
    scenes: tuple[RecoveryScenePlan, ...] = ()
    cancellations: tuple[RecoveryCancellation, ...] = ()
    blocked: tuple[int, ...] = ()
    invalidate_planning: bool = False
    rag_profile: RagRuntimeProfile | None = None
    drop_rag_receipt_keys: tuple[str, ...] = ()
    clear_final: bool = False


def _identity(scene: StoredSceneState) -> WorkIdentity:
    return WorkIdentity(scene.revision, scene.input_sha256)


def _clear_artifact(scene: StoredSceneState, *, visual_enabled: bool) -> None:
    scene.rendered = False
    scene.artifact = None
    scene.execution_verification = type(scene.execution_verification)()
    scene.visual_verification = type(scene.visual_verification)()
    scene.visual_status = "pending" if visual_enabled else "skipped"
    scene.visual_score = None
    scene.visual_report_file = scene.visual_report_sha256 = scene.visual_artifact_sha256 = ""
    scene.visual_feedback = ""
    scene.visual_best_candidate = None


def _next_action(scene: StoredSceneState, *, direct_render: bool) -> str:
    if scene.rendered:
        return "evaluate_visual"
    if scene.slurm_job is not None:
        return "poll_render"
    if not scene.plan_ready and not direct_render:
        return "detail_scene"
    if not scene.code_file:
        return (
            "propose_technical_spec"
            if scene.technical_status != "passed"
            else "propose_code_candidate"
        )
    return "submit_render" if direct_render or scene.reviewed else "review_code"


def _unsafe(scene: StoredSceneState) -> bool:
    job = scene.slurm_job
    if job is not None:
        return not job.cancelled and job.status not in TERMINAL_STATES
    attempt = scene.submission
    return attempt is not None and attempt.status in {"pending", "unknown", "submitted"}


def plan_retry(manifest: RunManifest, scene_id: int) -> RecoveryPlan:
    """Plan an explicit retry without spending or replenishing any budget.

    A live job produces a cancellation prerequisite, not a speculative scene
    replacement. The parent commits the cancellation tool and calls this again.
    ScenePatch owns the revision increment and recomputation of input digests.
    """
    if manifest.status == "completed":
        raise RuntimeError("Completed runs cannot retry in place; create a new incremental run")
    if scene_id not in manifest.scenes:
        raise ValueError(f"Run does not contain scene {scene_id}")
    previous = manifest.scenes[scene_id]
    job = previous.slurm_job
    if _unsafe(previous):
        if job is None or job.status == "CANCEL_FAILED" or job.backend == "local":
            return RecoveryPlan(blocked=(scene_id,))
        return RecoveryPlan(
            cancellations=(
                RecoveryCancellation(
                    scene_id=scene_id, job_id=job.job_id, identity=_identity(previous)
                ),
            )
        )
    scene = previous.model_copy(deep=True)
    scene.slurm_job = scene.submission = None
    _clear_artifact(scene, visual_enabled=manifest.visual_eval_profile.enabled)
    scene.failed = scene.give_up = False
    scene.failure_reason = scene.failure_category = ""
    scene.phase = "reviewed" if scene.reviewed else "coded" if scene.code_file else "pending"
    return RecoveryPlan(
        scenes=(
            RecoveryScenePlan(
                scene_id=scene_id,
                identity=_identity(previous),
                replacement=scene,
                next_action=_next_action(scene, direct_render=manifest.direct_render),
            ),
        ),
        clear_final=True,
    )


def plan_resume(
    manifest: RunManifest, *, current_rag_profile: RagRuntimeProfile | None = None
) -> RecoveryPlan:
    """Compute recovery invalidations over a snapshot, never touching transports.

    RequestBudget, agent attempts/usage/submissions and every scene retry counter
    remain owned by the existing manifest. No GenerationIdentity is fabricated or
    changed here. The parent must publish these decisions with typed patches.
    """
    if manifest.status == "completed":
        return RecoveryPlan()
    changed_rag = current_rag_profile is not None and current_rag_profile != manifest.rag_profile
    after = manifest.model_copy(deep=True)
    affected: set[int] = set()
    planning_changed = False
    detail_changed: set[int] = set()
    if changed_rag:
        after.rag_profile = current_rag_profile
        affected = DependencyService.affected_manifest(manifest, after, ())
        # Receipt keys are runtime stage:scene:index identities, never display aliases.
        for key, receipt in manifest.rag_receipts.items():
            parts = key.split(":")
            if receipt.stage in {"outline", "continuity"}:
                planning_changed = True
            elif len(parts) >= 2 and parts[1].isdigit() and int(parts[1]) in manifest.scenes:
                scene_id = int(parts[1])
                affected.add(scene_id)
                if receipt.stage == "detail":
                    detail_changed.add(scene_id)
            else:
                planning_changed = True
        if planning_changed:
            affected.update(manifest.scenes)
        # A receipt-dependent producer's future teaching/handoff output is unknown.
        plans = [scene.plan for scene in manifest.scenes.values()]
        graph = manifest.dependency_graph
        fallback = DependencyService.fallback_edges(
            plans, {edge.consumer_scene_id for edge in graph.bindings}, {}
        )
        frontier = set(affected)
        while frontier:
            source = frontier.pop()
            for edge in (*graph.bindings, *fallback):
                if edge.source_scene_id == source and edge.consumer_scene_id not in affected:
                    affected.add(edge.consumer_scene_id)
                    frontier.add(edge.consumer_scene_id)
    replacements = []
    cancellations = []
    blocked = []
    for scene_id, previous in sorted(manifest.scenes.items()):
        invalidate = scene_id in affected
        if not invalidate and not previous.failed and not previous.give_up:
            continue
        if _unsafe(previous):
            job = previous.slurm_job
            if invalidate and job and job.backend == "slurm" and job.status != "CANCEL_FAILED":
                cancellations.append(
                    RecoveryCancellation(
                        scene_id=scene_id, job_id=job.job_id, identity=_identity(previous)
                    )
                )
            else:
                blocked.append(scene_id)
            continue
        scene = previous.model_copy(deep=True)
        scene.failed = scene.give_up = False
        scene.failure_reason = scene.failure_category = ""
        if invalidate:
            scene.code_file = scene.code_sha256 = scene.class_name = ""
            scene.reviewed = False
            scene.technical_spec = None
            scene.technical_spec_sha256 = scene.technical_input_sha256 = ""
            scene.technical_status = "pending"
            scene.technical_error = ""
            scene.capability_contract = scene.resource_profile = None
            scene.capability_status = "pending"
            scene.static_verification = type(scene.static_verification)()
            scene.local_smoke_status = "pending"
            scene.inherited_elements_code = scene.exported_elements_code = ""
            scene.exported_elements = []
            scene.candidates = []
            if planning_changed or scene_id in detail_changed:
                scene.plan_ready = scene.plan_reviewed = False
            scene.slurm_job = scene.submission = None
            _clear_artifact(scene, visual_enabled=manifest.visual_eval_profile.enabled)
        elif scene.slurm_job and not scene.rendered:
            scene.slurm_job = scene.submission = None
            _clear_artifact(scene, visual_enabled=manifest.visual_eval_profile.enabled)
        scene.phase = (
            "rendered"
            if scene.rendered
            else "reviewed"
            if scene.reviewed
            else "coded"
            if scene.code_file
            else "plan_reviewed"
            if scene.plan_reviewed
            else "detailed"
            if scene.plan_ready
            else "pending"
        )
        replacements.append(
            RecoveryScenePlan(
                scene_id=scene_id,
                identity=_identity(previous),
                replacement=scene,
                next_action=_next_action(scene, direct_render=manifest.direct_render),
            )
        )
    # Do not retire old RAG evidence until every required cancellation is confirmed.
    ready = not cancellations and not blocked
    return RecoveryPlan(
        scenes=tuple(replacements),
        cancellations=tuple(cancellations),
        blocked=tuple(blocked),
        invalidate_planning=planning_changed and ready,
        rag_profile=current_rag_profile if changed_rag and ready else None,
        drop_rag_receipt_keys=tuple(manifest.rag_receipts) if changed_rag and ready else (),
        clear_final=bool(replacements),
    )


def validate_completed_run(manifest: RunManifest, root: Path) -> Path:
    """A completed run never falls back to rerendering a missing/corrupt final."""
    if manifest.status != "completed":
        raise ValueError("Final validation requires a completed run")
    if not manifest.final_video or not manifest.final_video_sha256:
        raise RuntimeError("Completed run lacks its final video identity")
    path = Path(manifest.final_video)
    if not path.is_absolute():
        path = restore_run_path(root, manifest.final_video)
    expected = Path(manifest.output_path)
    if not expected.is_absolute():
        expected = root / expected
    if (
        path.is_symlink()
        or any(parent.is_symlink() for parent in path.parents)
        or path.absolute() != expected.absolute()
        or not path.is_file()
        or sha256_file(path) != manifest.final_video_sha256
    ):
        raise RuntimeError("Completed run final video is missing or its identity/hash differs")
    private = manifest.artifact_refs.get("final_video")
    if manifest.schema_version == 11 and not private:
        raise RuntimeError("Completed run lacks its immutable final video reference")
    if private:
        snapshot = restore_run_path(root, private)
        if not snapshot.is_file() or sha256_file(snapshot) != manifest.final_video_sha256:
            raise RuntimeError("Completed run immutable final video is missing or corrupt")
    return path


class RecoveryPipeline:
    def __init__(self, backend, *, control=None):
        self.backend = backend
        self.control = control

    def _checkpoint(self, context: ToolContext) -> None:
        context.checkpoint()
        if self.control is not None:
            self.control.check()

    @staticmethod
    def _durable(inputs: SceneInputs, action_kind: str) -> dict:
        try:
            raw = json.loads(
                restore_run_path(inputs.root, "manifest.json").read_text(encoding="utf-8")
            )
            agent = raw["agent"]
            current = {"revision": inputs.scene.revision, "input_sha256": inputs.scene.input_sha256}
            intents = [
                item
                for item in agent["intents"]
                if (
                    item["action"]["kind"] == action_kind
                    and item["action"]["run_id"] == inputs.run_id
                    and item["action"]["scene_id"] == inputs.scene.plan.scene_id
                    and item["action"]["resource_id"]
                    == f"run:{inputs.run_id}/scene:{inputs.scene.plan.scene_id}"
                    and item["action"]["work_identity"] == current
                    and item["action"]["input_sha256"] == current["input_sha256"]
                    and item["status"] == "dispatched"
                )
            ]
            if not any(
                invocation["intent_id"] == intent["intent_id"]
                and invocation["action"] == action_kind
                and invocation["status"] == "running"
                and invocation["work_identity"] == current
                for intent in intents
                for invocation in agent["invocations"]
            ):
                raise ValueError("No durable reconciliation invocation")
            return raw
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError("Recovery requires a persisted reconciliation tool intent") from exc

    def _source(self, inputs: SceneInputs) -> Path:
        scene = inputs.scene
        expected = f"scenes/revisions/scene_{scene.plan.scene_id}/{scene.code_sha256}.py"
        if not scene.class_name or scene.code_file != expected:
            raise ValueError("Recovery source is not an immutable scene ObjectStore identity")
        source = restore_run_path(inputs.root, scene.code_file)
        if not source.is_file() or sha256_file(source) != scene.code_sha256:
            raise ValueError("Recovery source hash differs from the reserved code")
        if inputs.execution_spec.for_profile(inputs.render_profile) != inputs.execution_spec:
            raise ValueError("Recovery render profile differs from the execution identity")
        return source

    def _reservation(
        self, inputs: SceneInputs, attempt: SubmissionAttempt, agent: dict, *, manifest_state: str
    ) -> tuple[str, str]:
        identity = inputs.submission_identity
        scene = inputs.scene
        if (
            identity is None
            or identity.revision != attempt.scene_revision
            or identity.input_sha256 != scene.input_sha256
            or attempt.backend != self.backend.name
            or attempt.code_sha256 != scene.code_sha256
            or (scene.submission is not None and scene.submission != attempt)
        ):
            raise ValueError("Recovery attempt differs from its original dispatch identity")
        matches = [
            item for item in agent["submissions"] if item["attempt"]["token"] == attempt.token
        ]
        if len(matches) != 1:
            raise ValueError("Recovery requires one exact durable submission reservation")
        saved = matches[0]
        if (
            saved["scene_id"] != scene.plan.scene_id
            or SubmissionAttempt.model_validate(saved["attempt"]) != attempt
        ):
            raise ValueError("Recovery reservation differs from the supplied attempt")
        expected = {"revision": identity.revision, "input_sha256": identity.input_sha256}
        associated_intent = saved.get("intent_id")
        intents = [
            item
            for item in agent["intents"]
            if (
                item["action"]["kind"] == "submit_render"
                and (associated_intent is None or item["intent_id"] == associated_intent)
                and item["action"]["run_id"] == inputs.run_id
                and item["action"]["scene_id"] == scene.plan.scene_id
                and item["action"]["resource_id"]
                == f"run:{inputs.run_id}/scene:{scene.plan.scene_id}"
                and item["action"]["work_identity"] == expected
                and item["action"]["input_sha256"] == expected["input_sha256"]
                and item["action"]["backend"] == attempt.backend
                and item["action"]["class_name"] == scene.class_name
                and item["action"]["source"]["kind"] == "code"
                and item["action"]["source"]["path"] == scene.code_file
                and item["action"]["source"]["sha256"] == attempt.code_sha256
            )
        ]
        if len(intents) != 1:
            raise ValueError("Recovery reservation lacks its exact original dispatch intent")
        invocations = [
            item for item in agent["invocations"] if item["intent_id"] == intents[0]["intent_id"]
        ]
        identifiers = {
            item["external_job_id"] for item in invocations if item.get("external_job_id")
        }
        if len(identifiers) > 1 or (attempt.job_id and identifiers - {attempt.job_id}):
            raise ValueError("Recovery dispatch has conflicting durable returned IDs")
        returned_id = next(iter(identifiers), attempt.job_id)
        cancellation = saved.get("cancellation_status", "not_requested")
        if cancellation not in {"not_requested", "unknown", "succeeded", "failed"}:
            raise ValueError("Recovery reservation has an invalid cancellation status")
        committed = set(agent.get("committed_invocations", ()))
        result_committed = any(item.get("invocation_id") in committed for item in invocations)
        if (
            cancellation == "not_requested"
            and returned_id
            and manifest_state == "ERROR"
            and not result_committed
        ):
            # Failure to persist the cancellation marker is itself uncertain.
            cancellation = "unknown"
        return returned_id, cancellation

    @staticmethod
    def _restore_cancellation(job, cancellation: str) -> None:
        if job.cancelled or job.status in TERMINAL_STATES:
            return
        if cancellation == "succeeded":
            job.cancelled = True
            job.status = "CANCELLED"
            job.failure_reason = "Submission result persistence failed; cancellation was confirmed"
        elif cancellation in {"unknown", "failed"}:
            job.status = "CANCEL_FAILED"
            job.failure_reason = (
                "Submission cancellation failed or is unconfirmed; resubmission forbidden"
            )

    def _job(self, inputs: SceneInputs, job, attempt: SubmissionAttempt | None) -> None:
        scene = inputs.scene
        if (
            job.scene_id != scene.plan.scene_id
            or job.backend != self.backend.name
            or job.execution_sha256 != inputs.execution_spec.digest()
            or job.code_sha256 != scene.code_sha256
            or job.scene_class_name != scene.class_name
            or job.render_profile != inputs.render_profile
            or job.resource_profile != scene.resource_profile
        ):
            raise ValueError("Recovered job differs from immutable scene/execution inputs")
        for path in (job.script_path, job.log_out, job.log_err, job.media_dir):
            _private_path(inputs.root, path)
        root = inputs.root.resolve()
        if attempt is not None and (
            job.media_dir
            != root / "videos" / f"scene_{scene.plan.scene_id}" / f"attempt_{attempt.token}"
            or attempt.backend != job.backend
            or attempt.code_sha256 != job.code_sha256
            or (attempt.job_id and attempt.job_id != job.job_id)
        ):
            raise ValueError("Recovered job differs from the reserved token or returned ID")
        stored = scene.slurm_job
        if stored is not None and stored.job_id != job.job_id:
            raise ValueError("Recovered job differs from the already persisted job ID")
        log_out = (
            restore_run_path(inputs.root, stored.log_out)
            if stored
            else (root / "logs" / f"scene_{scene.plan.scene_id}_{job.job_id}.out")
        )
        log_err = (
            restore_run_path(inputs.root, stored.log_err)
            if stored
            else (root / "logs" / f"scene_{scene.plan.scene_id}_{job.job_id}.err")
        )
        if job.backend == "slurm" and (
            not re.fullmatch(r"\d+", job.job_id)
            or job.script_path != root / "scenes" / f"render_{scene.plan.scene_id}.sh"
            or job.log_out != log_out
            or job.log_out.parent != root / "logs"
            or job.log_err != log_err
            or job.log_err.parent != root / "logs"
        ):
            raise ValueError("Recovered job has an invalid current private path or Slurm ID")

    def reconcile_submission(
        self, inputs: SceneInputs, attempt: SubmissionAttempt, context: ToolContext
    ) -> RecoveryOutput:
        self._checkpoint(context)
        raw = self._durable(inputs, "reconcile_submission")
        source = self._source(inputs)
        returned_id, cancellation = self._reservation(
            inputs, attempt, raw["agent"], manifest_state=raw.get("state", "")
        )
        stored = inputs.scene.slurm_job
        if stored is not None:
            if returned_id and stored.job_id != returned_id:
                raise ValueError("Saved render job differs from the durable dispatch returned ID")
            job = restore_render_job(stored, inputs.root)
            self._job(inputs, job, attempt)
            self._restore_cancellation(job, cancellation)
            replacement = store_render_job(job, inputs.root)
            return RecoveryOutput(
                status="unchanged" if replacement == stored else "adopted",
                job=replacement,
                submission=attempt,
            )
        if attempt.status == "cancelled":
            return RecoveryOutput(status="restart", submission=attempt)
        if attempt.backend == "local":
            return RecoveryOutput(
                status="blocked",
                submission=attempt,
                reason="Historical local processes cannot be adopted or restarted without confirmed termination",
            )
        # A returned ID is already authoritative: reconstruct that exact job,
        # never search by name for another ID and never dispatch another job.
        if returned_id:
            from kd1_anime.cluster.jobs import RenderJob

            root = inputs.root.resolve()
            job = RenderJob(
                job_id=returned_id,
                scene_id=inputs.scene.plan.scene_id,
                script_path=root / "scenes" / f"render_{inputs.scene.plan.scene_id}.sh",
                log_out=root / "logs" / f"scene_{inputs.scene.plan.scene_id}_{returned_id}.out",
                log_err=root / "logs" / f"scene_{inputs.scene.plan.scene_id}_{returned_id}.err",
                media_dir=root
                / "videos"
                / f"scene_{inputs.scene.plan.scene_id}"
                / f"attempt_{attempt.token}",
                scene_class_name=inputs.scene.class_name,
                submitted_at=attempt.created_at.timestamp(),
                execution_sha256=inputs.execution_spec.digest(),
                code_sha256=attempt.code_sha256,
                render_profile=inputs.render_profile,
                resource_profile=inputs.scene.resource_profile,
            )
        else:
            if attempt.status not in {"pending", "unknown"}:
                return RecoveryOutput(
                    status="blocked",
                    submission=attempt,
                    reason="Submitted attempt has no durable returned ID",
                )
            recover = getattr(self.backend, "recover_submission", None)
            if not callable(recover):
                return RecoveryOutput(
                    status="blocked",
                    submission=attempt,
                    reason="Backend cannot confirm the unique historical submission",
                )
            with inputs.config.activate():
                inputs.execution_spec.verify_image()
                self.backend.execution_spec = inputs.execution_spec
                self._checkpoint(context)
                try:
                    job = recover(
                        attempt.model_copy(deep=True),
                        scene_id=inputs.scene.plan.scene_id,
                        source=source,
                        root=inputs.root.resolve(),
                        render_profile=inputs.render_profile,
                    )
                except (OSError, RuntimeError, TimeoutError) as exc:
                    return RecoveryOutput(status="blocked", submission=attempt, reason=str(exc))
            if job is None:
                return RecoveryOutput(
                    status="blocked",
                    submission=attempt,
                    reason="No unique confirmed Slurm record; automatic resubmission forbidden",
                )
            # The scheduler lookup contains no class/resource metadata. Restore
            # those only from the immutable reserved scene, not a guessed class.
            if job.scene_class_name not in {"", inputs.scene.class_name} or (
                job.resource_profile is not None
                and job.resource_profile != inputs.scene.resource_profile
            ):
                raise ValueError("Recovered scheduler metadata conflicts with the reserved scene")
            job.scene_class_name = inputs.scene.class_name
            job.resource_profile = (
                inputs.scene.resource_profile.model_copy(deep=True)
                if inputs.scene.resource_profile
                else None
            )
        self._job(inputs, job, attempt)
        self._restore_cancellation(job, cancellation)
        context.record_external_id(job.job_id)
        confirmed = attempt.model_copy(update={"status": "submitted", "job_id": job.job_id})
        return RecoveryOutput(
            status="adopted", job=store_render_job(job, inputs.root), submission=confirmed
        )

    def _artifact(self, inputs: SceneInputs, job) -> SceneArtifact:
        self._job(inputs, job, inputs.scene.submission)
        output = _exact_output(inputs.root, job)
        if not inputs.execution_spec.matches_environment(job.environment_fingerprint) or (
            inputs.render_profile.renderer == "opengl"
            and job.environment_fingerprint.get("pyopengl_platform")
            != inputs.render_profile.opengl_platform
        ):
            raise ValueError(
                "Recovered artifact environment differs from the pinned execution identity"
            )
        return SceneArtifact(
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

    def reconcile_scene(self, inputs: SceneInputs, context: ToolContext) -> RecoveryOutput:
        """Observe one restored job without JobMonitor's implicit cancellations."""
        self._checkpoint(context)
        raw = self._durable(inputs, "reconcile_render")
        self._source(inputs)
        scene = inputs.scene
        invalid = False
        if scene.rendered:
            try:
                validate_artifact(inputs.root, inputs.run_id, scene, inputs.render_profile)
                if scene.slurm_job.execution_sha256 != inputs.execution_spec.digest():
                    raise ValueError("Rendered scene execution identity differs")
            except (OSError, RuntimeError, ValueError) as exc:
                invalid = True
                reason = str(exc)
            else:
                return RecoveryOutput(
                    status="rendered", job=scene.slurm_job, artifact=scene.artifact
                )
        else:
            reason = ""
        if scene.slurm_job is None:
            if _unsafe(scene):
                return RecoveryOutput(
                    status="blocked",
                    invalidate_artifact=invalid,
                    reason="Unconfirmed submission must be reconciled before restarting",
                )
            return RecoveryOutput(
                status="restart" if invalid else "unchanged",
                invalidate_artifact=invalid,
                reason=reason,
            )
        job = restore_render_job(scene.slurm_job, inputs.root)
        self._job(inputs, job, scene.submission)
        if scene.submission is not None:
            returned_id, cancellation = self._reservation(
                inputs, scene.submission, raw["agent"], manifest_state=raw.get("state", "")
            )
            if returned_id and returned_id != job.job_id:
                raise ValueError("Restored job differs from the durable dispatch returned ID")
            self._restore_cancellation(job, cancellation)
        if job.backend == "local":
            restart = job.cancelled or job.status in TERMINAL_STATES
            return RecoveryOutput(
                status="restart" if restart else "blocked",
                job=store_render_job(job, inputs.root),
                invalidate_artifact=invalid,
                reason="Historical local processes are not adopted; confirmed termination is required",
            )
        with inputs.config.activate():
            inputs.execution_spec.verify_image()
            self.backend.execution_spec = inputs.execution_spec
            self._checkpoint(context)
            try:
                observation = self.backend.poll([job.job_id])
            except (OSError, RuntimeError, TimeoutError) as exc:
                return RecoveryOutput(
                    status="blocked",
                    job=store_render_job(job, inputs.root),
                    invalidate_artifact=invalid,
                    reason=str(exc),
                )
            status = observation.statuses.get(job.job_id, "UNKNOWN")
            if job.job_id in observation.start_times:
                job.started_at = observation.start_times[job.job_id]
            # Failed cancellation cannot be erased by an absent record.
            if job.status == "CANCEL_FAILED" and status not in TERMINAL_STATES:
                return RecoveryOutput(
                    status="blocked",
                    job=store_render_job(job, inputs.root),
                    invalidate_artifact=invalid,
                    reason="Cancellation failed; resubmission forbidden",
                )
            job.status = status
            if invalid and status in TERMINAL_STATES | {"GONE"}:
                return RecoveryOutput(
                    status="restart",
                    job=store_render_job(job, inputs.root),
                    invalidate_artifact=True,
                    reason=reason,
                )
            if status == "GONE":
                outcome = self.backend.classify_gone(job)
                if outcome is None:
                    return RecoveryOutput(
                        status="restart",
                        job=store_render_job(job, inputs.root),
                        invalidate_artifact=invalid,
                        reason="Confirmed absent from queue/accounting with no execution evidence",
                    )
                job.status = outcome
            if job.status == "COMPLETED":
                try:
                    valid = self.backend.validate_completed_job(job)
                    artifact = self._artifact(inputs, job) if valid else None
                except (OSError, RuntimeError, ValueError) as exc:
                    artifact = None
                    job.failure_reason = str(exc)
                if artifact is not None:
                    return RecoveryOutput(
                        status="rendered", job=store_render_job(job, inputs.root), artifact=artifact
                    )
                # Keep completed jobs during output propagation; parent monitoring
                # owns its existing bounded artifact grace/cancellation policy.
                return RecoveryOutput(
                    status="pending",
                    job=store_render_job(job, inputs.root),
                    invalidate_artifact=invalid,
                    reason=job.failure_reason,
                )
        if job.cancelled or job.status == "CANCELLED":
            result = "restart"
        elif job.status in FAILURE_STATES:
            result = "failed"
        elif job.status in {"UNKNOWN", "GONE", "CANCEL_FAILED"}:
            result = "blocked"
        else:
            result = "pending"
        return RecoveryOutput(
            status=result,
            job=store_render_job(job, inputs.root),
            invalidate_artifact=invalid,
            reason=job.failure_reason or reason,
        )
