"""Read-only incremental eligibility and exact, self-contained media reuse.

The caller validates its generation identity before probes, publishes reviewed
continuity in scene order, and commits the returned snapshot through CommitCommand.
This module never loads a manifest, contacts a renderer, or mutates its base run.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from kd1_anime.agent_runtime.contracts import ObjectRef, RuntimeRecord
from kd1_anime.agent_runtime.tools import ToolContext
from kd1_anime.domain.dependencies import WorkIdentity, digest
from kd1_anime.domain.runtime import ExecutionSpec
from kd1_anime.objects import ObjectStore
from kd1_anime.pipeline.contracts import SceneInputs
from kd1_anime.prompt_identity import GenerationIdentity
from kd1_anime.rag.models import RagRuntimeProfile
from kd1_anime.rendering import RenderProfile, SceneArtifact, sha256_file
from kd1_anime.run_store import (
    RunManifest,
    StoredSceneState,
    get_reusable_video_path,
    restore_run_path,
    scene_snapshot,
    sha256_text,
)
from kd1_anime.services.dependencies import DependencyService
from kd1_anime.verification import ExecutionVerification, VisualVerification


@dataclass(frozen=True, slots=True)
class IncrementalInputs:
    base: RunManifest
    base_root: Path
    current: SceneInputs
    generation_identity: GenerationIdentity
    current_scenes: tuple[StoredSceneState, ...]
    rag_profile: RagRuntimeProfile
    continuity_ready: bool
    backend: Literal["slurm", "local"] = "slurm"


class ReuseProvenance(RuntimeRecord):
    """All upstream evidence needed to validate a copy without reopening its base."""

    kind: Literal["incremental_reuse"] = "incremental_reuse"
    version: Literal[1] = 1
    base_run_id: str = Field(pattern=r"^\d{8}-\d{6}-[0-9a-f]{8}$")
    base_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    base_artifact: SceneArtifact
    run_id: str = Field(pattern=r"^\d{8}-\d{6}-[0-9a-f]{8}$")
    scene_id: int = Field(ge=1)
    work_identity: WorkIdentity
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_spec: ExecutionSpec
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    technical_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    inherited_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dependencies_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    globals_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def valid_work_identity(self):
        if self.work_identity.revision < 0 or not self.work_identity.input_sha256:
            raise ValueError("Reuse provenance requires a complete dispatch work identity")
        return self


class IncrementalOutput(RuntimeRecord):
    eligible: bool
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reasons: tuple[str, ...] = ()
    scene: StoredSceneState | None = None
    snapshot: ObjectRef | None = None
    video: ObjectRef | None = None
    provenance: ObjectRef | None = None

    @model_validator(mode="after")
    def complete_copy(self):
        refs = (self.scene, self.snapshot, self.video, self.provenance)
        if self.eligible and (self.reasons or any(item is None for item in refs)):
            raise ValueError("Reusable output requires a complete immutable copy")
        if not self.eligible and (not self.reasons or any(item is not None for item in refs)):
            raise ValueError("Ineligible output cannot contain an artifact")
        return self


def _technical(scene: StoredSceneState) -> str:
    return digest(scene.technical_spec.model_dump(mode="json") if scene.technical_spec else None)


def _source(root: Path, scene: StoredSceneState) -> Path:
    expected = f"scenes/revisions/scene_{scene.plan.scene_id}/{scene.code_sha256}.py"
    if not scene.code_sha256 or scene.code_file != expected:
        raise ValueError("Incremental source must be an immutable content-addressed source")
    path = restore_run_path(root, expected)
    if not path.is_file() or sha256_file(path) != scene.code_sha256:
        raise ValueError("Incremental source hash mismatch")
    return path


def _media(root: Path, artifact: SceneArtifact, profile: RenderProfile) -> Path:
    path = restore_run_path(root, artifact.video_path)
    if (
        "partial_movie_files" in path.parts
        or "__smoke__" in path.parts
        or not path.is_file()
        or path.stat().st_size != artifact.metadata.size_bytes
        or sha256_file(path) != artifact.video_sha256
        or artifact.metadata.width != profile.pixel_width
        or artifact.metadata.height != profile.pixel_height
        or abs(artifact.metadata.frame_rate - profile.frame_rate) > 0.1
    ):
        raise ValueError("Incremental artifact hash, size, or render metadata mismatch")
    return path


def _receipt(scene: StoredSceneState) -> None:
    receipt = scene.static_verification
    if (
        not scene.reviewed
        or scene.failed
        or scene.give_up
        or receipt.status != "passed"
        or receipt.code_sha256 != scene.code_sha256
        or receipt.technical_spec_sha256 != scene.technical_spec_sha256
        or scene.technical_spec_sha256
        != (sha256_text(scene.technical_spec.model_dump_json()) if scene.technical_spec else "")
    ):
        raise ValueError("Incremental reuse requires current reviewed static verification")


def validate_reused_artifact(
    root: Path,
    run_id: str,
    scene: StoredSceneState,
    profile: RenderProfile,
    *,
    provenance: ObjectRef,
) -> Path:
    """Validate typed current-run evidence; never infer authorization from a path."""
    artifact = scene.artifact
    if (
        artifact is None
        or artifact.origin != "reused"
        or not artifact.verified
        or not scene.rendered
        or scene.slurm_job is not None
        or scene.submission is not None
        or artifact.job_id is not None
        or artifact.source_run_id != run_id
        or artifact.scene_id != scene.plan.scene_id
        or artifact.scene_class_name != scene.class_name
        or artifact.code_sha256 != scene.code_sha256
        or artifact.render_profile_sha256 != profile.digest()
        or not artifact.execution_sha256
        or artifact.environment_warning
    ):
        raise ValueError("Reused artifact differs from the current immutable scene")
    reference = provenance
    if (
        not isinstance(reference, ObjectRef)
        or reference.kind != "report"
        or reference.path != f"artifacts/objects/{reference.sha256}.json"
    ):
        raise ValueError("Reused artifact requires an immutable typed provenance reference")
    report = restore_run_path(root, reference.path)
    if not report.is_file() or sha256_file(report) != reference.sha256:
        raise ValueError("Reused artifact provenance hash mismatch")
    provenance = ReuseProvenance.model_validate_json(report.read_text(encoding="utf-8"))
    original = provenance.base_artifact
    if (
        provenance.run_id != run_id
        or provenance.base_run_id == run_id
        or provenance.scene_id != scene.plan.scene_id
        or provenance.work_identity.revision > scene.revision
        or provenance.work_identity.input_sha256 != scene.input_sha256
        or provenance.plan_sha256 != digest(scene.plan.model_dump(mode="json"))
        or provenance.technical_sha256 != _technical(scene)
        or provenance.inherited_sha256 != sha256_text(scene.inherited_elements_code)
        or provenance.execution_spec.digest() != artifact.execution_sha256
        or provenance.execution_spec.for_profile(profile) != provenance.execution_spec
        or not provenance.execution_spec.matches_environment(artifact.environment_fingerprint)
        or artifact.video_path != f"videos/reused/{reference.sha256}/{artifact.video_sha256}.mp4"
        or not original.verified
        or original.source_run_id != provenance.base_run_id
        or original.model_dump(exclude={"origin", "source_run_id", "job_id", "video_path"})
        != artifact.model_dump(exclude={"origin", "source_run_id", "job_id", "video_path"})
    ):
        raise ValueError("Reused artifact provenance differs from the current scene")
    _source(root, scene)
    _receipt(scene)
    execution = scene.execution_verification
    if (
        execution.status != "passed"
        or execution.scope != "formal_video"
        or execution.code_sha256 != scene.code_sha256
        or execution.artifact_sha256 != artifact.video_sha256
        or execution.duration_seconds != artifact.metadata.duration_seconds
    ):
        raise ValueError("Reused artifact lacks its exact formal execution receipt")
    return _media(root, artifact, profile)


def _dependencies(scene, scenes, plans):
    handoff_source = (
        scene.technical_spec.handoff_in.source_scene_id
        if scene.technical_spec and scene.technical_spec.handoff_in
        else None
    )
    dependencies = DependencyService.dependencies(
        scene.plan,
        plans,
        inherited_code=scene.inherited_elements_code,
        handoff_source=handoff_source,
    )
    result = {}
    for dependency in dependencies:
        source = scenes.get(dependency.source_scene_id)
        if source is None or source.plan.scene_id >= scene.plan.scene_id:
            raise ValueError("Incremental dependency has no exact preceding source scene")
        if dependency.kind == "teaching":
            value = DependencyService.teaching(source.plan)
        elif dependency.kind == "code_context":
            value = source.code_sha256
            if not value:
                raise ValueError("Incremental code-context dependency lacks reviewed source")
        else:
            value = digest(
                {
                    "contract": DependencyService.contract(source.plan, source.technical_spec),
                    "handoff": source.technical_spec.handoff_out.model_dump(mode="json")
                    if source.technical_spec and source.technical_spec.handoff_out
                    else None,
                }
            )
        result[f"{dependency.source_scene_id}:{dependency.kind}"] = {
            "source": value,
            "elements": sorted(dependency.element_ids),
        }
    return result


def _globals(lesson, teaching, bible, rag, config):
    return {
        "lesson": lesson.model_dump(mode="json"),
        "teaching": teaching.model_dump(mode="json"),
        "bible": bible.model_dump(mode="json") if bible else None,
        "rag": rag.model_dump(mode="json"),
        "config": {
            key: value
            for key, value in json.loads(config.values_json).items()
            if key != "OUTPUT_FILE"
        },
    }


def _copy_video(root, relative, source, artifact, context):
    destination = restore_run_path(root, relative)
    missing = []
    parent = destination.parent
    while not parent.exists():
        missing.append(parent)
        parent = parent.parent
    for directory in reversed(missing):
        directory.mkdir(mode=0o700, exist_ok=True)
        directory.chmod(0o700)
    # Recheck after mkdir, including pre-existing ancestors; never overwrite an object.
    destination = restore_run_path(root, relative)
    if destination.exists():
        if (
            destination.stat().st_size != artifact.metadata.size_bytes
            or sha256_file(destination) != artifact.video_sha256
        ):
            raise ValueError("Immutable reused video already exists with different bytes")
        return destination
    descriptor, name = tempfile.mkstemp(prefix=".reuse-", dir=destination.parent)
    temporary = Path(name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as output, source.open("rb") as original:
            while chunk := original.read(1024 * 1024):
                context.checkpoint()
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        if (
            temporary.stat().st_size != artifact.metadata.size_bytes
            or sha256_file(temporary) != artifact.video_sha256
        ):
            raise ValueError("Base artifact changed while copying immutable video")
        context.checkpoint()
        try:
            os.link(temporary, destination)
        except FileExistsError:
            destination = restore_run_path(root, relative)
            if sha256_file(destination) != artifact.video_sha256:
                raise ValueError("Immutable reused video identity collision") from None
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


class IncrementalPipeline:
    @staticmethod
    def validate_base(
        base: RunManifest,
        root: Path,
        *,
        generation_identity: GenerationIdentity,
        execution_spec: ExecutionSpec,
        backend: Literal["slurm", "local"],
    ) -> None:
        """Validate every referenced base source/media before any new-run side effect."""
        base.validate_for_resume(runtime="agent")
        if base.agent.generation_identity != generation_identity:
            raise ValueError("Incompatible incremental generation identity")
        if root.is_symlink() or not root.is_dir() or root.name != base.run_id:
            raise ValueError("Incremental base root differs from its manifest identity")
        if base.status not in {"completed", "dry_run_complete"}:
            raise ValueError("Incremental base run is not complete")
        if base.backend != backend:
            raise ValueError("Incompatible incremental render backend")
        # Profile changes are legitimate new scene inputs; all other execution
        # changes (isolation, resources, image, Python) are incompatible identities.
        if execution_spec.for_profile(base.render_profile) != base.execution_spec:
            raise ValueError("Incompatible incremental execution identity")
        base.execution_spec.verify_image()
        if base.status == "completed":
            from kd1_anime.pipeline.recovery import validate_completed_run

            validate_completed_run(base, root)
        from kd1_anime.pipeline.render import validate_artifact

        for scene in base.scenes.values():
            if scene.code_file:
                _source(root, scene)
            if scene.rendered:
                artifact = scene.artifact
                if (
                    artifact is None
                    or artifact.source_run_id != base.run_id
                    or artifact.execution_sha256 != base.execution_spec.digest()
                    or artifact.environment_warning
                    or not base.execution_spec.matches_environment(artifact.environment_fingerprint)
                ):
                    raise ValueError(
                        "Incremental base artifact has incompatible execution evidence"
                    )
                exact = validate_artifact(
                    root,
                    base.run_id,
                    scene,
                    base.render_profile,
                    provenance=base.agent.reuse_receipts.get(scene.plan.scene_id),
                )
                if get_reusable_video_path(base, scene.plan.scene_id, root) != exact:
                    raise ValueError("Incremental base media is not its exact verified artifact")

    def reuse(self, inputs: IncrementalInputs, context: ToolContext) -> IncrementalOutput:
        context.checkpoint()
        # Worker-owned copies ensure no result construction can alter either input.
        base = inputs.base.model_copy(deep=True)
        current = inputs.current
        scene = current.scene.model_copy(deep=True)
        self.validate_base(
            base,
            inputs.base_root,
            generation_identity=inputs.generation_identity,
            execution_spec=current.execution_spec,
            backend=inputs.backend,
        )
        if current.run_id == base.run_id or current.root.resolve() == inputs.base_root.resolve():
            raise ValueError("Incremental reuse cannot mutate its base run")
        if current.root.is_symlink() or current.root.name != current.run_id:
            raise ValueError("Incremental current root differs from its run identity")
        if not inputs.continuity_ready:
            raise ValueError("Incremental reuse requires the ordered continuity barrier")
        if scene.rendered or scene.artifact or scene.slurm_job or scene.submission:
            raise ValueError("Incremental reuse must precede render submission")
        if not scene.input_sha256:
            raise ValueError("Incremental reuse requires the current full input digest")
        _source(current.root, scene)
        _receipt(scene)
        scenes = {item.plan.scene_id: item.model_copy(deep=True) for item in inputs.current_scenes}
        if (
            len(scenes) != len(inputs.current_scenes)
            or scenes.get(scene.plan.scene_id) != current.scene
        ):
            raise ValueError("Incremental current scene snapshot is inconsistent")
        dependencies = _dependencies(scene, scenes, current.plans)
        globals_ = _globals(
            current.lesson_spec,
            current.teaching_graph,
            current.continuity_bible,
            inputs.rag_profile,
            current.config,
        )
        previous_scenes = [item for key, item in base.scenes.items() if key < scene.plan.scene_id]
        previous_scene = (
            max(previous_scenes, key=lambda item: item.plan.scene_id) if previous_scenes else None
        )
        old_handoff = (
            previous_scene.technical_spec.handoff_out
            if previous_scene and previous_scene.technical_spec
            else None
        )
        identity = {
            "plan": digest(scene.plan.model_dump(mode="json")),
            "technical": _technical(scene),
            "code": scene.code_sha256,
            "class": scene.class_name,
            "inherited": sha256_text(scene.inherited_elements_code),
            "handoff": digest(
                current.previous_technical_handoff.model_dump(mode="json")
                if current.previous_technical_handoff
                else None
            ),
            "direct_render": current.direct_render,
            "dependencies": digest(dependencies),
            "globals": digest(globals_),
            "render": current.render_profile.digest(),
            "execution": current.execution_spec.digest(),
            "generation": inputs.generation_identity.digest(),
        }
        input_sha256 = digest(identity)
        previous = base.scenes.get(scene.plan.scene_id)
        if previous is None:
            return IncrementalOutput(
                eligible=False, input_sha256=input_sha256, reasons=("scene_missing",)
            )
        old = {
            "plan": digest(previous.plan.model_dump(mode="json")),
            "technical": _technical(previous),
            "code": previous.code_sha256,
            "class": previous.class_name,
            "inherited": sha256_text(previous.inherited_elements_code),
            "handoff": digest(old_handoff.model_dump(mode="json") if old_handoff else None),
            "direct_render": base.direct_render,
            "dependencies": digest(
                _dependencies(
                    previous, base.scenes, tuple(item.plan for item in base.scenes.values())
                )
            ),
            "globals": digest(
                _globals(
                    base.lesson_spec,
                    base.teaching_graph,
                    base.continuity_bible,
                    base.rag_profile,
                    base.run_config,
                )
            ),
            "render": base.render_profile.digest(),
            "execution": base.execution_spec.digest(),
            "generation": base.agent.generation_identity.digest(),
        }
        reasons = tuple(f"{key}_changed" for key in identity if identity[key] != old[key])
        if not previous.rendered:
            reasons += ("base_not_rendered",)
        if reasons:
            return IncrementalOutput(eligible=False, input_sha256=input_sha256, reasons=reasons)
        original = previous.artifact
        provenance = ReuseProvenance(
            base_run_id=base.run_id,
            base_manifest_sha256=digest(base.model_dump(mode="json")),
            base_artifact=original.model_copy(deep=True),
            run_id=current.run_id,
            scene_id=scene.plan.scene_id,
            work_identity=WorkIdentity(scene.revision, scene.input_sha256),
            input_sha256=input_sha256,
            generation_sha256=identity["generation"],
            execution_spec=current.execution_spec,
            plan_sha256=identity["plan"],
            technical_sha256=identity["technical"],
            inherited_sha256=identity["inherited"],
            dependencies_sha256=identity["dependencies"],
            globals_sha256=identity["globals"],
        )
        store = ObjectStore(current.root)
        provenance_path = store.report(provenance.model_dump_json())
        provenance_ref = ObjectRef(
            path=provenance_path, sha256=Path(provenance_path).stem, kind="report"
        )
        relative = f"videos/reused/{provenance_ref.sha256}/{original.video_sha256}.mp4"
        source = restore_run_path(inputs.base_root, original.video_path)
        _copy_video(current.root, relative, source, original, context)
        scene.artifact = original.model_copy(
            deep=True,
            update={
                "origin": "reused",
                "source_run_id": current.run_id,
                "job_id": None,
                "video_path": relative,
            },
        )
        scene.rendered = True
        scene.execution_verification = ExecutionVerification(
            status="passed",
            scope="formal_video",
            code_sha256=scene.code_sha256,
            artifact_sha256=original.video_sha256,
            duration_seconds=original.metadata.duration_seconds,
            checked_at=previous.execution_verification.checked_at,
        )
        scene.visual_verification = VisualVerification()
        scene.visual_status = (
            "pending"
            if current.visual_eval_profile and current.visual_eval_profile.enabled
            else "skipped"
        )
        scene.visual_score = None
        scene.visual_report_file = scene.visual_report_sha256 = scene.visual_artifact_sha256 = ""
        scene.visual_feedback = ""
        scene.visual_best_candidate = None
        scene.snapshot = scene_snapshot(scene)
        validate_reused_artifact(
            current.root, current.run_id, scene, current.render_profile, provenance=provenance_ref
        )
        snapshot_path = store.report(scene.model_dump_json())
        context.checkpoint()
        return IncrementalOutput(
            eligible=True,
            input_sha256=input_sha256,
            scene=scene,
            snapshot=ObjectRef(path=snapshot_path, sha256=Path(snapshot_path).stem, kind="report"),
            video=ObjectRef(path=relative, sha256=original.video_sha256, kind="video"),
            provenance=provenance_ref,
        )
