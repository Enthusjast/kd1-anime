"""Commands and snapshots at the single durable state boundary."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, fields, replace
from pathlib import Path
from threading import RLock
from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from kd1_anime.agent_runtime.state import AgentRunState, merge_transport_state
from kd1_anime.cluster.status import TERMINAL_STATES
from kd1_anime.domain.dependencies import DependencyGraph, StaleWorkError, WorkIdentity
from kd1_anime.domain.execution import FSMState, RunPhase
from kd1_anime.domain.planning import (
    ContinuityBible,
    ElementManifest,
    LessonSpec,
    SceneOutline,
    TeachingGraph,
)
from kd1_anime.domain.state_ledger import StateLedger
from kd1_anime.models import ValidatedModel
from kd1_anime.rag.models import RagReceipt, RagRuntimeProfile
from kd1_anime.request_budget import RequestReservation
from kd1_anime.run_store import (
    MANIFEST_NAME,
    WRITABLE_MANIFEST_SCHEMA_VERSIONS,
    RunManifest,
    RunStatus,
    StoredSceneState,
    write_manifest,
)
from kd1_anime.services.dependencies import DependencyService


class ScenePatch(ValidatedModel):
    """Replace one scene, bound to its previously observed inputs and dependencies."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scene_id: int = Field(ge=1, strict=True)
    identity: WorkIdentity
    replacement: StoredSceneState
    dependency_revisions: dict[int, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_scene(self):
        if self.replacement.plan.scene_id != self.scene_id:
            raise ValueError("Scene replacement does not match scene_id")
        if (
            isinstance(self.identity.revision, bool)
            or self.identity.revision < 0
            or any(
                source < 1 or source >= self.scene_id or isinstance(revision, bool) or revision < 0
                for source, revision in self.dependency_revisions.items()
            )
        ):
            raise ValueError("Invalid scene or dependency revision")
        return self


class RunPatch(ValidatedModel):
    """Closed writable outputs; pinned inputs and request limits cannot be replaced.

    ``model_fields_set`` distinguishes an omitted field from an explicit nullable
    output clear. Scene replacements never replace the entire scene collection.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenes: tuple[ScenePatch, ...] = ()
    scene_removals: dict[int, WorkIdentity] = Field(default_factory=dict)
    reservation_additions: tuple[RequestReservation, ...] = ()
    agent: AgentRunState | None = None
    state: FSMState | None = None
    status: RunStatus | None = None
    run_phase: RunPhase | None = None
    error: str | None = Field(default=None, max_length=50_000)
    artifact_refs: dict[str, str] | None = None
    dependency_graph: DependencyGraph | None = None
    outlines: list[SceneOutline] | None = None
    lesson_spec: LessonSpec | None = None
    teaching_graph: TeachingGraph | None = None
    state_ledger: StateLedger | None = None
    expected_final_duration: float | None = Field(default=None, ge=0, le=3_600)
    continuity_bible: ContinuityBible | None = None
    element_manifest: ElementManifest | None = None
    plan_approved: bool | None = None
    plan_review_status: Literal["pending", "reviewing", "passed", "failed", "skipped"] | None = None
    plan_review_cycle_signature: str | None = Field(default=None, pattern=r"^(?:[0-9a-f]{64})?$")
    plan_review_cycle_count: int | None = Field(default=None, ge=0)
    continuity_review_status: Literal["pending", "reviewing", "passed", "warning"] | None = None
    continuity_review_round: int | None = Field(default=None, ge=0)
    continuity_resume_recheck_used: bool | None = None
    continuity_warnings: list[str] | None = Field(default=None, max_length=100)
    fsm_warnings: list[str] | None = Field(default=None, max_length=100)
    final_video: str | None = None
    final_video_sha256: str | None = Field(default=None, pattern=r"^(?:[0-9a-f]{64})?$")
    eval_round: int | None = Field(default=None, ge=0)
    continuity_rebuild_required: bool | None = None
    rag_receipts: dict[str, RagReceipt] | None = Field(default=None, max_length=256)
    rag_profile: RagRuntimeProfile | None = None
    rag_warnings: list[str] | None = Field(default=None, max_length=100)

    @model_validator(mode="after")
    def validate_updates(self):
        scene_ids = [scene.scene_id for scene in self.scenes]
        if len(scene_ids) != len(set(scene_ids)):
            raise ValueError("A run patch cannot replace one scene twice")
        if set(scene_ids) & self.scene_removals.keys() or any(
            isinstance(key, bool) or key < 1 for key in self.scene_removals
        ):
            raise ValueError("Scene retirement must name distinct positive scene identities")
        nullable = {"expected_final_duration", "continuity_bible", "final_video"}
        for name in self.model_fields_set - nullable:
            if getattr(self, name) is None:
                raise ValueError(f"Run patch field {name} cannot be null")
        return self


@dataclass(frozen=True, slots=True)
class CommitCommand:
    expected_revision: int
    phase: str
    status: str = "running"
    error: str = ""
    scene_id: int | None = None
    identity: WorkIdentity | None = None
    apply: Callable | None = None
    queued_at: float | None = None
    patch: RunPatch | None = None

    def __post_init__(self):
        if (
            isinstance(self.expected_revision, bool)
            or not isinstance(self.expected_revision, int)
            or self.expected_revision < 0
        ):
            raise ValueError("Expected manifest revision must be a nonnegative integer")
        if self.patch is not None and self.apply is not None:
            raise ValueError("CommitCommand cannot mix patch and legacy apply")
        if self.patch is not None and not isinstance(self.patch, RunPatch):
            raise TypeError("CommitCommand patch must be a typed RunPatch")

    def check(self, context, current_identity=None):
        if self.expected_revision != context.manifest_revision:
            raise StaleWorkError("Commit was built from a superseded manifest")
        if self.identity is not None and self.identity != current_identity:
            raise StaleWorkError("Commit was built from superseded scene inputs")


class RunCommitter:
    """Serialize typed proposals and publish memory only after a durable v10 write.

    The application owns the run lock and external operations. This boundary owns
    only manifest publication; workers must explicitly rebase independent scene
    proposals rather than repeat their tools after a manifest revision conflict.
    """

    def __init__(self, run_root: Path, manifest: RunManifest):
        if manifest.schema_version not in WRITABLE_MANIFEST_SCHEMA_VERSIONS:
            raise ValueError("Typed commits require an executable manifest")
        manifest.validate_for_resume(runtime="agent" if manifest.schema_version == 11 else "legacy")
        self._run_root = Path(run_root)
        self._current = manifest
        self._lock = RLock()

    @property
    def current(self) -> RunManifest:
        with self._lock:
            return self._current

    @staticmethod
    def _typed_patch(command: CommitCommand) -> RunPatch:
        if command.apply is not None or command.patch is None:
            raise ValueError("Typed commits require a patch, never legacy apply")
        # Revalidate only the proposal's touched values. Keeping top-level field
        # presence separately preserves nested factory defaults such as request
        # IDs/timestamps, while still distinguishing null from unset run outputs.
        values = command.patch.model_dump()
        return RunPatch.model_validate(
            {name: values[name] for name in command.patch.model_fields_set}
        )

    def _check_inputs(self, command: CommitCommand, patch: RunPatch) -> None:
        current = self._current
        proposed = {scene.scene_id: scene for scene in patch.scenes}
        plans = [scene.plan for scene in current.scenes.values()]
        if command.identity is not None:
            scene = proposed.get(command.scene_id)
            if scene is None or scene.identity != command.identity:
                raise StaleWorkError("Command identity does not match its scene patch")
        elif command.scene_id is not None and command.scene_id not in proposed:
            raise ValueError("Command scene_id has no scene patch")
        if patch.scene_removals and current.schema_version != 11:
            raise ValueError("Scene retirement requires agent runtime ownership")
        for scene_id, identity in patch.scene_removals.items():
            stored = current.scenes.get(scene_id)
            if stored is None or identity != WorkIdentity(stored.revision, stored.input_sha256):
                raise StaleWorkError("Scene retirement was built from superseded inputs")
            if (
                stored.slurm_job is not None
                and not stored.slurm_job.cancelled
                and stored.slurm_job.status not in TERMINAL_STATES
            ):
                raise ValueError("Scene retirement requires confirmed job termination")
            if any(
                item.scene_id == scene_id and item.attempt.status in {"pending", "unknown"}
                for item in current.agent.submissions
            ):
                raise ValueError("Uncertain submission forbids scene retirement")
        for scene in patch.scenes:
            stored = current.scenes.get(scene.scene_id)
            if stored is None and current.schema_version == 11:
                known = current.agent.outline_identities if current.agent is not None else {}
                importing = (
                    patch.outlines is not None
                    and patch.agent is not None
                    and scene.scene_id in patch.agent.outline_identities
                    and any(outline.scene_id == scene.scene_id for outline in patch.outlines)
                )
                if scene.scene_id not in known and not importing:
                    raise StaleWorkError("New scene has no durable outline identity")
            identity = (
                WorkIdentity(stored.revision, stored.input_sha256)
                if stored is not None
                else (
                    current.agent.outline_identities.get(scene.scene_id, WorkIdentity(0, ""))
                    if current.schema_version == 11 and current.agent is not None
                    else WorkIdentity(0, "")
                )
            )
            if scene.identity != identity:
                raise StaleWorkError("Commit was built from superseded scene inputs")
            if stored is None and current.schema_version == 11:
                for source, revision in scene.dependency_revisions.items():
                    dependency = current.scenes.get(source)
                    if dependency is None or dependency.revision != revision:
                        raise StaleWorkError(
                            "Detail result was built from superseded dependency inputs"
                        )
                # Detail is bound to the durable outline, not sibling Detail results.
                continue
            required_dependencies = {
                binding.source_scene_id
                for binding in current.dependency_graph.bindings
                if binding.consumer_scene_id == scene.scene_id
            }
            for inputs in (stored, scene.replacement):
                if inputs is None:
                    continue
                handoff_source = (
                    inputs.technical_spec.handoff_in.source_scene_id
                    if inputs.technical_spec is not None
                    and inputs.technical_spec.handoff_in is not None
                    else None
                )
                required_dependencies.update(
                    dependency.source_scene_id
                    for dependency in DependencyService.dependencies(
                        inputs.plan,
                        plans,
                        inherited_code=inputs.inherited_elements_code,
                        handoff_source=handoff_source,
                    )
                )
            if patch.dependency_graph is not None:
                required_dependencies.update(
                    binding.source_scene_id
                    for binding in patch.dependency_graph.bindings
                    if binding.consumer_scene_id == scene.scene_id
                )
            if not required_dependencies.issubset(scene.dependency_revisions):
                raise ValueError("Scene patch is missing captured dependency revisions")
            for source, revision in scene.dependency_revisions.items():
                dependency = current.scenes.get(source)
                if dependency is None or dependency.revision != revision:
                    raise StaleWorkError("Commit was built from superseded dependency inputs")

    def _check_invalidations(self, candidate: RunManifest, patch: RunPatch) -> None:
        changed = {scene.scene_id for scene in patch.scenes} | patch.scene_removals.keys()
        affected = DependencyService.affected_manifest(self._current, candidate, tuple(changed))
        if not affected.issubset(changed):
            raise ValueError("Changed dependencies require explicit consumer invalidation patches")
        for scene_id in affected:
            if scene_id in patch.scene_removals:
                continue
            scene = candidate.scenes[scene_id]
            if (
                scene.code_file
                or scene.code_sha256
                or scene.class_name
                or scene.reviewed
                or (scene.plan_reviewed and candidate.schema_version != 11)
                or scene.local_smoke_status not in {"pending", "skipped"}
                or scene.technical_spec is not None
                or scene.technical_status != "pending"
                or scene.capability_contract is not None
                or scene.capability_status != "pending"
                or scene.rendered
                or scene.artifact is not None
                or scene.slurm_job is not None
                or scene.submission is not None
                or scene.inherited_elements_code
                or scene.exported_elements_code
                or scene.exported_elements
                or scene.candidates
                or scene.visual_best_candidate is not None
                or scene.visual_report_file
                or scene.visual_report_sha256
                or scene.visual_artifact_sha256
                or scene.visual_status not in {"pending", "skipped"}
                or scene.visual_score is not None
                or scene.static_verification.status != "not_run"
                or scene.execution_verification.status != "not_run"
                or scene.visual_verification.status != "not_run"
            ):
                raise ValueError("Consumer invalidation patch retains reusable stale outputs")

    def rebase(self, command: CommitCommand) -> CommitCommand:
        """Rebase unchanged scene/reservation results, never changed run outputs."""
        with self._lock:
            patch = self._typed_patch(command)
            if command.expected_revision > self._current.revision:
                raise StaleWorkError("Cannot rebase a future manifest revision")
            self._check_inputs(command, patch)
            if command.expected_revision == self._current.revision:
                return command
            if patch.model_fields_set - {"scenes", "reservation_additions"} or not (
                patch.scenes or patch.reservation_additions
            ):
                raise StaleWorkError("Only independent scene or reservation patches can be rebased")
            return replace(command, expected_revision=self._current.revision)

    def commit(self, command: CommitCommand) -> RunManifest:
        with self._lock:
            if command.expected_revision != self._current.revision:
                raise StaleWorkError("Commit was built from a superseded manifest")
            patch = self._typed_patch(command)
            self._check_inputs(command, patch)
            current = self._current
            candidate = current.model_copy()
            if candidate.schema_version == 11 and "plan_approved" in patch.model_fields_set:
                raise ValueError("Agent approvals belong to Session grants")
            for name in patch.model_fields_set - {
                "scenes",
                "scene_removals",
                "reservation_additions",
                "agent",
            }:
                setattr(candidate, name, getattr(patch, name))
            if patch.agent is not None:
                if current.agent is None:
                    raise ValueError("Legacy runs cannot acquire agent identity")
                candidate.agent = merge_transport_state(current.agent, patch.agent)
            if patch.scenes or patch.scene_removals:
                scenes = {
                    scene_id: scene
                    for scene_id, scene in current.scenes.items()
                    if scene_id not in patch.scene_removals
                }
                for scene in patch.scenes:
                    replacement = scene.replacement
                    # The proposal was reconstructed above, so all nested values
                    # are private to this commit, including plan/job structures.
                    replacement.snapshot = None
                    replacement.revision = scene.identity.revision + 1
                    previous = current.scenes.get(scene.scene_id)
                    attempts = {}
                    if previous is not None:
                        attempts = {
                            (counter.operation, counter.input_sha256): counter
                            for counter in previous.stage_attempts
                        }
                    for counter in replacement.stage_attempts:
                        key = (counter.operation, counter.input_sha256)
                        if key not in attempts or counter.used > attempts[key].used:
                            attempts[key] = counter
                    replacement.stage_attempts = list(attempts.values())
                    scenes[scene.scene_id] = replacement
                candidate.scenes = scenes
            if patch.reservation_additions:
                budget = current.request_budget.model_copy()
                budget.reservations = list(current.request_budget.reservations)
                reservations = {item.request_id: item for item in budget.reservations}
                for addition in patch.reservation_additions:
                    existing = reservations.get(addition.request_id)
                    if existing is not None:
                        if existing != addition:
                            raise ValueError(
                                "Conflicting request reservation with the same request_id"
                            )
                        continue
                    budget.reserve(addition)
                    reservations[addition.request_id] = addition
                candidate.request_budget = budget
            candidate.revision = current.revision + 1
            errors = candidate.integrity_errors()
            if errors:
                raise ValueError("运行清单完整性校验失败: " + "; ".join(errors))
            self._check_invalidations(candidate, patch)

            # write_manifest assigns EVERY scene.snapshot before writing. Shallow
            # persistence-only envelopes isolate those assignments for untouched
            # scenes without cloning their plans/jobs or mutating published state
            # on write failure. Only the touched scenes are actually replaced.
            durable = candidate.model_copy()
            touched = {scene.scene_id for scene in patch.scenes}
            durable.scenes = {
                scene_id: scene if scene_id in touched else scene.model_copy()
                for scene_id, scene in candidate.scenes.items()
            }
            write_manifest(self._run_root / MANIFEST_NAME, durable)
            candidate.updated_at = durable.updated_at
            self._current = candidate
            return candidate


def restore_snapshot(target, snapshot):
    """Preserve worker-held scene identities while replacing all proposed values."""
    existing = target.scene_states
    restored = {}
    for scene_id, stored in snapshot.scene_states.items():
        scene = existing.get(scene_id)
        if scene is None:
            restored[scene_id] = deepcopy(stored)
        else:
            values = deepcopy(stored.__dict__)
            previous_job = scene.slurm_job
            if (
                previous_job is not None
                and stored.slurm_job is not None
                and previous_job.job_id == stored.slurm_job.job_id
            ):
                previous_job.__dict__.update(deepcopy(stored.slurm_job.__dict__))
                values["execution"].slurm_job = previous_job
            scene.__dict__.clear()
            scene.__dict__.update(values)
            restored[scene_id] = scene
    for item in fields(target):
        if item.name != "scene_states":
            setattr(target, item.name, deepcopy(getattr(snapshot, item.name)))
    target.scene_states = restored
