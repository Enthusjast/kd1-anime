"""Manifest-owned bounded multiscene runtime, without a shared mutable Context."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from threading import Event, RLock

from kd1_anime.agent_runtime import contracts as actions
from kd1_anime.agent_runtime.contracts import (
    EventPayload,
    ObjectRef,
    RuntimeRecord,
    UsageReceipt,
    utc_now,
)
from kd1_anime.agent_runtime.loop import AgentLoop
from kd1_anime.agent_runtime.policy import (
    Policy,
    PolicyBudget,
    PolicyDependency,
    PolicySnapshot,
    ScenePolicyState,
    ToolPolicySpec,
)
from kd1_anime.agent_runtime.session import SessionStore, UserCommand
from kd1_anime.agent_runtime.state import (
    AgentPendingCandidate,
    AgentReviewState,
    AgentRunState,
    AgentStageAttempt,
    AgentSubmission,
    merge_transport_state,
)
from kd1_anime.agent_runtime.tools import (
    ExecutionAdapters,
    ToolDefinition,
    ToolExecutor,
    ToolRegistry,
    ToolResult,
    action_signature,
    current_invocation,
)
from kd1_anime.agents.auto_fixer import AutoFixerAgent
from kd1_anime.agents.base import BaseAgent
from kd1_anime.agents.capability import build_capability_contract
from kd1_anime.agents.coder import CoderAgent
from kd1_anime.agents.continuity import ContinuityReviewerAgent
from kd1_anime.agents.plan_reviewer import PlanReviewerAgent
from kd1_anime.agents.planner import PlannerAgent
from kd1_anime.agents.reviewer import ReviewerAgent
from kd1_anime.agents.technical_planner import TechnicalPlannerAgent
from kd1_anime.cluster.render_backend import create_render_backend
from kd1_anime.cluster.resource_estimator import estimate_render_resources
from kd1_anime.cluster.status import TERMINAL_STATES
from kd1_anime.commits import CommitCommand, RunCommitter, RunPatch, ScenePatch
from kd1_anime.config import resolve_runtime_path, settings
from kd1_anime.control import RunControl
from kd1_anime.domain.dependencies import DependencyGraph, WorkIdentity, digest
from kd1_anime.domain.diagnostics import Diagnostic
from kd1_anime.domain.execution import SubmissionAttempt
from kd1_anime.domain.planning import (
    ContinuityBible,
    ElementManifest,
    LessonSpec,
    SceneOutline,
    ScenePlan,
    TeachingGraph,
)
from kd1_anime.domain.runtime import ExecutionSpec, RunConfig, VisualEvalProfile
from kd1_anime.domain.state import RunPaths
from kd1_anime.domain.state_ledger import StateLedger
from kd1_anime.event_store import EventStore
from kd1_anime.events import CallbackAdapter, PipelineEvent
from kd1_anime.media.merger import VideoMerger
from kd1_anime.objects import ObjectStore
from kd1_anime.pipeline.contracts import (
    DetailInputs,
    DetailOutput,
    FinalizationInputs,
    FinalizationOutput,
    PlanningOutput,
    RenderingOutput,
    SceneInputs,
    SceneOutput,
    TechnicalOutput,
)
from kd1_anime.pipeline.finalization import FinalizationPipeline
from kd1_anime.pipeline.incremental import IncrementalInputs, IncrementalOutput, IncrementalPipeline
from kd1_anime.pipeline.planning import PlanningPipeline, PlanningStageError
from kd1_anime.pipeline.recovery import (
    RecoveryOutput,
    RecoveryPipeline,
    RecoveryPlan,
    plan_resume,
    plan_retry,
)
from kd1_anime.pipeline.render import RenderPipeline, validate_artifact
from kd1_anime.pipeline.repair import RenderRepairOutput, RenderRepairPipeline
from kd1_anime.pipeline.scene import CodeReviewOutput, InspectionOutput, ScenePipeline
from kd1_anime.pipeline.smoke import SmokePipeline
from kd1_anime.pipeline.visual import (
    BoundaryVisualOutput,
    VisualRepairProposal,
    VisualSceneOutput,
    scene_binding,
)
from kd1_anime.prompt_identity import capture_generation_identity, verify_generation_identity
from kd1_anime.rag.service import RagService
from kd1_anime.rendering import MergeProfile, RenderProfile, sha256_file
from kd1_anime.request_budget import RequestBudget, RequestContext, RequestReservation
from kd1_anime.run_store import (
    AGENT_MANIFEST_SCHEMA_VERSION,
    RunManifest,
    RunRepository,
    StoredSceneState,
    atomic_write_json,
    lock_run,
    restore_render_job,
    restore_run_path,
    sha256_text,
    write_manifest,
)
from kd1_anime.services.continuity_state import (
    ContinuityPublication,
    build_continuity_publication,
    prepare_inherited_inputs,
)
from kd1_anime.services.dependencies import DependencyService, scene_work_digest
from kd1_anime.services.rendering import RenderingService
from kd1_anime.services.results import StageResult
from kd1_anime.verification import ExecutionVerification

_active_work: ContextVar[tuple | None] = ContextVar("agent_work", default=None)


def _serialized(method):
    @wraps(method)
    def locked(self, *args, **kwargs):
        with self._runtime_lock:
            return method(self, *args, **kwargs)

    return locked


def _preflight(profile, backend):
    RenderingService(backend, backend.name).preflight(profile)


def _probe(agent):
    agent.check_api_available()


@dataclass
class AgentDependencies:
    planner_factory: Callable = PlannerAgent
    plan_reviewer_factory: Callable = PlanReviewerAgent
    continuity_reviewer_factory: Callable = ContinuityReviewerAgent
    technical_factory: Callable = TechnicalPlannerAgent
    coder_factory: Callable = CoderAgent
    reviewer_factory: Callable = ReviewerAgent
    backend_factory: Callable = create_render_backend
    merger_factory: Callable = VideoMerger
    preflight: Callable = _preflight
    probe: Callable = _probe
    after_commit: Callable | None = None
    rag_factory: Callable = RagService
    fixer_factory: Callable = AutoFixerAgent
    evaluator_factory: Callable | None = None


class CompletionOutput(RuntimeRecord):
    output: ObjectRef | None = None


class AgentRuntime:
    """One lock, one manifest authority, and one declared tool per durable stage."""

    def __init__(self, *, dependencies: AgentDependencies | None = None):
        self.dependencies = dependencies or AgentDependencies()
        self.control = RunControl(Event())
        self._callback = None
        self._runtime_lock = RLock()

    def generate(
        self,
        user_prompt: str,
        callback=None,
        dry_run=False,
        interactive=False,
        output_path: Path | None = None,
        approve_plan=False,
        smoke=False,
        backend=None,
    ):
        return self._start(
            user_prompt,
            callback=callback,
            dry_run=dry_run,
            interactive=interactive,
            output_path=output_path,
            approve_plan=approve_plan,
            smoke=smoke,
            backend=backend,
        )

    def plan_only(
        self,
        user_prompt: str,
        callback=None,
        interactive=False,
        output_path=None,
        approve_plan=False,
        review=True,
        preflight=True,
        *,
        export_path=None,
    ):
        self._start(
            user_prompt,
            callback=callback,
            interactive=interactive,
            output_path=output_path,
            approve_plan=approve_plan,
            operation="plan",
            planning_review=review,
            planning_preflight=preflight,
        )
        manifest = self.committer.current
        plans = [scene.plan.model_copy(deep=True) for _, scene in sorted(manifest.scenes.items())]
        if manifest.state != "DONE":
            raise RuntimeError(manifest.agent.stop_reason or "Planning did not complete")
        if export_path is not None:
            atomic_write_json(Path(export_path).expanduser().resolve(), self._export(manifest))
        return plans

    def run_from_plan(
        self,
        plan_file: Path,
        *,
        callback=None,
        dry_run=False,
        interactive=False,
        output_path=None,
        approve_plan=False,
        smoke=False,
        backend=None,
    ):
        payload = json.loads(Path(plan_file).read_text(encoding="utf-8"))
        self._parse_export(payload)
        return self._start(
            payload.get("user_prompt") or "Render approved plan",
            callback=callback,
            dry_run=dry_run,
            interactive=interactive,
            output_path=output_path,
            approve_plan=approve_plan,
            smoke=smoke,
            backend=backend,
            imported=payload,
        )

    def submit_existing_scene(
        self, source_code, class_name, *, scene_id=1, wait=False, backend=None
    ):
        if (backend or settings.RENDER_BACKEND) == "local" and not wait:
            raise ValueError("本地渲染必须等待完成")
        output = self._start(
            f"Render user source {class_name}",
            operation="render",
            backend=backend,
            direct_source=(source_code, class_name, scene_id),
            no_wait=not wait,
            planning_review=False,
            planning_preflight=False,
        )
        scene = self.manifest.scenes[scene_id]
        if scene.slurm_job is None or self.manifest.status == "failed":
            raise RuntimeError(self.manifest.agent.stop_reason or "Direct render submission failed")
        return restore_render_job(scene.slurm_job, self.root), output, self.manifest.run_id

    def run_incremental(
        self,
        user_prompt,
        base_run_id,
        *,
        base_manifest,
        callback=None,
        dry_run=False,
        interactive=False,
        output_path=None,
        approve_plan=False,
        smoke=False,
        backend=None,
    ):
        if base_manifest.run_id != base_run_id or base_manifest.schema_version != 11:
            raise ValueError("Incremental execution requires the exact v11 base manifest")
        verify_generation_identity(base_manifest.agent.generation_identity)
        return self._start(
            user_prompt,
            callback=callback,
            dry_run=dry_run,
            interactive=interactive,
            output_path=output_path,
            approve_plan=approve_plan,
            smoke=smoke,
            backend=backend,
            operation="incremental",
            base_manifest=base_manifest,
        )

    @staticmethod
    def _verify_resume_inputs(root, manifest):
        manifest.execution_spec.verify_image()
        actual = ExecutionSpec.capture(RenderProfile.current())
        if actual != manifest.execution_spec:
            raise ValueError("Actual execution environment differs from the pinned run identity")
        for identifier, scene in manifest.scenes.items():
            if not scene.code_file:
                continue
            expected = f"scenes/revisions/scene_{identifier}/{scene.code_sha256}.py"
            if scene.code_file != expected:
                raise ValueError("Resume source is not an immutable content object")
            source = restore_run_path(root, scene.code_file)
            if not source.is_file() or sha256_file(source) != scene.code_sha256:
                raise ValueError("Resume source hash mismatch")

    def _pending_submission(self, scene_id, *, manifest=None):
        manifest = manifest or self.manifest
        scene = manifest.scenes[scene_id]
        return next(
            (
                item.attempt
                for item in reversed(manifest.agent.submissions)
                if item.scene_id == scene_id
                and item.attempt.code_sha256 == scene.code_sha256
                and item.attempt.token not in manifest.agent.retired_submissions
            ),
            None,
        )

    def _recovery_action(self, name, scene_id=None):
        action = self._make_action(name, self._snapshot(), scene_id)
        result = self._run_action(action)
        if result.status != "CONTINUE":
            self._stop("Recovery requires an explicit safe decision: " + result.reason)
            return False
        return True

    def _reconcile_before_resume(self):
        for identifier in sorted(self.manifest.scenes):
            scene = self.manifest.scenes[identifier]
            if (
                not scene.rendered
                and scene.slurm_job is None
                and self._pending_submission(identifier) is not None
                and not self._recovery_action("reconcile_submission", identifier)
            ):
                return False
            scene = self.manifest.scenes[identifier]
            if (
                scene.slurm_job is not None or scene.artifact is not None
            ) and not self._recovery_action("reconcile_render", identifier):
                return False
        return True

    def _current_rag_profile(self):
        if self._rag is None:
            return self.manifest.rag_profile
        current = self._rag.runtime_profile()
        if current.model_dump(exclude={"status"}) == self.manifest.rag_profile.model_dump(
            exclude={"status"}
        ):
            return self.manifest.rag_profile
        return current

    def _cancel_recovery_prerequisites(self, plan):
        for cancellation in plan.cancellations:
            if not self._recovery_action("cancel_render", cancellation.scene_id):
                return False
            job = self.manifest.scenes[cancellation.scene_id].slurm_job
            if job is None or not job.cancelled or job.status != "CANCELLED":
                self._stop("Recovery cancellation failed; automatic resubmission forbidden")
                return False
        return True

    def resume(
        self,
        run_id: str,
        callback=None,
        interactive=False,
        approve_plan=False,
        retry_scene_id=None,
        *,
        base_manifest=None,
    ):
        repository = RunRepository(settings.WORKSPACE_DIR)
        before = repository.load_for_agent_resume(run_id)
        verify_generation_identity(before.agent.generation_identity)
        if retry_scene_id is not None and (before.state == "DONE" or before.status == "completed"):
            raise RuntimeError("Completed runs cannot retry in place; create a new incremental run")
        root = repository.run_root(run_id)
        with lock_run(root):
            manifest = repository.load_for_agent_resume(run_id)
            verify_generation_identity(manifest.agent.generation_identity)
            with manifest.run_config.activate():
                self._verify_resume_inputs(root, manifest)
                self._base_manifest = base_manifest
                self._bind(root, manifest, callback)
                if manifest.state == "DONE":
                    return self._completed_video()
                epoch = manifest.agent.recovery_epoch + 1
                pending = manifest.agent.pending_approval
                self._patch(
                    RunPatch(
                        agent=self._agent_update(
                            recovery_epoch=epoch,
                            pending_approval=pending.model_copy(update={"recovery_epoch": epoch})
                            if pending
                            else None,
                        )
                    )
                )
                self.sessions.command(
                    manifest.agent.session_id,
                    UserCommand(
                        session_id=manifest.agent.session_id,
                        kind="resume",
                        content=f"Resume run {run_id}",
                    ),
                )
                self._notify("run_started", {"run_id": run_id})
                if retry_scene_id is not None and self.manifest.agent.pending_approval is not None:
                    pending = self.manifest.agent.pending_approval
                    self.sessions.command(
                        manifest.agent.session_id,
                        UserCommand(
                            session_id=manifest.agent.session_id,
                            kind="reject",
                            action=pending,
                            content="Explicit scene retry supersedes this pending proposal",
                        ),
                    )
                    self._patch(RunPatch(agent=self._agent_update(pending_approval=None)))
                if not self._reconcile_before_resume():
                    return None
                plan = (
                    plan_retry(self.manifest, retry_scene_id)
                    if retry_scene_id is not None
                    else plan_resume(self.manifest, current_rag_profile=self._current_rag_profile())
                )
                if not self._cancel_recovery_prerequisites(plan):
                    return None
                plan = (
                    plan_retry(self.manifest, retry_scene_id)
                    if retry_scene_id is not None
                    else plan_resume(self.manifest, current_rag_profile=self._current_rag_profile())
                )
                if plan.blocked:
                    self._stop(
                        "Unconfirmed execution prevents recovery: "
                        + ",".join(map(str, plan.blocked))
                    )
                    return None
                if retry_scene_id is not None:
                    action = self._make_action("retry_scene", self._snapshot(), retry_scene_id)
                    self.sessions.command(
                        manifest.agent.session_id,
                        UserCommand(
                            session_id=manifest.agent.session_id,
                            kind="retry",
                            action=action,
                            content=f"Retry scene {retry_scene_id} without resetting budgets",
                        ),
                    )
                    result = self._run_action(action)
                    if result.status != "CONTINUE":
                        self._stop("Scene retry could not commit: " + result.reason)
                        return None
                elif (
                    self.manifest.state == "ERROR"
                    or self.manifest.agent.stop_reason
                    or plan.scenes
                    or plan.rag_profile is not None
                ):
                    if not self._recovery_action("resume_run"):
                        return None
                if (
                    self.manifest.state == "MERGING"
                    and any(
                        intent.action.kind in {"merge_video", "reconcile_merge"}
                        and intent.status != "completed"
                        for intent in self.manifest.agent.intents
                    )
                    and not self._recovery_action("reconcile_merge")
                ):
                    return None
                return self._drive(approve_plan=approve_plan, interactive=interactive)

    @staticmethod
    def _parse_export(payload):
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError("Expected exported plan JSON schema_version 1")
        plans = tuple(ScenePlan.model_validate(item) for item in payload.get("items", ()))
        if not plans or [plan.scene_id for plan in plans] != list(range(1, len(plans) + 1)):
            raise ValueError("Exported plans must have contiguous scene identities")
        return PlanningOutput(
            outlines=tuple(
                SceneOutline(
                    scene_id=p.scene_id,
                    title=p.title,
                    duration_seconds=p.duration_seconds,
                    purpose=p.purpose,
                    math_concept=p.math_concept,
                )
                for p in plans
            ),
            plans=plans,
            lesson_spec=LessonSpec.model_validate(payload.get("lesson_spec") or {}),
            teaching_graph=TeachingGraph.model_validate(payload.get("teaching_graph") or {}),
            continuity_bible=ContinuityBible.model_validate(payload["continuity_bible"])
            if payload.get("continuity_bible") is not None
            else None,
        )

    @staticmethod
    def _export(manifest):
        return {
            "schema_version": 1,
            "run_id": manifest.run_id,
            "user_prompt": manifest.user_prompt,
            "lesson_spec": manifest.lesson_spec.model_dump(mode="json"),
            "teaching_graph": manifest.teaching_graph.model_dump(mode="json"),
            "continuity_bible": manifest.continuity_bible.model_dump(mode="json")
            if manifest.continuity_bible
            else None,
            "items": [
                scene.plan.model_dump(mode="json") for _, scene in sorted(manifest.scenes.items())
            ],
        }

    def _start(
        self,
        prompt,
        *,
        callback=None,
        dry_run=False,
        interactive=False,
        output_path=None,
        approve_plan=False,
        smoke=False,
        backend=None,
        operation="generate",
        planning_review=True,
        planning_preflight=True,
        imported=None,
        direct_source=None,
        no_wait=False,
        base_manifest=None,
    ):
        prompt = actions.sanitize_record(prompt, self._secrets())
        config = RunConfig.capture()
        with config.activate():
            profile = RenderProfile.current()
            execution = ExecutionSpec.capture(profile)
            identity = capture_generation_identity()
            self._base_manifest = base_manifest
            if base_manifest is not None:
                verify_generation_identity(base_manifest.agent.generation_identity)
                IncrementalPipeline.validate_base(
                    base_manifest,
                    RunRepository(settings.WORKSPACE_DIR).run_root(base_manifest.run_id),
                    generation_identity=identity,
                    execution_spec=execution,
                    backend=backend or settings.RENDER_BACKEND,
                )
            paths = RunPaths.create(output_path)
            run_id, root = paths.run_id, paths.root
            sessions = SessionStore(
                resolve_runtime_path(settings.WORKSPACE_DIR) / "sessions", secrets=self._secrets()
            )
            session = sessions.create(
                prompt,
                run_ids=(run_id,),
                permissions=tuple(
                    actions.PermissionScope(permission=name, resource_id=f"run:{run_id}")
                    for name in ("read", "write", "exec", "cluster")
                ),
            )
            manifest = RunManifest(
                schema_version=AGENT_MANIFEST_SCHEMA_VERSION,
                run_id=run_id,
                user_prompt=prompt,
                run_config=config,
                execution_spec=execution,
                render_profile=profile,
                merge_profile=MergeProfile.current(),
                output_path=str(paths.output),
                backend=backend or settings.RENDER_BACKEND,
                generation_mode=settings.GENERATION_MODE,
                dry_run=dry_run,
                interactive=interactive,
                state="PLANNING",
                run_phase="planning",
                scene_codegen_mode=settings.SCENE_CODEGEN_MODE,
                incremental=base_manifest is not None,
                base_run_id=base_manifest.run_id if base_manifest is not None else None,
                visual_eval_profile=VisualEvalProfile(
                    enabled=settings.ENABLE_VISUAL_EVAL and operation != "render",
                    model=(settings.VISUAL_LLM_MODEL or settings.EVAL_VISUAL_MODEL or "")
                    if settings.ENABLE_VISUAL_EVAL
                    else "",
                    frame_count=settings.VISUAL_EVAL_FRAME_COUNT,
                    threshold=settings.VISUAL_EVAL_THRESHOLD,
                    max_fix_attempts=settings.MAX_VISUAL_FIX_ATTEMPTS,
                    repair_enabled=settings.GENERATION_MODE != "relaxed"
                    or settings.RELAXED_VISUAL_AUTO_FIX,
                ),
                request_budget=RequestBudget(
                    scene_limit=settings.MAX_SCENE_LLM_REQUESTS,
                    run_limit=settings.MAX_RUN_LLM_REQUESTS,
                ),
                agent=AgentRunState(
                    session_id=session.session_id,
                    generation_identity=identity,
                    operation=operation,
                    planning_review=planning_review,
                    planning_preflight=planning_preflight,
                    smoke_requested=smoke,
                    run_input_sha256=digest(
                        {
                            "prompt": prompt,
                            "config": config.model_dump(mode="json"),
                            "generation": identity.model_dump(mode="json"),
                        }
                    ),
                ),
            )
            with lock_run(root):
                if imported is not None:
                    text = json.dumps(imported, sort_keys=True, ensure_ascii=False)
                    manifest.artifact_refs = {
                        "imported_plan": ObjectStore(root).put("artifacts/objects", text, ".json")
                    }
                if direct_source is not None:
                    from kd1_anime.candidate_acceptor import CandidateAcceptor
                    from kd1_anime.verification import StaticVerification

                    source, class_name, scene_id = direct_source
                    plan = ScenePlan(
                        scene_id=scene_id,
                        title=class_name,
                        duration_seconds=1,
                        purpose="Render user source",
                        math_concept="User source",
                        visual_design="User source",
                        camera_movement="User source",
                        visual_flow=["Render supplied scene"],
                        key_moments=["User source"],
                        computation="User-supplied source; no generated mathematical contract",
                    )
                    inspection = CandidateAcceptor().inspect_candidate(
                        source,
                        plan,
                        renderer=profile.renderer,
                        manim_version=profile.manim_version,
                        expected_class_name=class_name,
                        allow_repair=False,
                        continuity_mode="none",
                    )
                    if not inspection.accepted:
                        raise ValueError(
                            "Direct render inspection rejected: " + inspection.feedback
                        )
                    path = ObjectStore(root).source(scene_id, source)
                    input_hash = digest(
                        {
                            "source": inspection.code_sha256,
                            "render": profile.digest(),
                            "execution": execution.digest(),
                        }
                    )
                    scene = StoredSceneState(
                        plan=plan,
                        input_sha256=input_hash,
                        code_file=path,
                        code_sha256=inspection.code_sha256,
                        class_name=class_name,
                        static_verification=StaticVerification(
                            status="passed",
                            code_sha256=inspection.code_sha256,
                            checked_at=utc_now().isoformat(),
                        ),
                        reviewed=True,
                        phase="reviewed",
                        plan_ready=True,
                        plan_reviewed=False,
                        technical_status="pending",
                    )
                    manifest.scenes = {scene_id: scene}
                    manifest.auto_fix = False
                    scene.input_sha256 = self._scene_digest(scene, manifest=manifest)
                    manifest.direct_render = True
                    manifest.state = "DISPATCHING"
                    manifest.agent = manifest.agent.model_copy(
                        update={
                            "next_action": "submit_render",
                            "continuity_published": True,
                            "reviewed_identities": {scene_id: WorkIdentity(0, scene.input_sha256)},
                            "scene_next_actions": {scene_id: "submit_render"},
                        }
                    )
                write_manifest(root / "manifest.json", manifest)
                self._bind(root, manifest, callback)
                self._notify("run_started", {"run_id": run_id})
                if no_wait:
                    self._run_action(self._next_action(self._snapshot()))
                    return None
                return self._drive(approve_plan=approve_plan, interactive=interactive)

    def _bind(self, root, manifest, callback):
        self.root = root
        self.committer = RunCommitter(root, manifest)
        self.sessions = SessionStore(
            resolve_runtime_path(settings.WORKSPACE_DIR) / "sessions", secrets=self._secrets()
        )
        self.events = EventStore(root, manifest.run_id, secrets=self._secrets())
        self._callback = callback
        self._backend = None
        self._render_pipeline = None
        self._rag = (
            self.dependencies.rag_factory(manifest.run_config.resolve())
            if manifest.run_config.resolve().RAG_ENABLED
            else None
        )
        self.planning = PlanningPipeline(
            self.dependencies.planner_factory,
            self.dependencies.plan_reviewer_factory,
            self.dependencies.continuity_reviewer_factory,
            self.dependencies.technical_factory,
            config=manifest.run_config,
            request_context=self._request_context,
            control=self.control,
            rag_service=self._rag,
            rag_profile=manifest.rag_profile if manifest.rag_profile.enabled else None,
        )
        self.scene_pipeline = ScenePipeline(
            self.dependencies.coder_factory,
            self.dependencies.reviewer_factory,
            request_context=self._request_context,
            control=self.control,
            rag_service=self._rag,
            rag_profile=manifest.rag_profile if manifest.rag_profile.enabled else None,
        )
        self.finalization = FinalizationPipeline(
            self.dependencies.merger_factory(),
            evaluator_factory=self.dependencies.evaluator_factory,
            fixer_factory=self.dependencies.fixer_factory,
            request_context=self._request_context,
            control=self.control,
        )
        self.visual = self.finalization.visual
        self.registry = ToolRegistry()
        declarations = (
            (actions.DetailScene, DetailOutput, "write", "commit", True),
            (actions.PublishContinuity, ContinuityPublication, "write", "commit", False),
            (actions.RevisePlan, PlanningOutput, "write", "commit", True),
            (
                actions.ProposePlan,
                PlanningOutput,
                "write",
                "commit",
                "imported_plan" not in manifest.artifact_refs,
            ),
            (actions.ReviewPlan, PlanningOutput, "write", "commit", manifest.agent.planning_review),
            (actions.ProposeTechnicalSpec, TechnicalOutput, "write", "commit", True),
            (actions.ProposeCodeCandidate, SceneOutput, "write", "commit", True),
            (actions.InspectCandidate, InspectionOutput, "write", "commit", False),
            (actions.ReviewCode, CodeReviewOutput, "write", "commit", True),
            (actions.SmokeScene, ObjectRef, "exec", "external", False),
            (
                actions.SubmitRender,
                RenderingOutput,
                "cluster" if manifest.backend == "slurm" else "exec",
                "external",
                False,
            ),
            (actions.PollRender, RenderingOutput, "read", "none", False),
            (actions.ReconcileSubmission, RecoveryOutput, "read", "none", False),
            (actions.ReconcileRender, RecoveryOutput, "read", "none", False),
            (actions.ResumeRun, RecoveryPlan, "write", "commit", False),
            (actions.RetryScene, RecoveryPlan, "write", "commit", False),
            (actions.ReuseScene, IncrementalOutput, "write", "commit", False),
            (actions.ReconcileMerge, FinalizationOutput, "exec", "external", False),
            (actions.ProposeRenderFix, RenderRepairOutput, "write", "commit", True),
            (
                actions.CancelRender,
                RenderingOutput,
                "cluster" if manifest.backend == "slurm" else "exec",
                "external",
                False,
            ),
            (actions.EvaluateVisual, VisualSceneOutput, "write", "commit", True),
            (actions.EvaluateBoundary, BoundaryVisualOutput, "write", "commit", True),
            (actions.ProposeVisualFix, VisualRepairProposal, "write", "commit", True),
            (actions.MergeVideo, FinalizationOutput, "exec", "external", False),
            (actions.EvaluateRun, FinalizationOutput, "write", "commit", False),
            (actions.Finish, CompletionOutput, "write", "commit", False),
        )
        for action, output, permission, side_effect, model in declarations:
            name = action.model_fields["kind"].default
            self.registry.register(
                ToolDefinition(
                    ToolPolicySpec(
                        action=name,
                        permission=permission,
                        side_effect=side_effect,
                        uses_model=model,
                        requires_approval=False,
                        timeout_seconds=86400,
                    ),
                    action,
                    output,
                    self._execute,
                )
            )
        adapters = ExecutionAdapters(
            self._latest_intent,
            self._persist_intent,
            self._persist_invocation,
            lambda grant, intent: self.sessions.consume_grant(
                manifest.agent.session_id, grant, intent
            ),
            self._commit_result,
            self._emit_committed,
        )
        self.executor = ToolExecutor(self.registry, adapters)
        self.events.reconcile(manifest, lock_held=True)

    @property
    def manifest(self):
        return self.committer.current

    @_serialized
    def _patch(self, patch, phase="agent_transport"):
        return self.committer.commit(
            CommitCommand(expected_revision=self.manifest.revision, phase=phase, patch=patch)
        )

    def _agent_update(self, **updates):
        return self.manifest.agent.model_copy(update=updates)

    @staticmethod
    def _replace(records, value, key):
        return (
            tuple(value if getattr(item, key) == getattr(value, key) else item for item in records)
            if any(getattr(item, key) == getattr(value, key) for item in records)
            else (*records, value)
        )

    @_serialized
    def _latest_intent(self, action, session_id):
        return next(
            (
                item
                for item in reversed(self.manifest.agent.intents)
                if item.session_id == session_id
                and action_signature(item.action) == action_signature(action)
            ),
            None,
        )

    @_serialized
    def _persist_intent(self, intent):
        previous = next(
            (item for item in self.manifest.agent.intents if item.intent_id == intent.intent_id),
            None,
        )
        if previous == intent or (previous is not None and previous.status == "completed"):
            return
        if previous is None and any(
            item.session_id == intent.session_id
            and action_signature(item.action) == action_signature(intent.action)
            for item in self.manifest.agent.intents
        ):
            raise ValueError(
                "Action already has a durable intent; reconcile instead of dispatching twice"
            )
        self._patch(
            RunPatch(
                agent=self._agent_update(
                    intents=self._replace(self.manifest.agent.intents, intent, "intent_id")
                )
            )
        )

    @_serialized
    def _persist_invocation(self, invocation):
        previous = next(
            (
                item
                for item in self.manifest.agent.invocations
                if item.invocation_id == invocation.invocation_id
            ),
            None,
        )
        if previous == invocation or (previous is not None and previous.status == "completed"):
            return
        submissions = self.manifest.agent.submissions
        if (
            invocation.action in {"submit_render", "reconcile_submission"}
            and invocation.external_job_id is not None
        ):
            bound_action = next(
                item.action
                for item in self.manifest.agent.intents
                if item.intent_id == invocation.intent_id
            )
            submissions = tuple(
                item.model_copy(
                    update={
                        "attempt": item.attempt.model_copy(
                            update={
                                "status": "submitted",
                                "job_id": invocation.external_job_id,
                            }
                        )
                    }
                )
                if (
                    (
                        invocation.action == "submit_render"
                        and item.intent_id in {None, invocation.intent_id}
                        and item.attempt.scene_revision == invocation.work_identity.revision
                    )
                    or (
                        invocation.action == "reconcile_submission"
                        and item.attempt.token == bound_action.attempt_token
                    )
                )
                and item.attempt.code_sha256 == self.manifest.scenes[item.scene_id].code_sha256
                and invocation.resource_id == f"run:{self.manifest.run_id}/scene:{item.scene_id}"
                else item
                for item in submissions
            )
        self._patch(
            RunPatch(
                agent=self._agent_update(
                    invocations=self._replace(
                        self.manifest.agent.invocations, invocation, "invocation_id"
                    ),
                    submissions=submissions,
                )
            )
        )

    @_serialized
    def _commit_result(self, intent, invocation, result):
        if intent.action.recovery_epoch != self.manifest.agent.recovery_epoch:
            from kd1_anime.domain.dependencies import StaleWorkError

            raise StaleWorkError("Result belongs to a superseded recovery invocation epoch")
        updates = {name: getattr(result.patch, name) for name in result.patch.model_fields_set}
        proposed = updates.get("agent") or self.manifest.agent
        if intent.action.scene_id is not None and updates.get("scenes"):
            proposed, expanded = self._invalidate_consumers(proposed, updates["scenes"])
            updates["scenes"] = expanded
        latest = self.manifest.agent
        if intent.action.scene_id is not None:
            stored = self.manifest.scenes.get(intent.action.scene_id)
            expected = (
                WorkIdentity(stored.revision, stored.input_sha256)
                if stored
                else latest.outline_identities[intent.action.scene_id]
            )
            if intent.action.work_identity != expected:
                from kd1_anime.domain.dependencies import StaleWorkError

                raise StaleWorkError("Scene result was built from superseded inputs")
            if stored is not None and stored.input_sha256 != self._scene_digest(
                stored, manifest=self.manifest
            ):
                from kd1_anime.domain.dependencies import StaleWorkError

                raise StaleWorkError("Scene result was built from superseded shared inputs")
        if intent.action.scene_id is None:
            if intent.action.work_identity != WorkIdentity(
                latest.run_revision, latest.run_input_sha256
            ):
                from kd1_anime.domain.dependencies import StaleWorkError

                raise StaleWorkError("Root result was built from superseded run inputs")
            if (
                intent.action.kind == "publish_continuity"
                and intent.action.publication != self._publication_binding(self.manifest)
            ):
                from kd1_anime.domain.dependencies import StaleWorkError

                raise StaleWorkError("Continuity result was built from superseded ordered sources")
            semantic = {
                name: getattr(proposed, name)
                for name in type(proposed).model_fields
                if name
                not in {
                    "intents",
                    "invocations",
                    "usage",
                    "attempts",
                    "submissions",
                    "committed_invocations",
                    "step_count",
                }
            }
        else:
            touched = {patch.scene_id for patch in updates.get("scenes", ())} | {
                intent.action.scene_id
            }
            semantic = {}
            for name in (
                "scene_next_actions",
                "scene_action_versions",
                "reviewed_identities",
                "pending_candidates",
                "scene_review_counters",
                "plan_repair_proposals",
                "visual_repairs",
                "incremental_checked",
                "reuse_receipts",
            ):
                values = dict(getattr(latest, name))
                incoming = getattr(proposed, name)
                for scene_id in touched:
                    if scene_id in incoming:
                        values[scene_id] = incoming[scene_id]
                    else:
                        values.pop(scene_id, None)
                semantic[name] = values
            if proposed.stop_reason or intent.action.kind in {"retry_scene", "resume_run"}:
                semantic["stop_reason"] = proposed.stop_reason
        reviewed = dict(semantic.get("reviewed_identities", latest.reviewed_identities))
        for patch in updates.get("scenes", ()):
            previous = self.manifest.scenes.get(patch.scene_id)
            replacement = patch.replacement
            if (
                previous is not None
                and previous.input_sha256 == replacement.input_sha256
                and previous.reviewed
                and replacement.reviewed
            ):
                reviewed[patch.scene_id] = WorkIdentity(
                    previous.revision + 1, replacement.input_sha256
                )
            elif previous is not None and previous.input_sha256 != replacement.input_sha256:
                semantic["continuity_published"] = False
        semantic["reviewed_identities"] = reviewed
        retired = list(dict.fromkeys((*latest.retired_submissions, *proposed.retired_submissions)))
        clearing = {
            patch.scene_id
            for patch in updates.get("scenes", ())
            if patch.replacement.submission is None
        } | set(updates.get("scene_removals", {}))
        for identifier in clearing:
            previous = self.manifest.scenes.get(identifier)
            if (
                previous
                and previous.submission
                and previous.slurm_job
                and (previous.slurm_job.cancelled or previous.slurm_job.status in TERMINAL_STATES)
                and previous.submission.token not in retired
            ):
                retired.append(previous.submission.token)
        semantic["retired_submissions"] = tuple(retired)
        if "artifact_refs" in updates:
            updates["artifact_refs"] = {**self.manifest.artifact_refs, **updates["artifact_refs"]}
        if "rag_receipts" in updates:
            dropped = (
                set(result.stage.value.drop_rag_receipt_keys)
                if intent.action.kind in {"resume_run", "retry_scene"}
                else set()
            )
            updates["rag_receipts"] = {
                **{
                    key: value
                    for key, value in self.manifest.rag_receipts.items()
                    if key not in dropped
                },
                **updates["rag_receipts"],
            }
        complete_intent = intent.model_copy(update={"status": "completed"})
        report = self._store_output(result.stage.value)
        complete_invocation = invocation.model_copy(
            update={"status": "completed", "finished_at": utc_now(), "result_ref": report}
        )
        latest = merge_transport_state(self.manifest.agent, proposed)
        updates["agent"] = latest.model_copy(
            update={
                **semantic,
                "step_count": latest.step_count + 1,
                "intents": self._replace(latest.intents, complete_intent, "intent_id"),
                "invocations": self._replace(
                    latest.invocations, complete_invocation, "invocation_id"
                ),
                "committed_invocations": tuple(
                    dict.fromkeys((*latest.committed_invocations, invocation.invocation_id))
                ),
            }
        )
        try:
            manifest = self._patch(RunPatch(**updates), phase=intent.action.kind)
        except Exception:
            if intent.action.kind == "submit_render" and invocation.external_job_id is not None:

                def record_cancellation(status):
                    self._patch(
                        RunPatch(
                            agent=self._agent_update(
                                submissions=tuple(
                                    item.model_copy(update={"cancellation_status": status})
                                    if item.intent_id == intent.intent_id
                                    else item
                                    for item in self.manifest.agent.submissions
                                )
                            )
                        )
                    )

                with suppress(Exception):
                    record_cancellation("unknown")
                cancelled = False
                with suppress(Exception):
                    cancelled = bool(self._render().backend.cancel_job(invocation.external_job_id))
                with suppress(Exception):
                    record_cancellation("succeeded" if cancelled else "failed")
                # The durable attempt and returned ID remain reserved even when
                # cancellation fails; recovery must reconcile, never resubmit.
            raise
        if "rag_profile" in updates and manifest.rag_profile.enabled:
            self.scene_pipeline.pin_profile(manifest.rag_profile)
        return manifest.revision

    def _invalidate_consumers(self, proposed, patches):
        """Publish dependency invalidations with the producer, never after it."""
        before = self.manifest
        after = before.model_copy()
        after.scenes = {**before.scenes, **{patch.scene_id: patch.replacement for patch in patches}}
        touched = {patch.scene_id for patch in patches}
        affected = DependencyService.affected_manifest(before, after, tuple(touched)) - touched
        if not affected:
            return proposed, patches
        actions_map = dict(proposed.scene_next_actions)
        versions = dict(proposed.scene_action_versions)
        reviewed = dict(proposed.reviewed_identities)
        candidates = dict(proposed.pending_candidates)
        for scene_id in affected:
            previous = before.scenes[scene_id]
            after.scenes[scene_id] = StoredSceneState(
                plan=previous.plan.model_copy(deep=True),
                plan_ready=True,
                plan_reviewed=previous.plan_reviewed,
                phase="plan_reviewed" if previous.plan_reviewed else "detailed",
            )
            actions_map[scene_id] = "propose_technical_spec"
            versions[scene_id] = versions.get(scene_id, 0) + 1
            reviewed.pop(scene_id, None)
            candidates.pop(scene_id, None)
        invalidations = []
        for scene_id in sorted(affected):
            replacement = after.scenes[scene_id]
            previous = before.scenes[scene_id]
            patch = self._scene_patch(
                replacement,
                manifest=before,
                identity=WorkIdentity(previous.revision, previous.input_sha256),
            )
            replacement.input_sha256 = self._scene_digest(replacement, manifest=after)
            invalidations.append(patch.model_copy(update={"replacement": replacement}))
        return proposed.model_copy(
            update={
                "scene_next_actions": actions_map,
                "scene_action_versions": versions,
                "reviewed_identities": reviewed,
                "pending_candidates": candidates,
                "continuity_published": False,
            }
        ), (*patches, *invalidations)

    @_serialized
    def _emit_committed(self, intent, invocation, revision):
        self.events.reconcile(self.manifest, lock_held=True)
        self.events.append(
            name="action_completed",
            durability="committed",
            manifest_revision=revision,
            intent_id=intent.intent_id,
            invocation_id=invocation.invocation_id,
            payload=EventPayload(action=intent.action.kind, state=self.manifest.state),
            lock_held=True,
        )

    @staticmethod
    def _secrets():
        return tuple(
            value
            for value in (
                settings.LLM_API_KEY,
                settings.VISUAL_LLM_API_KEY,
                settings.RAG_EMBEDDING_API_KEY,
                settings.RAG_RERANK_API_KEY,
            )
            if value
        )

    def _notify(self, name, data):
        CallbackAdapter(self._callback).publish(
            PipelineEvent(name, actions.sanitize_record(data, self._secrets()))
        )

    def _request_context(self, scene_id, stage):
        def reserve(profile, model, **metadata):
            invocation = current_invocation.get()
            with self._runtime_lock:
                receipt = RequestReservation(
                    scene_id=scene_id,
                    scene_revision=invocation.work_identity.revision if invocation else 0,
                    stage=stage,
                    profile=profile,
                    model=model,
                    **{
                        key: value
                        for key, value in metadata.items()
                        if key in {"call_id", "parent_span_id", "attempt", "json_repair_attempt"}
                    },
                )
                self._patch(RunPatch(reservation_additions=(receipt,)))
                return receipt

        def emit(name, data):
            if name != "model_request_finished":
                return
            invocation = current_invocation.get()
            usage = UsageReceipt(
                request_id=data["request_id"],
                invocation_id=invocation.invocation_id if invocation else None,
                usage_source="provider_reported" if data.get("usage") else "unknown",
                **(data.get("usage") or {}),
            )
            with self._runtime_lock:
                self._patch(
                    RunPatch(agent=self._agent_update(usage=(*self.manifest.agent.usage, usage)))
                )

        return RequestContext(reserve, emit)

    @_serialized
    def _reserve_stage(self, action):
        if not self.registry.get(action.kind).policy.uses_model:
            return
        if (
            action.kind == "revise_plan"
            and sum(
                item.used
                for item in self.manifest.agent.attempts
                if item.operation == "revise_plan"
            )
            >= self.manifest.run_config.resolve().MAX_PLAN_REPLAN_ATTEMPTS
        ):
            raise RuntimeError("Durable plan replan budget exhausted")
        previous = [
            attempt.used
            for attempt in self.manifest.agent.attempts
            if (attempt.operation, attempt.resource_id, attempt.input_sha256)
            == (action.kind, action.resource_id, action.input_sha256)
        ]
        used = max(previous, default=0) + 1
        limits = {
            "propose_code_candidate": "CODE_VALIDATION_ATTEMPTS",
            "propose_technical_spec": "MAX_TECHNICAL_SPEC_ATTEMPTS",
            "review_code": "MAX_REVIEW_ROUNDS",
            "review_plan": "MAX_PLAN_REVIEW_ROUNDS",
        }
        maximum = self.manifest.retry_limits.get(limits.get(action.kind), 1)
        if action.kind == "propose_render_fix":
            maximum = (
                self.manifest.run_config.resolve().MAX_FIX_ATTEMPTS
                + self.manifest.run_config.resolve().MAX_INFRA_RETRIES
                + 1
            )
        if used > maximum:
            raise RuntimeError("Durable stage attempt budget exhausted")
        attempt = AgentStageAttempt(
            operation=action.kind,
            resource_id=action.resource_id,
            input_sha256=action.input_sha256,
            used=used,
        )
        self._patch(
            RunPatch(
                agent=self._agent_update(
                    attempts=(
                        *(
                            item
                            for item in self.manifest.agent.attempts
                            if (item.operation, item.resource_id, item.input_sha256)
                            != (action.kind, action.resource_id, action.input_sha256)
                        ),
                        attempt,
                    )
                )
            )
        )

    @_serialized
    def _reserve_generation_attempt(self, action):
        attempts = self.manifest.agent.attempts
        key = ("code_generation", action.resource_id, action.input_sha256)
        used = (
            max(
                (
                    item.used
                    for item in attempts
                    if (item.operation, item.resource_id, item.input_sha256) == key
                ),
                default=0,
            )
            + 1
        )
        if used > self.manifest.run_config.resolve().CODE_VALIDATION_ATTEMPTS:
            raise RuntimeError("Durable generation validation budget exhausted")
        attempt = AgentStageAttempt(
            operation=key[0], resource_id=key[1], input_sha256=key[2], used=used
        )
        self._patch(
            RunPatch(
                agent=self._agent_update(
                    attempts=(
                        *(
                            item
                            for item in attempts
                            if (item.operation, item.resource_id, item.input_sha256) != key
                        ),
                        attempt,
                    )
                )
            )
        )
        return used

    def _store_output(self, value):
        text = json.dumps(
            actions.sanitize_record(value, self._secrets()), ensure_ascii=False, sort_keys=True
        )
        path = ObjectStore(self.root).report(text)
        return ObjectRef(path=path, sha256=sha256_text(text), kind="report")

    def _planning_output(self):
        manifest = self.manifest
        diagnostics = ()
        receipts = ()
        for invocation in reversed(manifest.agent.invocations):
            if invocation.status != "completed" or invocation.action not in {
                "propose_plan",
                "review_plan",
                "revise_plan",
            }:
                continue
            reference = invocation.result_ref
            if reference is None or reference.kind != "report":
                raise ValueError("Planning history is missing its immutable result receipt")
            path = restore_run_path(self.root, reference.path)
            if not path.is_file() or sha256_file(path) != reference.sha256:
                raise ValueError("Planning history result identity mismatch")
            historical = PlanningOutput.model_validate_json(path.read_text(encoding="utf-8"))
            diagnostics, receipts = historical.diagnostics, historical.rag_receipts
            break
        return PlanningOutput(
            outlines=tuple(item.model_copy(deep=True) for item in manifest.outlines),
            plans=tuple(
                scene.plan.model_copy(deep=True) for _, scene in sorted(manifest.scenes.items())
            ),
            lesson_spec=manifest.lesson_spec.model_copy(deep=True),
            teaching_graph=manifest.teaching_graph.model_copy(deep=True),
            continuity_bible=manifest.continuity_bible.model_copy(deep=True)
            if manifest.continuity_bible
            else None,
            rag_profile=manifest.rag_profile,
            diagnostics=diagnostics,
            rag_receipts=receipts,
        )

    @staticmethod
    def _dispatch_identity(manifest, scene_id, attempt=None):
        scene = manifest.scenes[scene_id]
        attempt = attempt or scene.submission
        if attempt is None:
            return None
        if not any(
            item.scene_id == scene_id and item.attempt == attempt
            for item in manifest.agent.submissions
        ):
            raise ValueError("Current submission differs from its durable attempt")
        reservation = next(
            item for item in manifest.agent.submissions if item.attempt.token == attempt.token
        )
        matches = [
            intent.action.work_identity
            for intent in manifest.agent.intents
            if intent.action.kind == "submit_render"
            and (reservation.intent_id is None or intent.intent_id == reservation.intent_id)
            and intent.action.scene_id == scene_id
            and intent.action.work_identity.revision == attempt.scene_revision
            and intent.action.source.sha256 == scene.code_sha256
            and intent.action.work_identity.input_sha256 == scene.input_sha256
        ]
        if len(matches) != 1:
            raise ValueError("Current submission has no unique durable dispatch identity")
        return matches[0]

    def _scene_inputs(self, scene_id, *, manifest=None, inherit=False, submission_attempt=None):
        manifest = manifest or self.manifest
        previous = manifest.scenes.get(scene_id - 1)
        inputs = SceneInputs(
            run_id=manifest.run_id,
            root=self.root,
            config=manifest.run_config,
            render_profile=manifest.render_profile,
            execution_spec=manifest.execution_spec,
            scene=manifest.scenes[scene_id].model_copy(deep=True),
            plans=tuple(
                scene.plan.model_copy(deep=True) for _, scene in sorted(manifest.scenes.items())
            ),
            lesson_spec=manifest.lesson_spec.model_copy(deep=True),
            teaching_graph=manifest.teaching_graph.model_copy(deep=True),
            continuity_bible=manifest.continuity_bible.model_copy(deep=True)
            if manifest.continuity_bible
            else None,
            element_manifest=manifest.element_manifest.model_copy(deep=True),
            state_ledger=manifest.state_ledger.model_copy(deep=True),
            previous_technical_handoff=previous.technical_spec.handoff_out.model_copy(deep=True)
            if previous and previous.technical_spec and previous.technical_spec.handoff_out
            else None,
            dry_run=manifest.dry_run,
            direct_render=manifest.direct_render,
            auto_fix=manifest.auto_fix,
            visual_eval_profile=manifest.visual_eval_profile,
            submission_identity=self._dispatch_identity(manifest, scene_id, submission_attempt),
            context_sha256=self._scene_context_digest(manifest.scenes[scene_id], manifest=manifest),
            reuse_provenance=manifest.agent.reuse_receipts.get(scene_id),
        )
        if inherit:
            prior = tuple(
                self._scene_inputs(identifier, manifest=manifest)
                for identifier in sorted(manifest.scenes)
                if identifier < scene_id
            )
            from dataclasses import replace

            prepared = prepare_inherited_inputs(inputs, prior)
            return replace(
                prepared,
                context_sha256=self._scene_context_digest(prepared.scene, manifest=manifest),
            )
        return inputs

    def _final_inputs(self):
        manifest = self.manifest
        return FinalizationInputs(
            manifest.run_id,
            self.root,
            manifest.run_config,
            manifest.render_profile,
            manifest.merge_profile,
            tuple(scene.model_copy(deep=True) for _, scene in sorted(manifest.scenes.items())),
            Path(manifest.output_path),
            user_prompt=manifest.user_prompt,
            lesson_spec=manifest.lesson_spec,
            teaching_graph=manifest.teaching_graph,
            continuity_bible=manifest.continuity_bible,
            state_ledger=manifest.state_ledger,
            allow_partial=manifest.run_config.resolve().ALLOW_PARTIAL_OUTPUT,
            visual_eval_profile=manifest.visual_eval_profile,
            auto_fix=manifest.auto_fix,
            eval_round=manifest.eval_round,
            existing_output_sha256=manifest.final_video_sha256
            or manifest.agent.previous_output_sha256
            or "",
            reuse_receipts=dict(manifest.agent.reuse_receipts),
        )

    def _render(self):
        if self._render_pipeline is None:
            self._backend = self.dependencies.backend_factory(self.manifest.backend)
            self._backend.execution_spec = self.manifest.execution_spec
            self._render_pipeline = RenderPipeline(
                self._backend, preflight=self.dependencies.preflight, control=self.control
            )
        return self._render_pipeline

    def _plan_digest(self, planning):
        return digest(
            {
                "planning": planning.model_dump(
                    mode="json", exclude={"diagnostics", "rag_receipts", "rag_profile"}
                ),
                "rag": self.manifest.rag_profile.model_dump(mode="json"),
                "config": self.manifest.run_config.model_dump(mode="json"),
                "renderer": self.manifest.render_profile.model_dump(mode="json"),
                "generation": self.manifest.agent.generation_identity.model_dump(mode="json"),
            }
        )

    def _scene_digest(self, scene, *, planning=None, manifest=None):
        return scene_work_digest(
            scene, self._scene_context_digest(scene, planning=planning, manifest=manifest)
        )

    def _scene_context_digest(self, scene, *, planning=None, manifest=None):
        manifest = manifest or self.manifest
        plans = (
            planning.plans
            if planning is not None
            else tuple(item.plan for item in manifest.scenes.values())
        )
        dependencies = DependencyService.dependencies(
            scene.plan,
            plans,
            inherited_code=scene.inherited_elements_code,
            handoff_source=scene.technical_spec.handoff_in.source_scene_id
            if scene.technical_spec and scene.technical_spec.handoff_in
            else None,
        )
        shared = planning
        return digest(
            {
                "lesson": (shared.lesson_spec if shared else manifest.lesson_spec).model_dump(
                    mode="json"
                ),
                "teaching": (
                    shared.teaching_graph if shared else manifest.teaching_graph
                ).model_dump(mode="json"),
                "bible": (
                    shared.continuity_bible if shared else manifest.continuity_bible
                ).model_dump(mode="json")
                if (shared.continuity_bible if shared else manifest.continuity_bible)
                else None,
                "rag": manifest.rag_profile.model_dump(mode="json")
                if manifest.rag_profile
                else None,
                "dependencies": {
                    f"{dependency.source_scene_id}:{dependency.kind}": (
                        manifest.scenes[dependency.source_scene_id].code_sha256
                        if dependency.kind == "code_context"
                        else DependencyService.teaching(
                            manifest.scenes[dependency.source_scene_id].plan
                        )
                        if dependency.kind == "teaching"
                        else DependencyService.contract(
                            manifest.scenes[dependency.source_scene_id].plan,
                            manifest.scenes[dependency.source_scene_id].technical_spec,
                        )
                    )
                    for dependency in dependencies
                    if dependency.source_scene_id in manifest.scenes
                    and scene.technical_spec is not None
                },
                "config": manifest.run_config.model_dump(mode="json"),
                "renderer": manifest.render_profile.model_dump(mode="json"),
                "generation": manifest.agent.generation_identity.model_dump(mode="json"),
            }
        )

    def _scene_patch(self, replacement, *, manifest=None, planning=None, identity=None):
        active = _active_work.get()
        manifest = manifest or (active[1] if active else self.manifest)
        scene_id = replacement.plan.scene_id
        stored = manifest.scenes.get(scene_id)
        if stored is not None:
            changed = stored.code_sha256 != replacement.code_sha256
            if (
                replacement.slurm_job is not None
                and replacement.slurm_job.code_sha256 != replacement.code_sha256
            ):
                replacement.slurm_job = replacement.submission = None
            if (
                replacement.artifact is not None
                and replacement.artifact.code_sha256 != replacement.code_sha256
            ):
                replacement.artifact, replacement.rendered = None, False
            if replacement.static_verification.code_sha256 != replacement.code_sha256:
                replacement.static_verification = type(replacement.static_verification)()
            receipt = replacement.execution_verification
            if receipt.code_sha256 != replacement.code_sha256 or (
                receipt.scope == "formal_video" and replacement.artifact is None
            ):
                replacement.execution_verification = type(receipt)()
            visual = replacement.visual_verification
            if (visual.code_sha256 != replacement.code_sha256 or replacement.artifact is None) and (
                visual.status != "pending"
                or replacement.visual_report_file
                or replacement.visual_status in {"passed", "warning", "unknown", "needs_fix"}
            ):
                replacement.visual_verification = type(visual)()
                replacement.visual_status, replacement.visual_score = "pending", None
                replacement.visual_report_file = replacement.visual_report_sha256 = (
                    replacement.visual_artifact_sha256
                ) = ""
            if changed:
                replacement.local_smoke_status = "pending"
        if identity is None:
            identity = (
                active[0].work_identity
                if active and active[0].scene_id == scene_id
                else (
                    WorkIdentity(stored.revision, stored.input_sha256)
                    if stored
                    else manifest.agent.outline_identities.get(scene_id, WorkIdentity(0, ""))
                )
            )
        replacement.input_sha256 = self._scene_digest(
            replacement, planning=planning, manifest=manifest
        )
        plans = (
            planning.plans
            if planning is not None
            else tuple(item.plan for item in manifest.scenes.values())
        )
        dependencies = DependencyService.dependencies(
            replacement.plan,
            plans,
            inherited_code=replacement.inherited_elements_code,
            handoff_source=replacement.technical_spec.handoff_in.source_scene_id
            if replacement.technical_spec and replacement.technical_spec.handoff_in
            else None,
        )
        return ScenePatch(
            scene_id=scene_id,
            identity=identity,
            replacement=replacement,
            dependency_revisions={
                item.source_scene_id: manifest.scenes[item.source_scene_id].revision
                for item in dependencies
                if item.source_scene_id in manifest.scenes
            },
        )

    @staticmethod
    def _resume_phase(manifest, agent):
        if not manifest.outlines:
            return "PLANNING", "propose_plan"
        if len(manifest.scenes) != len(manifest.outlines):
            return "DETAILING", "detail_scene"
        if not agent.plan_review_complete or any(
            not scene.plan_reviewed for scene in manifest.scenes.values()
        ):
            return "PLAN_REVIEWING", "review_plan"
        if (
            agent.operation != "render"
            and agent.approved_plan_sha256 is None
            and agent.operation != "plan"
        ):
            return "PLAN_REVIEWING", "review_plan"
        if agent.operation == "plan":
            return "EVALUATING", "finish"
        if (
            any(not scene.reviewed for scene in manifest.scenes.values())
            or not agent.continuity_published
        ):
            return "CODING", "publish_continuity"
        if manifest.dry_run:
            return "EVALUATING", "finish"
        if any(not scene.rendered and not scene.give_up for scene in manifest.scenes.values()):
            return "MONITORING", "poll_render"
        if manifest.final_video:
            return "EVALUATING", "finish" if agent.next_action == "finish" else "evaluate_run"
        return (
            ("VISUAL_EVALUATING", "evaluate_boundary")
            if manifest.visual_eval_profile.enabled
            else ("MERGING", "merge_video")
        )

    def _restore_retry_counters(self, scene, agent):
        reference = agent.retired_scenes.get(scene.plan.scene_id)
        if reference is None:
            return scene
        path = restore_run_path(self.root, reference.path)
        if reference.kind != "report" or sha256_file(path) != reference.sha256:
            raise ValueError("Retired scene counter snapshot hash mismatch")
        previous = StoredSceneState.model_validate_json(path.read_text())
        if previous.plan.scene_id != scene.plan.scene_id:
            raise ValueError("Retired scene counter snapshot identity mismatch")
        for name in (
            "review_round",
            "fix_attempts",
            "infra_retries",
            "plan_replan_attempts",
            "plan_review_round",
            "identical_plan_review_count",
            "identical_review_count",
            "visual_fix_attempts",
            "identical_error_count",
            "stagnant_repair_count",
        ):
            setattr(scene, name, max(getattr(scene, name), getattr(previous, name)))
        scene.stage_attempts = [item.model_copy(deep=True) for item in previous.stage_attempts]
        scene.submission_attempts = [
            item.model_copy(deep=True) for item in previous.submission_attempts
        ]
        for name in (
            "review_signature",
            "plan_review_signature",
            "last_error_fp",
            "last_repair_code_sha256",
            "last_repair_error_fp",
        ):
            setattr(scene, name, getattr(previous, name))
        return scene

    def _recovery_updates(self, plan, manifest, *, retry=False):
        if plan.blocked or plan.cancellations:
            raise RuntimeError("Recovery prerequisites must be confirmed before state publication")
        agent = manifest.agent
        updates = {"status": "running", "error": "", "run_phase": "executing"}
        future = manifest.model_copy()
        if plan.rag_profile is not None:
            future.rag_profile = plan.rag_profile
            updates["rag_profile"] = plan.rag_profile
            updates["rag_receipts"] = {}
            agent = agent.model_copy(
                update={
                    "plan_sha256": None,
                    "approved_plan_sha256": None,
                    "approved_plan_identity": None,
                    "plan_review_complete": False,
                    "pending_approval": None,
                    "run_revision": agent.run_revision + 1,
                    "run_input_sha256": digest(
                        {
                            "prior": agent.run_input_sha256,
                            "rag": plan.rag_profile.model_dump(mode="json"),
                            "revision": agent.run_revision + 1,
                        }
                    ),
                }
            )
        if plan.invalidate_planning:
            retired = dict(agent.retired_scenes)
            retired.update(
                (identifier, self._store_output(scene))
                for identifier, scene in manifest.scenes.items()
            )
            updates.update(
                scene_removals={
                    identifier: WorkIdentity(scene.revision, scene.input_sha256)
                    for identifier, scene in manifest.scenes.items()
                },
                outlines=[],
                dependency_graph=DependencyGraph(),
                state_ledger=StateLedger(),
                element_manifest=ElementManifest(),
                plan_review_status="pending",
            )
            future.scenes, future.outlines = {}, []
            agent = agent.model_copy(
                update={
                    "retired_scenes": retired,
                    "outline_identities": {},
                    "scene_next_actions": {},
                    "scene_action_versions": {},
                    "reviewed_identities": {},
                    "pending_candidates": {},
                    "visual_repairs": {},
                    "reuse_receipts": {},
                    "incremental_checked": {},
                    "continuity_published": False,
                }
            )
            patches = ()
        else:
            future.scenes = {
                **manifest.scenes,
                **{item.scene_id: item.replacement for item in plan.scenes},
            }
            patches = tuple(
                self._scene_patch(item.replacement, manifest=future, identity=item.identity)
                for item in plan.scenes
            )
            names, versions = dict(agent.scene_next_actions), dict(agent.scene_action_versions)
            candidates, visual, reuse = (
                dict(agent.pending_candidates),
                dict(agent.visual_repairs),
                dict(agent.reuse_receipts),
            )
            reviewed = dict(agent.reviewed_identities)
            for item, patch in zip(plan.scenes, patches, strict=True):
                identifier = item.scene_id
                names[identifier] = item.next_action
                versions[identifier] = versions.get(identifier, 0) + 1
                candidates.pop(identifier, None)
                visual.pop(identifier, None)
                if retry or patch.replacement.artifact is None:
                    reuse.pop(identifier, None)
                if patch.replacement.reviewed:
                    reviewed[identifier] = WorkIdentity(
                        item.identity.revision + 1, patch.replacement.input_sha256
                    )
                else:
                    reviewed.pop(identifier, None)
            agent = agent.model_copy(
                update={
                    "scene_next_actions": names,
                    "scene_action_versions": versions,
                    "pending_candidates": candidates,
                    "visual_repairs": visual,
                    "reuse_receipts": reuse,
                    "reviewed_identities": reviewed,
                    "pending_approval": None
                    if retry or plan.rag_profile is not None
                    else agent.pending_approval,
                    "continuity_published": False if plan.scenes else agent.continuity_published,
                }
            )
        if plan.clear_final or plan.invalidate_planning:
            updates.update(final_video=None, final_video_sha256="")
            future.final_video, future.final_video_sha256 = None, ""
            agent = agent.model_copy(
                update={
                    "previous_output_sha256": manifest.final_video_sha256
                    or agent.previous_output_sha256,
                    "final_visual_report": None,
                    "final_evaluation_report": None,
                    "final_visual_status": "skipped",
                }
            )
        agent = agent.model_copy(update={"stop_reason": ""})
        future.agent = agent
        state, next_action = self._resume_phase(future, agent)
        if manifest.direct_render:
            state = (
                "MONITORING"
                if any(scene.slurm_job for scene in future.scenes.values())
                else "DISPATCHING"
            )
            next_action = "poll_render" if state == "MONITORING" else "submit_render"
            agent = agent.model_copy(update={"continuity_published": True})
        updates.update(
            state=state, scenes=patches, agent=agent.model_copy(update={"next_action": next_action})
        )
        return updates

    def _execute(self, action, context):
        # Reservations are transport-only; they never change a scene identity.
        with self._runtime_lock:
            token = _active_work.set((action, self.manifest))
        try:
            self._reserve_stage(action)
            return self._execute_stage(action, context)
        except Exception as error:
            diagnostics = getattr(error, "diagnostics", ())
            if diagnostics:
                output = getattr(error, "output", None)
                text = json.dumps(
                    actions.sanitize_record(
                        {
                            "diagnostics": [item.model_dump(mode="json") for item in diagnostics],
                            "output": output.model_dump(mode="json")
                            if output is not None
                            else None,
                        }
                    ),
                    ensure_ascii=False,
                )
                path = ObjectStore(self.root).report(text)
                self._patch(
                    RunPatch(artifact_refs={**self.manifest.artifact_refs, "failed_stage": path})
                )
            detail = next((item for item in error.args if isinstance(item, str)), "")
            reason = (
                diagnostics[0].summary
                if diagnostics
                else f"{action.kind} failed ({type(error).__name__}): {detail}"
            )
            self._stop(reason)
            raise
        finally:
            _active_work.reset(token)

    def _execute_stage(self, action, context):
        manifest = _active_work.get()[1]
        agent = manifest.agent
        patches = ()
        updates = {}
        kind = action.kind
        if kind in {"resume_run", "retry_scene"}:
            if kind == "resume_run":
                expected = {item.scene_id: item.work_identity for item in self._snapshot().scenes}
                profile = self._current_rag_profile()
                if action.scenes != expected or action.rag_profile_sha256 != digest(
                    profile.model_dump(mode="json")
                ):
                    raise ValueError("Resume decision snapshot changed before dispatch")
                output = plan_resume(manifest, current_rag_profile=profile)
            else:
                output = plan_retry(manifest, action.scene_id)
            delta = self._recovery_updates(output, manifest, retry=kind == "retry_scene")
            patches = delta.pop("scenes")
            agent = delta.pop("agent")
            updates.update(delta)
            next_action = (
                agent.scene_next_actions[action.scene_id]
                if action.scene_id is not None
                else agent.next_action
            )
        elif kind == "reconcile_submission":
            attempt = self._pending_submission(action.scene_id, manifest=manifest)
            if attempt is None or attempt.token != action.attempt_token:
                raise ValueError("Outstanding submission changed before reconciliation")
            inputs = self._scene_inputs(
                action.scene_id, manifest=manifest, submission_attempt=attempt
            )
            output = RecoveryPipeline(
                self._render().backend, control=self.control
            ).reconcile_submission(inputs, attempt, context)
            if output.status != "adopted":
                raise RuntimeError(
                    output.reason
                    or "Submission remains uncertain; automatic resubmission forbidden"
                )
            scene = inputs.scene
            scene.slurm_job, scene.submission = output.job, output.submission
            scene.submission_attempts = [
                *{
                    item.token: item for item in (*scene.submission_attempts, output.submission)
                }.values()
            ]
            scene.failed = scene.give_up = False
            scene.phase = "monitoring"
            submissions = tuple(
                item.model_copy(update={"attempt": output.submission})
                if item.attempt.token == attempt.token
                else item
                for item in agent.submissions
            )
            reservation = next(
                item for item in agent.submissions if item.attempt.token == attempt.token
            )
            original = next(
                item
                for item in agent.intents
                if item.action.kind == "submit_render"
                and item.action.scene_id == action.scene_id
                and item.action.work_identity == inputs.submission_identity
                and (reservation.intent_id is None or item.intent_id == reservation.intent_id)
            )
            intents = self._replace(
                agent.intents, original.model_copy(update={"status": "completed"}), "intent_id"
            )
            invocations = agent.invocations
            committed = list(agent.committed_invocations)
            reference = self._store_output(RenderingOutput(status="pending", job=output.job))
            for item in agent.invocations:
                if item.intent_id == original.intent_id:
                    value = item.model_copy(
                        update={
                            "status": "completed",
                            "external_job_id": output.job.job_id,
                            "finished_at": utc_now(),
                            "result_ref": reference,
                        }
                    )
                    invocations = self._replace(invocations, value, "invocation_id")
                    if item.invocation_id not in committed:
                        committed.append(item.invocation_id)
            agent = agent.model_copy(
                update={
                    "submissions": submissions,
                    "intents": intents,
                    "invocations": invocations,
                    "committed_invocations": tuple(committed),
                }
            )
            patches = (self._scene_patch(scene),)
            next_action = "poll_render"
        elif kind == "reconcile_render":
            inputs = self._scene_inputs(action.scene_id, manifest=manifest)
            scene = inputs.scene
            expected_job = scene.slurm_job.job_id if scene.slurm_job else None
            expected_artifact = (
                ObjectRef(
                    path=scene.artifact.video_path, sha256=scene.artifact.video_sha256, kind="video"
                )
                if scene.artifact
                else None
            )
            if action.job_id != expected_job or action.artifact != expected_artifact:
                raise ValueError("Restored render identity changed before reconciliation")
            if scene.artifact is not None and scene.artifact.origin == "reused":
                try:
                    validate_artifact(
                        self.root,
                        manifest.run_id,
                        scene,
                        manifest.render_profile,
                        provenance=inputs.reuse_provenance,
                    )
                except (OSError, ValueError) as exc:
                    output = RecoveryOutput(
                        status="restart", invalidate_artifact=True, reason=str(exc)
                    )
                else:
                    output = RecoveryOutput(status="rendered", artifact=scene.artifact)
            else:
                output = RecoveryPipeline(
                    self._render().backend, control=self.control
                ).reconcile_scene(inputs, context)
            if output.status == "blocked":
                raise RuntimeError(output.reason or "Render state remains uncertain")
            if output.job is not None:
                scene.slurm_job = output.job
            if output.status == "rendered":
                was_rendered = scene.rendered
                scene.artifact, scene.rendered = output.artifact, True
                scene.failed = scene.give_up = False
                scene.execution_verification = ExecutionVerification(
                    status="passed",
                    scope="formal_video",
                    code_sha256=scene.code_sha256,
                    artifact_sha256=output.artifact.video_sha256,
                    duration_seconds=output.artifact.metadata.duration_seconds,
                    checked_at=utc_now().isoformat(),
                )
                if not was_rendered:
                    scene.visual_status = (
                        "pending" if manifest.visual_eval_profile.enabled else "skipped"
                    )
                scene.phase = (
                    "visual_evaluating"
                    if scene.visual_status == "pending" and manifest.visual_eval_profile.enabled
                    else "visual_accepted"
                )
                next_action = (
                    "evaluate_visual" if manifest.visual_eval_profile.enabled else "merge_video"
                )
            elif output.status == "restart":
                retired = list(agent.retired_submissions)
                if scene.submission is not None and scene.submission.token not in retired:
                    retired.append(scene.submission.token)
                scene.slurm_job = scene.submission = scene.artifact = None
                scene.rendered = scene.failed = scene.give_up = False
                scene.phase = "reviewed" if scene.reviewed else "coded"
                agent = agent.model_copy(
                    update={
                        "retired_submissions": tuple(retired),
                        "previous_output_sha256": manifest.final_video_sha256
                        or agent.previous_output_sha256,
                    }
                )
                updates.update(
                    state="DISPATCHING" if scene.reviewed else "CODING",
                    final_video=None,
                    final_video_sha256="",
                )
                next_action = "submit_render" if scene.reviewed else "review_code"
            else:
                scene.phase = "monitoring"
                next_action = "poll_render"
            patches = (self._scene_patch(scene),)
        elif kind == "reuse_scene":
            if (
                self._base_manifest is None
                or action.base_run_id != manifest.base_run_id
                or self._base_manifest.run_id != action.base_run_id
            ):
                raise ValueError(
                    "Pending incremental reuse requires an explicit verified base manifest"
                )
            output = IncrementalPipeline().reuse(
                IncrementalInputs(
                    base=self._base_manifest,
                    base_root=RunRepository(settings.WORKSPACE_DIR).run_root(action.base_run_id),
                    current=self._scene_inputs(action.scene_id, manifest=manifest),
                    generation_identity=agent.generation_identity,
                    current_scenes=tuple(
                        item.model_copy(deep=True) for item in manifest.scenes.values()
                    ),
                    rag_profile=manifest.rag_profile,
                    continuity_ready=agent.continuity_published,
                    backend=manifest.backend,
                ),
                context,
            )
            checked, receipts = dict(agent.incremental_checked), dict(agent.reuse_receipts)
            checked[action.scene_id] = manifest.scenes[action.scene_id].input_sha256
            if output.eligible:
                receipts[action.scene_id] = output.provenance
                scene = output.scene
                scene.phase = (
                    "visual_evaluating"
                    if manifest.visual_eval_profile.enabled
                    else "visual_accepted"
                )
                patches = (self._scene_patch(scene),)
                updates["artifact_refs"] = {
                    f"reuse_scene_{action.scene_id}": output.provenance.path
                }
                next_action = (
                    "evaluate_visual" if manifest.visual_eval_profile.enabled else "merge_video"
                )
            else:
                next_action = "submit_render"
            agent = agent.model_copy(
                update={"incremental_checked": checked, "reuse_receipts": receipts}
            )
        elif kind == "propose_plan":
            if "imported_plan" in manifest.artifact_refs:
                output = self._parse_export(
                    json.loads(
                        restore_run_path(
                            self.root, manifest.artifact_refs["imported_plan"]
                        ).read_text()
                    )
                )
            else:
                if agent.planning_preflight:
                    probe = BaseAgent()
                    probe.control = self.control
                    probe.request_context = self._request_context(None, "endpoint_probe")
                    self.dependencies.probe(probe)
                output = self.planning.propose(
                    manifest.user_prompt, manifest.render_profile.renderer
                )
            identities = {
                item.scene_id: WorkIdentity(
                    0,
                    digest(
                        {
                            "outline": item.model_dump(mode="json"),
                            "prompt": manifest.user_prompt,
                            "renderer": manifest.render_profile.model_dump(mode="json"),
                            "rag": (output.rag_profile or manifest.rag_profile).model_dump(
                                mode="json"
                            ),
                            "outlines": [
                                outline.model_dump(mode="json") for outline in output.outlines
                            ],
                            "lesson": output.lesson_spec.model_dump(mode="json"),
                            "teaching": output.teaching_graph.model_dump(mode="json"),
                            "bible": output.continuity_bible.model_dump(mode="json")
                            if output.continuity_bible
                            else None,
                            "config": manifest.run_config.model_dump(mode="json"),
                            "generation": agent.generation_identity.model_dump(mode="json"),
                        }
                    ),
                )
                for item in output.outlines
            }
            patches = tuple(
                self._scene_patch(
                    StoredSceneState(plan=plan, phase="detailed", plan_ready=True),
                    manifest=manifest,
                    planning=output,
                    identity=WorkIdentity(0, ""),
                )
                for plan in output.plans
            )
            agent = agent.model_copy(
                update={
                    "outline_identities": identities,
                    "scene_next_actions": {
                        item.scene_id: "review_plan" if output.plans else "detail_scene"
                        for item in output.outlines
                    },
                    "scene_action_versions": {item.scene_id: 0 for item in output.outlines},
                }
            )
            updates.update(
                outlines=list(output.outlines),
                lesson_spec=output.lesson_spec,
                teaching_graph=output.teaching_graph,
                continuity_bible=output.continuity_bible,
                state="PLAN_REVIEWING" if output.plans else "DETAILING",
                plan_review_status="pending",
            )
            if output.rag_profile is not None:
                updates["rag_profile"] = output.rag_profile
            next_action = "review_plan" if output.plans else "detail_scene"
        elif kind == "detail_scene":
            outline = next(item for item in manifest.outlines if item.scene_id == action.scene_id)
            output = self.planning.detail(
                DetailInputs(
                    run_id=manifest.run_id,
                    root=self.root,
                    config=manifest.run_config,
                    outline=outline.model_copy(deep=True),
                    outlines=tuple(item.model_copy(deep=True) for item in manifest.outlines),
                    lesson_spec=manifest.lesson_spec.model_copy(deep=True),
                    teaching_graph=manifest.teaching_graph.model_copy(deep=True),
                    continuity_bible=manifest.continuity_bible.model_copy(deep=True)
                    if manifest.continuity_bible
                    else None,
                    user_prompt=manifest.user_prompt,
                    renderer=manifest.render_profile.renderer,
                )
            )
            patches = (
                self._scene_patch(
                    self._restore_retry_counters(
                        StoredSceneState(plan=output.plan, phase="detailed", plan_ready=True), agent
                    )
                ),
            )
            next_action = "review_plan"
        elif kind == "review_plan":
            if agent.plan_review_complete:
                output = self._planning_output()
                agent = agent.model_copy(
                    update={
                        "approved_plan_sha256": agent.plan_sha256,
                        "approved_plan_identity": WorkIdentity(
                            agent.run_revision, agent.run_input_sha256
                        ),
                        "pending_approval": None,
                        "scene_next_actions": {
                            identifier: "propose_technical_spec" for identifier in manifest.scenes
                        },
                        "scene_action_versions": {
                            identifier: agent.scene_action_versions.get(identifier, 0) + 1
                            for identifier in manifest.scenes
                        },
                    }
                )
                updates.update(state="CODING", run_phase="executing")
                next_action = "propose_technical_spec"
            else:
                try:
                    output = self.planning.review(
                        self._planning_output(),
                        manifest.render_profile.renderer,
                        use_models=agent.planning_review,
                        user_prompt=manifest.user_prompt,
                    )
                except PlanningStageError as error:
                    if not error.repairable:
                        raise
                    output = error.output
                    if not isinstance(output, PlanningOutput):
                        raise
                    feedback = actions.sanitize_record(
                        "\n".join(
                            f"[{issue.code}] {issue.summary}\n{issue.hint}"
                            for issue in error.diagnostics
                            if issue.severity == "major"
                        ),
                        self._secrets(),
                    )
                    return ToolResult(
                        StageResult(action.work_identity, output),
                        RunPatch(
                            state="PLAN_REVIEWING",
                            plan_review_status="pending",
                            agent=agent.model_copy(
                                update={
                                    "next_action": "revise_plan",
                                    "plan_review_complete": False,
                                    "plan_repair_feedback": feedback,
                                    "pending_approval": None,
                                    "approved_plan_sha256": None,
                                    "approved_plan_identity": None,
                                }
                            ),
                        ),
                    )
                replacements = []
                for plan in output.plans:
                    scene = manifest.scenes[plan.scene_id].model_copy(deep=True)
                    scene.plan = plan
                    scene.plan_reviewed = True
                    scene.phase = "plan_reviewed"
                    replacements.append(self._scene_patch(scene, planning=output))
                patches = tuple(replacements)
                plan_hash = self._plan_digest(output)
                agent = agent.model_copy(
                    update={
                        "plan_sha256": plan_hash,
                        "plan_review_complete": True,
                        "run_revision": agent.run_revision + 1,
                        "run_input_sha256": plan_hash,
                        "approved_plan_sha256": None,
                        "approved_plan_identity": None,
                        "plan_repair_feedback": "",
                    }
                )
                updates.update(
                    outlines=list(output.outlines),
                    lesson_spec=output.lesson_spec,
                    teaching_graph=output.teaching_graph,
                    continuity_bible=output.continuity_bible,
                    plan_review_status="passed",
                    continuity_review_status="passed",
                    state="EVALUATING" if agent.operation == "plan" else "PLAN_REVIEWING",
                )
                next_action = "finish" if agent.operation == "plan" else "review_plan"
        elif kind == "revise_plan":
            if action.proposals != tuple(
                agent.plan_repair_proposals[key] for key in sorted(agent.plan_repair_proposals)
            ):
                raise ValueError("Safe plan proposal binding mismatch")
            proposed_plans = []
            for identifier, reference in sorted(agent.plan_repair_proposals.items()):
                path = restore_run_path(self.root, reference.path)
                if reference.kind != "report" or sha256_file(path) != reference.sha256:
                    raise ValueError("Safe plan report identity mismatch")
                report = json.loads(path.read_text())
                scene = manifest.scenes[identifier]
                if (
                    report.get("kind") != "render_repair"
                    or report.get("scene_id") != identifier
                    or report.get("original_code_sha256") != scene.code_sha256
                    or report.get("identity", {}).get("input_sha256") != scene.input_sha256
                ):
                    raise ValueError("Safe plan proposal is stale")
                proposal = ScenePlan.model_validate(report["proposed_plan"])
                if proposal.scene_id != identifier:
                    raise ValueError("Safe plan proposal resource mismatch")
                proposed_plans.append(proposal)
            output = self.planning.revise(
                self._planning_output(),
                action.feedback,
                manifest.render_profile.renderer,
                user_prompt=manifest.user_prompt,
                proposed_plans=tuple(proposed_plans),
            )
            replacements = []
            for plan in output.plans:
                scene = StoredSceneState(plan=plan, phase="detailed", plan_ready=True)
                replacements.append(self._scene_patch(scene, planning=output))
            patches = tuple(replacements)
            agent = agent.model_copy(
                update={
                    "plan_sha256": None,
                    "plan_review_complete": False,
                    "approved_plan_sha256": None,
                    "plan_repair_feedback": "",
                    "plan_repair_proposals": {},
                    "approved_plan_identity": None,
                    "run_revision": agent.run_revision + 1,
                    "run_input_sha256": digest(
                        {
                            "planning": output.model_dump(mode="json"),
                            "revision": agent.run_revision + 1,
                        }
                    ),
                    "reviewed_identities": {},
                    "pending_candidates": {},
                    "scene_review_counters": {},
                    "continuity_published": False,
                    "scene_next_actions": {plan.scene_id: "review_plan" for plan in output.plans},
                    "scene_action_versions": {
                        plan.scene_id: agent.scene_action_versions.get(plan.scene_id, 0) + 1
                        for plan in output.plans
                    },
                }
            )
            updates.update(
                outlines=list(output.outlines),
                lesson_spec=output.lesson_spec,
                teaching_graph=output.teaching_graph,
                continuity_bible=output.continuity_bible,
                plan_review_status="pending",
                state="PLAN_REVIEWING",
            )
            next_action = "review_plan"
        elif kind == "propose_technical_spec":
            output = self.planning.technical(self._scene_inputs(action.scene_id, manifest=manifest))
            scene = manifest.scenes[action.scene_id].model_copy(deep=True)
            scene.technical_spec = output.spec
            scene.technical_spec_sha256 = sha256_text(output.spec.model_dump_json())
            scene.technical_input_sha256 = scene.input_sha256
            scene.technical_status = "passed"
            pinned = manifest.run_config.resolve()
            scene.resource_profile = estimate_render_resources(
                scene.plan,
                output.spec,
                manifest.render_profile,
                cpus_per_task=pinned.SLURM_CPUS_PER_TASK,
                mem_gb=pinned.SLURM_MEM_GB,
                time_limit=pinned.SLURM_TIME_LIMIT,
                gpu_type=pinned.SLURM_GPU_TYPE,
                gpu_count=pinned.SLURM_GPU_COUNT,
                apply_estimate=pinned.AUTO_RESOURCE_ESTIMATION,
            )
            scene.capability_contract = build_capability_contract(
                scene.plan,
                output.spec,
                renderer=manifest.render_profile.renderer,
                render_profile=manifest.render_profile,
                gpu_type=manifest.run_config.resolve().SLURM_GPU_TYPE,
            )
            scene.capability_status = "passed"
            scene.phase = "technical_planning"
            patches = (self._scene_patch(scene),)
            next_action = "propose_code_candidate"
        elif kind == "propose_code_candidate":
            inputs = self._scene_inputs(action.scene_id, manifest=manifest, inherit=True)
            scene = inputs.scene.model_copy(deep=True)
            counters = agent.scene_review_counters.get(action.scene_id)
            used = max(
                (
                    item.used
                    for item in agent.attempts
                    if item.operation == "code_generation"
                    and item.resource_id == action.resource_id
                    and item.input_sha256 == action.input_sha256
                ),
                default=0,
            )
            output = self.scene_pipeline.generate(
                inputs,
                attempts_used=used,
                reserve_attempt=lambda: self._reserve_generation_attempt(action),
                feedback=counters.feedback if counters else scene.rewrite_feedback or None,
            )
            scene.code_file, scene.code_sha256, scene.class_name = (
                output.source.path,
                output.source.sha256,
                output.class_name,
            )
            from kd1_anime.verification import StaticVerification

            scene.static_verification = StaticVerification()
            scene.reviewed = False
            scene.exported_elements_code, scene.exported_elements = "", []
            scene.phase = "coded"
            patches = (self._scene_patch(scene),)
            next_action = "inspect_candidate"
        elif kind == "inspect_candidate":
            if action.proposal is not None:
                proposal = self._load_candidate(agent.pending_candidates[action.scene_id])
                if self._diff_binding(proposal) != action.proposal:
                    raise ValueError("Inspection proposal differs from approved binding")
                output = self.scene_pipeline.accept_proposal(
                    self._scene_inputs(action.scene_id, manifest=manifest, inherit=True), proposal
                )
                pending = dict(agent.pending_candidates)
                pending.pop(action.scene_id, None)
                agent = agent.model_copy(update={"pending_candidates": pending})
            else:
                output = self.scene_pipeline.inspect(
                    self._scene_inputs(action.scene_id, manifest=manifest, inherit=True)
                )
            scene = manifest.scenes[action.scene_id].model_copy(deep=True)
            scene.code_file, scene.code_sha256, scene.class_name = (
                output.source.path,
                output.source.sha256,
                output.class_name,
            )
            scene.static_verification = output.verification
            scene.exported_elements_code = output.exported_elements_code
            scene.exported_elements = list(output.exported_elements)
            scene.reviewed = False
            if output.accepted:
                self._record_candidate(scene, "validated")
            if not output.accepted:
                scene.phase = "failed"
                scene.failed = True
            patches = (self._scene_patch(scene),)
            if not output.accepted:
                updates.update(
                    state="ERROR",
                    status="failed",
                    run_phase="failed",
                    error="Candidate inspection failed",
                )
                agent = agent.model_copy(update={"stop_reason": updates["error"]})
            updates["artifact_refs"] = {f"inspection:{action.scene_id}": output.report.path}
            config = manifest.run_config.resolve()
            smoke_enabled = (
                agent.smoke_requested
                or (not manifest.dry_run and config.LOCAL_SMOKE_RENDER_ENABLED)
                or (manifest.dry_run and scene.unknown_animation_detected)
            )
            next_action = "smoke_scene" if smoke_enabled else "review_code"
        elif kind == "smoke_scene":
            inputs = self._scene_inputs(action.scene_id, manifest=manifest)
            with inputs.config.activate():
                output = SmokePipeline().run(inputs, context)
            scene = inputs.scene
            scene.local_smoke_status = "passed"
            scene.execution_verification = ExecutionVerification(
                status="passed",
                scope="short_video"
                if inputs.config.resolve().LOCAL_SMOKE_RENDER_MODE in {"video", "both"}
                or (manifest.dry_run and scene.unknown_animation_detected)
                else "frame",
                code_sha256=scene.code_sha256,
                checked_at=utc_now().isoformat(),
            )
            self._record_candidate(scene, "smoke")
            patches = (self._scene_patch(scene),)
            updates["artifact_refs"] = {f"smoke:{action.scene_id}": output.path}
            next_action = "submit_render" if scene.reviewed else "review_code"
        elif kind == "review_code":
            inputs = self._scene_inputs(action.scene_id, manifest=manifest, inherit=True)
            counters = agent.scene_review_counters.get(
                action.scene_id,
                AgentReviewState(
                    review_round=inputs.scene.review_round,
                    review_signature=inputs.scene.review_signature,
                    identical_review_count=inputs.scene.identical_review_count,
                ),
            )
            inputs.scene.review_round = counters.review_round
            inputs.scene.review_signature = counters.review_signature
            inputs.scene.identical_review_count = counters.identical_review_count
            from kd1_anime.agents.review_policy import review_budget

            config = manifest.run_config.resolve()
            maximum = review_budget(
                inputs.scene.plan,
                inputs.scene.technical_spec,
                global_max_rounds=config.MAX_REVIEW_ROUNDS,
                low_risk_max_rounds=config.MAX_LOW_RISK_REVIEW_ROUNDS,
                generation_mode=config.GENERATION_MODE,
            ).max_rounds
            eligible = any(
                candidate.verification in {"smoke", "rendered"}
                and candidate.code_sha256 != inputs.scene.code_sha256
                and candidate.inherited_elements_sha256
                == sha256_text(inputs.scene.inherited_elements_code)
                for candidate in inputs.scene.candidates
            )
            if eligible and counters.review_round == maximum - 1:
                output = self.scene_pipeline.rollback(inputs)
            else:
                output = self.scene_pipeline.review(inputs, identity=action.work_identity)
            review_counters = dict(agent.scene_review_counters)
            review_counters[action.scene_id] = AgentReviewState(
                review_round=output.review_round,
                review_signature=output.review_signature,
                identical_review_count=output.identical_review_count,
                feedback=actions.sanitize_record(output.feedback, self._secrets()),
            )
            candidates = dict(agent.pending_candidates)
            reviewed = dict(agent.reviewed_identities)
            if output.accepted:
                scene = manifest.scenes[action.scene_id].model_copy(deep=True)
                scene.review_round = output.review_round
                scene.review_signature = output.review_signature
                scene.identical_review_count = output.identical_review_count
                scene.reviewed, scene.phase, scene.failed = True, "reviewed", False
                if output.source is not None:
                    scene.code_file, scene.code_sha256 = output.source.path, output.source.sha256
                    scene.class_name = output.class_name or scene.class_name
                if output.inspection is not None:
                    scene.static_verification = output.inspection.verification
                    scene.exported_elements_code = output.inspection.exported_elements_code
                    scene.exported_elements = list(output.inspection.exported_elements)
                self._record_candidate(scene, "validated")
                patches = (self._scene_patch(scene),)
                reviewed[action.scene_id] = WorkIdentity(scene.revision + 1, scene.input_sha256)
                candidates.pop(action.scene_id, None)
                next_action = "submit_render"
                if scene.code_sha256 != inputs.scene.code_sha256 and (
                    agent.smoke_requested
                    or (not manifest.dry_run and config.LOCAL_SMOKE_RENDER_ENABLED)
                    or (manifest.dry_run and scene.unknown_animation_detected)
                ):
                    next_action = "smoke_scene"
            elif output.repair_route == "approval" and output.proposal is not None:
                ref = self._store_output(output.proposal)
                candidates[action.scene_id] = AgentPendingCandidate(
                    proposal=ref, diff_sha256=output.proposal.diff_sha256
                )
                next_action = "inspect_candidate"
            elif output.repair_route == "coder":
                next_action = "propose_code_candidate"
            elif output.repair_route == "plan":
                next_action = "revise_plan"
            else:
                reason = "Code review retry or stagnation budget exhausted"
                agent = agent.model_copy(update={"stop_reason": reason})
                updates.update(state="ERROR", status="failed", run_phase="failed", error=reason)
                next_action = "review_code"
            agent = agent.model_copy(
                update={
                    "reviewed_identities": reviewed,
                    "pending_candidates": candidates,
                    "scene_review_counters": review_counters,
                }
            )
            updates["artifact_refs"] = {f"code_review:{action.scene_id}": output.report.path}
        elif kind == "publish_continuity":
            if action.publication != self._publication_binding(manifest):
                raise ValueError("Continuity publication inputs differ from the declared action")
            output = build_continuity_publication(
                tuple(
                    self._scene_inputs(identifier, manifest=manifest)
                    for identifier in sorted(manifest.scenes)
                )
            )
            blocking = [item for item in output.diagnostics if item.severity == "major"]
            if blocking:
                raise ValueError(blocking[0].summary)
            from kd1_anime.domain.dependencies import DependencyGraph

            global_hash = digest(
                {
                    "lesson": manifest.lesson_spec.model_dump(mode="json"),
                    "teaching": manifest.teaching_graph.model_dump(mode="json"),
                    "bible": manifest.continuity_bible.model_dump(mode="json")
                    if manifest.continuity_bible
                    else None,
                    "renderer": manifest.render_profile.digest(),
                    "rag": manifest.rag_profile.model_dump(mode="json"),
                }
            )
            updates.update(
                element_manifest=output.element_manifest,
                state_ledger=output.state_ledger,
                dependency_graph=DependencyGraph(
                    bindings=list(output.dependency_bindings),
                    global_inputs={identifier: global_hash for identifier in manifest.scenes},
                ),
                state="EVALUATING" if manifest.dry_run else "DISPATCHING",
            )
            agent = agent.model_copy(update={"continuity_published": True})
            next_action = "finish" if manifest.dry_run else "submit_render"
        elif kind == "submit_render":
            scene = manifest.scenes[action.scene_id]
            if self._pending_submission(action.scene_id, manifest=manifest) is not None:
                raise RuntimeError("Outstanding submission must be reconciled before resubmission")
            attempt = SubmissionAttempt(
                backend=manifest.backend,
                scene_revision=scene.revision,
                code_sha256=scene.code_sha256,
            )
            self._patch(
                RunPatch(
                    agent=self._agent_update(
                        submissions=(
                            *self.manifest.agent.submissions,
                            AgentSubmission(
                                scene_id=action.scene_id,
                                attempt=attempt,
                                intent_id=current_invocation.get().intent_id,
                            ),
                        )
                    )
                )
            )
            output = self._render().submit(
                self._scene_inputs(action.scene_id, manifest=manifest), context, attempt
            )
            scene = scene.model_copy(deep=True)
            scene.submission = attempt.model_copy(
                update={"status": "submitted", "job_id": output.job.job_id}
            )
            scene.submission_attempts = [*scene.submission_attempts, scene.submission]
            scene.slurm_job = output.job
            scene.phase = "monitoring"
            patches = (self._scene_patch(scene),)
            updates["state"] = "MONITORING"
            next_action = "poll_render"
        elif kind == "poll_render":
            output = self._render().poll(
                self._scene_inputs(action.scene_id, manifest=manifest), context
            )
            scene = manifest.scenes[action.scene_id].model_copy(deep=True)
            scene.slurm_job = output.job
            next_action = "poll_render"
            if output.status == "rendered":
                scene.artifact = output.artifact
                scene.rendered = True
                scene.phase = "visual_accepted"
                scene.visual_status = "skipped"
                scene.execution_verification = ExecutionVerification(
                    status="passed",
                    scope="formal_video",
                    code_sha256=scene.code_sha256,
                    artifact_sha256=output.artifact.video_sha256,
                    checked_at=utc_now().isoformat(),
                )
                self._record_candidate(scene, "rendered")
                next_action = "merge_video"
                if manifest.visual_eval_profile.enabled:
                    scene.visual_status = "pending"
                    scene.phase = "visual_evaluating"
                    next_action = "evaluate_visual"
            elif output.status in {"failed", "unknown", "cancelled"}:
                scene.phase = "failed"
                scene.failed = True
                if output.status == "failed" and manifest.auto_fix and not manifest.direct_render:
                    updates.update(state="FIXING")
                    next_action = "propose_render_fix"
                else:
                    reason = "Render failed or requires safe submission/cancellation reconciliation"
                    agent = agent.model_copy(update={"stop_reason": reason})
                    updates.update(state="ERROR", status="failed", run_phase="failed", error=reason)
            patches = (self._scene_patch(scene),)
        elif kind == "propose_render_fix":
            inputs = self._scene_inputs(action.scene_id, manifest=manifest, inherit=True)
            output = RenderRepairPipeline(
                RenderingService(self._render().backend, manifest.backend),
                fixer_factory=self.dependencies.fixer_factory,
                request_context=self._request_context,
                control=self.control,
            ).repair(inputs, context)
            if (
                output.identity != action.work_identity
                or output.original_code_sha256 != inputs.scene.code_sha256
            ):
                raise ValueError("Render repair result identity mismatch")
            scene = output.scene
            if output.repair_route in {"review", "coder"}:
                scene.reviewed = scene.failed = False
                scene.phase = "coded"
                scene.slurm_job = scene.submission = scene.artifact = None
                scene.rendered = False
                scene.rewrite_feedback = output.feedback
                next_action = (
                    "inspect_candidate"
                    if output.repair_route == "review"
                    else "propose_code_candidate"
                )
                updates.update(state="CODING")
                counters = dict(agent.scene_review_counters)
                counters[action.scene_id] = AgentReviewState()
                agent = agent.model_copy(update={"scene_review_counters": counters})
                agent = agent.model_copy(update={"continuity_published": False})
            elif output.repair_route == "plan":
                proposals = dict(agent.plan_repair_proposals)
                if output.proposed_plan is not None:
                    proposals[action.scene_id] = output.report
                agent = agent.model_copy(update={"plan_repair_proposals": proposals})
                agent = agent.model_copy(
                    update={"plan_repair_feedback": output.feedback, "continuity_published": False}
                )
                updates.update(state="CODING")
                next_action = "revise_plan"
            elif output.repair_route == "infrastructure":
                scene.slurm_job = scene.submission = None
                scene.phase, scene.failed = "reviewed", False
                updates.update(state="DISPATCHING")
                next_action = "submit_render"
                reviewed = dict(agent.reviewed_identities)
                reviewed[action.scene_id] = WorkIdentity(scene.revision + 1, scene.input_sha256)
                agent = agent.model_copy(update={"reviewed_identities": reviewed})
            else:
                reason = output.feedback or "Finite render repair budget exhausted"
                if manifest.run_config.resolve().ALLOW_PARTIAL_OUTPUT:
                    scene.give_up = True
                    updates.update(state="MONITORING")
                    next_action = "merge_video"
                else:
                    agent = agent.model_copy(update={"stop_reason": reason})
                    updates.update(state="ERROR", status="failed", run_phase="failed", error=reason)
                    next_action = "propose_render_fix"
            patches = (self._scene_patch(scene),)
            updates["artifact_refs"] = {f"render_repair:{action.scene_id}": output.report.path}
        elif kind == "cancel_render":
            output = self._render().cancel(
                self._scene_inputs(action.scene_id, manifest=manifest), context
            )
            scene = manifest.scenes[action.scene_id].model_copy(deep=True)
            scene.slurm_job = output.job
            scene.failed, scene.phase = True, "failed"
            patches = (self._scene_patch(scene),)
            reason = (
                "Render cancelled"
                if output.status == "cancelled"
                else "Cancellation failed; resubmission forbidden"
            )
            updates.update(state="ERROR", status="failed", run_phase="failed", error=reason)
            agent = agent.model_copy(update={"stop_reason": reason})
            next_action = "cancel_render"
        elif kind in {"evaluate_visual", "evaluate_boundary"}:
            if kind == "evaluate_visual":
                scene = manifest.scenes[action.scene_id]
                if (
                    action.video.sha256 != scene.artifact.video_sha256
                    or action.snapshot_sha256 != digest(scene_binding(scene))
                ):
                    raise ValueError("Visual action input identity mismatch")
                output = self.visual.scene(
                    self._scene_inputs(action.scene_id, manifest=manifest), context
                )
                scenes, target_id = (output.scene,), action.scene_id
            else:
                if action.snapshot_sha256 != self._visual_input_digest(manifest):
                    raise ValueError("Boundary action input identity mismatch")
                output = self.visual.boundary(self._final_inputs(), context)
                scenes, target_id = output.scenes, output.scene_id
                updates["state_ledger"] = output.state_ledger
            route = output.repair_route
            receipts = dict(agent.reuse_receipts)
            for scene in scenes:
                if scene.artifact is not None and scene.artifact.origin == "rendered":
                    receipts.pop(scene.plan.scene_id, None)
            if (
                kind == "evaluate_visual"
                and output.restored
                and output.scene.artifact.origin == "reused"
            ):
                if output.reuse_provenance is None:
                    raise ValueError("Restored reused scene has no exact typed provenance")
                receipts[action.scene_id] = output.reuse_provenance
            agent = agent.model_copy(update={"reuse_receipts": receipts})
            next_action = "merge_video"
            if route in {"planner", "continuity"}:
                agent = agent.model_copy(
                    update={"plan_repair_feedback": output.feedback, "continuity_published": False}
                )
                updates.update(state="CODING")
                next_action = "revise_plan"
            elif route == "coder":
                pending = dict(agent.visual_repairs)
                pending[target_id] = output.report
                names = dict(agent.scene_next_actions)
                names[target_id] = "propose_visual_fix"
                agent = agent.model_copy(
                    update={"visual_repairs": pending, "scene_next_actions": names}
                )
                updates.update(state="VISUAL_EVALUATING")
                next_action = "propose_visual_fix"
            elif route == "rebuild":
                for scene in scenes:
                    if scene.plan.scene_id == target_id:
                        scene.reviewed, scene.phase = False, "coded"
                names = dict(agent.scene_next_actions)
                names[target_id] = "review_code"
                agent = agent.model_copy(
                    update={"scene_next_actions": names, "continuity_published": False}
                )
                updates.update(state="CODING")
                next_action = "review_code"
            else:
                for scene in scenes:
                    scene.phase = "visual_accepted"
                if kind == "evaluate_boundary":
                    agent = agent.model_copy(
                        update={"boundary_input_sha256": self._visual_input_digest(manifest)}
                    )
                    updates.update(state="MERGING", run_phase="merging")
                else:
                    updates.update(state="MONITORING")
            patches = tuple(self._scene_patch(scene) for scene in scenes)
            if output.report is not None:
                updates["artifact_refs"] = {f"{kind}:{target_id}": output.report.path}
        elif kind == "propose_visual_fix":
            inputs = self._scene_inputs(action.scene_id, manifest=manifest)
            if agent.visual_repairs.get(action.scene_id) != action.report:
                raise ValueError("Visual repair report identity mismatch")
            boundary = (
                action.report if action.report.path != inputs.scene.visual_report_file else None
            )
            output = self.visual.propose_fix(inputs, context, boundary_report=boundary)
            if (
                output.identity != action.work_identity
                or output.base_source.sha256 != inputs.scene.code_sha256
            ):
                raise ValueError("Visual repair candidate identity mismatch")
            scene = inputs.scene
            if output.accepted:
                scene.code_file, scene.code_sha256, scene.class_name = (
                    output.candidate_source.path,
                    output.candidate_source.sha256,
                    output.class_name,
                )
                scene.reviewed = scene.rendered = scene.failed = False
                scene.slurm_job = scene.submission = scene.artifact = None
                scene.visual_status, scene.phase = "pending", "coded"
                scene.static_verification = type(scene.static_verification)()
                scene.execution_verification = type(scene.execution_verification)()
                scene.visual_verification = type(scene.visual_verification)()
                updates.update(state="CODING")
                next_action = "inspect_candidate"
                agent = agent.model_copy(update={"continuity_published": False})
            else:
                restored = self.visual.reject_proposal(inputs, context)
                scene = restored.scene
                receipts = dict(agent.reuse_receipts)
                if scene.artifact.origin == "reused":
                    if restored.reuse_provenance is None:
                        raise ValueError("Visual rollback requires exact typed reuse provenance")
                    receipts[action.scene_id] = restored.reuse_provenance
                else:
                    receipts.pop(action.scene_id, None)
                agent = agent.model_copy(update={"reuse_receipts": receipts})
                scene.phase = "visual_accepted"
                next_action = "merge_video"
                updates.update(state="MONITORING")
                if restored.repair_route == "rebuild":
                    scene.reviewed, scene.phase = False, "coded"
                    next_action = "review_code"
                    updates.update(state="CODING")
                    agent = agent.model_copy(update={"continuity_published": False})
            pending = dict(agent.visual_repairs)
            pending.pop(action.scene_id, None)
            agent = agent.model_copy(update={"visual_repairs": pending})
            patches = (self._scene_patch(scene),)
        elif kind in {"merge_video", "reconcile_merge", "evaluate_run"}:
            if kind == "merge_video":
                output = self.finalization.merge(self._final_inputs(), context)
            elif kind == "reconcile_merge":
                output = self.finalization.reconcile_merge(self._final_inputs(), context)
            else:
                output = self.finalization.evaluate(
                    self._final_inputs(), self._final_output(), context
                )
            updates.update(
                final_video=output.output_path,
                final_video_sha256=output.video.sha256,
                artifact_refs={**manifest.artifact_refs, "final_video": output.video.path},
                state="EVALUATING",
                run_phase="evaluating",
            )
            next_action = "finish" if kind == "evaluate_run" else "evaluate_run"
            if kind == "evaluate_run":
                updates["eval_round"] = output.eval_round
            agent = agent.model_copy(
                update={
                    "final_visual_report": output.visual_report,
                    "final_evaluation_report": output.evaluation_report,
                    "final_visual_status": output.visual_status,
                }
            )
            if kind == "evaluate_run" and output.scenes_to_improve:
                changed = []
                names = dict(agent.scene_next_actions)
                for identifier in output.scenes_to_improve:
                    scene = manifest.scenes[identifier].model_copy(deep=True)
                    scene.reviewed = scene.rendered = False
                    scene.phase = "coded"
                    scene.slurm_job = scene.submission = scene.artifact = None
                    scene.rewrite_feedback = "Improve the committed run evaluation: " + str(
                        output.evaluation_report
                    )
                    changed.append(self._scene_patch(scene))
                    names[identifier] = "propose_code_candidate"
                patches = tuple(changed)
                agent = agent.model_copy(
                    update={"scene_next_actions": names, "continuity_published": False}
                )
                updates.update(state="CODING")
                next_action = "propose_code_candidate"
        elif kind == "finish":
            output = CompletionOutput(
                output=self._final_output().video if manifest.final_video else None
            )
            updates.update(
                state="DONE",
                run_phase="finished",
                status="completed" if manifest.final_video else "dry_run_complete",
            )
            next_action = "finish"
        else:
            raise RuntimeError("Unsupported domain action")
        updates["scenes"] = patches
        if getattr(output, "rag_receipts", ()):
            updates["rag_receipts"] = {
                f"{kind}:{action.scene_id}:{index}": receipt
                for index, receipt in enumerate(output.rag_receipts)
            }
        if action.scene_id is not None:
            scene_actions = dict(agent.scene_next_actions)
            scene_actions[action.scene_id] = next_action
            versions = dict(agent.scene_action_versions)
            if kind != "retry_scene":
                versions[action.scene_id] = versions.get(action.scene_id, 0) + 1
            agent = agent.model_copy(
                update={"scene_next_actions": scene_actions, "scene_action_versions": versions}
            )
        else:
            agent = agent.model_copy(update={"next_action": next_action})
        updates["agent"] = agent
        return ToolResult(
            StageResult(action.work_identity, output),
            RunPatch(**updates),
            waiting_for_job=kind == "poll_render" and output.status == "pending",
        )

    def _final_output(self):
        manifest = self.manifest
        return FinalizationOutput(
            video=ObjectRef(
                path=manifest.artifact_refs["final_video"],
                sha256=manifest.final_video_sha256,
                kind="video",
            ),
            output_path=manifest.final_video,
            visual_report=manifest.agent.final_visual_report,
            evaluation_report=manifest.agent.final_evaluation_report,
            visual_status=manifest.agent.final_visual_status,
            eval_round=manifest.eval_round,
        )

    def _completed_video(self):
        if not self.manifest.final_video:
            return None
        result = self._final_output()
        self.finalization._validate_video(self._final_inputs(), result)
        private = restore_run_path(self.root, result.video.path)
        public = Path(result.output_path)
        if (
            not private.is_file()
            or sha256_file(private) != result.video.sha256
            or public.is_symlink()
            or not public.is_file()
            or sha256_file(public) != result.video.sha256
        ):
            raise ValueError("Completed video identity mismatch")
        return public

    @_serialized
    def _snapshot(self):
        manifest = self.manifest
        agent = manifest.agent
        planning = self._planning_output()
        if agent.plan_sha256 is not None and self._plan_digest(planning) != agent.plan_sha256:
            raise ValueError("Committed plan differs from its approved semantic identity")
        if any(
            scene.input_sha256 != self._scene_digest(scene, planning=planning)
            for scene in manifest.scenes.values()
        ):
            raise ValueError("Committed scene inputs differ from their work identity")
        scenes = tuple(
            ScenePolicyState(
                scene_id=scene_id,
                phase=scene.phase,
                give_up=scene.give_up,
                work_identity=WorkIdentity(scene.revision, scene.input_sha256),
                dependencies=tuple(
                    PolicyDependency(
                        scene_id=item.source_scene_id,
                        work_identity=WorkIdentity(
                            manifest.scenes[item.source_scene_id].revision,
                            manifest.scenes[item.source_scene_id].input_sha256,
                        ),
                    )
                    for item in self._dependencies(scene, manifest)
                    if item.source_scene_id in manifest.scenes
                ),
                code_sha256=scene.code_sha256 or None,
                approved_code_sha256=scene.code_sha256 if scene.reviewed else None,
                approved_code_identity=agent.reviewed_identities.get(scene_id),
                diff_sha256=agent.pending_candidates[scene_id].diff_sha256
                if scene_id in agent.pending_candidates
                else None,
                external_job_id=scene.slurm_job.job_id if scene.slurm_job else None,
                static_verified=scene.static_verification.status == "passed"
                and scene.static_verification.code_sha256 == scene.code_sha256,
                source_ref=ObjectRef(path=scene.code_file, sha256=scene.code_sha256, kind="code")
                if scene.code_file
                else None,
            )
            for scene_id, scene in sorted(manifest.scenes.items())
        )
        scenes += tuple(
            ScenePolicyState(
                scene_id=item.scene_id,
                phase="pending",
                work_identity=agent.outline_identities[item.scene_id],
            )
            for item in manifest.outlines
            if item.scene_id not in manifest.scenes
        )
        return PolicySnapshot(
            run_id=manifest.run_id,
            revision=manifest.revision,
            state=manifest.state,
            work_identity=WorkIdentity(agent.run_revision, agent.run_input_sha256),
            scenes=scenes,
            plan_sha256=agent.plan_sha256,
            approved_plan_sha256=agent.approved_plan_sha256,
            approved_plan_identity=agent.approved_plan_identity,
            waiting_for_approval=agent.pending_approval is not None,
            plan_only=agent.operation == "plan",
            dry_run=manifest.dry_run,
            allow_partial=manifest.run_config.resolve().ALLOW_PARTIAL_OUTPUT,
            recovery_epoch=agent.recovery_epoch,
        )

    def _budget(self):
        budget = self.manifest.request_budget
        config = self.manifest.run_config.resolve()
        polling_steps = (
            config.MONITOR_QUEUE_TIMEOUT
            + config.MONITOR_RUN_TIMEOUT
            + config.MONITOR_UNKNOWN_TIMEOUT
            + config.MONITOR_ARTIFACT_GRACE
        ) // config.MONITOR_POLL_INTERVAL + 1
        return PolicyBudget(
            steps_remaining=max(
                0,
                (100 + polling_steps) * max(1, len(self.manifest.outlines))
                - self.manifest.agent.step_count,
            ),
            requests_remaining=max(0, budget.run_limit - len(budget.reservations)),
        )

    def _dependencies(self, scene, manifest):
        return DependencyService.dependencies(
            scene.plan,
            tuple(item.plan for item in manifest.scenes.values()),
            inherited_code=scene.inherited_elements_code,
            handoff_source=scene.technical_spec.handoff_in.source_scene_id
            if scene.technical_spec and scene.technical_spec.handoff_in
            else None,
        )

    @staticmethod
    def _record_candidate(scene, verification):
        from kd1_anime.domain.candidates import CodeCandidate
        from kd1_anime.services.policies import candidate_rank

        candidate = CodeCandidate(
            code_file=scene.code_file,
            code_sha256=scene.code_sha256,
            class_name=scene.class_name,
            verification=verification,
            inherited_elements_sha256=sha256_text(scene.inherited_elements_code),
            exported_elements_code=scene.exported_elements_code,
            exported_elements=[item.model_copy(deep=True) for item in scene.exported_elements],
            artifact=scene.artifact if verification == "rendered" else None,
        )
        records = {item.code_sha256: item for item in scene.candidates}
        previous = records.get(candidate.code_sha256)
        if previous is None or candidate_rank(candidate) >= candidate_rank(previous):
            records[candidate.code_sha256] = candidate
        scene.candidates = sorted(records.values(), key=candidate_rank, reverse=True)[:3]

    def _load_candidate(self, pending):
        from kd1_anime.pipeline.scene import CandidateProposal

        path = restore_run_path(self.root, pending.proposal.path)
        if path.is_symlink() or not path.is_file() or sha256_file(path) != pending.proposal.sha256:
            raise ValueError("Pending candidate report identity mismatch")
        proposal = CandidateProposal.model_validate_json(path.read_text())
        if (
            proposal.diff_sha256 != pending.diff_sha256
            or proposal.bound_digest() != pending.diff_sha256
        ):
            raise ValueError("Pending candidate diff identity mismatch")
        return proposal

    @staticmethod
    def _diff_binding(proposal):
        return actions.DiffBinding(
            work_identity=proposal.identity,
            base_source=proposal.base_source,
            candidate_source=proposal.candidate_source,
            inspection_report=proposal.inspection_report,
            review_report=proposal.review_report,
            context_sha256=proposal.context_sha256,
        )

    @staticmethod
    def _publication_binding(manifest):
        return actions.ContinuityBinding(
            plan_identity=WorkIdentity(
                manifest.agent.run_revision, manifest.agent.run_input_sha256
            ),
            scenes=tuple(
                actions.ContinuitySource(
                    scene_id=identifier,
                    work_identity=WorkIdentity(scene.revision, scene.input_sha256),
                    source=ObjectRef(path=scene.code_file, sha256=scene.code_sha256, kind="code"),
                )
                for identifier, scene in sorted(manifest.scenes.items())
            ),
        )

    def _make_action(self, name, snapshot, scene_id=None):
        manifest = self.manifest
        identity = (
            next(item.work_identity for item in snapshot.scenes if item.scene_id == scene_id)
            if scene_id
            else snapshot.work_identity
        )
        values = dict(
            kind=name,
            recovery_epoch=manifest.agent.recovery_epoch,
            run_id=manifest.run_id,
            resource_id=f"run:{manifest.run_id}" + (f"/scene:{scene_id}" if scene_id else ""),
            input_sha256=identity.input_sha256,
            work_identity=identity,
        )
        if scene_id is not None:
            values["scene_id"] = scene_id
        if name == "inspect_candidate":
            scene = manifest.scenes[scene_id]
            values["candidate"] = ObjectRef(
                path=scene.code_file, sha256=scene.code_sha256, kind="code"
            )
            pending = manifest.agent.pending_candidates.get(scene_id)
            if pending is not None:
                proposal = self._load_candidate(pending)
                binding = self._diff_binding(proposal)
                values.update(
                    candidate=proposal.candidate_source,
                    proposal=binding,
                    input_sha256=binding.digest(),
                )
        elif name == "publish_continuity":
            binding = self._publication_binding(manifest)
            values.update(publication=binding, input_sha256=binding.digest())
        elif name == "submit_render":
            scene = manifest.scenes[scene_id]
            values.update(
                backend=manifest.backend,
                source=ObjectRef(path=scene.code_file, sha256=scene.code_sha256, kind="code"),
                class_name=scene.class_name,
            )
        elif name == "evaluate_visual":
            scene = manifest.scenes[scene_id]
            values.update(
                video=ObjectRef(
                    path=scene.artifact.video_path, sha256=scene.artifact.video_sha256, kind="video"
                ),
                snapshot_sha256=digest(scene_binding(scene)),
            )
        elif name == "evaluate_boundary":
            values["snapshot_sha256"] = self._visual_input_digest(manifest)
        elif name == "propose_visual_fix":
            values["report"] = manifest.agent.visual_repairs[scene_id]
        elif name == "resume_run":
            values.update(
                scenes={item.scene_id: item.work_identity for item in snapshot.scenes},
                rag_profile_sha256=digest(self._current_rag_profile().model_dump(mode="json")),
            )
        elif name == "reconcile_submission":
            attempt = self._pending_submission(scene_id)
            if attempt is None:
                raise ValueError("No outstanding submission to reconcile")
            values["attempt_token"] = attempt.token
        elif name == "reconcile_render":
            scene = manifest.scenes[scene_id]
            values.update(
                job_id=scene.slurm_job.job_id if scene.slurm_job else None,
                artifact=ObjectRef(
                    path=scene.artifact.video_path, sha256=scene.artifact.video_sha256, kind="video"
                )
                if scene.artifact
                else None,
            )
        elif name == "reuse_scene":
            values["base_run_id"] = manifest.base_run_id
        elif name in {"poll_render", "cancel_render"}:
            values["job_id"] = manifest.scenes[scene_id].slurm_job.job_id
        elif name == "propose_render_fix":
            scene = manifest.scenes[scene_id]
            values["diagnostic"] = Diagnostic(
                code="render.failed",
                stage="render",
                category="execution",
                summary=scene.slurm_job.failure_reason or "Render failed",
                code_sha256=scene.code_sha256,
                scene_id=scene_id,
                revision=scene.revision,
            )
        elif name in {"merge_video", "reconcile_merge"}:
            values["artifacts"] = tuple(
                ObjectRef(
                    path=scene.artifact.video_path, sha256=scene.artifact.video_sha256, kind="video"
                )
                for _, scene in sorted(manifest.scenes.items())
                if scene.rendered
            )
        elif name == "revise_plan":
            values["proposals"] = tuple(
                manifest.agent.plan_repair_proposals[key]
                for key in sorted(manifest.agent.plan_repair_proposals)
            )
            values["feedback"] = manifest.agent.plan_repair_feedback or "\n".join(
                counters.feedback
                for identifier, counters in manifest.agent.scene_review_counters.items()
                if manifest.agent.scene_next_actions.get(identifier) == "revise_plan"
            )
        elif name == "evaluate_run":
            values["video"] = self._final_output().video
        elif name == "finish":
            values["output"] = self._final_output().video if manifest.final_video else None
        return actions.ACTION_ADAPTER.validate_python(values)

    @_serialized
    def _ready_actions(self, snapshot):
        manifest = self.manifest
        agent = manifest.agent
        if manifest.state == "DETAILING":
            return tuple(
                self._make_action("detail_scene", snapshot, item.scene_id)
                for item in manifest.outlines
                if item.scene_id not in manifest.scenes
            )
        if manifest.state == "CODING":
            if any(name == "revise_plan" for name in agent.scene_next_actions.values()):
                return (self._make_action("revise_plan", snapshot),)
            for identifier, name in sorted(agent.scene_next_actions.items()):
                if (
                    name == "smoke_scene"
                    and manifest.scenes[identifier].local_smoke_status != "passed"
                ):
                    return (self._make_action(name, snapshot, identifier),)
            if manifest.scenes and all(scene.reviewed for scene in manifest.scenes.values()):
                return (self._make_action("publish_continuity", snapshot),)
            technical = [
                identifier
                for identifier, scene in sorted(manifest.scenes.items())
                if scene.technical_status != "passed"
            ]
            if technical:
                return (self._make_action("propose_technical_spec", snapshot, technical[0]),)
            ready = []
            for identifier, scene in sorted(manifest.scenes.items()):
                if scene.reviewed:
                    continue
                dependencies = self._dependencies(scene, manifest)
                if any(
                    item.source_scene_id not in manifest.scenes
                    or not manifest.scenes[item.source_scene_id].reviewed
                    for item in dependencies
                ):
                    continue
                name = agent.scene_next_actions[identifier]
                ready.append(self._make_action(name, snapshot, identifier))
            coding = tuple(action for action in ready if action.kind == "propose_code_candidate")
            if coding:
                return coding
            return tuple(ready[:1])
        if manifest.state == "FIXING":
            for identifier, name in sorted(agent.scene_next_actions.items()):
                if name == "propose_render_fix":
                    return (self._make_action(name, snapshot, identifier),)
        if manifest.state == "VISUAL_EVALUATING":
            for identifier, scene in sorted(manifest.scenes.items()):
                if scene.give_up:
                    continue
                if agent.scene_next_actions.get(identifier) == "propose_visual_fix":
                    return (self._make_action("propose_visual_fix", snapshot, identifier),)
                if scene.visual_status not in {"passed", "warning", "unknown", "skipped"}:
                    return (self._make_action("evaluate_visual", snapshot, identifier),)
            return (self._make_action("evaluate_boundary", snapshot),)
        if manifest.state in {"DISPATCHING", "MONITORING"}:
            for identifier, scene in sorted(manifest.scenes.items()):
                if not scene.rendered and not scene.give_up:
                    if scene.slurm_job is None and self._pending_submission(identifier) is not None:
                        return (self._make_action("reconcile_submission", snapshot, identifier),)
                    if (
                        scene.slurm_job is None
                        and manifest.incremental
                        and agent.incremental_checked.get(identifier) != scene.input_sha256
                    ):
                        return (self._make_action("reuse_scene", snapshot, identifier),)
                    name = "poll_render" if scene.slurm_job is not None else "submit_render"
                    return (self._make_action(name, snapshot, identifier),)
        return (self._make_action(agent.next_action, snapshot),)

    def _next_action(self, snapshot):
        ready = self._ready_actions(snapshot)
        return ready[0] if ready else None

    @staticmethod
    def _visual_input_digest(manifest):
        return digest(
            {
                "scenes": [scene_binding(scene) for _, scene in sorted(manifest.scenes.items())],
                "profile": manifest.visual_eval_profile.digest(),
                "ledger": manifest.state_ledger.digest(),
            }
        )

    @_serialized
    def _advance_barriers(self):
        manifest = self.manifest
        agent = manifest.agent
        if manifest.state == "DETAILING" and len(manifest.scenes) == len(manifest.outlines):
            self._patch(
                RunPatch(
                    state="PLAN_REVIEWING", agent=self._agent_update(next_action="review_plan")
                )
            )
        elif (
            manifest.state == "CODING"
            and manifest.scenes
            and all(scene.reviewed for scene in manifest.scenes.values())
        ):
            if agent.next_action != "publish_continuity":
                self._patch(RunPatch(agent=self._agent_update(next_action="publish_continuity")))
        elif manifest.state == "CODING":
            patches = []
            for identifier, scene in sorted(manifest.scenes.items()):
                if (
                    agent.scene_next_actions.get(identifier) != "propose_code_candidate"
                    or scene.technical_status != "passed"
                ):
                    continue
                if any(
                    not manifest.scenes[item.source_scene_id].reviewed
                    for item in self._dependencies(scene, manifest)
                ):
                    continue
                prepared = self._scene_inputs(identifier, manifest=manifest, inherit=True)
                if prepared.scene.inherited_elements_code != scene.inherited_elements_code:
                    patches.append(self._scene_patch(prepared.scene, manifest=manifest))
            if patches:
                self._patch(RunPatch(scenes=tuple(patches)))
        elif manifest.state in {"DISPATCHING", "MONITORING"}:
            if all(scene.rendered or scene.give_up for scene in manifest.scenes.values()):
                self._patch(
                    RunPatch(
                        state="VISUAL_EVALUATING"
                        if manifest.visual_eval_profile.enabled
                        else "MERGING",
                        run_phase="evaluating"
                        if manifest.visual_eval_profile.enabled
                        else "merging",
                        agent=self._agent_update(
                            next_action="evaluate_boundary"
                            if manifest.visual_eval_profile.enabled
                            else "merge_video"
                        ),
                    )
                )
            else:
                scene = next(
                    scene
                    for _, scene in sorted(manifest.scenes.items())
                    if not scene.rendered and not scene.give_up
                )
                target = "MONITORING" if scene.slurm_job else "DISPATCHING"
                if target != manifest.state:
                    self._patch(RunPatch(state=target))

    def _configure_tool(self, action):
        if action.kind == "review_plan":
            continuation = self.manifest.agent.plan_review_complete
            self.registry.configure(
                "review_plan",
                requires_approval=continuation,
                uses_model=not continuation and self.manifest.agent.planning_review,
            )
        elif action.kind == "propose_render_fix":
            job = self.manifest.scenes[action.scene_id].slurm_job
            self.registry.configure(
                "propose_render_fix",
                uses_model=job.status
                not in {
                    "PREEMPTED",
                    "NODE_FAIL",
                    "BOOT_FAIL",
                    "DEADLINE",
                    "REVOKED",
                    "SPECIAL_EXIT",
                },
            )
        elif action.kind == "propose_code_candidate":
            self.registry.configure(
                "propose_code_candidate",
                uses_model=self.manifest.run_config.resolve().CODEGEN_MODE != "ir",
            )

    def _run_action(self, action):
        self._configure_tool(action)
        loop = AgentLoop(
            policy=Policy(),
            executor=self.executor,
            snapshot=self._snapshot,
            session=lambda: self.sessions.load(self.manifest.agent.session_id),
            budget=self._budget,
            next_action=lambda snapshot: action,
            after_step=(lambda: self.dependencies.after_commit(self.manifest))
            if self.dependencies.after_commit
            else None,
        )
        with self.manifest.run_config.activate():
            return loop.step(cancelled=self.control.cancelled.is_set)

    @_serialized
    def _stop(self, reason):
        reason = actions.sanitize_record(reason, self._secrets())
        self._patch(
            RunPatch(
                agent=self._agent_update(stop_reason=reason),
                error=reason,
                status="failed",
                state="ERROR",
                run_phase="failed",
            )
        )
        self.events.reconcile(self.manifest, lock_held=True)

    def _drive(self, *, approve_plan, interactive):
        while True:
            if self.manifest.state == "DONE":
                return self._completed_video()
            if self.manifest.state == "ERROR":
                return None
            agent = self.manifest.agent
            if agent.pending_approval is not None:
                action = agent.pending_approval
                self._configure_tool(action)
                decision = Policy().evaluate(
                    self._snapshot(),
                    action,
                    self.registry.get(action.kind).policy,
                    self.sessions.load(agent.session_id),
                    self._budget(),
                )
                if decision.effect != "allow":
                    approved = (
                        approve_plan and action.kind == "review_plan" and action.scene_id is None
                    )
                    if not approved and interactive and sys.stdin.isatty():
                        description = (
                            "complete plan" if action.kind == "review_plan" else "exact code diff"
                        )
                        approved = input(
                            f"Approve this {description} ({action.input_sha256})? [y/N] "
                        ).strip().lower() in {"y", "yes"}
                    if not approved:
                        return None
                    self.sessions.issue_grant(
                        agent.session_id,
                        UserCommand(session_id=agent.session_id, kind="approve", action=action),
                    )
                self._patch(RunPatch(agent=self._agent_update(pending_approval=None)))
            self._advance_barriers()
            if self.manifest.state == "ERROR":
                return None
            snapshot = self._snapshot()
            ready = self._ready_actions(snapshot)
            if not ready:
                self._stop("No dependency-ready scene action")
                return None
            workers = max(1, self.manifest.run_config.resolve().LLM_PARALLEL_WORKERS)
            batch = (
                ready[:workers]
                if ready[0].kind in {"detail_scene", "propose_code_candidate"}
                else ready[:1]
            )
            if len(batch) > 1:
                with ThreadPoolExecutor(
                    max_workers=workers, thread_name_prefix="agent-scene"
                ) as pool:
                    results = tuple(pool.map(self._run_action, batch))
            else:
                results = (self._run_action(batch[0]),)
            for result in results:
                if result.status == "CONTINUE":
                    continue
                if result.status == "DONE":
                    return self._completed_video()
                if result.reason == "approval_required" and result.action is not None:
                    self._patch(RunPatch(agent=self._agent_update(pending_approval=result.action)))
                    self.events.append(
                        name="approval_requested",
                        durability="committed",
                        manifest_revision=self.manifest.revision,
                        scene_revision=result.action.work_identity.revision
                        if result.action.scene_id is not None
                        else None,
                        payload=EventPayload(
                            action=result.action.kind,
                            scene_id=result.action.scene_id,
                            metadata={
                                "session_id": self.manifest.agent.session_id,
                                "resource_id": result.action.resource_id,
                                "input_sha256": result.action.input_sha256,
                                "work_identity": {
                                    "revision": result.action.work_identity.revision,
                                    "input_sha256": result.action.work_identity.input_sha256,
                                },
                                **(
                                    {"proposal": result.action.proposal.model_dump(mode="json")}
                                    if isinstance(result.action, actions.InspectCandidate)
                                    and result.action.proposal is not None
                                    else {}
                                ),
                            },
                        ),
                        lock_held=True,
                    )
                    break
                if result.reason == "waiting_for_job":
                    self.control.wait(self.manifest.run_config.resolve().MONITOR_POLL_INTERVAL)
                    continue
                if self.manifest.state != "ERROR":
                    self._stop("Agent stopped safely: " + result.reason)
                return None
