"""Manifest-owned bounded multiscene runtime, without a shared mutable Context."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
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
from kd1_anime.commits import CommitCommand, RunCommitter, RunPatch, ScenePatch
from kd1_anime.config import resolve_runtime_path, settings
from kd1_anime.control import RunControl
from kd1_anime.domain.dependencies import WorkIdentity, digest
from kd1_anime.domain.execution import SubmissionAttempt
from kd1_anime.domain.planning import (
    ContinuityBible,
    LessonSpec,
    SceneOutline,
    ScenePlan,
    TeachingGraph,
)
from kd1_anime.domain.runtime import ExecutionSpec, RunConfig
from kd1_anime.domain.state import RunPaths
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
from kd1_anime.pipeline.planning import PlanningPipeline, PlanningStageError
from kd1_anime.pipeline.render import RenderPipeline
from kd1_anime.pipeline.scene import CodeReviewOutput, InspectionOutput, ScenePipeline
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
    restore_run_path,
    sha256_text,
    write_manifest,
)
from kd1_anime.services.continuity_state import (
    ContinuityPublication,
    build_continuity_publication,
    prepare_inherited_inputs,
)
from kd1_anime.services.dependencies import DependencyService
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

    def resume(
        self,
        run_id: str,
        callback=None,
        interactive=False,
        approve_plan=False,
        retry_scene_id=None,
        **unsupported,
    ):
        repository = RunRepository(settings.WORKSPACE_DIR)
        before = repository.load_for_agent_resume(run_id)
        verify_generation_identity(before.agent.generation_identity)
        root = repository.run_root(run_id)
        with lock_run(root), before.run_config.activate():
            manifest = repository.load_for_agent_resume(run_id)
            verify_generation_identity(manifest.agent.generation_identity)
            self._bind(root, manifest, callback)
            self._notify("run_started", {"run_id": run_id})
            if retry_scene_id is not None or unsupported:
                self._stop("P4 does not support scene retry or incremental resume")
                return None
            if manifest.state == "DONE":
                return self._completed_video()
            if manifest.agent.stop_reason or manifest.state == "ERROR":
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
    ):
        prompt = actions.sanitize_record(prompt, self._secrets())
        config = RunConfig.capture()
        with config.activate():
            profile = RenderProfile.current()
            execution = ExecutionSpec.capture(profile)
            paths = RunPaths.create(output_path)
            run_id, root = paths.run_id, paths.root
            identity = capture_generation_identity()
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
                write_manifest(root / "manifest.json", manifest)
                self._bind(root, manifest, callback)
                self._notify("run_started", {"run_id": run_id})
                if (
                    smoke
                    or (
                        (settings.ENABLE_VISUAL_EVAL or settings.ENABLE_AUTO_EVAL)
                        and operation != "plan"
                    )
                    or settings.CODEGEN_MODE != "python"
                ):
                    self._stop(
                        "P5 does not support smoke, automated visual evaluation/repair or standalone IR generation"
                    )
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
        self.finalization = FinalizationPipeline(self.dependencies.merger_factory())
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
            (
                actions.SubmitRender,
                RenderingOutput,
                "cluster" if manifest.backend == "slurm" else "exec",
                "external",
                False,
            ),
            (actions.PollRender, RenderingOutput, "read", "none", False),
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
        if invocation.action == "submit_render" and invocation.external_job_id is not None:
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
                if item.attempt.scene_revision == invocation.work_identity.revision
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
            ):
                values = dict(getattr(latest, name))
                incoming = getattr(proposed, name)
                for scene_id in touched:
                    if scene_id in incoming:
                        values[scene_id] = incoming[scene_id]
                    else:
                        values.pop(scene_id, None)
                semantic[name] = values
            if proposed.stop_reason:
                semantic["stop_reason"] = proposed.stop_reason
        if "artifact_refs" in updates:
            updates["artifact_refs"] = {**self.manifest.artifact_refs, **updates["artifact_refs"]}
        if "rag_receipts" in updates:
            updates["rag_receipts"] = {**self.manifest.rag_receipts, **updates["rag_receipts"]}
        complete_intent = intent.model_copy(update={"status": "completed"})
        report = self._store_output(result.stage.value)
        complete_invocation = invocation.model_copy(
            update={"status": "completed", "finished_at": utc_now(), "result_ref": report}
        )
        latest = self.manifest.agent
        updates["agent"] = latest.model_copy(
            update={
                **semantic,
                "step_count": latest.step_count + 1,
                "intents": self._replace(latest.intents, complete_intent, "intent_id"),
                "invocations": self._replace(
                    latest.invocations, complete_invocation, "invocation_id"
                ),
                "committed_invocations": (*latest.committed_invocations, invocation.invocation_id),
            }
        )
        manifest = self._patch(RunPatch(**updates), phase=intent.action.kind)
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

    def _scene_inputs(self, scene_id, *, manifest=None, inherit=False):
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
        )
        if inherit:
            prior = tuple(
                self._scene_inputs(identifier, manifest=manifest)
                for identifier in sorted(manifest.scenes)
                if identifier < scene_id
            )
            return prepare_inherited_inputs(inputs, prior)
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
                "plan": scene.plan.model_dump(mode="json"),
                "technical": scene.technical_spec.model_dump(mode="json")
                if scene.technical_spec
                else None,
                "source": scene.code_sha256,
                "inherited": scene.inherited_elements_code,
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
        if kind == "propose_plan":
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
                    StoredSceneState(plan=output.plan, phase="detailed", plan_ready=True)
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
            output = self.planning.revise(
                self._planning_output(),
                action.feedback,
                manifest.render_profile.renderer,
                user_prompt=manifest.user_prompt,
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
            next_action = "review_code"
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
                            AgentSubmission(scene_id=action.scene_id, attempt=attempt),
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
            elif output.status in {"failed", "unknown", "cancelled"}:
                reason = "Render did not complete safely; P4 does not resubmit or run render fixes"
                agent = agent.model_copy(update={"stop_reason": reason})
                updates.update(state="ERROR", status="failed", run_phase="failed", error=reason)
                scene.phase = "failed"
                scene.failed = True
            patches = (self._scene_patch(scene),)
        elif kind in {"merge_video", "evaluate_run"}:
            output = (
                self.finalization.merge(self._final_inputs(), context)
                if kind == "merge_video"
                else self.finalization.evaluate(self._final_inputs(), self._final_output(), context)
            )
            updates.update(
                final_video=output.output_path,
                final_video_sha256=output.video.sha256,
                artifact_refs={**manifest.artifact_refs, "final_video": output.video.path},
                state="EVALUATING",
                run_phase="evaluating",
            )
            next_action = "evaluate_run" if kind == "merge_video" else "finish"
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
        )

    def _completed_video(self):
        if not self.manifest.final_video:
            return None
        result = self._final_output()
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
        elif name == "poll_render":
            values["job_id"] = manifest.scenes[scene_id].slurm_job.job_id
        elif name == "merge_video":
            values["artifacts"] = tuple(
                ObjectRef(
                    path=scene.artifact.video_path, sha256=scene.artifact.video_sha256, kind="video"
                )
                for _, scene in sorted(manifest.scenes.items())
            )
        elif name == "revise_plan":
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
            if manifest.scenes and all(scene.reviewed for scene in manifest.scenes.values()):
                return (self._make_action("publish_continuity", snapshot),)
            if any(name == "revise_plan" for name in agent.scene_next_actions.values()):
                return (self._make_action("revise_plan", snapshot),)
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
        if manifest.state in {"DISPATCHING", "MONITORING"}:
            for identifier, scene in sorted(manifest.scenes.items()):
                if not scene.rendered:
                    name = "poll_render" if scene.slurm_job is not None else "submit_render"
                    return (self._make_action(name, snapshot, identifier),)
        return (self._make_action(agent.next_action, snapshot),)

    def _next_action(self, snapshot):
        ready = self._ready_actions(snapshot)
        return ready[0] if ready else None

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
            if all(scene.rendered for scene in manifest.scenes.values()):
                self._patch(
                    RunPatch(
                        state="MERGING",
                        run_phase="merging",
                        agent=self._agent_update(next_action="merge_video"),
                    )
                )
            else:
                scene = next(
                    scene for _, scene in sorted(manifest.scenes.items()) if not scene.rendered
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
