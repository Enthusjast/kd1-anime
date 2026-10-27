"""Manifest-owned single-scene runtime; no legacy coordinator or mutable Context."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import Event

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
    PolicySnapshot,
    ScenePolicyState,
    ToolPolicySpec,
)
from kd1_anime.agent_runtime.session import SessionStore, UserCommand
from kd1_anime.agent_runtime.state import AgentRunState, AgentStageAttempt, AgentSubmission
from kd1_anime.agent_runtime.tools import (
    ExecutionAdapters,
    ToolDefinition,
    ToolExecutor,
    ToolRegistry,
    ToolResult,
    action_signature,
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
    FinalizationInputs,
    FinalizationOutput,
    PlanningOutput,
    RenderingOutput,
    SceneInputs,
    SceneOutput,
    TechnicalOutput,
)
from kd1_anime.pipeline.finalization import FinalizationPipeline
from kd1_anime.pipeline.planning import PlanningPipeline
from kd1_anime.pipeline.render import RenderPipeline
from kd1_anime.pipeline.scene import CodeReviewOutput, InspectionOutput, ScenePipeline
from kd1_anime.prompt_identity import capture_generation_identity, verify_generation_identity
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
from kd1_anime.services.rendering import RenderingService
from kd1_anime.services.results import StageResult
from kd1_anime.verification import ExecutionVerification


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


class CompletionOutput(RuntimeRecord):
    output: ObjectRef | None = None


class AgentRuntime:
    """One lock, one manifest authority, and one declared tool per durable stage."""

    def __init__(self, *, dependencies: AgentDependencies | None = None):
        self.dependencies = dependencies or AgentDependencies()
        self.control = RunControl(Event())
        self._callback = None

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
                    or settings.RAG_ENABLED
                    or (
                        (settings.ENABLE_VISUAL_EVAL or settings.ENABLE_AUTO_EVAL)
                        and operation != "plan"
                    )
                    or settings.SCENE_CODEGEN_MODE != "llm"
                    or settings.CODEGEN_MODE != "python"
                ):
                    self._stop(
                        "P4 single-scene runtime does not support RAG, smoke, automated visual evaluation/repair or compiler/IR generation"
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
        self.planning = PlanningPipeline(
            self.dependencies.planner_factory,
            self.dependencies.plan_reviewer_factory,
            self.dependencies.continuity_reviewer_factory,
            self.dependencies.technical_factory,
            config=manifest.run_config,
            request_context=self._request_context,
            control=self.control,
        )
        self.scene_pipeline = ScenePipeline(
            self.dependencies.coder_factory,
            self.dependencies.reviewer_factory,
            request_context=self._request_context,
            control=self.control,
        )
        self.finalization = FinalizationPipeline(self.dependencies.merger_factory())
        self.registry = ToolRegistry()
        declarations = (
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
            (actions.InspectCandidate, InspectionOutput, "read", "none", False),
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
                        requires_approval=name == "propose_technical_spec",
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

    def _persist_intent(self, intent):
        previous = next(
            (item for item in self.manifest.agent.intents if item.intent_id == intent.intent_id),
            None,
        )
        if previous == intent or (previous is not None and previous.status == "completed"):
            return
        self._patch(
            RunPatch(
                agent=self._agent_update(
                    intents=self._replace(self.manifest.agent.intents, intent, "intent_id")
                )
            )
        )

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

    def _commit_result(self, intent, invocation, result):
        updates = {name: getattr(result.patch, name) for name in result.patch.model_fields_set}
        agent = updates.get("agent") or self.manifest.agent
        # Stage outputs were built before request/transport updates; always merge
        # their semantic fields with the latest append-only transport authority.
        semantic = {
            name: getattr(agent, name)
            for name in (
                "next_action",
                "plan_sha256",
                "run_revision",
                "run_input_sha256",
                "approved_plan_sha256",
                "approved_plan_identity",
                "reviewed_identities",
                "pending_approval",
                "stop_reason",
            )
        }
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
        return manifest.revision

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
            scene = self.manifest.scenes.get(scene_id)
            receipt = RequestReservation(
                scene_id=scene_id,
                scene_revision=scene.revision if scene else 0,
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
            tokens = data.get("usage") or {}
            invocation = next(
                (
                    item
                    for item in reversed(self.manifest.agent.invocations)
                    if item.status == "running"
                ),
                None,
            )
            usage = UsageReceipt(
                request_id=data["request_id"],
                invocation_id=invocation.invocation_id if invocation else None,
                usage_source="provider_reported" if tokens else "unknown",
                **tokens,
            )
            self._patch(
                RunPatch(agent=self._agent_update(usage=(*self.manifest.agent.usage, usage)))
            )

        return RequestContext(reserve, emit)

    def _reserve_stage(self, action):
        if not self.registry.get(action.kind).policy.uses_model:
            return
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
            RunPatch(agent=self._agent_update(attempts=(*self.manifest.agent.attempts, attempt)))
        )

    def _store_output(self, value):
        text = value.model_dump_json()
        path = ObjectStore(self.root).report(text)
        return ObjectRef(path=path, sha256=sha256_text(text), kind="report")

    def _planning_output(self):
        manifest = self.manifest
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
        )

    def _scene_inputs(self):
        manifest = self.manifest
        return SceneInputs(
            manifest.run_id,
            self.root,
            manifest.run_config,
            manifest.render_profile,
            manifest.execution_spec,
            manifest.scenes[1].model_copy(deep=True),
            tuple(scene.plan.model_copy(deep=True) for _, scene in sorted(manifest.scenes.items())),
            manifest.lesson_spec.model_copy(deep=True),
            manifest.teaching_graph.model_copy(deep=True),
            manifest.continuity_bible.model_copy(deep=True) if manifest.continuity_bible else None,
        )

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
                "planning": planning.model_dump(mode="json", exclude={"diagnostics"}),
                "renderer": self.manifest.render_profile.model_dump(mode="json"),
                "generation": self.manifest.agent.generation_identity.model_dump(mode="json"),
            }
        )

    def _scene_digest(self, scene, *, planning=None):
        return digest(
            {
                "plan": scene.plan.model_dump(mode="json"),
                "technical": scene.technical_spec.model_dump(mode="json")
                if scene.technical_spec
                else None,
                "source": scene.code_sha256,
                "inherited": scene.inherited_elements_code,
                "planning": (planning or self._planning_output()).model_dump(
                    mode="json", exclude={"diagnostics"}
                ),
                "config": self.manifest.run_config.model_dump(mode="json"),
                "renderer": self.manifest.render_profile.model_dump(mode="json"),
                "generation": self.manifest.agent.generation_identity.model_dump(mode="json"),
            }
        )

    def _scene_patch(self, replacement):
        scene = self.manifest.scenes[replacement.plan.scene_id]
        replacement.input_sha256 = self._scene_digest(replacement)
        return ScenePatch(
            scene_id=scene.plan.scene_id,
            identity=WorkIdentity(scene.revision, scene.input_sha256),
            replacement=replacement,
        )

    def _execute(self, action, context):
        # Reservations are transport-only; they never change a scene identity.
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

    def _execute_stage(self, action, context):
        manifest = self.manifest
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
            if len(output.plans) != 1:
                from kd1_anime.domain.diagnostics import Diagnostic
                from kd1_anime.pipeline.planning import UnsupportedPlanningError

                diagnostic = Diagnostic(
                    code="planning.unsupported_multiple_scenes",
                    stage="plan",
                    category="contract",
                    summary="P4 supports exactly one scene; multi-scene execution is not supported",
                )
                raise UnsupportedPlanningError(diagnostic.summary, diagnostics=(diagnostic,))
            scene = StoredSceneState(plan=output.plans[0], phase="detailed", plan_ready=True)
            scene.input_sha256 = self._scene_digest(scene, planning=output)
            patches = (ScenePatch(scene_id=1, identity=WorkIdentity(0, ""), replacement=scene),)
            updates.update(
                outlines=list(output.outlines),
                lesson_spec=output.lesson_spec,
                teaching_graph=output.teaching_graph,
                continuity_bible=output.continuity_bible,
                state="PLAN_REVIEWING",
                plan_review_status="pending",
            )
            next_action = "review_plan"
        elif kind == "review_plan":
            output = self.planning.review(
                self._planning_output(),
                manifest.render_profile.renderer,
                use_models=agent.planning_review,
            )
            scene = manifest.scenes[1].model_copy(deep=True)
            scene.plan = output.plans[0]
            scene.plan_reviewed = True
            scene.phase = "plan_reviewed"
            scene.input_sha256 = self._scene_digest(scene, planning=output)
            patches = (
                ScenePatch(
                    scene_id=1,
                    identity=WorkIdentity(
                        manifest.scenes[1].revision, manifest.scenes[1].input_sha256
                    ),
                    replacement=scene,
                ),
            )
            plan_hash = self._plan_digest(output)
            agent = agent.model_copy(
                update={
                    "plan_sha256": plan_hash,
                    "run_revision": agent.run_revision + 1,
                    "run_input_sha256": plan_hash,
                }
            )
            updates.update(
                outlines=list(output.outlines),
                lesson_spec=output.lesson_spec,
                teaching_graph=output.teaching_graph,
                continuity_bible=output.continuity_bible,
                plan_review_status="passed",
                continuity_review_status="passed",
                state="EVALUATING" if agent.operation == "plan" else "CODING",
                run_phase="planning" if agent.operation == "plan" else "executing",
            )
            next_action = "finish" if agent.operation == "plan" else "propose_technical_spec"
        elif kind == "propose_technical_spec":
            output = self.planning.technical(self._scene_inputs())
            scene = manifest.scenes[1].model_copy(deep=True)
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
            agent = agent.model_copy(
                update={
                    "approved_plan_sha256": agent.plan_sha256,
                    "approved_plan_identity": WorkIdentity(
                        agent.run_revision, agent.run_input_sha256
                    ),
                    "pending_approval": None,
                }
            )
            next_action = "propose_code_candidate"
        elif kind == "propose_code_candidate":
            output = self.scene_pipeline.generate(self._scene_inputs())
            scene = manifest.scenes[1].model_copy(deep=True)
            scene.code_file, scene.code_sha256, scene.class_name = (
                output.source.path,
                output.source.sha256,
                output.class_name,
            )
            scene.phase = "coded"
            patches = (self._scene_patch(scene),)
            next_action = "inspect_candidate"
        elif kind == "inspect_candidate":
            output = self.scene_pipeline.inspect(self._scene_inputs())
            scene = manifest.scenes[1].model_copy(deep=True)
            scene.code_file, scene.code_sha256, scene.class_name = (
                output.source.path,
                output.source.sha256,
                output.class_name,
            )
            scene.static_verification = output.verification
            scene.exported_elements_code = output.exported_elements_code
            scene.exported_elements = list(output.exported_elements)
            if not output.accepted:
                scene.phase = "failed"
                scene.failed = True
            patches = (self._scene_patch(scene),)
            if not output.accepted:
                updates.update(
                    state="ERROR",
                    status="failed",
                    run_phase="failed",
                    error="Candidate inspection failed; P4 does not perform LLM code fixes",
                )
                agent = agent.model_copy(update={"stop_reason": updates["error"]})
            updates["artifact_refs"] = {**manifest.artifact_refs, "inspection": output.report.path}
            next_action = "review_code"
        elif kind == "review_code":
            output = self.scene_pipeline.review(self._scene_inputs())
            scene = manifest.scenes[1].model_copy(deep=True)
            scene.review_round += 1
            scene.reviewed = output.accepted
            scene.phase = "reviewed" if output.accepted else "failed"
            scene.failed = not output.accepted
            patch = self._scene_patch(scene)
            patches = (patch,)
            next_action = "finish" if manifest.dry_run else "submit_render"
            if output.accepted:
                reviewed = dict(agent.reviewed_identities)
                reviewed[1] = WorkIdentity(scene.revision + 1, scene.input_sha256)
                agent = agent.model_copy(update={"reviewed_identities": reviewed})
                updates["state"] = "EVALUATING" if manifest.dry_run else "DISPATCHING"
            else:
                reason = "Code review rejected candidate; P4 does not perform LLM code fixes"
                agent = agent.model_copy(update={"stop_reason": reason})
                updates.update(state="ERROR", status="failed", run_phase="failed", error=reason)
            updates["artifact_refs"] = {**manifest.artifact_refs, "code_review": output.report.path}
        elif kind == "submit_render":
            scene = manifest.scenes[1]
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
                            AgentSubmission(scene_id=1, attempt=attempt),
                        )
                    )
                )
            )
            output = self._render().submit(self._scene_inputs(), context, attempt)
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
            output = self._render().poll(self._scene_inputs(), context)
            scene = manifest.scenes[1].model_copy(deep=True)
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
                updates.update(state="MERGING", run_phase="merging")
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
            raise RuntimeError("Unsupported P4 domain action")
        updates["scenes"] = patches
        updates["agent"] = agent.model_copy(update={"next_action": next_action})
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
                code_sha256=scene.code_sha256 or None,
                approved_code_sha256=scene.code_sha256 if scene.reviewed else None,
                approved_code_identity=agent.reviewed_identities.get(scene_id),
                external_job_id=scene.slurm_job.job_id if scene.slurm_job else None,
                static_verified=scene.static_verification.status == "passed"
                and scene.static_verification.code_sha256 == scene.code_sha256,
            )
            for scene_id, scene in sorted(manifest.scenes.items())
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
            steps_remaining=max(0, 100 + polling_steps - self.manifest.agent.step_count),
            requests_remaining=max(0, budget.run_limit - len(budget.reservations)),
        )

    def _next_action(self, snapshot):
        manifest = self.manifest
        name = manifest.agent.next_action
        scene_action = name in {
            "propose_technical_spec",
            "propose_code_candidate",
            "inspect_candidate",
            "review_code",
            "submit_render",
            "poll_render",
        }
        identity = snapshot.scenes[0].work_identity if scene_action else snapshot.work_identity
        values = {
            "kind": name,
            "run_id": manifest.run_id,
            "resource_id": f"run:{manifest.run_id}" + ("/scene:1" if scene_action else ""),
            "input_sha256": identity.input_sha256,
            "work_identity": identity,
        }
        if scene_action:
            values["scene_id"] = 1
        if name == "inspect_candidate":
            values["candidate"] = ObjectRef(
                path=manifest.scenes[1].code_file,
                sha256=manifest.scenes[1].code_sha256,
                kind="code",
            )
        elif name == "submit_render":
            scene = manifest.scenes[1]
            values.update(
                backend=manifest.backend,
                source=ObjectRef(path=scene.code_file, sha256=scene.code_sha256, kind="code"),
                class_name=scene.class_name,
            )
        elif name == "poll_render":
            values["job_id"] = manifest.scenes[1].slurm_job.job_id
        elif name == "merge_video":
            artifact = manifest.scenes[1].artifact
            values["artifacts"] = (
                ObjectRef(path=artifact.video_path, sha256=artifact.video_sha256, kind="video"),
            )
        elif name == "evaluate_run":
            values["video"] = self._final_output().video
        elif name == "finish":
            values["output"] = self._final_output().video if manifest.final_video else None
        return actions.ACTION_ADAPTER.validate_python(values)

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
            agent = self.manifest.agent
            if agent.pending_approval is not None:
                approved = approve_plan
                if not approved and interactive and sys.stdin.isatty():
                    approved = input(
                        "Approve this complete scene plan? [y/N] "
                    ).strip().lower() in {"y", "yes"}
                if not approved:
                    return None
                self.sessions.issue_grant(
                    agent.session_id,
                    UserCommand(
                        session_id=agent.session_id, kind="approve", action=agent.pending_approval
                    ),
                )
                self._patch(RunPatch(agent=self._agent_update(pending_approval=None)))
            loop = AgentLoop(
                policy=Policy(),
                executor=self.executor,
                snapshot=self._snapshot,
                session=lambda: self.sessions.load(self.manifest.agent.session_id),
                budget=self._budget,
                next_action=self._next_action,
                after_step=(lambda: self.dependencies.after_commit(self.manifest))
                if self.dependencies.after_commit
                else None,
            )
            result = loop.run()
            if result.status == "DONE":
                return self._completed_video()
            if result.reason == "approval_required" and result.action is not None:
                self._patch(RunPatch(agent=self._agent_update(pending_approval=result.action)))
                self.events.append(
                    name="approval_requested",
                    durability="committed",
                    manifest_revision=self.manifest.revision,
                    payload=EventPayload(action=result.action.kind),
                    lock_held=True,
                )
                continue
            if result.reason == "waiting_for_job":
                self.control.wait(self.manifest.run_config.resolve().MONITOR_POLL_INTERVAL)
                continue
            if self.manifest.state != "ERROR":
                self._stop("Agent stopped safely: " + result.reason)
            return None
