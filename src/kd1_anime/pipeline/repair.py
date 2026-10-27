"""Bounded render repair proposals; ReviewCode and state publication stay separate.

Every installed proposal invalidates execution/visual receipts and asks the
coordinator to stop the render batch and rebuild dependent continuity first.
The worker never cancels an unconfirmed job, submits work, or changes a plan.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from typing import Literal

from pydantic import Field

from kd1_anime.agent_runtime.contracts import (
    ObjectRef,
    RuntimeRecord,
    Sha256,
    sanitize_record,
    utc_now,
)
from kd1_anime.agent_runtime.tools import ToolCancelled, ToolContext
from kd1_anime.agents.auto_fixer import AutoFixerAgent
from kd1_anime.agents.failure_corpus import FailureCase, FailureCaseStore
from kd1_anime.agents.failure_router import classify_failure
from kd1_anime.agents.progress import ProgressSnapshot, classify_progress
from kd1_anime.agents.render_error_parser import extract_render_error
from kd1_anime.agents.safe_fallback import (
    build_safe_fallback_plan,
    fallback_reason_summary,
    is_high_confidence_geometry_conflict,
)
from kd1_anime.agents.scene_ir import build_scene_program_from_contract, compile_scene_program
from kd1_anime.agents.scene_templates import build_safe_scene_code
from kd1_anime.candidate_acceptor import CandidateAcceptor
from kd1_anime.cluster.status import TERMINAL_STATES
from kd1_anime.config import settings
from kd1_anime.control import OperationDeadlineExceeded, RunCancelled
from kd1_anime.domain.candidates import CodeCandidate
from kd1_anime.domain.dependencies import WorkIdentity, digest
from kd1_anime.domain.planning import ContinuityBible, ScenePlan
from kd1_anime.objects import ObjectStore
from kd1_anime.pipeline.contracts import SceneInputs
from kd1_anime.rendering import sha256_file
from kd1_anime.request_budget import RequestBudgetExceeded
from kd1_anime.run_store import StoredSceneState, restore_render_job, restore_run_path, sha256_text
from kd1_anime.services.policies import candidate_rank
from kd1_anime.verification import ExecutionVerification, StaticVerification, VisualVerification

RepairRoute = Literal["review", "coder", "plan", "exhausted", "infrastructure"]


class RenderRepairOutput(RuntimeRecord):
    identity: WorkIdentity
    original_code_sha256: Sha256
    error_fingerprint: str
    technical_spec_sha256: str
    context_sha256: Sha256
    source: ObjectRef
    report: ObjectRef
    scene: StoredSceneState
    fix_attempts: int = Field(ge=0)
    infra_retries: int = Field(ge=0)
    identical_error_count: int = Field(ge=0)
    stagnant_repair_count: int = Field(ge=0)
    repair_route: RepairRoute
    strategy: str
    feedback: str = ""
    proposed_plan: ScenePlan | None = None
    continuity_rebuild_required: bool = False


class RenderRepairPipeline:
    def __init__(
        self,
        render_service=None,
        *,
        fixer_factory: Callable,
        request_context: Callable,
        control,
        failure_cases=None,
    ):
        self.render_service = render_service
        self.fixer_factory = fixer_factory
        self.request_context = request_context
        self.control = control
        self.failure_cases = failure_cases
        self.acceptor = CandidateAcceptor()

    def _checkpoint(self, context):
        context.checkpoint()
        if self.control is not None:
            checkpoint = getattr(self.control, "checkpoint", None) or self.control.check
            checkpoint()

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

    @staticmethod
    def _read_source(inputs, path, expected_hash):
        expected = f"scenes/revisions/scene_{inputs.scene.plan.scene_id}/{expected_hash}.py"
        if path != expected:
            raise ValueError("Repair requires an immutable ObjectStore source")
        source = restore_run_path(inputs.root, path)
        if source.is_symlink() or not source.is_file() or sha256_file(source) != expected_hash:
            raise ValueError("Repair source identity mismatch")
        return source.read_text(encoding="utf-8")

    @staticmethod
    def _bind(inputs):
        scene, job = inputs.scene, inputs.scene.slurm_job
        if inputs.direct_render:
            raise ValueError("Direct render does not invoke code repair agents")
        if job is None or job.status not in TERMINAL_STATES:
            raise ValueError("Repair requires a confirmed terminal render job; reconcile first")
        if job.cancelled or job.status in {"CANCELLED", "COMPLETED"}:
            raise ValueError("Cancelled or successful render jobs cannot enter repair")
        if (
            job.scene_id != scene.plan.scene_id
            or job.code_sha256 != scene.code_sha256
            or job.scene_class_name != scene.class_name
            or job.render_profile != inputs.render_profile
            or job.resource_profile != scene.resource_profile
            or job.execution_sha256 != inputs.execution_spec.digest()
        ):
            raise ValueError("Repair job identity differs from the immutable scene inputs")
        if any(
            attempt.status in {"pending", "unknown"}
            for attempt in (
                *scene.submission_attempts,
                *((scene.submission,) if scene.submission else ()),
            )
        ):
            raise ValueError("Uncertain submission forbids repair and automatic resubmission")
        attempt = scene.submission
        dispatch = inputs.submission_identity
        if attempt is not None and (
            dispatch is None
            or attempt.status != "submitted"
            or attempt.job_id != job.job_id
            or attempt.code_sha256 != scene.code_sha256
            or attempt.scene_revision != dispatch.revision
            or dispatch.input_sha256 != scene.input_sha256
            or attempt.backend != job.backend
        ):
            raise ValueError("Repair submission identity mismatch")
        technical_hash = (
            sha256_text(scene.technical_spec.model_dump_json()) if scene.technical_spec else ""
        )
        if scene.technical_contract_stale or technical_hash != scene.technical_spec_sha256:
            raise ValueError("Repair TechnicalSpec identity mismatch")

    def _inspect(self, inputs, context, code, *, class_name=None, allow_repair=True):
        self._checkpoint(context)
        inspection = self.acceptor.inspect_candidate(
            code,
            inputs.scene.plan,
            technical_spec=inputs.scene.technical_spec,
            renderer=inputs.render_profile.renderer,
            manim_version=inputs.render_profile.manim_version,
            expected_class_name=class_name,
            allow_repair=allow_repair,
        )
        # Acceptor repairs can change bytes. Bind a fresh read-only inspection to
        # the exact final bytes, never to a pre-normalization candidate.
        if inspection.accepted and allow_repair:
            self._checkpoint(context)
            inspection = self.acceptor.inspect_candidate(
                inspection.code,
                inputs.scene.plan,
                technical_spec=inputs.scene.technical_spec,
                renderer=inputs.render_profile.renderer,
                manim_version=inputs.render_profile.manim_version,
                expected_class_name=inspection.class_name,
                allow_repair=False,
            )
        return inspection

    def repair(self, inputs: SceneInputs, context: ToolContext) -> RenderRepairOutput:
        """Propose one bounded attempt; all changed code routes to ReviewCode.

        ``scene`` is a worker-owned copy, not a committed replacement. ``plan``
        means RevisePlan must validate/approve ``proposed_plan`` before any code
        generation. Neither that route nor a failed proposal changes input state.
        """
        with inputs.config.activate():
            self._checkpoint(context)
            self._bind(inputs)
            # Freeze nested mutable models before passing anything to agents.
            scene = inputs.scene.model_copy(update={"snapshot": None}, deep=True)
            inputs = replace(inputs, scene=scene)
            original = self._read_source(inputs, scene.code_file, scene.code_sha256)
            original_hash = scene.code_sha256
            identity = WorkIdentity(scene.revision, scene.input_sha256)
            secrets = self._secrets()
            context_hash = digest(
                {
                    "run_id": inputs.run_id,
                    "identity": {
                        "revision": identity.revision,
                        "input_sha256": identity.input_sha256,
                    },
                    "plan": scene.plan.model_dump(mode="json"),
                    "technical_spec_sha256": scene.technical_spec_sha256,
                    "inherited": sha256_text(scene.inherited_elements_code),
                    "elements": inputs.element_manifest.model_dump(mode="json"),
                    "lesson": inputs.lesson_spec.model_dump(mode="json"),
                    "teaching": inputs.teaching_graph.model_dump(mode="json"),
                    "render": inputs.render_profile.digest(),
                    "execution": inputs.execution_spec.digest(),
                }
            )
            job = restore_render_job(scene.slurm_job, inputs.root)
            if self.render_service is not None:
                error_log = self.render_service.get_error_log(job=job) or ""
            else:
                error_log = (
                    job.log_err.read_text(encoding="utf-8", errors="replace")
                    if job.log_err.is_file()
                    else ""
                )
            self._checkpoint(context)
            error_log = sanitize_record(error_log, secrets)
            evidence = extract_render_error(
                error_log,
                source_code=original,
                renderer=inputs.render_profile.renderer,
                secrets=secrets,
            )
            fp = evidence.fingerprint
            reports = []
            corpus = self.failure_cases

            def finish(route, strategy, feedback="", *, inspection=None, proposed_plan=None):
                self._checkpoint(context)
                source = ObjectRef(path=scene.code_file, sha256=scene.code_sha256, kind="code")
                feedback = sanitize_record(feedback, secrets)
                payload = {
                    "kind": "render_repair",
                    "run_id": inputs.run_id,
                    "scene_id": scene.plan.scene_id,
                    "job_id": job.job_id,
                    "identity": {
                        "revision": identity.revision,
                        "input_sha256": identity.input_sha256,
                    },
                    "original_code_sha256": original_hash,
                    "candidate_code_sha256": source.sha256,
                    "technical_spec_sha256": scene.technical_spec_sha256,
                    "context_sha256": context_hash,
                    "error_fingerprint": fp,
                    "error_evidence": evidence.to_dict(),
                    "repair_route": route,
                    "strategy": strategy,
                    "feedback": feedback,
                    "fix_attempts": scene.fix_attempts,
                    "infra_retries": scene.infra_retries,
                    "identical_error_count": scene.identical_error_count,
                    "stagnant_repair_count": scene.stagnant_repair_count,
                    "inspections": reports,
                    "proposed_plan": proposed_plan.model_dump(mode="json")
                    if proposed_plan
                    else None,
                }
                text = json.dumps(
                    sanitize_record(payload, secrets), ensure_ascii=False, sort_keys=True
                )
                report = ObjectRef(
                    path=ObjectStore(inputs.root).report(text),
                    sha256=sha256_text(text),
                    kind="report",
                )
                return RenderRepairOutput(
                    identity=identity,
                    original_code_sha256=original_hash,
                    error_fingerprint=fp,
                    technical_spec_sha256=scene.technical_spec_sha256,
                    context_sha256=context_hash,
                    source=source,
                    report=report,
                    scene=scene,
                    fix_attempts=scene.fix_attempts,
                    infra_retries=scene.infra_retries,
                    identical_error_count=scene.identical_error_count,
                    stagnant_repair_count=scene.stagnant_repair_count,
                    repair_route=route,
                    strategy=strategy,
                    feedback=feedback,
                    proposed_plan=proposed_plan,
                    continuity_rebuild_required=inspection is not None or proposed_plan is not None,
                )

            def inspect(code, strategy, *, class_name=None, allow_repair=True):
                result = self._inspect(
                    inputs, context, code, class_name=class_name, allow_repair=allow_repair
                )
                reports.append(
                    {
                        "strategy": strategy,
                        "accepted": result.accepted,
                        "code_sha256": result.code_sha256,
                        "diagnostics": [
                            item.model_dump(mode="json") for item in result.diagnostics
                        ],
                    }
                )
                return result

            def install(result, strategy):
                nonlocal corpus
                if not result.accepted or result.code_sha256 == original_hash:
                    return None
                path = ObjectStore(inputs.root).source(scene.plan.scene_id, result.code)
                scene.code_file, scene.code_sha256, scene.class_name = (
                    path,
                    result.code_sha256,
                    result.class_name,
                )
                scene.reviewed = False
                scene.review_round = scene.identical_review_count = 0
                scene.review_signature = scene.rewrite_feedback = ""
                scene.slurm_job = scene.submission = scene.artifact = None
                scene.rendered = scene.failed = scene.give_up = False
                scene.phase = "coded"
                scene.failure_reason = scene.failure_category = ""
                scene.infra_retries = 0
                scene.local_smoke_status = "pending"
                scene.static_verification = StaticVerification(
                    status="passed",
                    code_sha256=result.code_sha256,
                    technical_spec_sha256=scene.technical_spec_sha256,
                    checked_at=utc_now().isoformat(),
                )
                scene.execution_verification = ExecutionVerification()
                scene.visual_verification = VisualVerification()
                scene.exported_elements_code = ""
                scene.exported_elements = []
                scene.unknown_animation_detected = bool(result.unknown_animations)
                scene.unknown_animation_details = list(result.unknown_animations[-30:])
                scene.visual_status = "pending"
                scene.visual_score = None
                scene.visual_report_file = scene.visual_report_sha256 = ""
                scene.visual_artifact_sha256 = scene.visual_feedback = ""
                scene.visual_best_candidate = None
                scene.last_repair_code_sha256, scene.last_repair_error_fp = result.code_sha256, fp
                if strategy in {"scene_ir", "safe_template", "best_rollback"}:
                    scene.stagnant_repair_count = 0
                scene.candidates = sorted(
                    [item for item in scene.candidates if item.code_sha256 != result.code_sha256]
                    + [
                        CodeCandidate(
                            code_file=path,
                            code_sha256=result.code_sha256,
                            class_name=result.class_name,
                            verification="validated",
                            inherited_elements_sha256=sha256_text(scene.inherited_elements_code),
                        )
                    ],
                    key=candidate_rank,
                    reverse=True,
                )[:3]
                if corpus is None:
                    corpus = FailureCaseStore(secrets=secrets)
                self._checkpoint(context)
                corpus.record(
                    FailureCase(
                        category=evidence.category,
                        fingerprint=fp,
                        error_type=evidence.error_type,
                        message=evidence.message,
                        original_code_sha256=original_hash,
                        fixed_code_sha256=result.code_sha256,
                        verification="validated",
                        renderer=inputs.render_profile.renderer,
                        patch_summary=strategy,
                        source_run_id=inputs.run_id,
                    )
                )
                return finish("review", strategy, inspection=result)

            def rollback():
                for candidate in sorted(scene.candidates, key=candidate_rank, reverse=True):
                    if (
                        candidate.verification not in {"smoke", "rendered"}
                        or candidate.code_sha256 == original_hash
                        or candidate.inherited_elements_sha256
                        != sha256_text(scene.inherited_elements_code)
                    ):
                        continue
                    try:
                        code = self._read_source(inputs, candidate.code_file, candidate.code_sha256)
                    except (OSError, UnicodeError, ValueError):
                        continue
                    result = inspect(
                        code, "best_rollback", class_name=candidate.class_name, allow_repair=False
                    )
                    output = install(result, "best_rollback")
                    if output is not None:
                        return output
                return None

            config = inputs.config.resolve()
            route = classify_failure(error_log, phase="render", status=job.status)
            if route.category == "infrastructure" or AutoFixerAgent.is_infrastructure_error(
                error_log
            ):
                if scene.infra_retries >= config.MAX_INFRA_RETRIES:
                    return finish(
                        "exhausted",
                        "infrastructure_budget",
                        "Maximum infrastructure retries reached",
                    )
                scene.infra_retries += 1
                return finish("infrastructure", "infrastructure", route.reason or error_log)
            if not error_log:
                return rollback() or finish(
                    "exhausted",
                    "missing_error_log",
                    job.failure_reason or "Render failed without an error log",
                )
            previous = ProgressSnapshot(scene.last_repair_code_sha256, scene.last_repair_error_fp)
            progress = classify_progress(previous, ProgressSnapshot(original_hash, fp))
            scene.stagnant_repair_count = (
                scene.stagnant_repair_count + 1 if progress == "unchanged" else 0
            )
            scene.identical_error_count = (
                scene.identical_error_count + 1 if fp == scene.last_error_fp else 1
            )
            scene.last_error_fp = fp
            if scene.fix_attempts >= config.MAX_FIX_ATTEMPTS:
                return rollback() or finish(
                    "exhausted", "attempt_budget", "Maximum render repair attempts reached"
                )
            scene.fix_attempts += 1

            # Pure, uniquely matched API patches do not spend an LLM request.
            patches = AutoFixerAgent.deterministic_patches(original, error_log)
            candidate = original
            for patch in patches:
                if not patch.find or candidate.count(patch.find) != 1:
                    candidate = original
                    break
                candidate = candidate.replace(patch.find, patch.replace, 1)
            if candidate != original:
                output = install(
                    inspect(candidate, "deterministic_patch", class_name=scene.class_name),
                    "deterministic_patch",
                )
                if output is not None:
                    return output

            stagnated = scene.stagnant_repair_count >= max(1, config.MAX_STAGNANT_ATTEMPTS)
            if stagnated:
                builders = []
                if config.CODEGEN_MODE != "python":
                    builders.append(
                        (
                            "scene_ir",
                            lambda: compile_scene_program(
                                build_scene_program_from_contract(scene.plan, scene.technical_spec),
                                scene.plan,
                            ),
                        )
                    )
                builders.append(
                    (
                        "safe_template",
                        lambda: build_safe_scene_code(scene.plan, scene.technical_spec),
                    )
                )
                for strategy, builder in builders:
                    self._checkpoint(context)
                    try:
                        candidate = builder()
                    except (ValueError, RuntimeError):
                        continue
                    output = install(inspect(candidate, strategy), strategy)
                    if output is not None:
                        return output
                feedback = evidence.prompt_text()
                if (
                    config.SAFE_FALLBACK_ENABLED
                    and not scene.safe_fallback_used
                    and is_high_confidence_geometry_conflict(scene.plan, feedback)
                ):
                    proposed = build_safe_fallback_plan(
                        scene.plan,
                        inputs.continuity_bible or ContinuityBible(),
                        reason=fallback_reason_summary(feedback),
                    )
                    return finish("plan", "safe_teaching_plan", feedback, proposed_plan=proposed)
                return rollback() or finish(
                    "exhausted",
                    "stagnation",
                    "Repeated repair made no progress; deterministic fallback was rejected",
                )
            if (
                config.GENERATION_MODE != "relaxed"
                and scene.identical_error_count >= config.MAX_FIX_IDENTICAL_ERRORS
                and scene.fix_attempts > 2
            ):
                return rollback() or finish(
                    "exhausted", "identical_error", "Repeated identical render error"
                )

            self._checkpoint(context)
            fixer = self.fixer_factory()
            fixer.control = self.control
            fixer.request_context = self.request_context(scene.plan.scene_id, "fix")
            if corpus is None:
                corpus = FailureCaseStore(secrets=secrets)
            failure_context = sanitize_record(
                corpus.context(category=evidence.category, limit=3), secrets
            )
            try:
                candidate = fixer.fix(
                    original,
                    error_log,
                    renderer=inputs.render_profile.renderer,
                    technical_spec=scene.technical_spec.model_copy(deep=True)
                    if scene.technical_spec
                    else None,
                    rag_context=sanitize_record(inputs.rag_context, secrets),
                    lesson_spec=inputs.lesson_spec.model_copy(deep=True),
                    teaching_graph=inputs.teaching_graph.model_copy(deep=True),
                    error_evidence=evidence,
                    failure_case_context=failure_context,
                )
            except (
                ToolCancelled,
                RunCancelled,
                OperationDeadlineExceeded,
                RequestBudgetExceeded,
                TimeoutError,
            ):
                raise
            except Exception as exc:
                self._checkpoint(context)
                scene.last_repair_code_sha256, scene.last_repair_error_fp = original_hash, fp
                return rollback() or finish("coder", "fixer_failed", str(exc))
            self._checkpoint(context)
            result = inspect(candidate, "auto_fix", class_name=scene.class_name)
            output = install(result, "auto_fix")
            if output is not None:
                return output
            # Persist unchanged snapshots too; otherwise restarts erase stagnation.
            scene.last_repair_code_sha256, scene.last_repair_error_fp = original_hash, fp
            return rollback() or finish(
                "coder", "rejected_auto_fix", result.feedback or "AutoFix returned unchanged source"
            )
