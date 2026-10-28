"""Evidence-bound visual gates; workers return snapshots, never mutate run state."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from contextlib import suppress
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

from kd1_anime.agent_runtime.contracts import ObjectRef, RuntimeRecord, sanitize_record
from kd1_anime.agent_runtime.tools import ToolCancelled, ToolContext
from kd1_anime.candidate_acceptor import CandidateAcceptor
from kd1_anime.config import settings
from kd1_anime.control import OperationDeadlineExceeded, RunCancelled
from kd1_anime.domain.dependencies import WorkIdentity, digest
from kd1_anime.domain.runtime import VisualEvalProfile
from kd1_anime.domain.state_ledger import StateLedger
from kd1_anime.eval import Evaluator
from kd1_anime.eval.boundary_checks import check_boundary_samples
from kd1_anime.objects import ObjectStore
from kd1_anime.pipeline.contracts import FinalizationInputs, SceneInputs
from kd1_anime.pipeline.render import validate_artifact
from kd1_anime.rendering import sha256_file
from kd1_anime.request_budget import RequestBudgetExceeded
from kd1_anime.run_store import (
    StoredSceneState,
    StoredVisualCandidate,
    restore_run_path,
    sha256_text,
)
from kd1_anime.services.dependencies import scene_work_digest
from kd1_anime.services.policies import visual_repair_target
from kd1_anime.services.visual_evaluation import (
    SceneVisualInput,
    VisualEvaluationService,
    VisualIdentityError,
    verify_samples,
)
from kd1_anime.verification import ExecutionVerification, StaticVerification, VisualVerification

VisualRoute = Literal["none", "coder", "planner", "continuity", "rebuild"]
_PROPAGATE = (
    VisualIdentityError,
    ToolCancelled,
    RunCancelled,
    OperationDeadlineExceeded,
    RequestBudgetExceeded,
)


class VisualSceneOutput(RuntimeRecord):
    """Return historical reuse authority only when restoring a verified best candidate."""

    scene: StoredSceneState
    reuse_provenance: ObjectRef | None = None
    report: ObjectRef | None = None
    repair_route: VisualRoute = "none"
    feedback: str = ""
    visual_fix_attempts: int = 0
    restored: bool = False


class BoundaryVisualOutput(RuntimeRecord):
    scenes: tuple[StoredSceneState, ...]
    state_ledger: StateLedger
    report: ObjectRef | None = None
    status: Literal["passed", "warning", "unknown", "skipped"] = "skipped"
    repair_route: VisualRoute = "none"
    scene_id: int | None = None
    feedback: str = ""
    visual_fix_attempts: int = 0


class VisualRepairProposal(RuntimeRecord):
    identity: WorkIdentity
    base_source: ObjectRef
    candidate_source: ObjectRef
    visual_report: ObjectRef
    inspection_report: ObjectRef
    inherited_sha256: str
    technical_sha256: str
    context_sha256: str
    class_name: str
    accepted: bool
    visual_fix_attempts: int
    repair_route: Literal["review_code", "exhausted"]


def visual_profile(inputs) -> VisualEvalProfile:
    if inputs.visual_eval_profile is not None:
        return inputs.visual_eval_profile
    config = inputs.config.resolve()
    return VisualEvalProfile(
        enabled=config.ENABLE_VISUAL_EVAL,
        model=config.visual_llm_profile().model,
        frame_count=config.VISUAL_EVAL_FRAME_COUNT,
        threshold=config.VISUAL_EVAL_THRESHOLD,
        max_fix_attempts=config.MAX_VISUAL_FIX_ATTEMPTS,
        repair_enabled=(config.GENERATION_MODE != "relaxed" or config.RELAXED_VISUAL_AUTO_FIX),
    )


def _secrets():
    return tuple(
        value for key, value in settings.model_dump().items() if key.endswith("API_KEY") and value
    )


def store_report(inputs, payload) -> ObjectRef:
    text = json.dumps(sanitize_record(payload, _secrets()), ensure_ascii=False, sort_keys=True)
    return ObjectRef(
        path=ObjectStore(inputs.root).report(text), sha256=sha256_text(text), kind="report"
    )


def _source(inputs, scene=None) -> str:
    scene = inputs.scene if scene is None else scene
    path = restore_run_path(inputs.root, scene.code_file)
    if path.is_symlink() or not path.is_file() or sha256_file(path) != scene.code_sha256:
        raise VisualIdentityError("Visual source identity mismatch")
    return path.read_text(encoding="utf-8")


def scene_binding(scene):
    return {
        "scene_id": scene.plan.scene_id,
        "code_sha256": scene.code_sha256,
        "inherited_elements_sha256": sha256_text(scene.inherited_elements_code),
        "video_sha256": scene.artifact.video_sha256 if scene.artifact else "",
        "technical_spec_sha256": sha256_text(scene.technical_spec.model_dump_json())
        if scene.technical_spec
        else "",
        "plan_sha256": digest(scene.plan.model_dump(mode="json")),
    }


def _read_report(inputs, reference):
    path = restore_run_path(inputs.root, reference.path)
    if reference.kind != "report" or not path.is_file() or sha256_file(path) != reference.sha256:
        raise VisualIdentityError("Visual report identity mismatch")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_report(inputs, scene, reference):
    report = _read_report(inputs, reference)
    if report.get("scope") != "scene" or report.get("binding") != scene_binding(scene):
        raise VisualIdentityError("Visual report is stale for current code, context or video")
    if report.get("visual_profile_sha256") != visual_profile(inputs).digest():
        raise VisualIdentityError("Visual report profile mismatch")
    for frame in report.get("frames", []):
        path = restore_run_path(inputs.root, frame["path"])
        if not frame.get("sha256") or not path.is_file() or sha256_file(path) != frame["sha256"]:
            raise VisualIdentityError("Visual report frame identity mismatch")
    return report


def verify_scene_receipt(inputs, scene):
    reference = ObjectRef(
        path=scene.visual_report_file, sha256=scene.visual_report_sha256, kind="report"
    )
    report = verify_report(inputs, scene, reference)
    receipt = scene.visual_verification
    if (
        scene.artifact is None
        or scene.visual_artifact_sha256 != scene.artifact.video_sha256
        or receipt.code_sha256 != scene.code_sha256
        or receipt.artifact_sha256 != scene.artifact.video_sha256
        or receipt.report_sha256 != reference.sha256
        or receipt.status
        != ("failed" if scene.visual_status == "needs_fix" else scene.visual_status)
        or receipt.score != scene.visual_score
    ):
        raise VisualIdentityError("Visual verification receipt identity mismatch")
    result = report.get("result")
    if result is None:
        if scene.visual_status != "unknown" or scene.visual_score is not None:
            raise VisualIdentityError("Unknown visual evidence cannot supply a score")
    elif scene.visual_score is None or abs(result["overall_score"] - scene.visual_score) > 0.0001:
        raise VisualIdentityError("Visual score does not match the bound report")
    return report


def _preserve_frames(inputs, samples):
    """Copy extractor scratch files into private content-addressed frame objects."""
    verify_samples(samples)
    preserved = []
    for sample in samples:
        if not sample.path.resolve().is_relative_to(inputs.root.resolve()):
            raise VisualIdentityError("Visual frame escapes the private run")
        relative_source = sample.path.resolve().relative_to(inputs.root.resolve()).as_posix()
        try:
            restore_run_path(inputs.root, relative_source)
        except ValueError as exc:
            raise VisualIdentityError("Unsafe visual frame path") from exc
        suffix = sample.path.suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
            raise VisualIdentityError("Unsupported visual frame format")
        relative = f"artifacts/frames/{sample.image_sha256}{suffix}"
        target = restore_run_path(inputs.root, relative)
        target.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        target.parent.chmod(0o700)
        if not target.exists():
            descriptor, name = tempfile.mkstemp(prefix=".frame-", dir=target.parent)
            temporary = Path(name)
            try:
                os.fchmod(descriptor, 0o600)
                with os.fdopen(descriptor, "wb") as output, sample.path.open("rb") as source:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                    output.flush()
                    os.fsync(output.fileno())
                if sha256_file(temporary) != sample.image_sha256:
                    raise VisualIdentityError("Visual frame changed while preserving evidence")
                with suppress(FileExistsError):
                    os.link(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
        if target.is_symlink() or sha256_file(target) != sample.image_sha256:
            raise VisualIdentityError("Immutable visual frame identity mismatch")
        preserved.append(sample.model_copy(update={"path": target}))
    return preserved


def frame_bindings(inputs, samples):
    return [
        {
            "frame_id": sample.frame_id,
            "scene_id": sample.scene_id,
            "boundary_id": sample.boundary_id,
            "role": sample.role,
            "timestamp_seconds": sample.timestamp_seconds,
            "path": sample.path.relative_to(inputs.root.resolve()).as_posix(),
            "sha256": sample.image_sha256,
        }
        for sample in samples
    ]


class VisualPipeline:
    def __init__(
        self, evaluator_factory=Evaluator, fixer_factory=None, *, request_context, control
    ):
        if fixer_factory is None:
            from kd1_anime.agents.auto_fixer import AutoFixerAgent

            fixer_factory = AutoFixerAgent
        self.evaluator_factory = evaluator_factory
        self.fixer_factory = fixer_factory
        self.request_context = request_context
        self.control = control
        self.service = VisualEvaluationService()
        self.acceptor = CandidateAcceptor()

    def _checkpoint(self, context):
        context.checkpoint()
        if self.control is not None:
            self.control.check()

    def evaluator(self, inputs, context, *, scene_id=None, visual=True):
        self._checkpoint(context)
        if visual and (not callable(self.request_context) or self.control is None):
            raise RuntimeError("Visual dispatch requires injected request_context and control")
        evaluator = self.evaluator_factory(
            enable_visual_eval=visual,
            visual_eval_model=visual_profile(inputs).model,
            output_dir=inputs.root / "eval_reports",
        )
        if visual:
            visual_evaluator = evaluator.visual_evaluator
            if visual_evaluator is None:
                raise RuntimeError("视觉评估器未初始化")
            agent = visual_evaluator.agent
            agent.request_context = self.request_context(scene_id, "visual_evaluating")
            agent.control = self.control
        return evaluator

    @staticmethod
    def _description(inputs):
        return getattr(inputs, "user_prompt", "") or (
            inputs.lesson_spec.topic if inputs.lesson_spec is not None else ""
        )

    @staticmethod
    def _scene_context(inputs):
        return json.dumps(
            {
                "scene_plan": inputs.scene.plan.model_dump(mode="json"),
                "lesson_spec": inputs.lesson_spec.model_dump(mode="json"),
                "teaching_graph": inputs.teaching_graph.model_dump(mode="json"),
                "global_continuity": inputs.continuity_bible.model_dump(mode="json")
                if inputs.continuity_bible
                else None,
                "element_manifest": inputs.element_manifest.model_dump(mode="json"),
                "state_ledger": inputs.state_ledger.model_dump(mode="json"),
            },
            ensure_ascii=False,
        )[:30_000]

    def _receipt(self, inputs, scene, reference, status, score, feedback):
        scene.visual_status = status
        scene.visual_score = score
        scene.visual_report_file = reference.path
        scene.visual_report_sha256 = reference.sha256
        scene.visual_artifact_sha256 = scene.artifact.video_sha256
        scene.visual_feedback = feedback[:20_000]
        scene.visual_verification = VisualVerification(
            status="failed" if status == "needs_fix" else status,
            code_sha256=scene.code_sha256,
            artifact_sha256=scene.artifact.video_sha256,
            report_sha256=reference.sha256,
            score=score,
            feedback=feedback[:20_000],
            checked_at=datetime.now(timezone.utc).isoformat(),
        )

    def _restore_best(self, inputs, context):
        scene = inputs.scene.model_copy(deep=True)
        candidate = scene.visual_best_candidate
        if candidate is None:
            raise VisualIdentityError("No visual rollback candidate")
        if candidate.inherited_elements_sha256 != sha256_text(scene.inherited_elements_code):
            raise VisualIdentityError("Visual rollback inherited context mismatch")
        scene.code_file = candidate.code_file
        scene.code_sha256 = candidate.code_sha256
        scene.class_name = candidate.class_name
        scene.artifact = candidate.artifact
        scene.slurm_job = candidate.slurm_job
        scene.submission = None
        scene.phase = "visual_accepted"
        scene.rendered = scene.reviewed = True
        scene.failed = scene.give_up = False
        scene.failure_reason = scene.failure_category = scene.rewrite_feedback = ""
        scene.exported_elements_code = candidate.exported_elements_code
        scene.exported_elements = list(candidate.exported_elements)
        if inputs.context_sha256:
            scene.input_sha256 = scene_work_digest(scene, inputs.context_sha256)
        code = _source(inputs, scene)
        inspection = self.acceptor.inspect_candidate(
            code,
            scene.plan,
            technical_spec=scene.technical_spec,
            renderer=inputs.render_profile.renderer,
            manim_version=inputs.render_profile.manim_version,
            expected_class_name=scene.class_name,
            allow_repair=False,
        )
        if not inspection.accepted or inspection.code_sha256 != scene.code_sha256:
            raise VisualIdentityError("Visual rollback failed readonly candidate inspection")
        scene.exported_elements_code = inspection.exported_elements_code
        scene.exported_elements = list(inspection.exported_elements)
        scene.unknown_animation_detected = bool(inspection.unknown_animations)
        scene.unknown_animation_details = list(inspection.unknown_animations)[-30:]
        reference = ObjectRef(
            path=candidate.report_file, sha256=candidate.report_sha256, kind="report"
        )
        report = verify_report(inputs, scene, reference)
        # Only the candidate's immutable report can authorize historical reuse;
        # the current typed receipt may already belong to a later code attempt.
        reuse_provenance = None
        if scene.artifact.origin == "reused":
            try:
                reuse_provenance = ObjectRef.model_validate(report.get("reuse_provenance"))
            except (TypeError, ValueError) as exc:
                raise VisualIdentityError(
                    "Visual rollback lacks exact typed reuse provenance"
                ) from exc
        result = report.get("result")
        if (
            not result
            or abs(result["overall_score"] - candidate.score) > 0.0001
            or result["has_major_issue"] != candidate.has_major_issue
            or candidate.passed
            != (
                not candidate.has_major_issue
                and candidate.score >= visual_profile(inputs).threshold
            )
        ):
            raise VisualIdentityError("Visual rollback rank does not match its report")
        feedback = "视觉修复链路失败或预算耗尽，已恢复最佳可用版本"
        self._receipt(inputs, scene, reference, "warning", candidate.score, feedback)
        scene.static_verification = StaticVerification(
            status="passed",
            code_sha256=scene.code_sha256,
            technical_spec_sha256=sha256_text(scene.technical_spec.model_dump_json())
            if scene.technical_spec
            else "",
        )
        scene.execution_verification = ExecutionVerification(
            status="passed",
            scope="formal_video",
            code_sha256=scene.code_sha256,
            artifact_sha256=scene.artifact.video_sha256,
            duration_seconds=scene.artifact.metadata.duration_seconds,
        )
        validate_artifact(
            inputs.root,
            inputs.run_id,
            scene,
            inputs.render_profile,
            provenance=reuse_provenance,
        )
        self._checkpoint(context)
        return VisualSceneOutput(
            scene=scene,
            report=reference,
            reuse_provenance=reuse_provenance,
            restored=True,
            feedback=feedback,
            visual_fix_attempts=scene.visual_fix_attempts,
            repair_route="rebuild" if inputs.scene.code_sha256 != scene.code_sha256 else "none",
        )

    def reject_proposal(self, inputs: SceneInputs, context: ToolContext) -> VisualSceneOutput:
        """End a rejected repair without rerunning its reserved model attempt."""
        self._checkpoint(context)
        if inputs.scene.visual_best_candidate is not None:
            return self._restore_best(inputs, context)
        scene = inputs.scene.model_copy(deep=True)
        reference = ObjectRef(
            path=scene.visual_report_file, sha256=scene.visual_report_sha256, kind="report"
        )
        verify_scene_receipt(inputs, scene)
        self._receipt(
            inputs,
            scene,
            reference,
            "warning",
            scene.visual_score,
            "Visual repair rejected; retained the verified current render",
        )
        return VisualSceneOutput(
            scene=scene, report=reference, visual_fix_attempts=scene.visual_fix_attempts
        )

    def scene(self, inputs: SceneInputs, context: ToolContext) -> VisualSceneOutput:
        self._checkpoint(context)
        scene = inputs.scene.model_copy(deep=True)
        profile = visual_profile(inputs)
        if not profile.enabled:
            scene.visual_status = "skipped"
            scene.visual_verification = VisualVerification()
            return VisualSceneOutput(scene=scene, visual_fix_attempts=scene.visual_fix_attempts)
        if not scene.rendered and scene.visual_best_candidate and (scene.failed or scene.give_up):
            return self._restore_best(inputs, context)
        _source(inputs)
        video = validate_artifact(
            inputs.root,
            inputs.run_id,
            scene,
            inputs.render_profile,
            provenance=inputs.reuse_provenance,
        )
        if scene.visual_status == "needs_fix":
            reference = ObjectRef(
                path=scene.visual_report_file, sha256=scene.visual_report_sha256, kind="report"
            )
            report = verify_scene_receipt(inputs, scene)
            if report.get("attempt") != scene.visual_fix_attempts - 1 or not report.get("result"):
                raise VisualIdentityError("Visual repair reservation mismatch")
            route = visual_repair_target(
                SimpleNamespace(
                    issues=[SimpleNamespace(**issue) for issue in report["result"]["issues"]]
                )
            )
            if route not in {"coder", "planner", "continuity"}:
                raise VisualIdentityError("Visual repair receipt has no actionable route")
            return VisualSceneOutput(
                scene=scene,
                report=reference,
                repair_route=route,
                feedback=scene.visual_feedback,
                visual_fix_attempts=scene.visual_fix_attempts,
            )
        if scene.visual_status in {"passed", "warning", "unknown"}:
            reference = ObjectRef(
                path=scene.visual_report_file, sha256=scene.visual_report_sha256, kind="report"
            )
            verify_scene_receipt(inputs, scene)
            if scene.visual_artifact_sha256 != scene.artifact.video_sha256:
                raise VisualIdentityError("Visual receipt belongs to a different video")
            return VisualSceneOutput(
                scene=scene, report=reference, visual_fix_attempts=scene.visual_fix_attempts
            )
        result, samples, error = None, [], ""
        with inputs.config.activate():
            try:
                evaluator = self.evaluator(inputs, context, scene_id=scene.plan.scene_id)
                evaluated = self.service.evaluate_scene(
                    evaluator,
                    SceneVisualInput(
                        video,
                        scene.artifact.video_sha256,
                        scene.code_sha256,
                        sha256_text(scene.inherited_elements_code),
                        self._description(inputs),
                        self._scene_context(inputs),
                    ),
                    output_dir=inputs.root
                    / "eval_frames"
                    / f"scene_{scene.plan.scene_id}"
                    / scene.artifact.video_sha256,
                    frame_count=profile.frame_count,
                )
                result, samples = evaluated.analysis, list(evaluated.samples)
            except _PROPAGATE:
                raise
            except Exception as exc:
                error = sanitize_record(str(exc) or type(exc).__name__, _secrets())
            self._checkpoint(context)
            _source(inputs)
            validate_artifact(
                inputs.root,
                inputs.run_id,
                scene,
                inputs.render_profile,
                provenance=inputs.reuse_provenance,
            )
            samples = _preserve_frames(inputs, samples)
            reference = store_report(
                inputs,
                {
                    "schema_version": 1,
                    "scope": "scene",
                    "binding": scene_binding(scene),
                    "reuse_provenance": inputs.reuse_provenance.model_dump(mode="json")
                    if scene.artifact.origin == "reused" and inputs.reuse_provenance is not None
                    else None,
                    "visual_profile_sha256": profile.digest(),
                    "model": profile.model,
                    "attempt": scene.visual_fix_attempts,
                    "frames": frame_bindings(inputs, samples),
                    "error": error,
                    "result": result.to_dict() if result is not None else None,
                },
            )
            if result is None:
                self._receipt(inputs, scene, reference, "unknown", None, error)
                return VisualSceneOutput(
                    scene=scene,
                    report=reference,
                    feedback=error,
                    visual_fix_attempts=scene.visual_fix_attempts,
                )
            score, passed = result.overall_score, not result.needs_fix(profile.threshold)
            feedback = sanitize_record(result.feedback(), _secrets())
            candidate = StoredVisualCandidate(
                score=score,
                passed=passed,
                has_major_issue=result.has_major_issue,
                inherited_elements_sha256=sha256_text(scene.inherited_elements_code),
                code_file=scene.code_file,
                code_sha256=scene.code_sha256,
                class_name=scene.class_name,
                slurm_job=scene.slurm_job,
                artifact=scene.artifact,
                exported_elements_code=scene.exported_elements_code,
                exported_elements=list(scene.exported_elements),
                report_file=reference.path,
                report_sha256=reference.sha256,
            )
            previous = scene.visual_best_candidate

            def rank(item):
                return item.passed, not item.has_major_issue, item.score

            if previous is None or rank(candidate) > rank(previous):
                scene.visual_best_candidate = candidate
            route = visual_repair_target(result) if not passed else "unknown"
            can_fix = (
                inputs.auto_fix
                and profile.repair_enabled
                and scene.visual_fix_attempts < profile.max_fix_attempts
                and route in {"coder", "planner", "continuity"}
            )
            if passed:
                feedback, status, route = "", "passed", "none"
            elif can_fix:
                scene.visual_fix_attempts += 1
                status = "needs_fix"
            else:
                best = scene.visual_best_candidate
                if best is not None and best.artifact.video_sha256 != scene.artifact.video_sha256:
                    return self._restore_best(replace(inputs, scene=scene), context)
                status, route = "warning", "none"
                if not profile.repair_enabled:
                    feedback += "\nrelaxed 模式仅做视觉诊断"
            self._receipt(inputs, scene, reference, status, score, feedback)
            return VisualSceneOutput(
                scene=scene,
                report=reference,
                repair_route=route,
                feedback=feedback,
                visual_fix_attempts=scene.visual_fix_attempts,
            )

    def propose_fix(
        self,
        inputs: SceneInputs,
        context: ToolContext,
        *,
        boundary_report: ObjectRef | None = None,
        feedback: str = "",
    ) -> VisualRepairProposal:
        """Propose a readonly-inspected source; the coordinator must Review before installing."""
        self._checkpoint(context)
        inputs = replace(inputs, scene=inputs.scene.model_copy(deep=True))
        if not callable(self.request_context) or self.control is None:
            raise RuntimeError("Visual repair requires injected request_context and control")
        scene, profile = inputs.scene, visual_profile(inputs)
        if (
            not profile.enabled
            or not profile.repair_enabled
            or not inputs.auto_fix
            or (boundary_report is None and scene.visual_status != "needs_fix")
            or scene.visual_fix_attempts < 1
            or scene.visual_fix_attempts > profile.max_fix_attempts
        ):
            raise ValueError("Visual repair was not durably reserved or is disabled")
        reference = boundary_report or ObjectRef(
            path=scene.visual_report_file, sha256=scene.visual_report_sha256, kind="report"
        )
        if boundary_report is None:
            report = verify_scene_receipt(inputs, scene)
            reserved = report.get("attempt") == scene.visual_fix_attempts - 1
        else:
            report = _read_report(inputs, reference)
            reserved = (
                report.get("attempts", {}).get(str(scene.plan.scene_id))
                == scene.visual_fix_attempts - 1
            )
            if (
                report.get("scope") != "scene_boundaries"
                or scene_binding(scene) not in report.get("scene_bindings", [])
                or report.get("visual_profile_sha256") != profile.digest()
            ):
                raise VisualIdentityError("Boundary repair report does not bind the current scene")
            for frame in report.get("frames", []):
                path = restore_run_path(inputs.root, frame["path"])
                if (
                    not frame.get("sha256")
                    or not path.is_file()
                    or sha256_file(path) != frame["sha256"]
                ):
                    raise VisualIdentityError("Boundary repair frame hash mismatch")
        if not reserved or not report.get("result"):
            raise VisualIdentityError("Visual repair attempt does not match the report")
        route = visual_repair_target(
            SimpleNamespace(
                issues=[SimpleNamespace(**issue) for issue in report["result"]["issues"]]
            )
        )
        if route != "coder":
            raise VisualIdentityError(
                "Visual evidence must be routed to plan repair, not a code fixer"
            )
        original = _source(inputs)
        validate_artifact(
            inputs.root,
            inputs.run_id,
            scene,
            inputs.render_profile,
            provenance=inputs.reuse_provenance,
        )
        with inputs.config.activate():
            fixer = self.fixer_factory()
            fixer.request_context = self.request_context(scene.plan.scene_id, "visual_fixing")
            fixer.control = self.control
            candidate = fixer.fix(
                original,
                "## Visual Evaluation Feedback\n" + (feedback or scene.visual_feedback),
                renderer=inputs.render_profile.renderer,
                technical_spec=scene.technical_spec,
                rag_context=inputs.rag_context,
                lesson_spec=inputs.lesson_spec,
                teaching_graph=inputs.teaching_graph,
            )
            self._checkpoint(context)
            _source(inputs)
            validate_artifact(
                inputs.root,
                inputs.run_id,
                scene,
                inputs.render_profile,
                provenance=inputs.reuse_provenance,
            )
            _read_report(inputs, reference)
            for frame in report.get("frames", []):
                path = restore_run_path(inputs.root, frame["path"])
                if not path.is_file() or sha256_file(path) != frame["sha256"]:
                    raise VisualIdentityError("Visual repair frame changed during proposal")
            inspection = self.acceptor.inspect_candidate(
                candidate,
                scene.plan,
                technical_spec=scene.technical_spec,
                renderer=inputs.render_profile.renderer,
                manim_version=inputs.render_profile.manim_version,
                expected_class_name=scene.class_name,
                allow_repair=False,
            )
            code_file = ObjectStore(inputs.root).source(scene.plan.scene_id, inspection.code)
            inspection_ref = store_report(
                inputs,
                {
                    "kind": "visual_fix_inspection",
                    "base": scene_binding(scene),
                    "candidate_sha256": inspection.code_sha256,
                    "accepted": inspection.accepted,
                    "feedback": inspection.feedback,
                    "visual_report_sha256": reference.sha256,
                },
            )
            return VisualRepairProposal(
                identity=WorkIdentity(scene.revision, scene.input_sha256),
                base_source=ObjectRef(path=scene.code_file, sha256=scene.code_sha256, kind="code"),
                candidate_source=ObjectRef(
                    path=code_file, sha256=inspection.code_sha256, kind="code"
                ),
                visual_report=reference,
                inspection_report=inspection_ref,
                inherited_sha256=sha256_text(scene.inherited_elements_code),
                technical_sha256=sha256_text(scene.technical_spec.model_dump_json())
                if scene.technical_spec
                else "",
                context_sha256=digest(scene_binding(scene)),
                class_name=inspection.class_name,
                accepted=inspection.accepted,
                visual_fix_attempts=scene.visual_fix_attempts,
                repair_route="review_code" if inspection.accepted else "exhausted",
            )

    def boundary(self, inputs: FinalizationInputs, context: ToolContext) -> BoundaryVisualOutput:
        self._checkpoint(context)
        scenes = tuple(
            scene.model_copy(deep=True)
            for scene in sorted(inputs.scenes, key=lambda item: item.plan.scene_id)
        )
        profile = visual_profile(inputs)
        common = dict(scenes=scenes, state_ledger=inputs.state_ledger)
        if not profile.enabled or len([scene for scene in scenes if scene.rendered]) < 2:
            return BoundaryVisualOutput(**common)
        rendered = [
            (
                scene.plan.scene_id,
                validate_artifact(
                    inputs.root,
                    inputs.run_id,
                    scene,
                    inputs.render_profile,
                    provenance=inputs.reuse_receipts.get(scene.plan.scene_id),
                ),
            )
            for scene in scenes
            if scene.rendered
        ]
        for scene in scenes:
            if scene.rendered:
                _source(inputs, scene)
        result, samples, error, deterministic = None, [], "", None
        ledger = inputs.state_ledger.model_copy(deep=True)
        with inputs.config.activate():
            try:
                evaluator = self.evaluator(inputs, context)
                samples = evaluator.extract_boundary_samples(
                    rendered,
                    inputs.root
                    / "eval_frames"
                    / "boundaries"
                    / digest([scene_binding(scene) for scene in scenes]),
                    max_boundaries=min(3, max(1, profile.frame_count // 2)),
                )
                samples = _preserve_frames(inputs, samples)
                if not samples:
                    raise RuntimeError("没有可用于边界视觉评估的关键帧")
                deterministic = check_boundary_samples(samples)
                boundaries = dict(ledger.boundaries)
                for sample in samples:
                    if sample.scene_id in boundaries:
                        field = {
                            "boundary_end": "ending_frame_sha256",
                            "boundary_start": "opening_frame_sha256",
                        }.get(sample.role)
                        if field:
                            boundaries[sample.scene_id] = boundaries[sample.scene_id].model_copy(
                                update={field: sample.image_sha256}
                            )
                ledger = ledger.model_copy(update={"boundaries": boundaries})
                result = self.service.evaluate_frames(
                    evaluator.visual_evaluator,
                    samples,
                    self._description(inputs),
                    scene_context=json.dumps(
                        {
                            "state_ledger": ledger.model_dump(mode="json"),
                            "scene_plans": [s.plan.model_dump(mode="json") for s in scenes],
                        },
                        ensure_ascii=False,
                    )[:30_000],
                    scope="complete video",
                )
            except _PROPAGATE:
                raise
            except Exception as exc:
                error = sanitize_record(str(exc) or type(exc).__name__, _secrets())
            self._checkpoint(context)
            for scene in scenes:
                if scene.rendered:
                    _source(inputs, scene)
                    validate_artifact(
                        inputs.root,
                        inputs.run_id,
                        scene,
                        inputs.render_profile,
                        provenance=inputs.reuse_receipts.get(scene.plan.scene_id),
                    )
            verify_samples(samples)
            reference = store_report(
                inputs,
                {
                    "schema_version": 1,
                    "scope": "scene_boundaries",
                    "visual_profile_sha256": profile.digest(),
                    "model": profile.model,
                    "scene_bindings": [scene_binding(s) for s in scenes if s.rendered],
                    "attempts": {str(s.plan.scene_id): s.visual_fix_attempts for s in scenes},
                    "state_ledger_digest": ledger.digest(),
                    "frames": frame_bindings(inputs, samples),
                    "deterministic_checks": deterministic.to_dict() if deterministic else None,
                    "result": result.to_dict() if result is not None else None,
                    "error": error,
                },
            )
            common.update(state_ledger=ledger, report=reference)
            if result is None:
                return BoundaryVisualOutput(status="unknown", feedback=error, **common)
            if not result.needs_fix(profile.threshold):
                return BoundaryVisualOutput(status="passed", **common)
            feedback = sanitize_record(result.feedback(), _secrets())
            route = visual_repair_target(result)
            issue_boundaries = {bid for issue in result.issues for bid in issue.boundary_ids}
            starts = [
                sample.scene_id
                for sample in samples
                if sample.role == "boundary_start"
                and sample.scene_id is not None
                and (not issue_boundaries or sample.boundary_id in issue_boundaries)
            ]
            referenced = [sample.scene_id for sample in samples if sample.scene_id is not None]
            scene_id = min(starts or referenced) if starts or referenced else rendered[0][0]
            scene = next(s for s in scenes if s.plan.scene_id == scene_id)
            can_fix = (
                inputs.auto_fix
                and profile.repair_enabled
                and scene.visual_fix_attempts < profile.max_fix_attempts
                and route in {"coder", "planner", "continuity"}
            )
            if can_fix:
                scene.visual_fix_attempts += 1
                scene.visual_feedback = feedback
            return BoundaryVisualOutput(
                status="warning",
                repair_route=route if can_fix else "none",
                scene_id=scene_id,
                feedback=feedback,
                visual_fix_attempts=scene.visual_fix_attempts,
                **common,
            )
