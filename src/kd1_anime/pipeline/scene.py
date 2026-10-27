"""Bounded source generation and immutable, evidence-bound scene review."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from itertools import pairwise
from typing import Literal

from kd1_anime.agent_runtime.contracts import (
    DiffBinding,
    ObjectRef,
    RuntimeRecord,
    Sha256,
    sanitize_record,
    utc_now,
)
from kd1_anime.agents.review_policy import review_budget
from kd1_anime.agents.validator import validate_manim_code
from kd1_anime.candidate_acceptor import CandidateAcceptor
from kd1_anime.config import settings
from kd1_anime.domain.dependencies import WorkIdentity, digest
from kd1_anime.domain.diagnostics import Diagnostic, diagnostic_from_review_finding
from kd1_anime.domain.planning import ExtractedElement
from kd1_anime.domain.review import ReviewResult
from kd1_anime.objects import ObjectStore
from kd1_anime.pipeline.contracts import SceneInputs, SceneOutput
from kd1_anime.rendering import sha256_file
from kd1_anime.run_store import restore_run_path, sha256_text
from kd1_anime.services.generation import GenerationService
from kd1_anime.services.policies import (
    accepts_review,
    candidate_rank,
    review_finding_requires_plan_repair,
)
from kd1_anime.services.program_compiler import ProgramCompileError, compile_program
from kd1_anime.services.reviewing import ReviewService
from kd1_anime.verification import StaticVerification


class SceneStageError(RuntimeError):
    """A failed stage, never a request to invoke the legacy orchestrator."""


class InspectionOutput(RuntimeRecord):
    source: ObjectRef
    class_name: str
    accepted: bool
    verification: StaticVerification
    report: ObjectRef
    exported_elements_code: str = ""
    exported_elements: tuple[ExtractedElement, ...] = ()
    diagnostics: tuple[Diagnostic, ...] = ()


class CandidateProposal(RuntimeRecord):
    """An inspected diff, not permission to install or execute its source."""

    identity: WorkIdentity
    base_source: ObjectRef
    candidate_source: ObjectRef
    inspection_report: ObjectRef
    review_report: ObjectRef
    context_sha256: Sha256
    diff_sha256: Sha256

    def bound_digest(self) -> str:
        return DiffBinding(
            work_identity=self.identity,
            base_source=self.base_source,
            candidate_source=self.candidate_source,
            inspection_report=self.inspection_report,
            review_report=self.review_report,
            context_sha256=self.context_sha256,
        ).digest()


class CodeReviewOutput(RuntimeRecord):
    accepted: bool
    code_sha256: str
    report: ObjectRef
    result: ReviewResult | None = None
    diagnostics: tuple[Diagnostic, ...] = ()
    repair_route: Literal["none", "coder", "plan", "approval", "exhausted"] = "none"
    feedback: str = ""
    proposal: CandidateProposal | None = None
    source: ObjectRef | None = None
    class_name: str = ""
    inspection: InspectionOutput | None = None
    review_round: int = 0
    review_signature: str = ""
    identical_review_count: int = 0


class ScenePipeline:
    def __init__(
        self,
        coder_factory,
        reviewer_factory,
        *,
        request_context: Callable,
        control,
        rag_service=None,
        rag_profile=None,
    ):
        self.coder_factory = coder_factory
        self.reviewer_factory = reviewer_factory
        self.request_context = request_context
        self.control = control
        self.rag_service = rag_service
        self.rag_profile = rag_profile
        self.acceptor = CandidateAcceptor()

    def pin_profile(self, profile) -> None:
        """Pin the persisted RAG identity at a coordinator barrier, before workers."""
        if profile.enabled:
            if self.rag_service is None:
                raise SceneStageError("RAG is enabled but no pinned RagService was provided")
            self.rag_service.require_profile(profile)
        self.rag_profile = profile

    def _checkpoint(self):
        checkpoint = getattr(self.control, "checkpoint", None) or self.control.check
        checkpoint()

    def _agent(self, factory, inputs, stage):
        self._checkpoint()
        agent = factory()
        agent.control = self.control
        agent.request_context = self.request_context(inputs.scene.plan.scene_id, stage)
        return agent

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

    @classmethod
    def _report(cls, inputs, payload):
        text = json.dumps(
            sanitize_record(payload, cls._secrets()), ensure_ascii=False, sort_keys=True
        )
        return ObjectRef(
            path=ObjectStore(inputs.root).report(text), sha256=sha256_text(text), kind="report"
        )

    @staticmethod
    def _read_ref(inputs, ref):
        path = restore_run_path(inputs.root, ref.path)
        if path.is_symlink() or not path.is_file() or sha256_file(path) != ref.sha256:
            raise ValueError("Scene object identity mismatch")
        return path.read_text(encoding="utf-8")

    @classmethod
    def _source(cls, inputs):
        return cls._read_ref(
            inputs,
            ObjectRef(path=inputs.scene.code_file, sha256=inputs.scene.code_sha256, kind="code"),
        )

    @staticmethod
    def _context_digest(inputs):
        return digest(
            {
                "run": inputs.run_id,
                "plan": inputs.scene.plan.model_dump(mode="json"),
                "technical": inputs.scene.technical_spec.model_dump(mode="json")
                if inputs.scene.technical_spec is not None
                else None,
                "inherited": inputs.scene.inherited_elements_code,
                "elements": inputs.element_manifest.model_dump(mode="json"),
                "lesson": inputs.lesson_spec.model_dump(mode="json"),
                "teaching": inputs.teaching_graph.model_dump(mode="json"),
                "bible": inputs.continuity_bible.model_dump(mode="json")
                if inputs.continuity_bible is not None
                else None,
                "renderer": inputs.render_profile.digest(),
                "execution": inputs.execution_spec.digest(),
                "config": inputs.config.values_json,
                "rag": inputs.rag_context,
            }
        )

    def _inspect(self, inputs, code, *, allow_repair, expected_class_name=None):
        self._checkpoint()
        return self.acceptor.inspect_candidate(
            code,
            inputs.scene.plan,
            technical_spec=inputs.scene.technical_spec,
            renderer=inputs.render_profile.renderer,
            manim_version=inputs.render_profile.manim_version,
            expected_class_name=expected_class_name,
            allow_repair=allow_repair,
        )

    def _inspection_output(self, inputs, inspection):
        secrets = self._secrets()
        path = ObjectStore(inputs.root).source(inputs.scene.plan.scene_id, inspection.code)
        report = self._report(
            inputs,
            {
                "kind": "candidate_inspection",
                "code_sha256": inspection.code_sha256,
                "technical_spec_sha256": inputs.scene.technical_spec_sha256,
                "accepted": inspection.accepted,
                "repairs": inspection.repairs,
                "diagnostics": [item.model_dump(mode="json") for item in inspection.diagnostics],
            },
        )
        return InspectionOutput(
            source=ObjectRef(path=path, sha256=inspection.code_sha256, kind="code"),
            class_name=inspection.class_name or inputs.scene.class_name,
            accepted=inspection.accepted,
            verification=StaticVerification(
                status="passed" if inspection.accepted else "failed",
                code_sha256=inspection.code_sha256,
                technical_spec_sha256=inputs.scene.technical_spec_sha256,
                checked_at=utc_now().isoformat(),
                error=sanitize_record(inspection.feedback, secrets),
            ),
            report=report,
            exported_elements_code=inspection.exported_elements_code,
            exported_elements=inspection.exported_elements,
            diagnostics=tuple(
                Diagnostic.model_validate(sanitize_record(item, secrets))
                for item in inspection.diagnostics
            ),
        )

    def _require_rag(self):
        if self.rag_service is None:
            raise SceneStageError("RAG is enabled but no pinned RagService was provided")
        if self.rag_profile is not None:
            self.rag_service.require_profile(self.rag_profile)
        else:
            self.rag_service.require_ready()

    def _rag(self, inputs):
        if not inputs.config.resolve().RAG_ENABLED:
            return inputs.rag_context, ()
        self._require_rag()
        plan = inputs.scene.plan
        result = self.rag_service.search(
            "\n".join((plan.title, plan.math_concept, plan.visual_design, plan.computation)),
            stage="code",
            code_sha256=inputs.scene.code_sha256,
            inherited_elements_sha256=sha256_text(inputs.scene.inherited_elements_code)
            if inputs.scene.inherited_elements_code
            else "",
        )
        if (
            self.rag_profile is not None
            and result.receipt.index_sha256
            and result.receipt.index_sha256 != self.rag_profile.index_sha256
        ):
            raise SceneStageError("RAG index identity changed during scene retrieval")
        return "\n\n".join(item for item in (inputs.rag_context, result.context) if item), (
            result.receipt,
        )

    def _fallback_source(self, inputs, *, use_ir, reason):
        from kd1_anime.agents.scene_ir import (
            build_scene_program_from_contract,
            compile_scene_program,
        )
        from kd1_anime.agents.scene_templates import build_safe_scene_code

        if use_ir:
            code = compile_scene_program(
                build_scene_program_from_contract(inputs.scene.plan, inputs.scene.technical_spec),
                inputs.scene.plan,
            )
        else:
            code = build_safe_scene_code(inputs.scene.plan, inputs.scene.technical_spec)
        inspection = self._inspect(inputs, code, allow_repair=True)
        if not inspection.accepted:
            raise SceneStageError(
                "Explicit fallback failed complete inspection: " + inspection.feedback
            )
        return SceneOutput(
            source=ObjectRef(
                path=ObjectStore(inputs.root).source(inputs.scene.plan.scene_id, inspection.code),
                sha256=inspection.code_sha256,
                kind="code",
            ),
            class_name=inspection.class_name,
            diagnostics=(
                *inspection.diagnostics,
                Diagnostic(
                    code="scene_codegen_fallback",
                    stage="ast",
                    category="compiler",
                    severity="minor",
                    scene_id=inputs.scene.plan.scene_id,
                    summary="Explicit degraded generation; teaching equivalence is not guaranteed",
                    detail=sanitize_record(reason, self._secrets()),
                ),
            ),
        )

    def generate(
        self,
        inputs: SceneInputs,
        *,
        attempts_used=0,
        reserve_attempt=None,
        feedback=None,
        previous_code=None,
    ) -> SceneOutput:
        """Generate only statically accepted source; Review remains a separate action."""
        with inputs.config.activate():
            config = inputs.config.resolve()
            feedback = inputs.scene.rewrite_feedback if feedback is None else feedback
            previous_code = (
                (self._source(inputs) if inputs.scene.code_file and feedback else "")
                if previous_code is None
                else previous_code
            )
            diagnostics = []
            if config.SCENE_CODEGEN_MODE == "compiler_first" and not feedback:
                GenerationService.validate_technical_contract(
                    inputs.scene.plan,
                    inputs.scene.technical_spec,
                    renderer=inputs.render_profile.renderer,
                )
                try:
                    if inputs.scene.plan.program is None:
                        raise ProgramCompileError("The approved plan has no complete SceneProgram")
                    code = compile_program(inputs.scene.plan.program, inputs.scene.plan)
                    inspection = self._inspect(inputs, code, allow_repair=True)
                    if not inspection.accepted:
                        raise ProgramCompileError(inspection.feedback)
                except ProgramCompileError as exc:
                    diagnostics.append(
                        Diagnostic(
                            code="scene_program_unsupported",
                            stage="ast",
                            category="compiler",
                            severity="minor",
                            scene_id=inputs.scene.plan.scene_id,
                            summary="Complete program compilation unavailable; using LLM generation",
                            detail=str(exc),
                            repair_target="coder",
                        )
                    )
                else:
                    if config.RAG_ENABLED:
                        self._require_rag()
                    return SceneOutput(
                        source=ObjectRef(
                            path=ObjectStore(inputs.root).source(
                                inputs.scene.plan.scene_id, inspection.code
                            ),
                            sha256=inspection.code_sha256,
                            kind="code",
                        ),
                        class_name=inspection.class_name,
                        diagnostics=inspection.diagnostics,
                    )
            if config.CODEGEN_MODE == "ir":
                return self._fallback_source(inputs, use_ir=True, reason="Explicit IR mode")
            rag_context, receipts = self._rag(inputs)
            try:
                generated = GenerationService(
                    agent=self._agent(self.coder_factory, inputs, "code"),
                    validator=validate_manim_code,
                    candidate_acceptor=self.acceptor,
                    generation_mode=config.GENERATION_MODE,
                    manim_version=inputs.render_profile.manim_version,
                    run_config=inputs.config,
                ).generate(
                    inputs.scene.plan,
                    feedback=feedback,
                    previous_code=previous_code,
                    stream=False,
                    renderer=inputs.render_profile.renderer,
                    continuity_bible=inputs.continuity_bible,
                    inherited_elements_code=inputs.scene.inherited_elements_code,
                    inherited_elements=inputs.scene.plan.inherited_elements,
                    elements_to_remove=inputs.scene.plan.elements_to_remove,
                    element_manifest=inputs.element_manifest,
                    technical_spec=inputs.scene.technical_spec,
                    rag_context=rag_context,
                    lesson_spec=inputs.lesson_spec,
                    teaching_graph=inputs.teaching_graph,
                    attempts_used=attempts_used,
                    reserve_attempt=reserve_attempt,
                )
            except Exception as exc:
                from kd1_anime.agent_runtime.tools import ToolCancelled
                from kd1_anime.control import OperationDeadlineExceeded, RunCancelled
                from kd1_anime.exceptions import ValidationError
                from kd1_anime.request_budget import RequestBudgetExceeded

                if isinstance(
                    exc,
                    (
                        RequestBudgetExceeded,
                        OperationDeadlineExceeded,
                        RunCancelled,
                        ToolCancelled,
                        ValidationError,
                        OSError,
                    ),
                ):
                    raise
                if config.CODEGEN_MODE == "python" and config.GENERATION_MODE != "relaxed":
                    raise
                if config.CODEGEN_MODE == "hybrid":
                    try:
                        return self._fallback_source(inputs, use_ir=True, reason=str(exc))
                    except SceneStageError:
                        pass
                return self._fallback_source(inputs, use_ir=False, reason=str(exc))
            return SceneOutput(
                source=ObjectRef(
                    path=ObjectStore(inputs.root).source(
                        inputs.scene.plan.scene_id, generated.code
                    ),
                    sha256=sha256_text(generated.code),
                    kind="code",
                ),
                class_name=generated.class_name,
                diagnostics=tuple(diagnostics),
                rag_receipts=receipts,
            )

    def inspect(self, inputs: SceneInputs) -> InspectionOutput:
        with inputs.config.activate():
            return self._inspection_output(
                inputs, self._inspect(inputs, self._source(inputs), allow_repair=True)
            )

    @staticmethod
    def _precise_source(code, result):
        """Resolve all nonoverlapping replacements against the original source."""
        edits = []
        for fix in result.fixes:
            if fix.find == fix.replace:
                continue
            if code.count(fix.find) != 1:
                return None
            start = code.index(fix.find)
            edits.append((start, start + len(fix.find), fix.replace))
        edits.sort()
        if not edits or any(left[1] > right[0] for left, right in pairwise(edits)):
            return None
        for start, end, replacement in reversed(edits):
            code = code[:start] + replacement + code[end:]
        return code

    @staticmethod
    def _automatic_fix(result, *, require_confirmation):
        return (
            not require_confirmation
            and result.severity == "minor"
            and bool(result.findings)
            and all(
                item.severity == "minor"
                and item.evidence_type == "deterministic"
                and item.confidence == "high"
                for item in result.findings
            )
        )

    def _approve_source(self, inputs, proposal, base_code):
        if proposal.identity != WorkIdentity(inputs.scene.revision, inputs.scene.input_sha256):
            raise ValueError("Candidate work identity mismatch")
        if proposal.bound_digest() != proposal.diff_sha256:
            raise ValueError("Candidate diff identity mismatch")
        if proposal.base_source.sha256 != sha256_text(base_code):
            raise ValueError("Candidate base source identity mismatch")
        if proposal.context_sha256 != self._context_digest(inputs):
            raise ValueError("Candidate context identity mismatch")
        self._read_ref(inputs, proposal.base_source)
        inspection_report = json.loads(self._read_ref(inputs, proposal.inspection_report))
        review_report = json.loads(self._read_ref(inputs, proposal.review_report))
        if (
            review_report.get("kind") != "code_review"
            or review_report.get("code_sha256") != proposal.base_source.sha256
        ):
            raise ValueError("Candidate review identity mismatch")
        if (
            inspection_report.get("kind") != "candidate_inspection"
            or inspection_report.get("code_sha256") != proposal.candidate_source.sha256
            or inspection_report.get("technical_spec_sha256") != inputs.scene.technical_spec_sha256
            or inspection_report.get("accepted") is not True
        ):
            raise ValueError("Candidate inspection identity mismatch")
        return self._read_ref(inputs, proposal.candidate_source)

    def accept_proposal(self, inputs: SceneInputs, proposal: CandidateProposal) -> InspectionOutput:
        """Readonly diff acceptance after the coordinator consumes its exact grant.

        No model is called here. The installed source remains unreviewed until a
        separate ReviewCode action reviews these exact bytes.
        """
        with inputs.config.activate():
            code = self._approve_source(inputs, proposal, self._source(inputs))
            return self._inspection_output(
                inputs,
                self._inspect(
                    inputs,
                    code,
                    allow_repair=False,
                    expected_class_name=inputs.scene.class_name or None,
                ),
            )

    def review(
        self,
        inputs: SceneInputs,
        *,
        review_round=None,
        review_signature=None,
        identical_review_count=None,
        identity: WorkIdentity | None = None,
        require_confirmation=False,
    ) -> CodeReviewOutput:
        """Return immutable review evidence and routing, never mutate the scene."""
        with inputs.config.activate():
            config = inputs.config.resolve()
            secrets = self._secrets()
            scene = inputs.scene
            rounds = scene.review_round if review_round is None else review_round
            signature = scene.review_signature if review_signature is None else review_signature
            identical = (
                scene.identical_review_count
                if identical_review_count is None
                else identical_review_count
            )
            budget = review_budget(
                scene.plan,
                scene.technical_spec,
                global_max_rounds=config.MAX_REVIEW_ROUNDS,
                low_risk_max_rounds=config.MAX_LOW_RISK_REVIEW_ROUNDS,
                generation_mode=config.GENERATION_MODE,
            ).max_rounds
            base_code = self._source(inputs)
            code = base_code
            diagnostics = []
            source_changed = code != base_code
            while True:
                raw_inspection = self._inspect(
                    inputs, code, allow_repair=False, expected_class_name=scene.class_name or None
                )
                inspection = self._inspection_output(inputs, raw_inspection)
                diagnostics.extend(inspection.diagnostics)
                if not inspection.accepted:
                    result = ReviewResult(
                        is_valid=False, severity="major", feedback=raw_inspection.feedback
                    )
                    route = "coder"
                elif rounds >= budget or identical >= config.MAX_IDENTICAL_REVIEW_ATTEMPTS:
                    result = ReviewResult(
                        is_valid=False,
                        severity="major",
                        feedback="Finite source review budget exhausted",
                    )
                    route = "exhausted"
                else:
                    reviewed = ReviewService.review(
                        self._agent(self.reviewer_factory, inputs, "code_review"),
                        code,
                        scene.plan,
                        {
                            "renderer": inputs.render_profile.renderer,
                            "technical_spec": scene.technical_spec,
                            "continuity_bible": inputs.continuity_bible,
                            "inherited_elements_code": scene.inherited_elements_code,
                            "lesson_spec": inputs.lesson_spec,
                            "generation_mode": config.GENERATION_MODE,
                        },
                    )
                    result = reviewed.result
                    rounds += 1
                    next_signature = sha256_text(code + "\n" + result.model_dump_json())[:16]
                    identical = identical + 1 if signature == next_signature else 1
                    signature = next_signature
                    diagnostics.extend(
                        diagnostic_from_review_finding(
                            item,
                            secrets=secrets,
                            scene_id=scene.plan.scene_id,
                            code_sha256=sha256_text(code),
                            run_id=inputs.run_id,
                            revision=scene.revision,
                        )
                        for item in result.findings
                    )
                    route = (
                        "none"
                        if accepts_review(
                            result,
                            relaxed=config.GENERATION_MODE == "relaxed",
                            allow_soft_pass=True,
                        )
                        else "coder"
                    )
                report = self._report(
                    inputs,
                    {
                        "kind": "code_review",
                        "code_sha256": sha256_text(code),
                        "result": result.model_dump(mode="json"),
                        "inspection": inspection.report.model_dump(mode="json"),
                        "review_round": rounds,
                        "review_signature": signature,
                        "identical_review_count": identical,
                    },
                )
                common = dict(
                    code_sha256=sha256_text(code),
                    report=report,
                    result=ReviewResult.model_validate(sanitize_record(result, secrets)),
                    diagnostics=tuple(diagnostics),
                    feedback=sanitize_record(result.feedback, secrets),
                    review_round=rounds,
                    review_signature=signature,
                    identical_review_count=identical,
                )
                if route == "none":
                    return CodeReviewOutput(
                        accepted=True,
                        source=inspection.source if source_changed else None,
                        class_name=inspection.class_name,
                        inspection=inspection,
                        **common,
                    )
                plan_finding = next(
                    (
                        item
                        for item in result.findings
                        if item.severity == "major"
                        and item.category in {"math", "continuity"}
                        and review_finding_requires_plan_repair(item)
                    ),
                    None,
                )
                if plan_finding is not None:
                    common["feedback"] = sanitize_record(
                        plan_finding.repair or plan_finding.why or result.feedback, secrets
                    )
                    return CodeReviewOutput(accepted=False, repair_route="plan", **common)
                if route == "exhausted" or not inspection.accepted:
                    return CodeReviewOutput(accepted=False, repair_route=route, **common)
                if rounds >= budget or identical >= config.MAX_IDENTICAL_REVIEW_ATTEMPTS:
                    return CodeReviewOutput(accepted=False, repair_route="exhausted", **common)
                if source_changed:
                    # A rejected automatic repair is never installed. Any next
                    # diff must be proposed against the durable base, not against
                    # an uncommitted intermediate candidate.
                    return CodeReviewOutput(accepted=False, repair_route="coder", **common)
                candidate_code = self._precise_source(code, result)
                if candidate_code is not None:
                    candidate_raw = self._inspect(
                        inputs,
                        candidate_code,
                        allow_repair=True,
                        expected_class_name=scene.class_name or None,
                    )
                    candidate = self._inspection_output(inputs, candidate_raw)
                    if candidate.accepted:
                        if self._automatic_fix(result, require_confirmation=require_confirmation):
                            # The loop checks the final repaired bytes readonly and
                            # calls the Reviewer again before publishing any change.
                            code, source_changed = candidate_raw.code, True
                            continue
                        proposal = CandidateProposal(
                            identity=identity or WorkIdentity(scene.revision, scene.input_sha256),
                            base_source=ObjectRef(
                                path=scene.code_file, sha256=scene.code_sha256, kind="code"
                            ),
                            candidate_source=candidate.source,
                            inspection_report=candidate.report,
                            review_report=report,
                            context_sha256=self._context_digest(inputs),
                            diff_sha256="0" * 64,
                        )
                        proposal = proposal.model_copy(
                            update={"diff_sha256": proposal.bound_digest()}
                        )
                        return CodeReviewOutput(
                            accepted=False, repair_route="approval", proposal=proposal, **common
                        )
                    common["diagnostics"] = (*diagnostics, *candidate.diagnostics)
                return CodeReviewOutput(accepted=False, repair_route="coder", **common)

    def rollback(self, inputs: SceneInputs) -> CodeReviewOutput:
        """Select a still-bound proven candidate, then inspect and Review it anew."""
        with inputs.config.activate():
            rounds = inputs.scene.review_round
            signature = inputs.scene.review_signature
            identical = inputs.scene.identical_review_count
            last_output = None
            for candidate in sorted(inputs.scene.candidates, key=candidate_rank, reverse=True):
                if (
                    candidate.code_sha256 == inputs.scene.code_sha256
                    or candidate.verification not in {"smoke", "rendered"}
                    or candidate.inherited_elements_sha256
                    != sha256_text(inputs.scene.inherited_elements_code)
                ):
                    continue
                try:
                    code = self._read_ref(
                        inputs,
                        ObjectRef(
                            path=candidate.code_file, sha256=candidate.code_sha256, kind="code"
                        ),
                    )
                    inspected = self._inspect(
                        inputs, code, allow_repair=False, expected_class_name=candidate.class_name
                    )
                except (OSError, UnicodeError, ValueError):
                    continue
                if not inspected.accepted:
                    continue
                snapshot = inputs.scene.model_copy(
                    update={
                        "code_file": candidate.code_file,
                        "code_sha256": candidate.code_sha256,
                        "class_name": candidate.class_name,
                    },
                    deep=True,
                )
                output = self.review(
                    replace(inputs, scene=snapshot),
                    review_round=rounds,
                    review_signature=signature,
                    identical_review_count=identical,
                )
                rounds, signature, identical = (
                    output.review_round,
                    output.review_signature,
                    output.identical_review_count,
                )
                last_output = output
                if output.accepted:
                    return output.model_copy(
                        update={
                            "source": output.source
                            or ObjectRef(
                                path=candidate.code_file, sha256=candidate.code_sha256, kind="code"
                            ),
                            "class_name": output.class_name or candidate.class_name,
                        }
                    )
                if output.repair_route in {"exhausted", "plan"}:
                    return output
            if last_output is not None:
                # A rejected historical candidate is not the durable base for a
                # new approval. Preserve consumed counters/evidence, not its diff.
                return last_output.model_copy(
                    update={
                        "proposal": None,
                        "repair_route": "coder"
                        if last_output.repair_route == "approval"
                        else last_output.repair_route,
                    }
                )
            raise SceneStageError(
                "No matching inspected and reviewed rollback candidate is available"
            )
