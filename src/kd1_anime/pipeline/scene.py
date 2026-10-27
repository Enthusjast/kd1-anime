"""Single-scene generation, deterministic inspection and evidence-bound review."""

from __future__ import annotations

import ast
import json
from collections.abc import Callable

from kd1_anime.agent_runtime.contracts import ObjectRef, RuntimeRecord, sanitize_record, utc_now
from kd1_anime.candidate_acceptor import CandidateAcceptor
from kd1_anime.domain.diagnostics import Diagnostic
from kd1_anime.domain.planning import ExtractedElement
from kd1_anime.objects import ObjectStore
from kd1_anime.pipeline.contracts import SceneInputs, SceneOutput
from kd1_anime.rendering import sha256_file
from kd1_anime.run_store import restore_run_path, sha256_text
from kd1_anime.services.policies import accepts_review
from kd1_anime.services.reviewing import ReviewService
from kd1_anime.verification import StaticVerification


class SceneStageError(RuntimeError):
    """An unsupported repair or failed stage, never a legacy fallback request."""


class InspectionOutput(RuntimeRecord):
    source: ObjectRef
    class_name: str
    accepted: bool
    verification: StaticVerification
    report: ObjectRef
    exported_elements_code: str = ""
    exported_elements: tuple[ExtractedElement, ...] = ()
    diagnostics: tuple[Diagnostic, ...] = ()


class CodeReviewOutput(RuntimeRecord):
    accepted: bool
    code_sha256: str
    report: ObjectRef
    diagnostics: tuple[Diagnostic, ...] = ()


class ScenePipeline:
    def __init__(self, coder_factory, reviewer_factory, *, request_context: Callable, control):
        self.coder_factory = coder_factory
        self.reviewer_factory = reviewer_factory
        self.request_context = request_context
        self.control = control
        self.acceptor = CandidateAcceptor()

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
    def _report(inputs, payload):
        text = json.dumps(sanitize_record(payload), ensure_ascii=False, sort_keys=True)
        return ObjectRef(
            path=ObjectStore(inputs.root).report(text), sha256=sha256_text(text), kind="report"
        )

    @staticmethod
    def _source(inputs):
        path = restore_run_path(inputs.root, inputs.scene.code_file)
        if not path.is_file() or sha256_file(path) != inputs.scene.code_sha256:
            raise ValueError("Scene source identity mismatch")
        return path.read_text(encoding="utf-8")

    def generate(self, inputs: SceneInputs) -> SceneOutput:
        agent = self._agent(self.coder_factory, inputs, "code")
        code = agent.generate_code(
            inputs.scene.plan.model_copy(deep=True),
            renderer=inputs.render_profile.renderer,
            technical_spec=inputs.scene.technical_spec,
            continuity_bible=inputs.continuity_bible,
            inherited_elements_code=inputs.scene.inherited_elements_code,
            lesson_spec=inputs.lesson_spec,
            teaching_graph=inputs.teaching_graph,
            stream=False,
        )
        if not isinstance(code, str):
            raise SceneStageError("Coder did not return source text")
        path = ObjectStore(inputs.root).source(inputs.scene.plan.scene_id, code)
        # Class discovery does not execute generated code. Inspection remains a
        # separate durable action and owns all safety/contract acceptance.
        try:
            classes = [node.name for node in ast.parse(code).body if isinstance(node, ast.ClassDef)]
        except SyntaxError:
            classes = []
        return SceneOutput(
            source=ObjectRef(path=path, sha256=sha256_text(code), kind="code"),
            class_name=classes[0] if classes else "",
        )

    def inspect(self, inputs: SceneInputs) -> InspectionOutput:
        self._checkpoint()
        inspection = self.acceptor.inspect_candidate(
            self._source(inputs),
            inputs.scene.plan,
            technical_spec=inputs.scene.technical_spec,
            renderer=inputs.render_profile.renderer,
            manim_version=inputs.render_profile.manim_version,
            allow_repair=True,
        )
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
                error=inspection.feedback,
            ),
            report=report,
            exported_elements_code=inspection.exported_elements_code,
            exported_elements=inspection.exported_elements,
            diagnostics=inspection.diagnostics,
        )

    def review(self, inputs: SceneInputs) -> CodeReviewOutput:
        agent = self._agent(self.reviewer_factory, inputs, "code_review")
        reviewed = ReviewService.review(
            agent,
            self._source(inputs),
            inputs.scene.plan,
            {
                "renderer": inputs.render_profile.renderer,
                "technical_spec": inputs.scene.technical_spec,
                "continuity_bible": inputs.continuity_bible,
                "lesson_spec": inputs.lesson_spec,
                "generation_mode": inputs.config.resolve().GENERATION_MODE,
            },
        )
        report = self._report(
            inputs,
            {
                "kind": "code_review",
                "code_sha256": reviewed.code_sha256,
                "result": reviewed.result.model_dump(mode="json"),
            },
        )
        accepted = accepts_review(reviewed.result, relaxed=False, allow_soft_pass=False)
        return CodeReviewOutput(accepted=accepted, code_sha256=reviewed.code_sha256, report=report)
