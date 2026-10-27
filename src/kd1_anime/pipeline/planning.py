"""Bounded single-scene planning stages over public agents and pure validators."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from typing import Literal

from kd1_anime.agents.capability import build_capability_contract, validate_capability_contract
from kd1_anime.agents.continuity import (
    deterministic_continuity_issues,
    normalize_scene_plan_contract,
)
from kd1_anime.agents.plan_compiler import PlanCompiler, normalize_scene_timeline_contract
from kd1_anime.agents.plan_reviewer import (
    classify_plan_review_issues,
    deterministic_plan_issues,
)
from kd1_anime.agents.technical_planner import (
    compile_technical_spec,
    normalize_technical_spec_contract,
)
from kd1_anime.config import settings
from kd1_anime.control import RunControl
from kd1_anime.domain.diagnostics import Diagnostic, diagnostic_from_plan_issue
from kd1_anime.domain.plan_review import ContinuityIssue
from kd1_anime.domain.planning import PlanningDraft
from kd1_anime.domain.runtime import RunConfig
from kd1_anime.models import validated_copy
from kd1_anime.pipeline.contracts import PlanningOutput, SceneInputs, TechnicalOutput
from kd1_anime.request_budget import RequestContext
from kd1_anime.services.planning import PlanningService

Renderer = Literal["cairo", "opengl"]


class PlanningStageError(RuntimeError):
    """A stage cannot advance; evidence is available for the driver's durable stop."""

    def __init__(
        self,
        message: str,
        *,
        diagnostics: tuple[Diagnostic, ...],
        output: PlanningOutput | TechnicalOutput | None = None,
    ) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics
        self.output = output


class UnsupportedPlanningError(PlanningStageError):
    """The current vertical slice cannot execute this planning shape."""


def _report(
    code: str,
    summary: str,
    evidence: dict,
    *,
    severity: Literal["info", "minor", "major"] = "info",
    scene_id: int | None = None,
    stage: Literal["plan", "continuity"] = "plan",
    category: str = "contract",
) -> Diagnostic:
    return Diagnostic(
        code=code,
        stage=stage,
        category=category,
        severity=severity,
        confidence="high",
        scene_id=scene_id,
        summary=summary,
        evidence=evidence,
        repair_target="continuity" if stage == "continuity" else "planner",
    )


def _continuity_diagnostic(issue: ContinuityIssue, *, deterministic: bool) -> Diagnostic:
    return Diagnostic(
        code="planning.continuity_issue",
        stage="continuity",
        category=issue.category,
        severity=issue.severity,
        confidence="high" if deterministic else "unknown",
        scene_id=issue.scene_ids[0] if len(issue.scene_ids) == 1 else None,
        summary=issue.message,
        hint=issue.fix_instruction,
        evidence={"issue": issue.model_dump(mode="json"), "deterministic": deterministic},
        repair_target="continuity",
        suggested_action="revise_plan",
    )


class PlanningPipeline:
    """No Context, orchestration methods, retries, replanning, or code generation."""

    def __init__(
        self,
        planner_factory: Callable,
        plan_reviewer_factory: Callable,
        continuity_reviewer_factory: Callable,
        technical_factory: Callable,
        *,
        config: RunConfig,
        request_context: Callable[[int | None, str], RequestContext],
        control: RunControl,
    ) -> None:
        self._planner_factory = planner_factory
        self._plan_reviewer_factory = plan_reviewer_factory
        self._continuity_reviewer_factory = continuity_reviewer_factory
        self._technical_factory = technical_factory
        self._config = config
        self._request_context = request_context
        self._control = control

    def _agent(self, factory: Callable, scene_id: int | None, stage: str):
        self._control.check()
        agent = factory()
        agent.control = self._control
        agent.request_context = self._request_context(scene_id, stage)
        return agent

    @staticmethod
    def _require_single_scene(scene_ids: list[int], *, evidence: dict | None = None) -> None:
        if scene_ids != [1]:
            diagnostic = _report(
                "planning.unsupported_multiple_scenes",
                "Agent planning currently supports exactly one scene with scene_id=1; no legacy fallback is permitted.",
                {"scene_ids": scene_ids, **(evidence or {})},
                severity="major",
            )
            raise UnsupportedPlanningError(diagnostic.summary, diagnostics=(diagnostic,))

    def propose(self, user_prompt: str, renderer: Renderer) -> PlanningOutput:
        with self._config.activate():
            planner = self._agent(self._planner_factory, None, "planning_draft")
            draft = planner.plan_draft(user_prompt)
            if not isinstance(draft, PlanningDraft):
                raise TypeError("Planner must return PlanningDraft")
            draft = validated_copy(draft, deep=True)
            self._require_single_scene(
                [outline.scene_id for outline in draft.items],
                evidence={"draft": draft.model_dump(mode="json")},
            )
            planner = self._agent(self._planner_factory, None, "planning_bible")
            bible = planner.plan_continuity_bible(
                user_prompt,
                [validated_copy(outline, deep=True) for outline in draft.items],
                renderer=renderer,
                stream=False,
                lesson_spec=validated_copy(draft.lesson_spec, deep=True),
                teaching_graph=validated_copy(draft.teaching_graph, deep=True),
            )
            bible = validated_copy(bible, deep=True)
            planner = self._agent(self._planner_factory, 1, "planning_detail")
            planned = PlanningService.detail(
                planner,
                draft.items[0],
                draft.items,
                user_prompt,
                {
                    "renderer": renderer,
                    "stream": False,
                    "continuity_bible": bible,
                    "lesson_spec": draft.lesson_spec,
                    "teaching_graph": draft.teaching_graph,
                },
            )
            proposed_plan = planned.plan.model_dump(mode="json")
            plan, continuity_repairs = normalize_scene_plan_contract(
                planned.plan,
                bible,
                has_next_scene=False,
            )
            plan, timeline_repairs = normalize_scene_timeline_contract(plan)
            diagnostics = ()
            if continuity_repairs or timeline_repairs:
                diagnostics = (
                    _report(
                        "planning.normalization",
                        "Applied deterministic scene contract normalization",
                        {
                            "continuity_repairs": list(continuity_repairs),
                            "timeline_repairs": list(timeline_repairs),
                            "proposed_plan": proposed_plan,
                        },
                        scene_id=1,
                    ),
                )
            return PlanningOutput(
                outlines=tuple(validated_copy(item, deep=True) for item in draft.items),
                plans=(validated_copy(plan, deep=True),),
                lesson_spec=validated_copy(draft.lesson_spec, deep=True),
                teaching_graph=validated_copy(draft.teaching_graph, deep=True),
                continuity_bible=bible,
                diagnostics=diagnostics,
            )

    def review(
        self,
        output: PlanningOutput,
        renderer: Renderer,
        *,
        use_models: bool = True,
    ) -> PlanningOutput:
        self._require_single_scene([item.scene_id for item in output.plans])
        with self._config.activate():
            self._control.check()
            snapshot = validated_copy(output, deep=True)
            plans = list(snapshot.plans)
            bible = snapshot.continuity_bible
            if bible is None:
                diagnostic = _report(
                    "planning.missing_bible",
                    "Planning review requires a continuity bible",
                    {},
                    severity="major",
                )
                raise PlanningStageError(
                    diagnostic.summary,
                    diagnostics=(*snapshot.diagnostics, diagnostic),
                    output=snapshot,
                )
            compiled = PlanCompiler().compile(
                list(snapshot.outlines),
                plans,
                bible,
                lesson_spec=snapshot.lesson_spec,
                teaching_graph=snapshot.teaching_graph,
            )
            local_plan_issues = deterministic_plan_issues(
                plans[0],
                bible,
                lesson_spec=snapshot.lesson_spec,
            )
            local_continuity = deterministic_continuity_issues(
                plans,
                bible,
                snapshot.lesson_spec,
                snapshot.teaching_graph,
            )
            diagnostics = [
                *snapshot.diagnostics,
                _report(
                    "planning.compiler",
                    "Deterministic plan compiler result",
                    compiled.model_dump(mode="json"),
                    severity="major"
                    if any(issue.severity == "major" for issue in compiled.issues)
                    else "info",
                ),
            ]
            for issue in compiled.issues:
                diagnostics.append(
                    _report(
                        "planning.compiler_issue",
                        issue.message,
                        issue.model_dump(mode="json"),
                        severity=issue.severity,
                        scene_id=issue.scene_ids[0] if len(issue.scene_ids) == 1 else None,
                        category=issue.category,
                    )
                )
            diagnostics.extend(
                diagnostic_from_plan_issue(issue, scene_id=1) for issue in local_plan_issues
            )
            diagnostics.extend(
                _continuity_diagnostic(issue, deterministic=True) for issue in local_continuity
            )
            # Deterministic major findings cannot be downgraded by model votes,
            # relaxed mode, or the model-only false-positive classifier.
            if any(item.severity == "major" for item in diagnostics):
                rejected = snapshot.model_copy(update={"diagnostics": tuple(diagnostics)})
                raise PlanningStageError(
                    "Deterministic planning validation failed",
                    diagnostics=rejected.diagnostics,
                    output=rejected,
                )
            if not use_models:
                return snapshot.model_copy(update={"diagnostics": tuple(diagnostics)})
            reviewer = self._agent(self._plan_reviewer_factory, 1, "plan_review")
            plan_review = reviewer.review(
                validated_copy(plans[0], deep=True),
                all_plans=[validated_copy(plan, deep=True) for plan in plans],
                continuity_bible=validated_copy(bible, deep=True),
                deterministic_issues=[
                    validated_copy(issue, deep=True) for issue in local_plan_issues
                ],
                renderer=renderer,
                lesson_spec=validated_copy(snapshot.lesson_spec, deep=True),
                teaching_graph=validated_copy(snapshot.teaching_graph, deep=True),
                generation_mode=settings.GENERATION_MODE,
            )
            effective, blocking, _ = classify_plan_review_issues(
                plans[0],
                deterministic_issues=[],
                result=plan_review,
            )
            diagnostics.append(
                _report(
                    "planning.plan_review",
                    "Independent plan review result",
                    plan_review.model_dump(mode="json"),
                    severity="major" if blocking else "minor" if effective else "info",
                    scene_id=1,
                )
            )
            diagnostics.extend(diagnostic_from_plan_issue(issue, scene_id=1) for issue in effective)
            reviewer = self._agent(self._continuity_reviewer_factory, None, "continuity_review")
            continuity_review = reviewer.review(
                validated_copy(bible, deep=True),
                [validated_copy(item, deep=True) for item in snapshot.outlines],
                [validated_copy(plan, deep=True) for plan in plans],
                deterministic_issues=[
                    validated_copy(issue, deep=True) for issue in local_continuity
                ],
                renderer=renderer,
                stream=False,
                lesson_spec=validated_copy(snapshot.lesson_spec, deep=True),
                teaching_graph=validated_copy(snapshot.teaching_graph, deep=True),
            )
            continuity_blocking = any(
                issue.severity == "major" for issue in continuity_review.issues
            )
            diagnostics.append(
                _report(
                    "planning.continuity_review",
                    "Independent continuity review result",
                    continuity_review.model_dump(mode="json"),
                    severity="major"
                    if continuity_blocking
                    else "minor"
                    if continuity_review.issues
                    else "info",
                    stage="continuity",
                )
            )
            diagnostics.extend(
                _continuity_diagnostic(issue, deterministic=False)
                for issue in continuity_review.issues
            )
            result = snapshot.model_copy(update={"diagnostics": tuple(diagnostics)})
            if blocking or continuity_blocking:
                raise PlanningStageError(
                    "Independent planning review failed",
                    diagnostics=result.diagnostics,
                    output=result,
                )
            return result

    def technical(self, inputs: SceneInputs) -> TechnicalOutput:
        self._require_single_scene([item.scene_id for item in inputs.plans])
        with inputs.config.activate():
            scene = validated_copy(inputs.scene, deep=True)
            self._require_single_scene([scene.plan.scene_id])
            renderer = inputs.render_profile.renderer
            agent = self._agent(self._technical_factory, scene.plan.scene_id, "technical")
            spec = agent.plan(
                validated_copy(scene.plan, deep=True),
                continuity_bible=validated_copy(inputs.continuity_bible, deep=True)
                if inputs.continuity_bible is not None
                else None,
                renderer=renderer,
                stream=False,
                lesson_spec=validated_copy(inputs.lesson_spec, deep=True),
                teaching_graph=validated_copy(inputs.teaching_graph, deep=True),
            )
            proposed_spec = spec.model_dump(mode="json")
            spec, repairs = normalize_technical_spec_contract(
                scene.plan,
                validated_copy(spec, deep=True),
                renderer=renderer,
            )
            compiled = compile_technical_spec(scene.plan, spec, renderer=renderer)
            capability = build_capability_contract(
                scene.plan,
                spec,
                renderer=renderer,
                render_profile=inputs.render_profile,
                gpu_type=settings.SLURM_GPU_TYPE,
            )
            capability_result = validate_capability_contract(
                capability,
                render_profile=inputs.render_profile,
                gpu_type=settings.SLURM_GPU_TYPE,
            )
            diagnostics = (
                _report(
                    "planning.technical_normalization",
                    "Technical contract normalization result",
                    {"repairs": list(repairs), "proposed_spec": proposed_spec},
                    scene_id=scene.plan.scene_id,
                ),
                _report(
                    "planning.technical_compiler",
                    "Deterministic technical compiler result",
                    asdict(compiled),
                    severity="info" if compiled.is_valid else "major",
                    scene_id=scene.plan.scene_id,
                ),
                _report(
                    "planning.capability",
                    "Deterministic capability validation result",
                    {
                        "contract": capability.model_dump(mode="json"),
                        "validation": capability_result.model_dump(mode="json"),
                    },
                    severity="info" if capability_result.is_valid else "major",
                    scene_id=scene.plan.scene_id,
                    category="renderer",
                ),
            )
            result = TechnicalOutput(spec=validated_copy(spec, deep=True), diagnostics=diagnostics)
            if not compiled.is_valid or not capability_result.is_valid:
                raise PlanningStageError(
                    "Technical specification validation failed",
                    diagnostics=diagnostics,
                    output=result,
                )
            return result


__all__ = ["PlanningPipeline", "PlanningStageError", "UnsupportedPlanningError"]
