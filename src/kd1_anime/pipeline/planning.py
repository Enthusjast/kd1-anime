"""Bounded planning actions over public agents and deterministic domain contracts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from hashlib import sha256
from typing import Literal

from kd1_anime.agents.capability import build_capability_contract, validate_capability_contract
from kd1_anime.agents.continuity import deterministic_continuity_issues
from kd1_anime.agents.plan_compiler import PlanCompiler
from kd1_anime.agents.plan_reviewer import classify_plan_review_issues, deterministic_plan_issues
from kd1_anime.agents.review_policy import review_mode_policy
from kd1_anime.agents.technical_planner import (
    build_technical_handoff,
    compile_technical_spec,
    normalize_technical_spec_contract,
)
from kd1_anime.config import settings
from kd1_anime.control import OperationDeadlineExceeded, RunCancelled, RunControl
from kd1_anime.domain.diagnostics import Diagnostic, diagnostic_from_plan_issue
from kd1_anime.domain.plan_review import ContinuityIssue
from kd1_anime.domain.planning import PlanningDraft
from kd1_anime.domain.runtime import RunConfig
from kd1_anime.models import validated_copy
from kd1_anime.pipeline.contracts import (
    DetailInputs,
    DetailOutput,
    PlanningOutput,
    SceneInputs,
    TechnicalOutput,
)
from kd1_anime.rag.models import RagReceipt, RagRuntimeProfile
from kd1_anime.rag.service import RagService
from kd1_anime.request_budget import RequestBudgetExceeded, RequestContext
from kd1_anime.services.planning import PlanningService

Renderer = Literal["cairo", "opengl"]


class PlanningStageError(RuntimeError):
    """A barrier failed; its immutable evidence remains available to the driver."""

    def __init__(
        self,
        message: str,
        *,
        diagnostics: tuple[Diagnostic, ...],
        output: PlanningOutput | DetailOutput | TechnicalOutput | None = None,
        repairable: bool = False,
    ) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics
        self.output = output
        self.repairable = repairable


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
        suggested_action="revise_plan" if severity == "major" else None,
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
    """Outline, independent Detail, whole-plan review, and ordered technical actions.

    Each real call owns an agent/request context. No action receives a mutable run
    Context, shares a model client with a sibling, or publishes a shared ledger.
    """

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
        rag_service: RagService | None = None,
        rag_profile: RagRuntimeProfile | None = None,
    ) -> None:
        self._planner_factory = planner_factory
        self._plan_reviewer_factory = plan_reviewer_factory
        self._continuity_reviewer_factory = continuity_reviewer_factory
        self._technical_factory = technical_factory
        self._config = config
        self._request_context = request_context
        self._control = control
        resolved = config.resolve()
        self._rag = (
            rag_service
            if rag_service is not None
            else (RagService(resolved) if resolved.RAG_ENABLED else None)
        )
        self._rag_profile = (
            validated_copy(rag_profile, deep=True)
            if rag_profile is not None
            else self._rag.runtime_profile()
            if self._rag is not None
            else None
        )

    def _agent(self, factory: Callable, scene_id: int | None, stage: str):
        self._control.check()
        agent = factory()
        agent.control = self._control
        agent.request_context = self._request_context(scene_id, stage)
        return agent

    def _retrieve(
        self,
        query: str,
        stage: str,
        *,
        supplied: str = "",
        profile: RagRuntimeProfile | None = None,
        inherited_elements_sha256: str = "",
    ) -> tuple[str, tuple[RagReceipt, ...], tuple[Diagnostic, ...], RagRuntimeProfile | None]:
        expected = profile or self._rag_profile
        if self._rag is None:
            if settings.RAG_ENABLED or (expected is not None and expected.enabled):
                diagnostic = _report(
                    "planning.rag_unavailable",
                    "Enabled RAG requires its fixed service",
                    {},
                    severity="major",
                )
                raise PlanningStageError(diagnostic.summary, diagnostics=(diagnostic,))
            return supplied, (), (), None
        self._control.check()
        try:
            self._rag.require_ready()
            current = self._rag.runtime_profile()
        except (RunCancelled, OperationDeadlineExceeded, RequestBudgetExceeded):
            raise
        except Exception as exc:
            diagnostic = _report("planning.rag_unavailable", str(exc), {}, severity="major")
            raise PlanningStageError(diagnostic.summary, diagnostics=(diagnostic,)) from exc
        if expected is not None and current.model_dump(exclude={"status"}) != expected.model_dump(
            exclude={"status"}
        ):
            diagnostic = _report(
                "planning.rag_profile_mismatch",
                "RAG index/model/chunker differs from the fixed run profile",
                {
                    "expected": expected.model_dump(mode="json"),
                    "actual": current.model_dump(mode="json"),
                },
                severity="major",
            )
            raise PlanningStageError(diagnostic.summary, diagnostics=(diagnostic,))
        if settings.RAG_ENABLED and not current.enabled:
            diagnostic = _report(
                "planning.rag_unavailable",
                "Enabled RAG cannot use a disabled retrieval service",
                {},
                severity="major",
            )
            raise PlanningStageError(diagnostic.summary, diagnostics=(diagnostic,))
        result = self._rag.search(
            query[:50_000],
            stage=stage,
            source_kinds={"manim_doc", "example", "recipe"},
            preferred_source_kinds={"recipe"} if stage == "technical" else None,
            inherited_elements_sha256=inherited_elements_sha256,
        )
        receipt = validated_copy(result.receipt, deep=True)
        if receipt.index_sha256 != current.index_sha256 and (
            receipt.index_sha256 or receipt.status == "active"
        ):
            diagnostic = _report(
                "planning.rag_profile_mismatch",
                "Retrieval receipt does not match the fixed index",
                {
                    "profile": current.model_dump(mode="json"),
                    "receipt": receipt.model_dump(mode="json"),
                },
                severity="major",
            )
            raise PlanningStageError(diagnostic.summary, diagnostics=(diagnostic,))
        diagnostics = ()
        if receipt.warning or receipt.status == "degraded":
            diagnostics = (
                _report(
                    "planning.rag_degraded",
                    receipt.warning or "RAG retrieval degraded",
                    receipt.model_dump(mode="json"),
                    severity="minor",
                ),
            )
        # Never associate a fresh receipt with caller-supplied, unbound RAG text.
        return result.context, (receipt,), diagnostics, current

    def propose(self, user_prompt: str, renderer: Renderer) -> PlanningOutput:
        """Fix global contracts only; Detail is a separate identity-bound action."""
        with self._config.activate():
            rag_context, receipts, diagnostics, profile = self._retrieve(user_prompt, "outline")
            planner = self._agent(self._planner_factory, None, "planning_draft")
            draft = planner.plan_draft(
                user_prompt, **({"rag_context": rag_context} if rag_context else {})
            )
            if not isinstance(draft, PlanningDraft):
                raise TypeError("Planner must return PlanningDraft")
            draft = validated_copy(draft, deep=True)
            ids = [item.scene_id for item in draft.items]
            if ids != list(range(1, len(ids) + 1)) or not ids:
                diagnostic = _report(
                    "planning.invalid_outlines",
                    "Outlines must have unique ordered scene IDs 1..N",
                    {"scene_ids": ids},
                    severity="major",
                )
                raise PlanningStageError(diagnostic.summary, diagnostics=(*diagnostics, diagnostic))
            bible_context, bible_receipts, bible_diagnostics, _ = self._retrieve(
                user_prompt + "\n" + "\n".join(item.model_dump_json() for item in draft.items),
                "continuity",
                profile=profile,
            )
            planner = self._agent(self._planner_factory, None, "planning_bible")
            bible = planner.plan_continuity_bible(
                user_prompt,
                [validated_copy(item, deep=True) for item in draft.items],
                renderer=renderer,
                stream=False,
                rag_context=bible_context,
                lesson_spec=validated_copy(draft.lesson_spec, deep=True),
                teaching_graph=validated_copy(draft.teaching_graph, deep=True),
            )
            return PlanningOutput(
                outlines=tuple(validated_copy(item, deep=True) for item in draft.items),
                plans=(),
                lesson_spec=validated_copy(draft.lesson_spec, deep=True),
                teaching_graph=validated_copy(draft.teaching_graph, deep=True),
                continuity_bible=validated_copy(bible, deep=True),
                diagnostics=(*diagnostics, *bible_diagnostics),
                rag_profile=profile,
                rag_receipts=(*receipts, *bible_receipts),
            )

    def detail(self, inputs: DetailInputs) -> DetailOutput:
        with inputs.config.activate():
            if inputs.continuity_bible is None:
                diagnostic = _report(
                    "planning.missing_bible",
                    "Detail requires a fixed continuity bible",
                    {},
                    severity="major",
                )
                raise PlanningStageError(diagnostic.summary, diagnostics=(diagnostic,))
            rag_context, receipts, diagnostics, _ = self._retrieve(
                "\n".join(
                    (inputs.outline.title, inputs.outline.purpose, inputs.outline.math_concept)
                ),
                "detail",
                supplied=inputs.rag_context,
            )
            agent = self._agent(self._planner_factory, inputs.outline.scene_id, "planning_detail")
            proposed = PlanningService.detail(
                agent,
                inputs.outline,
                inputs.outlines,
                inputs.user_prompt,
                {
                    "renderer": inputs.renderer,
                    "stream": False,
                    "continuity_bible": inputs.continuity_bible,
                    "continuity_context": inputs.continuity_context,
                    "rag_context": rag_context,
                    "lesson_spec": inputs.lesson_spec,
                    "teaching_graph": inputs.teaching_graph,
                },
            ).plan
            position = next(
                index
                for index, item in enumerate(inputs.outlines)
                if item.scene_id == inputs.outline.scene_id
            )
            plan, repairs = PlanningService.normalize(
                proposed,
                inputs.continuity_bible,
                inputs.lesson_spec,
                has_next_scene=position + 1 < len(inputs.outlines),
            )
            findings = PlanCompiler().compile_scene(
                plan, inputs.continuity_bible, inputs.lesson_spec
            )
            local = deterministic_plan_issues(
                plan, inputs.continuity_bible, lesson_spec=inputs.lesson_spec
            )
            reports = [*diagnostics]
            if repairs:
                reports.append(
                    _report(
                        "planning.normalization",
                        "Applied deterministic scene contract normalization",
                        {
                            "repairs": list(repairs),
                            "proposed_plan": proposed.model_dump(mode="json"),
                        },
                        scene_id=plan.scene_id,
                    )
                )
            reports.extend(
                _report(
                    "planning.compiler_issue",
                    issue.message,
                    issue.model_dump(mode="json"),
                    severity=issue.severity,
                    scene_id=plan.scene_id,
                    category=issue.category,
                )
                for issue in findings
            )
            reports.extend(
                diagnostic_from_plan_issue(issue, scene_id=plan.scene_id) for issue in local
            )
            result = DetailOutput(
                plan=validated_copy(plan, deep=True),
                diagnostics=tuple(reports),
                rag_receipts=receipts,
            )
            # Detail is an unapproved proposal. Keep its deterministic findings
            # for the mandatory full-plan gate, which routes repair through an
            # explicit RevisePlan action; no candidate reaches coding here.
            return result

    def _validate(self, output: PlanningOutput):
        """Normalize and compile a complete snapshot; historic failures are evidence only."""
        self._control.check()
        snapshot = validated_copy(output, deep=True)
        ids = [item.scene_id for item in snapshot.outlines]
        if (
            sorted(item.scene_id for item in snapshot.plans) != ids
            or len(set(ids)) != len(ids)
            or not ids
        ):
            diagnostic = _report(
                "planning.incomplete_detail",
                "Every outline must have exactly one completed Detail before review",
                {"outline_ids": ids, "plan_ids": [item.scene_id for item in snapshot.plans]},
                severity="major",
            )
            rejected = snapshot.model_copy(
                update={"diagnostics": (*snapshot.diagnostics, diagnostic)}
            )
            raise PlanningStageError(
                diagnostic.summary, diagnostics=rejected.diagnostics, output=rejected
            )
        if snapshot.continuity_bible is None:
            diagnostic = _report(
                "planning.missing_bible",
                "Planning review requires a continuity bible",
                {},
                severity="major",
            )
            rejected = snapshot.model_copy(
                update={"diagnostics": (*snapshot.diagnostics, diagnostic)}
            )
            raise PlanningStageError(
                diagnostic.summary, diagnostics=rejected.diagnostics, output=rejected
            )
        plans, repairs = PlanningService.normalize_plans(
            snapshot.plans, snapshot.continuity_bible, snapshot.lesson_spec
        )
        compiled = PlanCompiler().compile(
            list(snapshot.outlines),
            list(plans),
            snapshot.continuity_bible,
            lesson_spec=snapshot.lesson_spec,
            teaching_graph=snapshot.teaching_graph,
        )
        local = {
            plan.scene_id: deterministic_plan_issues(
                plan, snapshot.continuity_bible, lesson_spec=snapshot.lesson_spec
            )
            for plan in plans
        }
        continuity = deterministic_continuity_issues(
            list(plans), snapshot.continuity_bible, snapshot.lesson_spec, snapshot.teaching_graph
        )
        current = [
            _report(
                "planning.compiler",
                "Deterministic plan compiler result",
                compiled.model_dump(mode="json"),
                severity="major"
                if any(issue.severity == "major" for issue in compiled.issues)
                else "info",
            )
        ]
        current.extend(
            _report(
                "planning.normalization",
                "Applied deterministic scene contract normalization",
                {"repairs": list(messages)},
                scene_id=scene_id,
            )
            for scene_id, messages in repairs.items()
        )
        current.extend(
            _report(
                "planning.compiler_issue",
                issue.message,
                issue.model_dump(mode="json"),
                severity=issue.severity,
                scene_id=issue.scene_ids[0] if len(issue.scene_ids) == 1 else None,
                category=issue.category,
            )
            for issue in compiled.issues
        )
        current.extend(
            diagnostic_from_plan_issue(issue, scene_id=scene_id)
            for scene_id, issues in local.items()
            for issue in issues
        )
        current.extend(_continuity_diagnostic(issue, deterministic=True) for issue in continuity)
        result = snapshot.model_copy(
            update={"plans": plans, "diagnostics": (*snapshot.diagnostics, *current)}
        )
        if any(item.severity == "major" for item in current):
            raise PlanningStageError(
                "Deterministic planning validation failed",
                diagnostics=result.diagnostics,
                output=result,
                repairable=True,
            )
        return result, local, continuity

    def _revise_targets(
        self, output: PlanningOutput, feedback: dict[int, str], renderer: Renderer, user_prompt: str
    ) -> PlanningOutput:
        plans = {item.scene_id: validated_copy(item, deep=True) for item in output.plans}
        diagnostics = list(output.diagnostics)
        receipts = list(output.rag_receipts)
        for outline in output.outlines:
            if outline.scene_id not in feedback:
                continue
            used = sum(
                item.code == "planning.revision" and item.scene_id == outline.scene_id
                for item in output.diagnostics
            )
            if used >= settings.MAX_PLAN_REPLAN_ATTEMPTS:
                self._reject(
                    output,
                    "planning.replan_budget",
                    "Scene replan attempts exhausted",
                    {
                        "scene_id": outline.scene_id,
                        "used": used,
                        "limit": settings.MAX_PLAN_REPLAN_ATTEMPTS,
                    },
                )
            rag_context, fresh, rag_diagnostics, _ = self._retrieve(
                "\n".join((outline.title, outline.math_concept, feedback[outline.scene_id])),
                "detail",
                profile=output.rag_profile,
            )
            agent = self._agent(self._planner_factory, outline.scene_id, "planning_revise")
            proposed = PlanningService.detail(
                agent,
                outline,
                output.outlines,
                user_prompt,
                {
                    "renderer": renderer,
                    "stream": False,
                    "continuity_bible": output.continuity_bible,
                    "continuity_feedback": feedback[outline.scene_id],
                    "continuity_context": "\n".join(
                        item.model_dump_json() for item in output.plans
                    ),
                    "rag_context": rag_context,
                    "lesson_spec": output.lesson_spec,
                    "teaching_graph": output.teaching_graph,
                },
            ).plan
            plans[outline.scene_id] = proposed
            diagnostics.extend(rag_diagnostics)
            diagnostics.append(
                _report(
                    "planning.revision",
                    "Replanned scene from exact review feedback",
                    {
                        "feedback": feedback[outline.scene_id],
                        "proposed_plan": proposed.model_dump(mode="json"),
                    },
                    scene_id=outline.scene_id,
                )
            )
            receipts.extend(fresh)
        revised = output.model_copy(
            update={
                "plans": tuple(plans[item.scene_id] for item in output.outlines),
                "diagnostics": tuple(diagnostics),
                "rag_receipts": tuple(receipts),
            }
        )
        return self._validate(revised)[0]

    def revise(
        self, output: PlanningOutput, feedback: str, renderer: Renderer, *, user_prompt: str = ""
    ) -> PlanningOutput:
        """Explicit root repair; callers must invalidate affected technical/code consumers."""
        with self._config.activate():
            return self._revise_targets(
                validated_copy(output, deep=True),
                {item.scene_id: feedback for item in output.outlines},
                renderer,
                user_prompt,
            )

    def review(
        self,
        output: PlanningOutput,
        renderer: Renderer,
        *,
        use_models: bool = True,
        user_prompt: str = "",
    ) -> PlanningOutput:
        with self._config.activate():
            current, local, deterministic_continuity = self._validate(output)
            if not use_models:
                return current
            policy = review_mode_policy(settings.GENERATION_MODE)
            max_reviews = policy.limit(settings.MAX_PLAN_REVIEW_ROUNDS)
            max_replans = policy.limit(settings.MAX_PLAN_REPLAN_ATTEMPTS)
            max_continuity = policy.limit(settings.MAX_CONTINUITY_FIX_ROUNDS)
            replans = {
                item.scene_id: sum(
                    report.code == "planning.revision" and report.scene_id == item.scene_id
                    for report in current.diagnostics
                )
                for item in current.plans
            }
            reviews: dict[str, int] = {}
            for report in current.diagnostics:
                identity = report.evidence.get("input_sha256")
                if report.code == "planning.plan_review" and isinstance(identity, str):
                    reviews[identity] = reviews.get(identity, 0) + 1
            seen_cycles: set[str] = set()
            continuity_fixes = sum(
                item.code == "planning.continuity_repair" for item in current.diagnostics
            )
            # The outer guard is independent of per-scene budgets: plan and
            # continuity reviewers cannot bounce changing plans indefinitely.
            for cycle in range(4):
                reports = list(current.diagnostics)
                feedback: dict[int, list[str]] = {}
                shared_review_input = {
                    "plans": [item.model_dump(mode="json") for item in current.plans],
                    "lesson_spec": current.lesson_spec.model_dump(mode="json"),
                    "teaching_graph": current.teaching_graph.model_dump(mode="json"),
                    "continuity_bible": current.continuity_bible.model_dump(mode="json"),
                    "renderer": renderer,
                    "generation_mode": settings.GENERATION_MODE,
                }
                for plan in current.plans:
                    signature = PlanningService.cycle_signature(
                        {"scene_id": plan.scene_id, **shared_review_input}
                    )
                    if reviews.get(signature, 0) >= max_reviews:
                        self._reject(
                            current,
                            "planning.review_budget",
                            "Plan review rounds exhausted",
                            {"scene_id": plan.scene_id, "limit": max_reviews},
                        )
                    reviews[signature] = reviews.get(signature, 0) + 1
                    try:
                        reviewer = self._agent(
                            self._plan_reviewer_factory, plan.scene_id, "plan_review"
                        )
                        result = reviewer.review(
                            validated_copy(plan, deep=True),
                            user_prompt=user_prompt,
                            all_plans=[validated_copy(item, deep=True) for item in current.plans],
                            continuity_bible=validated_copy(current.continuity_bible, deep=True),
                            deterministic_issues=[
                                validated_copy(item, deep=True) for item in local[plan.scene_id]
                            ],
                            renderer=renderer,
                            lesson_spec=validated_copy(current.lesson_spec, deep=True),
                            teaching_graph=validated_copy(current.teaching_graph, deep=True),
                            generation_mode=settings.GENERATION_MODE,
                        )
                    except (RunCancelled, OperationDeadlineExceeded, RequestBudgetExceeded):
                        raise
                    except Exception as exc:
                        if not policy.relaxed:
                            diagnostic = _report(
                                "planning.review_error",
                                str(exc),
                                {},
                                severity="major",
                                scene_id=plan.scene_id,
                            )
                            rejected = current.model_copy(
                                update={"diagnostics": (*reports, diagnostic)}
                            )
                            raise PlanningStageError(
                                diagnostic.summary,
                                diagnostics=rejected.diagnostics,
                                output=rejected,
                            ) from exc
                        reports.append(
                            _report(
                                "planning.review_warning",
                                "Relaxed plan review unavailable",
                                {"error": str(exc)},
                                severity="minor",
                                scene_id=plan.scene_id,
                            )
                        )
                        continue
                    effective, blocking, _ = classify_plan_review_issues(
                        plan, deterministic_issues=[], result=result
                    )
                    reports.append(
                        _report(
                            "planning.plan_review",
                            "Independent plan review result",
                            {**result.model_dump(mode="json"), "input_sha256": signature},
                            severity="major" if blocking else "minor" if effective else "info",
                            scene_id=plan.scene_id,
                        )
                    )
                    reports.extend(
                        diagnostic_from_plan_issue(issue, scene_id=plan.scene_id)
                        for issue in effective
                    )
                    if blocking:
                        feedback[plan.scene_id] = [
                            f"{issue.message}\n{issue.fix_instruction}" for issue in blocking
                        ]
                continuity_issues = []
                if not policy.relaxed or deterministic_continuity:
                    try:
                        reviewer = self._agent(
                            self._continuity_reviewer_factory, None, "continuity_review"
                        )
                        result = reviewer.review(
                            validated_copy(current.continuity_bible, deep=True),
                            [validated_copy(item, deep=True) for item in current.outlines],
                            [validated_copy(item, deep=True) for item in current.plans],
                            deterministic_issues=[
                                validated_copy(item, deep=True) for item in deterministic_continuity
                            ],
                            renderer=renderer,
                            stream=False,
                            lesson_spec=validated_copy(current.lesson_spec, deep=True),
                            teaching_graph=validated_copy(current.teaching_graph, deep=True),
                        )
                        continuity_issues = list(result.issues)
                        reports.append(
                            _report(
                                "planning.continuity_review",
                                "Independent continuity review result",
                                result.model_dump(mode="json"),
                                severity="major"
                                if any(item.severity == "major" for item in continuity_issues)
                                else "minor"
                                if continuity_issues
                                else "info",
                                stage="continuity",
                            )
                        )
                        reports.extend(
                            _continuity_diagnostic(item, deterministic=False)
                            for item in continuity_issues
                        )
                    except (RunCancelled, OperationDeadlineExceeded, RequestBudgetExceeded):
                        raise
                    except Exception as exc:
                        reports.append(
                            _report(
                                "planning.continuity_warning",
                                "Continuity model unavailable; deterministic contracts passed",
                                {"error": str(exc)},
                                severity="minor",
                                stage="continuity",
                            )
                        )
                plan_blocking = bool(feedback)
                continuity_blocking = any(item.severity == "major" for item in continuity_issues)
                minor_targets = {
                    scene_id for item in continuity_issues for scene_id in item.scene_ids
                }
                repair_minor = (
                    not policy.relaxed
                    and continuity_fixes < max_continuity
                    and cycle < 3
                    and all(
                        scene_id in replans and replans[scene_id] < max_replans
                        for scene_id in minor_targets
                    )
                )
                continuity_repair_requested = False
                for issue in continuity_issues:
                    if issue.severity == "major" or repair_minor:
                        continuity_repair_requested = True
                        for scene_id in issue.scene_ids or [
                            item.scene_id for item in current.plans
                        ]:
                            feedback.setdefault(scene_id, []).append(
                                f"{issue.message}\n{issue.fix_instruction}"
                            )
                if continuity_issues and not continuity_blocking and not repair_minor:
                    reports.append(
                        _report(
                            "planning.continuity_warning",
                            "Nonblocking continuity repair reached its finite budget",
                            {"rounds": continuity_fixes, "limit": max_continuity},
                            severity="minor",
                            stage="continuity",
                        )
                    )
                current = current.model_copy(update={"diagnostics": tuple(reports)})
                if not feedback:
                    return current
                signature = PlanningService.cycle_signature(
                    {
                        "plans": [item.model_dump(mode="json") for item in current.plans],
                        "feedback": feedback,
                    }
                )
                if signature in seen_cycles:
                    self._reject(
                        current,
                        "planning.stagnation",
                        "Planning review repeated identical inputs and findings",
                        {"cycle": cycle},
                    )
                seen_cycles.add(signature)
                if cycle == 3:
                    self._reject(
                        current,
                        "planning.cycle_budget",
                        "Planning and continuity did not converge within four cycles",
                        {"limit": 4},
                    )
                if continuity_repair_requested:
                    if continuity_fixes >= max_continuity:
                        self._reject(
                            current,
                            "planning.continuity_budget",
                            "Continuity repair rounds exhausted",
                            {"limit": max_continuity},
                        )
                    continuity_fixes += 1
                    current = current.model_copy(
                        update={
                            "diagnostics": (
                                *current.diagnostics,
                                _report(
                                    "planning.continuity_repair",
                                    "Reserved bounded continuity repair round",
                                    {"round": continuity_fixes, "limit": max_continuity},
                                    stage="continuity",
                                ),
                            )
                        }
                    )
                for scene_id in feedback:
                    if scene_id not in replans or replans[scene_id] >= max_replans:
                        self._reject(
                            current,
                            "planning.replan_budget",
                            "Scene replan attempts exhausted",
                            {"scene_id": scene_id, "limit": max_replans},
                        )
                    replans[scene_id] += 1
                previous = tuple(item.model_dump(mode="json") for item in current.plans)
                current = self._revise_targets(
                    current,
                    {scene_id: "\n".join(items) for scene_id, items in feedback.items()},
                    renderer,
                    user_prompt,
                )
                if tuple(item.model_dump(mode="json") for item in current.plans) == previous:
                    if not plan_blocking and not continuity_blocking:
                        warning = _report(
                            "planning.continuity_warning",
                            "Nonblocking continuity repair made no progress; deterministic contracts passed",
                            {"cycle": cycle},
                            severity="minor",
                            stage="continuity",
                        )
                        return current.model_copy(
                            update={"diagnostics": (*current.diagnostics, warning)}
                        )
                    self._reject(
                        current,
                        "planning.stagnation",
                        "Planner returned unchanged plans for blocking feedback",
                        {"cycle": cycle},
                    )
                current, local, deterministic_continuity = self._validate(current)
            self._reject(
                current,
                "planning.cycle_budget",
                "Planning and continuity did not converge within four cycles",
                {"limit": 4},
            )

    @staticmethod
    def _reject(output: PlanningOutput, code: str, message: str, evidence: dict) -> None:
        diagnostic = _report(code, message, evidence, severity="major")
        rejected = output.model_copy(update={"diagnostics": (*output.diagnostics, diagnostic)})
        raise PlanningStageError(message, diagnostics=rejected.diagnostics, output=rejected)

    def technical(self, inputs: SceneInputs) -> TechnicalOutput:
        with inputs.config.activate():
            self._control.check()
            scene = validated_copy(inputs.scene, deep=True)
            # This is also required for direct public callers: a technical model
            # must not turn an invalid mathematical plan into usable code input.
            issues = PlanCompiler().compile_scene(
                scene.plan, inputs.continuity_bible, inputs.lesson_spec
            )
            local = deterministic_plan_issues(
                scene.plan, inputs.continuity_bible, lesson_spec=inputs.lesson_spec
            )
            blockers = [
                *(
                    _report(
                        "planning.compiler_issue",
                        item.message,
                        item.model_dump(mode="json"),
                        severity=item.severity,
                        scene_id=scene.plan.scene_id,
                        category=item.category,
                    )
                    for item in issues
                ),
                *(diagnostic_from_plan_issue(item, scene_id=scene.plan.scene_id) for item in local),
            ]
            if inputs.continuity_bible is None:
                blockers.append(
                    _report(
                        "planning.missing_bible",
                        "Technical planning requires a fixed continuity bible",
                        {},
                        severity="major",
                    )
                )
            else:
                blockers.extend(
                    _continuity_diagnostic(item, deterministic=True)
                    for item in deterministic_continuity_issues(
                        [validated_copy(item, deep=True) for item in inputs.plans],
                        validated_copy(inputs.continuity_bible, deep=True),
                        inputs.lesson_spec,
                        inputs.teaching_graph,
                    )
                )
            if any(item.severity == "major" for item in blockers):
                raise PlanningStageError(
                    "Technical input plan failed deterministic validation",
                    diagnostics=tuple(blockers),
                )
            renderer = inputs.render_profile.renderer
            inherited = (
                inputs.previous_technical_handoff.model_dump_json()
                if inputs.previous_technical_handoff is not None
                else scene.inherited_elements_code
            )
            rag_context, receipts, rag_diagnostics, _ = self._retrieve(
                "\n".join(
                    (
                        scene.plan.title,
                        scene.plan.math_concept,
                        scene.plan.computation,
                        scene.plan.camera_movement,
                        renderer,
                    )
                ),
                "technical",
                supplied=inputs.rag_context,
                inherited_elements_sha256=sha256(inherited.encode("utf-8")).hexdigest()
                if inherited
                else "",
            )
            diagnostics = [*blockers, *rag_diagnostics]
            feedback = ""
            result = None
            for attempt in range(max(1, settings.MAX_TECHNICAL_SPEC_ATTEMPTS)):
                agent = self._agent(self._technical_factory, scene.plan.scene_id, "technical")
                spec = agent.plan(
                    validated_copy(scene.plan, deep=True),
                    continuity_bible=validated_copy(inputs.continuity_bible, deep=True)
                    if inputs.continuity_bible is not None
                    else None,
                    renderer=renderer,
                    stream=False,
                    rag_context=rag_context,
                    feedback=feedback,
                    element_manifest=validated_copy(inputs.element_manifest, deep=True),
                    inherited_elements_code=scene.inherited_elements_code,
                    previous_technical_handoff=validated_copy(
                        inputs.previous_technical_handoff, deep=True
                    )
                    if inputs.previous_technical_handoff is not None
                    else None,
                    lesson_spec=validated_copy(inputs.lesson_spec, deep=True),
                    teaching_graph=validated_copy(inputs.teaching_graph, deep=True),
                )
                proposed = spec.model_dump(mode="json")
                spec, repairs = normalize_technical_spec_contract(
                    scene.plan, validated_copy(spec, deep=True), renderer=renderer
                )
                compiled = compile_technical_spec(scene.plan, spec, renderer=renderer)
                handoff_errors = PlanningService.technical_handoff_errors(
                    scene.plan, spec, inputs.previous_technical_handoff
                )
                if spec.handoff_in is None and inputs.previous_technical_handoff is not None:
                    spec = spec.model_copy(
                        update={
                            "handoff_in": validated_copy(
                                inputs.previous_technical_handoff, deep=True
                            )
                        }
                    )
                spec = spec.model_copy(update={"handoff_out": build_technical_handoff(spec)})
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
                diagnostics.extend(
                    (
                        _report(
                            "planning.technical_normalization",
                            "Technical contract normalization result",
                            {
                                "repairs": list(repairs),
                                "proposed_spec": proposed,
                                "attempt": attempt + 1,
                            },
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
                )
                diagnostics.append(
                    _report(
                        "planning.technical_handoff",
                        "Structured technical boundary validation",
                        {"errors": list(handoff_errors)},
                        severity="major" if handoff_errors else "info",
                        scene_id=scene.plan.scene_id,
                    )
                )
                result = TechnicalOutput(
                    spec=validated_copy(spec, deep=True),
                    diagnostics=tuple(diagnostics),
                    rag_receipts=receipts,
                )
                if compiled.is_valid and capability_result.is_valid and not handoff_errors:
                    return result
                # Hardware/profile failures cannot be repaired by asking a model
                # to change its answer; preserve the real failed capability gate.
                if not capability_result.is_valid:
                    break
                next_feedback = "\n".join((*compiled.errors, *handoff_errors))
                if next_feedback == feedback:
                    break
                feedback = next_feedback
            assert result is not None
            raise PlanningStageError(
                "Technical specification validation failed",
                diagnostics=result.diagnostics,
                output=result,
            )


__all__ = ["PlanningPipeline", "PlanningStageError"]
