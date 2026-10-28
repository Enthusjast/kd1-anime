"""Real planning/validation code with only the model response boundary replaced."""

from threading import Event

import pytest

from kd1_anime.agents.base import BaseAgent
from kd1_anime.agents.continuity import ContinuityReviewerAgent
from kd1_anime.agents.plan_reviewer import PlanReviewerAgent
from kd1_anime.agents.planner import PlannerAgent
from kd1_anime.agents.technical_planner import TechnicalPlannerAgent
from kd1_anime.config import settings
from kd1_anime.control import RunControl
from kd1_anime.domain.plan_review import (
    ContinuityIssue,
    ContinuityReviewResult,
    PlanReviewIssue,
    PlanReviewResult,
)
from kd1_anime.domain.planning import (
    ContinuityBible,
    LessonSpec,
    MathClaim,
    PlanningDraft,
    SceneDetail,
    SceneOutline,
    ScenePlan,
    TeachingGraph,
)
from kd1_anime.domain.runtime import ExecutionSpec, RunConfig
from kd1_anime.domain.technical import TechnicalAnimation, TechnicalSpec
from kd1_anime.pipeline.contracts import DetailInputs, PlanningOutput, SceneInputs
from kd1_anime.pipeline.planning import (
    PlanningPipeline,
    PlanningStageError,
)
from kd1_anime.rendering import RenderProfile
from kd1_anime.request_budget import RequestContext
from kd1_anime.run_store import StoredSceneState


def outline(scene_id=1):
    return SceneOutline(
        scene_id=scene_id,
        title="Pause",
        duration_seconds=1,
        purpose="Show elapsed time",
        math_concept="Time",
    )


def detail():
    return SceneDetail(
        visual_design="Empty frame",
        camera_movement="Fixed camera",
        visual_flow=["Wait one second"],
        key_moments=["One second elapses"],
        computation="No computation",
        opening_state=["Empty frame"],
        closing_state=["Empty frame"],
        transition_in="Start",
        transition_out="End",
    )


def output():
    return PlanningOutput(
        outlines=(outline(),),
        plans=(ScenePlan(**outline().model_dump(), **detail().model_dump()),),
        lesson_spec=LessonSpec(topic="Time"),
        teaching_graph=TeachingGraph(),
        continuity_bible=ContinuityBible(),
    )


@pytest.fixture
def boundary(monkeypatch):
    monkeypatch.setattr(settings, "SCENE_CODEGEN_MODE", "llm")
    monkeypatch.setattr(settings, "GENERATION_MODE", "strict")
    monkeypatch.setattr(settings, "SLURM_GPU_TYPE", "")
    monkeypatch.setattr(settings, "SLURM_CONTAINER_IMAGE", None)
    responses = {
        PlanningDraft: PlanningDraft(items=[outline()], lesson_spec=LessonSpec(topic="Time")),
        SceneDetail: detail(),
        ContinuityBible: ContinuityBible(),
        PlanReviewResult: PlanReviewResult(is_valid=True, summary="Independent plan evidence"),
        ContinuityReviewResult: ContinuityReviewResult(
            is_valid=True, summary="Independent continuity evidence"
        ),
        TechnicalSpec: TechnicalSpec(scene_id=1),
    }
    calls = []
    contexts = []
    control = RunControl(Event())

    def fake_model(agent, **kwargs):
        assert agent.control is control
        assert kwargs.get("stream", False) is False
        calls.append((type(agent), kwargs))
        return responses[kwargs["response_model"]].model_copy(deep=True)

    monkeypatch.setattr(BaseAgent, "call_llm_json", fake_model)

    def request_context(scene_id, stage):
        context = RequestContext(reserve=lambda *a, **k: None, emit=lambda *a: None)
        contexts.append(context)
        return context

    def pipeline():
        return PlanningPipeline(
            PlannerAgent,
            PlanReviewerAgent,
            ContinuityReviewerAgent,
            TechnicalPlannerAgent,
            config=RunConfig.capture(),
            request_context=request_context,
            control=control,
        )

    return responses, calls, contexts, pipeline


def inputs(tmp_path, planned=None, *, renderer="cairo"):
    planned = planned or output()
    profile = RenderProfile(
        renderer=renderer,
        quality="l",
        pixel_width=854,
        pixel_height=480,
        frame_rate=15,
    )
    return SceneInputs(
        run_id="planning-test",
        root=tmp_path,
        config=RunConfig.capture(),
        render_profile=profile,
        execution_spec=ExecutionSpec.capture(profile),
        scene=StoredSceneState(plan=planned.plans[0]),
        plans=planned.plans,
        lesson_spec=planned.lesson_spec,
        teaching_graph=planned.teaching_graph,
        continuity_bible=planned.continuity_bible,
    )


def test_propose_builds_outline_bible_without_detail_or_placeholder(boundary):
    responses, calls, _, factory = boundary
    result = factory().propose("Show time", "cairo")
    assert result.lesson_spec.topic == "Time"
    assert result.teaching_graph == TeachingGraph()
    assert result.plans == ()
    assert [entry[1]["response_model"] for entry in calls] == [PlanningDraft, ContinuityBible]
    responses[PlanningDraft].lesson_spec.topic = "Changed outside worker"
    assert result.lesson_spec.topic == "Time"
    assert not hasattr(result, "context")


def test_multiple_scene_draft_returns_all_outlines_before_detail(boundary):
    responses, calls, _, factory = boundary
    responses[PlanningDraft] = PlanningDraft(items=[outline(), outline(2)])
    result = factory().propose("Two separate chapters", "cairo")
    assert [item.scene_id for item in result.outlines] == [1, 2]
    assert result.plans == ()
    assert [item[1]["response_model"] for item in calls] == [PlanningDraft, ContinuityBible]


def test_review_preserves_full_independent_review_evidence(boundary):
    responses, calls, _, factory = boundary
    source = output()
    result = factory().review(source, "cairo")
    evidence = {item.code: item.evidence for item in result.diagnostics}
    assert evidence["planning.plan_review"]["summary"] == "Independent plan evidence"
    assert evidence["planning.continuity_review"]["summary"] == "Independent continuity evidence"
    assert evidence["planning.compiler"] == {"is_valid": True, "issues": []}
    assert [kind for kind, _ in calls] == [PlanReviewerAgent, ContinuityReviewerAgent]
    responses[PlanReviewResult].summary = "mutated"
    assert evidence["planning.plan_review"]["summary"] == "Independent plan evidence"
    assert source.diagnostics == ()


@pytest.mark.parametrize("mode", ["strict", "relaxed"])
def test_deterministic_math_failure_cannot_be_approved_by_model(boundary, monkeypatch, mode):
    _, calls, _, factory = boundary
    monkeypatch.setattr(settings, "GENERATION_MODE", mode)
    planned = output()
    plan = planned.plans[0].model_copy(
        update={
            "math_claims": [
                MathClaim(
                    claim_id="wrong",
                    statement="1 equals 2",
                    expression_before="1",
                    expression_after="2",
                    relation="equivalent",
                )
            ]
        }
    )
    planned = planned.model_copy(update={"plans": (plan,)})
    with pytest.raises(PlanningStageError) as caught:
        factory().review(planned, "cairo")
    assert not calls
    assert caught.value.output.plans[0].math_claims == planned.plans[0].math_claims
    assert planned.plans[0].math_claims[0].expression_after == "2"
    assert any(
        item.category == "math" and item.severity == "major" for item in caught.value.diagnostics
    )
    assert any(item.code == "planning.compiler" for item in caught.value.diagnostics)


def test_deterministic_continuity_failure_stops_before_independent_models(boundary):
    _, calls, _, factory = boundary
    planned = output().model_copy(update={"continuity_bible": ContinuityBible(palette=[])})
    with pytest.raises(PlanningStageError) as caught:
        factory().review(planned, "cairo")
    assert not calls
    assert any(item.code == "planning.continuity_issue" for item in caught.value.diagnostics)


def test_independent_continuity_failure_retains_plan_review_evidence(boundary):
    responses, calls, _, factory = boundary
    issue = ContinuityIssue(
        scene_ids=[1],
        category="narrative",
        message="Unsupported ending",
        fix_instruction="Correct ending",
    )
    responses[ContinuityReviewResult] = ContinuityReviewResult(
        is_valid=False, summary="Continuity rejected", issues=[issue]
    )
    with pytest.raises(PlanningStageError) as caught:
        factory().review(output(), "cairo")
    assert 2 <= len(calls) <= 12
    evidence = {item.code: item.evidence for item in caught.value.diagnostics}
    assert evidence["planning.plan_review"]["summary"] == "Independent plan evidence"
    assert evidence["planning.continuity_review"]["issues"] == [issue.model_dump(mode="json")]


def test_model_plan_failure_remains_bounded_and_evidence_complete(boundary):
    responses, calls, _, factory = boundary
    issue = PlanReviewIssue(
        category="math",
        confidence="high",
        evidence_type="calculation",
        evidence="2+2=4, not 5",
        message="Incorrect arithmetic",
        fix_instruction="Correct arithmetic",
    )
    responses[PlanReviewResult] = PlanReviewResult(
        is_valid=False, summary="Arithmetic rejected", issues=[issue]
    )
    with pytest.raises(PlanningStageError) as caught:
        factory().review(output(), "cairo")
    assert 1 <= sum(kind is PlanReviewerAgent for kind, _ in calls) <= 4
    assert any(
        item.evidence.get("issues") == [issue.model_dump(mode="json")]
        for item in caught.value.diagnostics
    )


def test_technical_normalizes_compiles_and_never_mutates_scene_inputs(boundary, tmp_path):
    _, calls, _, factory = boundary
    source = inputs(tmp_path)
    before = source.scene.model_dump(mode="json")
    result = factory().technical(source)
    assert result.spec.scene_id == 1
    assert source.scene.model_dump(mode="json") == before
    assert len(calls) == 1
    assert any(item.code == "planning.technical_compiler" for item in result.diagnostics)
    assert any(item.code == "planning.capability" for item in result.diagnostics)


def test_invalid_technical_spec_raises_before_any_codegen(boundary, tmp_path):
    responses, calls, _, factory = boundary
    responses[TechnicalSpec] = TechnicalSpec(
        scene_id=1,
        animations=[
            TechnicalAnimation(
                event_id="bad",
                start_seconds=0,
                end_seconds=1,
                semantic_action="introduce",
                target_element_ids=["missing"],
                create_element_ids=["missing"],
            )
        ],
    )
    with pytest.raises(PlanningStageError) as caught:
        factory().technical(inputs(tmp_path))
    assert 1 <= len(calls) <= settings.MAX_TECHNICAL_SPEC_ATTEMPTS
    assert all(kind is TechnicalPlannerAgent for kind, _ in calls)
    evidence = next(
        item.evidence
        for item in caught.value.diagnostics
        if item.code == "planning.technical_compiler"
    )
    assert evidence["is_valid"] is False
    assert any("missing" in message for message in evidence["errors"])
    assert caught.value.output.spec.scene_id == 1


def test_capability_failure_prevents_returning_usable_technical_output(boundary, tmp_path):
    _, calls, _, factory = boundary
    with pytest.raises(PlanningStageError) as caught:
        factory().technical(inputs(tmp_path, renderer="opengl"))
    assert len(calls) == 1
    evidence = next(
        item.evidence for item in caught.value.diagnostics if item.code == "planning.capability"
    )
    assert evidence["validation"]["is_valid"] is False
    assert any("SLURM_GPU_TYPE" in message for message in evidence["validation"]["errors"])


def test_review_without_models_still_runs_deterministic_gates(boundary):
    _, calls, _, factory = boundary
    result = factory().review(output(), "cairo", use_models=False)
    assert not calls
    assert any(item.code == "planning.compiler" for item in result.diagnostics)
    invalid = output().model_copy(update={"continuity_bible": ContinuityBible(palette=[])})
    with pytest.raises(PlanningStageError):
        factory().review(invalid, "cairo", use_models=False)
    assert not calls


def test_mutating_first_reviewer_cannot_change_second_reviewer_or_output(boundary, monkeypatch):
    _, calls, _, factory = boundary
    original = PlanReviewerAgent.review

    def mutate(agent, plan, **kwargs):
        result = original(agent, plan, **kwargs)
        plan.title = "Mutated plan"
        kwargs["continuity_bible"].palette.clear()
        kwargs["lesson_spec"].topic = "Mutated lesson"
        kwargs["all_plans"][0].title = "Mutated neighbor"
        return result

    monkeypatch.setattr(PlanReviewerAgent, "review", mutate)
    source = output()
    before = source.model_dump(mode="json")
    result = factory().review(source, "cairo")
    assert result.plans[0].title == "Pause"
    assert result.lesson_spec.topic == "Time"
    assert source.model_dump(mode="json") == before
    continuity_prompt = calls[-1][1]["user_message"]
    assert "Mutated plan" not in continuity_prompt
    assert "Mutated lesson" not in continuity_prompt


def test_detail_is_independent_and_never_mutates_closed_inputs(boundary, tmp_path):
    responses, calls, _, factory = boundary
    planned = output()
    source = DetailInputs(
        run_id="detail-test",
        root=tmp_path,
        config=RunConfig.capture(),
        outline=planned.outlines[0],
        outlines=planned.outlines,
        lesson_spec=planned.lesson_spec,
        teaching_graph=planned.teaching_graph,
        continuity_bible=planned.continuity_bible,
        user_prompt="Show time",
        renderer="cairo",
    )
    before = source.outline.model_dump(mode="json")
    result = factory().detail(source)
    assert result.plan.scene_id == 1
    assert source.outline.model_dump(mode="json") == before
    assert result.plan.global_visual_state == planned.continuity_bible.global_visual_state
    assert [item[1]["response_model"] for item in calls] == [SceneDetail]
    responses[SceneDetail].visual_design = "Changed outside worker"
    assert result.plan.visual_design == "Empty frame"


def test_review_rejects_missing_scene_detail_before_any_model(boundary):
    _, calls, _, factory = boundary
    planned = output().model_copy(update={"outlines": (outline(), outline(2))})
    with pytest.raises(PlanningStageError) as caught:
        factory().review(planned, "cairo")
    assert not calls
    assert any(item.code == "planning.incomplete_detail" for item in caught.value.diagnostics)


@pytest.mark.parametrize("mode", ["strict", "relaxed"])
def test_identical_replan_stops_without_dropping_original_evidence(boundary, monkeypatch, mode):
    responses, calls, _, factory = boundary
    monkeypatch.setattr(settings, "GENERATION_MODE", mode)
    responses[PlanReviewResult] = PlanReviewResult(
        is_valid=False,
        issues=[
            PlanReviewIssue(
                category="math",
                confidence="high",
                evidence_type="calculation",
                evidence="2+2=4, not 5",
                message="Incorrect arithmetic",
                fix_instruction="Fix arithmetic",
            )
        ],
    )
    with pytest.raises(PlanningStageError) as caught:
        factory().review(output(), "cairo")
    assert sum(kind is PlannerAgent for kind, _ in calls) <= settings.MAX_PLAN_REPLAN_ATTEMPTS
    assert any(item.code == "planning.stagnation" for item in caught.value.diagnostics)
    assert any(item.evidence.get("issues") for item in caught.value.diagnostics)
