"""Closed planning actions, fixed RAG identities and bounded whole-plan barriers."""

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from threading import Barrier, Event, Lock

import pytest

from kd1_anime.config import settings
from kd1_anime.control import RunControl
from kd1_anime.domain.planning import (
    ContinuityBible,
    LessonSpec,
    SceneOutline,
    ScenePlan,
    TeachingGraph,
)
from kd1_anime.domain.runtime import RunConfig
from kd1_anime.pipeline.contracts import DetailInputs, PlanningOutput
from kd1_anime.pipeline.planning import PlanningPipeline, PlanningStageError
from kd1_anime.rag.models import RagReceipt, RagRuntimeProfile, RagSearchResult


def outline(scene_id):
    return SceneOutline(
        scene_id=scene_id,
        title=f"Pause {scene_id}",
        duration_seconds=1,
        purpose="Show time",
        math_concept="Time",
    )


def plan(item):
    return ScenePlan(
        **item.model_dump(),
        visual_design="Empty frame",
        camera_movement="Fixed",
        visual_flow=["Wait"],
        key_moments=["One second"],
        computation="No computation",
        opening_state=["Empty frame"],
        closing_state=["Empty frame"],
        transition_in="Start",
        transition_out="End",
    )


def detail_inputs(tmp_path, scene_id):
    return DetailInputs(
        run_id="parallel-detail",
        root=tmp_path,
        config=RunConfig.capture(),
        outline=outline(scene_id),
        outlines=(outline(1), outline(2)),
        lesson_spec=LessonSpec(topic="Time"),
        teaching_graph=TeachingGraph(),
        continuity_bible=ContinuityBible(),
        user_prompt="Show two pauses",
        renderer="cairo",
    )


def pipeline(factory, *, rag_service=None, rag_profile=None):
    return PlanningPipeline(
        factory,
        factory,
        factory,
        factory,
        config=RunConfig.capture(),
        request_context=lambda scene_id, stage: (scene_id, stage),
        control=RunControl(Event()),
        rag_service=rag_service,
        rag_profile=rag_profile,
    )


def test_parallel_details_have_separate_agents_and_scene_request_context(tmp_path):
    barrier = Barrier(2)
    lock = Lock()
    agents = []

    class Planner:
        def plan_detail(self, item, outlines, prompt, **kwargs):
            with lock:
                agents.append(self)
            assert self.request_context == (item.scene_id, "planning_detail")
            assert kwargs["stream"] is False
            assert prompt == "Show two pauses"
            barrier.wait(timeout=5)
            return plan(item)

    worker = pipeline(Planner)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(worker.detail, [detail_inputs(tmp_path, 1), detail_inputs(tmp_path, 2)])
        )
    assert len({id(item) for item in agents}) == 2
    assert [item.plan.scene_id for item in results] == [1, 2]


class FakeRag:
    def __init__(self):
        self.ready_calls = 0
        self.index_sha256 = "a" * 64
        self.queries = []

    def require_ready(self):
        self.ready_calls += 1

    def runtime_profile(self):
        return RagRuntimeProfile(
            enabled=True,
            status="active",
            index_sha256=self.index_sha256,
            embedding_model="embedding-fixed",
            reranker_model="rerank-fixed",
            chunker_version="1",
            chunk_size=1000,
            chunk_overlap=100,
        )

    def search(self, query, *, stage, **kwargs):
        self.queries.append((query, stage))
        return RagSearchResult(
            context="Exact retrieved evidence",
            receipt=RagReceipt(
                stage=stage,
                query_sha256=sha256(query.encode()).hexdigest(),
                status="active",
                index_sha256=self.index_sha256,
            ),
        )


def test_detail_preserves_exact_rag_receipt_and_prompt(tmp_path):
    rag = FakeRag()

    class Planner:
        def plan_detail(self, item, outlines, prompt, **kwargs):
            assert kwargs["rag_context"] == "Exact retrieved evidence"
            return plan(item)

    result = pipeline(Planner, rag_service=rag).detail(detail_inputs(tmp_path, 1))
    assert rag.ready_calls == 1
    assert result.rag_receipts[0].index_sha256 == "a" * 64
    query, stage = rag.queries[0]
    assert result.rag_receipts[0].query_sha256 == sha256(query.encode()).hexdigest()
    assert stage == "detail"


def test_changed_rag_index_cannot_silently_replace_pinned_profile(tmp_path):
    rag = FakeRag()
    profile = rag.runtime_profile()
    rag.index_sha256 = "b" * 64

    class NeverPlanner:
        def plan_detail(self, *args, **kwargs):
            pytest.fail("Changed retrieval identity reached a model")

    with pytest.raises(PlanningStageError) as caught:
        pipeline(NeverPlanner, rag_service=rag, rag_profile=profile).detail(
            detail_inputs(tmp_path, 1)
        )
    assert any(item.code == "planning.rag_profile_mismatch" for item in caught.value.diagnostics)
    assert rag.queries == []


def test_multi_scene_review_checks_every_scene_before_models(tmp_path):
    source = detail_inputs(tmp_path, 1)
    first, second = plan(outline(1)), plan(outline(2))
    second = second.model_copy(update={"computation": "1+1=3"})
    output = PlanningOutput(
        outlines=source.outlines,
        plans=(first, second),
        lesson_spec=source.lesson_spec,
        teaching_graph=source.teaching_graph,
        continuity_bible=source.continuity_bible,
    )

    class NeverReviewer:
        def review(self, *args, **kwargs):
            pytest.fail("Bad downstream math reached an independent model")

    with pytest.raises(PlanningStageError) as caught:
        pipeline(NeverReviewer).review(output, "cairo")
    assert any(
        item.category == "math" and item.severity == "major" for item in caught.value.diagnostics
    )


def test_detail_pinned_config_overrides_changed_global_codegen_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "GENERATION_MODE", "strict")
    source = detail_inputs(tmp_path, 1)
    observed = []

    class Planner:
        def plan_detail(self, item, outlines, prompt, **kwargs):
            observed.append(settings.GENERATION_MODE)
            return plan(item)

    monkeypatch.setattr(settings, "GENERATION_MODE", "relaxed")
    pipeline(Planner).detail(source)
    assert observed == ["strict"]


def test_replanned_scene_forces_fresh_whole_plan_review(tmp_path, monkeypatch):
    from kd1_anime.domain.plan_review import (
        ContinuityReviewResult,
        PlanReviewIssue,
        PlanReviewResult,
    )

    monkeypatch.setattr(settings, "GENERATION_MODE", "strict")
    source = detail_inputs(tmp_path, 1)
    original = PlanningOutput(
        outlines=source.outlines,
        plans=tuple(plan(item) for item in source.outlines),
        lesson_spec=source.lesson_spec,
        teaching_graph=source.teaching_graph,
        continuity_bible=source.continuity_bible,
    )
    reviewed = []
    revised = []

    class Planner:
        def plan_detail(self, item, outlines, prompt, **kwargs):
            assert prompt == "Show two pauses"
            assert "Correct" in kwargs["continuity_feedback"]
            revised.append(item.scene_id)
            return plan(item).model_copy(update={"visual_design": "Corrected empty frame"})

    class PlanReviewer:
        def review(self, item, **kwargs):
            reviewed.append((item.scene_id, item.visual_design))
            if item.scene_id == 1 and item.visual_design == "Empty frame":
                return PlanReviewResult(
                    is_valid=False,
                    issues=[
                        PlanReviewIssue(
                            category="math",
                            confidence="high",
                            evidence_type="calculation",
                            evidence="1+1=2",
                            message="Correct the arithmetic demonstration",
                            fix_instruction="Correct frame",
                        )
                    ],
                )
            return PlanReviewResult(is_valid=True)

    class ContinuityReviewer:
        def review(self, *args, **kwargs):
            return ContinuityReviewResult(is_valid=True)

    worker = PlanningPipeline(
        Planner,
        PlanReviewer,
        ContinuityReviewer,
        Planner,
        config=source.config,
        request_context=lambda scene_id, stage: (scene_id, stage),
        control=RunControl(Event()),
    )
    result = worker.review(original, "cairo", user_prompt="Show two pauses")
    assert revised == [1]
    assert [scene_id for scene_id, _ in reviewed] == [1, 2, 1, 2]
    assert result.plans[0].visual_design == "Corrected empty frame"
    assert original.plans[0].visual_design == "Empty frame"
    assert any(item.code == "planning.revision" for item in result.diagnostics)


def test_explicit_revision_rejects_new_bad_math_before_returning(tmp_path):
    source = detail_inputs(tmp_path, 1)
    original = PlanningOutput(
        outlines=source.outlines,
        plans=tuple(plan(item) for item in source.outlines),
        lesson_spec=source.lesson_spec,
        teaching_graph=source.teaching_graph,
        continuity_bible=source.continuity_bible,
    )

    class Planner:
        def plan_detail(self, item, outlines, prompt, **kwargs):
            return plan(item).model_copy(update={"computation": "1+1=3"})

    with pytest.raises(PlanningStageError) as caught:
        pipeline(Planner).revise(original, "Repair plan", "cairo", user_prompt="Show time")
    assert any(
        item.category == "math" and item.severity == "major" for item in caught.value.diagnostics
    )


@pytest.mark.parametrize("mode", ["strict", "relaxed"])
def test_changed_plans_do_not_reset_total_replan_budget(tmp_path, monkeypatch, mode):
    from kd1_anime.domain.plan_review import (
        ContinuityReviewResult,
        PlanReviewIssue,
        PlanReviewResult,
    )

    monkeypatch.setattr(settings, "GENERATION_MODE", mode)
    monkeypatch.setattr(settings, "MAX_PLAN_REPLAN_ATTEMPTS", 1)
    source = detail_inputs(tmp_path, 1)
    original = PlanningOutput(
        outlines=(source.outline,),
        plans=(plan(source.outline),),
        lesson_spec=source.lesson_spec,
        teaching_graph=source.teaching_graph,
        continuity_bible=source.continuity_bible,
    )
    replans = []

    class Planner:
        def plan_detail(self, item, outlines, prompt, **kwargs):
            replans.append(item.scene_id)
            return plan(item).model_copy(update={"visual_design": f"Revision {len(replans)}"})

    class PlanReviewer:
        def review(self, item, **kwargs):
            return PlanReviewResult(
                is_valid=False,
                issues=[
                    PlanReviewIssue(
                        category="math",
                        confidence="high",
                        evidence_type="calculation",
                        evidence="1+1=2",
                        message="Incorrect arithmetic",
                        fix_instruction="Correct arithmetic",
                    )
                ],
            )

    class ContinuityReviewer:
        def review(self, *args, **kwargs):
            return ContinuityReviewResult(is_valid=True)

    worker = PlanningPipeline(
        Planner,
        PlanReviewer,
        ContinuityReviewer,
        Planner,
        config=source.config,
        request_context=lambda scene_id, stage: (scene_id, stage),
        control=RunControl(Event()),
    )
    with pytest.raises(PlanningStageError) as caught:
        worker.review(original, "cairo", user_prompt="Show time")
    assert replans == [1]
    assert any(item.code == "planning.replan_budget" for item in caught.value.diagnostics)
    with pytest.raises(PlanningStageError) as resumed:
        worker.revise(caught.value.output, "Try again", "cairo", user_prompt="Show time")
    assert replans == [1]
    assert any(item.code == "planning.replan_budget" for item in resumed.value.diagnostics)


def test_strict_minor_continuity_stagnation_becomes_warning_not_a_false_major(
    tmp_path, monkeypatch
):
    from kd1_anime.domain.plan_review import (
        ContinuityIssue,
        ContinuityReviewResult,
        PlanReviewResult,
    )

    monkeypatch.setattr(settings, "GENERATION_MODE", "strict")
    source = detail_inputs(tmp_path, 1)
    original = PlanningOutput(
        outlines=(source.outline,),
        plans=(plan(source.outline),),
        lesson_spec=source.lesson_spec,
        teaching_graph=source.teaching_graph,
        continuity_bible=source.continuity_bible,
    )
    replans = []

    class Planner:
        def plan_detail(self, item, outlines, prompt, **kwargs):
            replans.append(item.scene_id)
            return plan(item)

    class PlanReviewer:
        def review(self, item, **kwargs):
            return PlanReviewResult(is_valid=True)

    class ContinuityReviewer:
        def review(self, *args, **kwargs):
            return ContinuityReviewResult(
                is_valid=False,
                issues=[
                    ContinuityIssue(
                        scene_ids=[1],
                        category="narrative",
                        severity="minor",
                        message="Optional pacing improvement",
                        fix_instruction="Consider a gentler pause",
                    )
                ],
            )

    worker = PlanningPipeline(
        Planner,
        PlanReviewer,
        ContinuityReviewer,
        Planner,
        config=source.config,
        request_context=lambda scene_id, stage: (scene_id, stage),
        control=RunControl(Event()),
    )
    result = worker.review(original, "cairo", user_prompt="Show time")
    assert replans == [1]
    assert any(item.code == "planning.continuity_warning" for item in result.diagnostics)
    assert not any(
        item.code == "planning.stagnation" and item.severity == "major"
        for item in result.diagnostics
    )
