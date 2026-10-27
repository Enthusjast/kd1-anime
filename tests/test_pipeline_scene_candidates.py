import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_pipeline_scene import SOURCE, inputs

from kd1_anime.domain.candidates import CodeCandidate
from kd1_anime.domain.dependencies import WorkIdentity
from kd1_anime.domain.program import ProgramEvent, SceneProgram
from kd1_anime.domain.review import FixSuggestion, ReviewFinding, ReviewResult
from kd1_anime.domain.runtime import RunConfig
from kd1_anime.exceptions import ValidationError
from kd1_anime.objects import ObjectStore
from kd1_anime.pipeline.scene import ScenePipeline
from kd1_anime.request_budget import RequestContext
from kd1_anime.run_store import sha256_text


def configured(scene, **values):
    config = json.loads(scene.config.values_json)
    config.update(values)
    return replace(scene, config=RunConfig(values_json=json.dumps(config)))


def stage(codes=(SOURCE,), reviews=()):
    calls = SimpleNamespace(code=[], review=[])
    code_iter = iter(codes)
    review_iter = iter(reviews)

    class Coder:
        def generate_code(self, plan, **kwargs):
            calls.code.append(kwargs)
            return next(code_iter)

    class Reviewer:
        def review(self, code, plan, **kwargs):
            calls.review.append(code)
            return next(review_iter, ReviewResult(is_valid=True))

    pipeline = ScenePipeline(
        Coder,
        Reviewer,
        request_context=lambda *_: RequestContext(lambda *a, **k: None, lambda *a: None),
        control=SimpleNamespace(checkpoint=lambda: None),
    )
    return pipeline, calls


def source(scene, text=SOURCE):
    scene.scene.code_file = ObjectStore(scene.root).source(scene.scene.plan.scene_id, text)
    scene.scene.code_sha256 = sha256_text(text)
    scene.scene.class_name = "Pause"
    return scene


def fix_result(
    *, minor=False, deterministic=False, find="self.wait(1)", replacement="self.wait(2)"
):
    return ReviewResult(
        is_valid=False,
        severity="minor" if minor else "major",
        feedback="Adjust pause",
        fixes=[FixSuggestion(find=find, replace=replacement)],
        findings=[
            ReviewFinding(
                category="runtime",
                severity="minor" if minor else "major",
                evidence_type="deterministic" if deterministic else "source_code",
                confidence="high",
                evidence=find,
                why="Adjust pause",
            )
        ],
    )


def test_generation_retries_real_static_failures_and_reserves_each_attempt(tmp_path):
    scene = configured(inputs(tmp_path), CODE_VALIDATION_ATTEMPTS=2)
    pipeline, calls = stage(("import os\nos.system('bad')", SOURCE))
    reserved = []

    def reserve():
        reserved.append(len(reserved) + 1)
        return reserved[-1]

    output = pipeline.generate(scene, reserve_attempt=reserve)
    assert reserved == [1, 2]
    assert (tmp_path / output.source.path).read_text() == SOURCE
    assert calls.code[0]["stream"] is False
    assert calls.code[1]["feedback"]
    assert "os.system" in calls.code[1]["previous_code"]


@pytest.mark.parametrize("mode", ["strict", "relaxed"])
def test_generation_never_softpasses_unsafe_source_and_honors_pinned_limit(tmp_path, mode):
    scene = configured(inputs(tmp_path), GENERATION_MODE=mode, CODE_VALIDATION_ATTEMPTS=2)
    pipeline, calls = stage(("import os\nos.system('bad')",) * 2)
    with pytest.raises(ValidationError):
        pipeline.generate(scene)
    assert len(calls.code) == 2


def test_compiler_first_uses_complete_program_without_coder(tmp_path):
    scene = configured(inputs(tmp_path), SCENE_CODEGEN_MODE="compiler_first")
    scene.scene.plan.program = SceneProgram(
        scene_id=1,
        class_name="Pause",
        duration_seconds=1,
        events=(ProgramEvent(event_id="hold", operation="wait", start_seconds=0, end_seconds=1),),
    )
    pipeline, calls = stage(())
    output = pipeline.generate(scene)
    assert not calls.code
    source(scene, (tmp_path / output.source.path).read_text())
    assert pipeline.inspect(scene).accepted
    assert pipeline.review(scene).accepted


def test_unsupported_compiler_is_explicit_and_falls_back_to_llm(tmp_path):
    scene = configured(inputs(tmp_path), SCENE_CODEGEN_MODE="compiler_first")
    pipeline, calls = stage()
    result = pipeline.generate(scene)
    assert len(calls.code) == 1
    assert any(item.code == "scene_program_unsupported" for item in result.diagnostics)


def test_review_does_not_call_model_for_static_rejection(tmp_path):
    scene = source(inputs(tmp_path), SOURCE.replace("self.wait(1)", "open('oops')"))
    pipeline, calls = stage()
    result = pipeline.review(scene)
    assert not result.accepted and result.repair_route == "coder"
    assert not calls.review
    assert result.diagnostics


def test_coder_cannot_mutate_boundary_contract_to_bypass_common_acceptor(tmp_path):
    from kd1_anime.domain.planning import VisualElementState

    scene = configured(inputs(tmp_path), CODE_VALIDATION_ATTEMPTS=1)
    scene.scene.plan.new_elements = [
        VisualElementState(element_id="required", variable_name="required")
    ]

    class Coder:
        def generate_code(self, plan, **kwargs):
            plan.new_elements.clear()
            return SOURCE

    pipeline = ScenePipeline(
        Coder,
        lambda: None,
        request_context=lambda *_: RequestContext(lambda *a, **k: None, lambda *a: None),
        control=SimpleNamespace(checkpoint=lambda: None),
    )
    with pytest.raises(ValidationError):
        pipeline.generate(scene)
    assert scene.scene.plan.new_elements


def test_minor_deterministic_precise_fix_is_reinspected_and_reviewed(tmp_path):
    scene = source(inputs(tmp_path))
    pipeline, calls = stage(
        reviews=(fix_result(minor=True, deterministic=True), ReviewResult(is_valid=True))
    )
    result = pipeline.review(scene)
    assert result.accepted and result.source is not None
    assert result.code_sha256 == sha256_text(SOURCE.replace("self.wait(1)", "self.wait(2)"))
    assert len(calls.review) == 2 and calls.review[1] != calls.review[0]
    assert result.inspection.accepted
    assert scene.scene.code_sha256 == sha256_text(SOURCE)


def test_major_precise_fix_is_immutable_proposal_until_exact_authorized_review(tmp_path):
    scene = source(inputs(tmp_path))
    identity = WorkIdentity(4, "a" * 64)
    scene.scene.revision, scene.scene.input_sha256 = identity.revision, identity.input_sha256
    pipeline, calls = stage(reviews=(fix_result(), ReviewResult(is_valid=True)))
    proposed = pipeline.review(scene, identity=identity)
    assert not proposed.accepted and proposed.repair_route == "approval"
    assert proposed.source is None and proposed.proposal.identity == identity
    assert proposed.proposal.base_source.sha256 == scene.scene.code_sha256
    assert len(calls.review) == 1
    installed = pipeline.accept_proposal(scene, proposed.proposal)
    assert installed.accepted and installed.source == proposed.proposal.candidate_source
    assert len(calls.review) == 1
    changed = scene.scene.model_copy(
        update={
            "code_file": installed.source.path,
            "code_sha256": installed.source.sha256,
            "class_name": installed.class_name,
            "reviewed": False,
            "review_round": proposed.review_round,
        }
    )
    approved = pipeline.review(replace(scene, scene=changed))
    assert approved.accepted and approved.code_sha256 == installed.source.sha256
    assert len(calls.review) == 2
    assert scene.scene.code_sha256 == sha256_text(SOURCE)


def test_pending_proposal_rejects_modified_context_and_diff(tmp_path):
    scene = source(inputs(tmp_path))
    pipeline, _ = stage(reviews=(fix_result(),))
    proposal = pipeline.review(scene).proposal
    changed = replace(scene, rag_context="changed retrieval")
    with pytest.raises(ValueError):
        pipeline.accept_proposal(changed, proposal)
    forged = proposal.model_copy(update={"diff_sha256": "f" * 64})
    with pytest.raises(ValueError):
        pipeline.accept_proposal(scene, forged)
    stale = replace(scene, scene=scene.scene.model_copy(update={"revision": 1}))
    with pytest.raises(ValueError):
        pipeline.accept_proposal(stale, proposal)


def test_ambiguous_or_unsafe_precise_fix_never_becomes_proposal(tmp_path):
    scene = source(inputs(tmp_path))
    pipeline, calls = stage(reviews=(fix_result(find="self", replacement="scene"),))
    result = pipeline.review(scene)
    assert not result.accepted and result.proposal is None
    assert len(calls.review) == 1
    pipeline, calls = stage(reviews=(fix_result(replacement="open('oops')"),))
    result = pipeline.review(scene)
    assert not result.accepted and result.proposal is None and result.diagnostics
    assert len(calls.review) == 1


def test_math_contract_findings_route_to_plan_without_code_fix(tmp_path):
    scene = source(inputs(tmp_path))
    finding = ReviewFinding(
        category="math",
        severity="major",
        evidence_type="contract",
        why="ScenePlan contains the wrong identity",
        repair="Revise ScenePlan",
    )
    pipeline, _ = stage(reviews=(ReviewResult(is_valid=False, findings=[finding]),))
    result = pipeline.review(scene)
    assert result.repair_route == "plan"
    assert result.feedback == "Revise ScenePlan"


@pytest.mark.parametrize("mode", ["strict", "relaxed"])
def test_review_budget_blocks_new_requests_after_exhaustion(tmp_path, mode):
    scene = configured(source(inputs(tmp_path)), GENERATION_MODE=mode, MAX_LOW_RISK_REVIEW_ROUNDS=1)
    scene.scene.review_round = 1
    pipeline, calls = stage()
    result = pipeline.review(scene)
    assert not result.accepted and result.repair_route == "exhausted"
    assert not calls.review


def test_rollback_chooses_highest_matching_candidate_then_reviews_again(tmp_path):
    scene = source(inputs(tmp_path))
    better = SOURCE.replace("self.wait(1)", "self.wait(2)")
    stale = SOURCE.replace("self.wait(1)", "self.wait(3)")
    scene.scene.candidates = [
        CodeCandidate(
            code_file=ObjectStore(tmp_path).source(1, stale),
            code_sha256=sha256_text(stale),
            class_name="Pause",
            verification="rendered",
            inherited_elements_sha256="0" * 64,
        ),
        CodeCandidate(
            code_file=ObjectStore(tmp_path).source(1, better),
            code_sha256=sha256_text(better),
            class_name="Pause",
            verification="smoke",
            inherited_elements_sha256=sha256_text(""),
        ),
    ]
    pipeline, calls = stage()
    result = pipeline.rollback(scene)
    assert result.accepted and result.code_sha256 == sha256_text(better)
    assert calls.review == [better]
    assert scene.scene.code_sha256 == sha256_text(SOURCE)


def test_enabled_rag_never_silently_uses_missing_service(tmp_path):
    from kd1_anime.pipeline.scene import SceneStageError

    scene = configured(inputs(tmp_path), RAG_ENABLED=True)
    pipeline, calls = stage()
    with pytest.raises(SceneStageError):
        pipeline.generate(scene)
    assert not calls.code


def test_code_retrieval_receipt_and_pinned_profile_are_preserved(tmp_path):
    from kd1_anime.rag.models import RagReceipt, RagRuntimeProfile, RagSearchResult

    scene = configured(inputs(tmp_path), RAG_ENABLED=True)
    receipt = RagReceipt(
        stage="code", query_sha256="a" * 64, index_sha256="b" * 64, status="active"
    )
    profile = RagRuntimeProfile(enabled=True, status="active", index_sha256="b" * 64)
    requested = []

    class Rag:
        def require_profile(self, expected):
            requested.append(expected)

        def search(self, query, **kwargs):
            assert kwargs["stage"] == "code"
            return RagSearchResult(context="verified retrieval", receipt=receipt)

    pipeline, calls = stage()
    pipeline.rag_service = Rag()
    pipeline.pin_profile(profile)
    result = pipeline.generate(scene)
    assert result.rag_receipts == (receipt,)
    assert calls.code[0]["rag_context"] == "verified retrieval"
    assert requested == [profile, profile]


def test_non_deterministic_minor_fix_still_requires_diff_approval(tmp_path):
    scene = source(inputs(tmp_path))
    pipeline, calls = stage(reviews=(fix_result(minor=True),))
    result = pipeline.review(scene)
    assert result.repair_route == "approval" and result.source is None
    assert result.proposal.diff_sha256 == result.proposal.bound_digest()
    assert len(calls.review) == 1


@pytest.mark.parametrize("mode", ["strict", "relaxed"])
def test_identical_review_stagnation_is_persistable_and_finite(tmp_path, mode):
    scene = configured(source(inputs(tmp_path)), GENERATION_MODE=mode, MAX_LOW_RISK_REVIEW_ROUNDS=8)
    rejected = ReviewResult(
        is_valid=False,
        severity="major",
        feedback="Wrong scene",
        findings=[
            ReviewFinding(
                category="runtime", severity="major", evidence_type="source_code", why="Wrong scene"
            )
        ],
    )
    pipeline, calls = stage(reviews=(rejected, rejected))
    first = pipeline.review(scene)
    second = pipeline.review(
        scene,
        review_round=first.review_round,
        review_signature=first.review_signature,
        identical_review_count=first.identical_review_count,
    )
    assert second.repair_route == "exhausted" and second.identical_review_count == 2
    third = pipeline.review(
        scene,
        review_round=second.review_round,
        review_signature=second.review_signature,
        identical_review_count=second.identical_review_count,
    )
    assert third.repair_route == "exhausted"
    assert len(calls.review) == 2


@pytest.mark.parametrize("field", ["candidate_source", "inspection_report", "review_report"])
def test_approved_diff_rejects_tampered_immutable_objects_without_model_call(tmp_path, field):
    scene = source(inputs(tmp_path))
    pipeline, calls = stage(reviews=(fix_result(),))
    proposal = pipeline.review(scene).proposal
    reference = getattr(proposal, field)
    (tmp_path / reference.path).write_text("tampered")
    with pytest.raises(ValueError):
        pipeline.accept_proposal(scene, proposal)
    assert len(calls.review) == 1


def test_manual_confirmation_prevents_automatic_minor_deterministic_install(tmp_path):
    scene = source(inputs(tmp_path))
    pipeline, calls = stage(reviews=(fix_result(minor=True, deterministic=True),))
    result = pipeline.review(scene, require_confirmation=True)
    assert result.repair_route == "approval" and result.source is None
    assert len(calls.review) == 1


@pytest.mark.parametrize("mode,accepted", [("strict", False), ("relaxed", True)])
def test_feedback_only_softpass_uses_pinned_generation_mode(tmp_path, mode, accepted):
    scene = configured(source(inputs(tmp_path)), GENERATION_MODE=mode)
    pipeline, _ = stage(
        reviews=(ReviewResult(is_valid=False, severity="major", feedback="Subjective preference"),)
    )
    assert pipeline.review(scene).accepted is accepted


def test_compiler_branch_cannot_bypass_invalid_technical_contract(tmp_path):
    from kd1_anime.domain.technical import TechnicalSpec

    scene = configured(inputs(tmp_path), SCENE_CODEGEN_MODE="compiler_first")
    scene.scene.plan.program = SceneProgram(
        scene_id=1,
        class_name="Pause",
        duration_seconds=1,
        events=(ProgramEvent(event_id="hold", operation="wait", start_seconds=0, end_seconds=1),),
    )
    scene.scene.technical_spec = TechnicalSpec(scene_id=2)
    pipeline, calls = stage()
    with pytest.raises(ValidationError):
        pipeline.generate(scene)
    assert not calls.code


@pytest.mark.parametrize("plan_route", [False, True])
def test_review_reports_and_outputs_redact_configured_credentials(
    tmp_path, monkeypatch, plan_route
):
    from kd1_anime.config import settings

    secret = "configured-private-credential-value"
    monkeypatch.setattr(settings, "LLM_API_KEY", secret)
    scene = configured(source(inputs(tmp_path)), GENERATION_MODE="strict")
    finding = ReviewFinding(
        category="math" if plan_route else "runtime",
        severity="major",
        evidence_type="contract",
        why=f"ScenePlan problem {secret}" if plan_route else f"Runtime problem {secret}",
        repair=f"Revise ScenePlan {secret}" if plan_route else "Adjust pause",
    )
    review = ReviewResult(
        is_valid=False,
        severity="major",
        feedback=f"Feedback {secret} <think>private chain</think>",
        findings=[finding],
        fixes=[]
        if plan_route
        else [
            FixSuggestion(find="self.wait(1)", replace="self.wait(2)", reason=f"Reason {secret}")
        ],
    )
    pipeline, _ = stage(reviews=(review,))
    result = pipeline.review(scene)
    visible = result.model_dump_json() + (tmp_path / result.report.path).read_text()
    assert secret not in visible and "private chain" not in visible
    assert "<redacted>" in visible


@pytest.mark.parametrize("mode", ["strict", "relaxed"])
@pytest.mark.parametrize(
    "limit,expected_calls,expected_route", [(2, 1, "exhausted"), (5, 3, "coder")]
)
def test_rejected_rollback_candidates_share_budget_and_return_consumed_counters(
    tmp_path,
    mode,
    limit,
    expected_calls,
    expected_route,
):
    scene = configured(
        source(inputs(tmp_path)), GENERATION_MODE=mode, MAX_LOW_RISK_REVIEW_ROUNDS=limit
    )
    scene.scene.review_round = 1
    candidates = [
        SOURCE.replace("self.wait(1)", f"self.wait({duration})") for duration in (2, 3, 4)
    ]
    scene.scene.candidates = [
        CodeCandidate(
            code_file=ObjectStore(tmp_path).source(1, text),
            code_sha256=sha256_text(text),
            class_name="Pause",
            verification="smoke",
            inherited_elements_sha256=sha256_text(""),
        )
        for text in candidates
    ]
    rejected = ReviewResult(
        is_valid=False,
        severity="major",
        feedback="Wrong implementation",
        findings=[
            ReviewFinding(
                category="runtime",
                severity="major",
                evidence_type="source_code",
                why="Wrong implementation",
            )
        ],
    )
    pipeline, calls = stage(reviews=(rejected,) * 3)
    result = pipeline.rollback(scene)
    assert not result.accepted and result.repair_route == expected_route
    assert len(calls.review) == expected_calls
    assert result.review_round == 1 + expected_calls
    assert result.review_signature and result.identical_review_count == 1
    assert scene.scene.review_round == 1 and scene.scene.code_sha256 == sha256_text(SOURCE)
