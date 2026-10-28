"""Render fixes use real static inspection and immutable objects, never execution."""

import json
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from test_pipeline_scene import SOURCE, inputs
from test_pipeline_scene_candidates import configured

from kd1_anime.agent_runtime.tools import ToolCancelled, ToolContext
from kd1_anime.agents.render_error_parser import extract_render_error
from kd1_anime.domain.candidates import CodeCandidate
from kd1_anime.domain.dependencies import WorkIdentity
from kd1_anime.domain.execution import SubmissionAttempt
from kd1_anime.objects import ObjectStore
from kd1_anime.pipeline.repair import RenderRepairOutput, RenderRepairPipeline
from kd1_anime.run_store import StoredSlurmJob, sha256_text

ERROR = "Traceback (most recent call last):\nValueError: animation failed\n"
FIXED = SOURCE.replace("self.wait(1)", "self.wait(2)")


def scene_input(tmp_path, *, code=SOURCE, error=ERROR, **config):
    scene = configured(inputs(tmp_path), MAX_FIX_ATTEMPTS=4, **config)
    scene.scene.code_file = ObjectStore(tmp_path).source(1, code)
    scene.scene.code_sha256 = sha256_text(code)
    scene.scene.class_name = "Pause"
    scene.scene.revision = 9
    scene.scene.input_sha256 = "b" * 64
    scene.scene.reviewed = True
    log = tmp_path / "error.log"
    log.write_text(error)
    scene.scene.slurm_job = StoredSlurmJob(
        job_id="42",
        scene_id=1,
        script_path=scene.scene.code_file,
        log_out="output.log",
        log_err="error.log",
        media_dir="videos/attempt_old",
        scene_class_name="Pause",
        submitted_at=1,
        status="FAILED",
        code_sha256=scene.scene.code_sha256,
        render_profile=scene.render_profile,
        execution_sha256=scene.execution_spec.digest(),
    )
    scene.scene.submission = SubmissionAttempt(
        backend="slurm",
        scene_revision=9,
        code_sha256=scene.scene.code_sha256,
        status="submitted",
        job_id="42",
    )
    return replace(scene, submission_identity=WorkIdentity(9, scene.scene.input_sha256))


def context(cancelled=lambda: False):
    return ToolContext(
        deadline=time.monotonic() + 60,
        cancelled=cancelled,
        record_external_id=lambda _: pytest.fail("Repair must not submit work"),
    )


def pipeline(*, candidate=FIXED, service=None, callback=None):
    calls = SimpleNamespace(agents=[], fixes=[], records=[], contexts=[], requests=[])
    from kd1_anime.request_budget import RequestContext

    request = RequestContext(reserve=lambda *a, **k: None, emit=lambda *a: None)
    control = SimpleNamespace(checkpoint=lambda: None)

    class Fixer:
        def __init__(self):
            calls.agents.append(self)

        def fix(self, original, error_log, **kwargs):
            calls.fixes.append((original, error_log, kwargs))
            assert self.control is control
            assert self.request_context is request
            if callback:
                callback(original, error_log, kwargs)
            return candidate

    def request_context(scene_id, stage):
        calls.requests.append((scene_id, stage))
        return request

    corpus = SimpleNamespace(
        context=lambda **kwargs: calls.contexts.append(kwargs) or "Known failure recipe",
        record=lambda case: calls.records.append(case) or True,
    )
    return RenderRepairPipeline(
        service,
        fixer_factory=Fixer,
        request_context=request_context,
        control=control,
        failure_cases=corpus,
    ), calls


def report(scene, output):
    return json.loads((scene.root / output.report.path).read_text())


def test_fixer_proposal_is_bound_immutable_and_requires_separate_review(tmp_path):
    scene = scene_input(tmp_path)
    before = scene.scene.model_dump(mode="json")
    stage, calls = pipeline()
    result = stage.repair(scene, context())
    assert scene.scene.model_dump(mode="json") == before
    assert (tmp_path / scene.scene.code_file).read_text() == SOURCE
    assert (tmp_path / result.source.path).read_text() == FIXED
    assert result.source.sha256 == sha256_text(FIXED)
    assert result.identity == WorkIdentity(9, "b" * 64)
    assert result.original_code_sha256 == scene.scene.code_sha256
    assert result.repair_route == "review" and result.continuity_rebuild_required
    assert result.fix_attempts == 1 and result.identical_error_count == 1
    assert not result.scene.reviewed and result.scene.review_round == 0
    assert result.scene.slurm_job is None and result.scene.submission is None
    assert result.scene.artifact is None and not result.scene.rendered
    assert result.scene.static_verification.code_sha256 == result.source.sha256
    assert result.scene.execution_verification.status == "not_run"
    assert result.scene.visual_verification.status == "not_run"
    assert calls.requests == [(1, "fix")] and len(calls.agents) == 1
    assert calls.fixes[0][2]["failure_case_context"] == "Known failure recipe"
    evidence = report(scene, result)
    assert evidence["original_code_sha256"] == scene.scene.code_sha256
    assert evidence["candidate_code_sha256"] == result.source.sha256
    assert evidence["error_fingerprint"] == result.error_fingerprint
    assert evidence["technical_spec_sha256"] == result.technical_spec_sha256
    assert evidence["context_sha256"] == result.context_sha256
    assert evidence["identity"]["revision"] == 9
    assert evidence["inspections"][-1]["accepted"]
    assert calls.records[0].verification == "validated"
    assert calls.records[0].fixed_code_sha256 == result.source.sha256
    assert (tmp_path / result.source.path).stat().st_mode & 0o777 == 0o600
    assert (tmp_path / result.report.path).stat().st_mode & 0o777 == 0o600
    with pytest.raises(ValidationError):
        RenderRepairOutput.model_validate({**result.model_dump(), "unexpected": True})


def test_deterministic_api_patch_does_not_construct_fixer(tmp_path):
    old = SOURCE.replace(
        "self.wait(1)",
        "dot = Dot()\n        dot.setColor(RED)\n        self.add(dot)\n        self.wait(1)",
    )
    scene = scene_input(tmp_path, code=old, error="AttributeError: Dot has no attribute setColor")
    stage, calls = pipeline()
    result = stage.repair(scene, context())
    assert result.strategy == "deterministic_patch" and result.repair_route == "review"
    assert ".set_color(" in (tmp_path / result.source.path).read_text()
    assert not calls.agents and not calls.fixes
    assert result.fix_attempts == 1 and len(calls.records) == 1


def test_normalized_candidate_gets_fresh_readonly_inspection(tmp_path, monkeypatch):
    scene = scene_input(tmp_path)
    stage, _ = pipeline()
    inspect = stage.acceptor.inspect_candidate
    inspections = []

    def observe(code, plan, **kwargs):
        inspections.append((sha256_text(code), kwargs["allow_repair"]))
        return inspect(code, plan, **kwargs)

    monkeypatch.setattr(stage.acceptor, "inspect_candidate", observe)
    result = stage.repair(scene, context())
    assert inspections[-2:] == [(result.source.sha256, True), (result.source.sha256, False)]
    assert result.scene.reviewed is False


@pytest.mark.parametrize(
    "status",
    [
        "RUNNING",
        "PENDING",
        "UNKNOWN",
        "GONE",
        "CANCEL_FAILED",
        "RUN_TIMEOUT",
        "CANCELLED",
        "COMPLETED",
    ],
)
def test_unconfirmed_or_cancelled_job_never_repairs(tmp_path, status):
    scene = scene_input(tmp_path)
    scene.scene.slurm_job.status = status
    stage, calls = pipeline()
    with pytest.raises(ValueError):
        stage.repair(scene, context())
    assert not calls.agents and not calls.records


@pytest.mark.parametrize("status", ["pending", "unknown"])
@pytest.mark.parametrize("historical", [False, True])
def test_uncertain_submission_cannot_be_erased_by_repair(tmp_path, status, historical):
    scene = scene_input(tmp_path)
    attempt = SubmissionAttempt(
        backend="slurm", scene_revision=9, code_sha256=scene.scene.code_sha256, status=status
    )
    if historical:
        scene.scene.submission_attempts.append(attempt)
    else:
        scene.scene.submission = attempt
    stage, calls = pipeline()
    with pytest.raises(ValueError, match="Uncertain"):
        stage.repair(scene, context())
    assert not calls.fixes


@pytest.mark.parametrize("mutation", ["source", "job_hash", "execution", "technical", "submission"])
def test_stale_identity_fails_closed_before_agent(tmp_path, mutation):
    scene = scene_input(tmp_path)
    if mutation == "source":
        (tmp_path / scene.scene.code_file).write_text("tampered")
    elif mutation == "job_hash":
        scene.scene.slurm_job.code_sha256 = "0" * 64
    elif mutation == "execution":
        scene.scene.slurm_job.execution_sha256 = "0" * 64
    elif mutation == "technical":
        scene.scene.technical_spec_sha256 = "0" * 64
    else:
        scene.scene.submission = SubmissionAttempt(
            backend="slurm",
            scene_revision=9,
            code_sha256=scene.scene.code_sha256,
            status="submitted",
            job_id="99",
        )
    stage, calls = pipeline()
    with pytest.raises(ValueError, match="identity"):
        stage.repair(scene, context())
    assert not calls.agents


def test_infrastructure_error_does_not_rewrite_code_or_spend_attempt(tmp_path):
    scene = scene_input(
        tmp_path, error="sbatch: error: Invalid account or account/partition combination specified"
    )
    stage, calls = pipeline()
    result = stage.repair(scene, context())
    assert result.repair_route == "infrastructure" and result.fix_attempts == 0
    assert result.infra_retries == result.scene.infra_retries == 1
    assert scene.scene.infra_retries == 0
    assert report(scene, result)["infra_retries"] == 1
    assert result.source.sha256 == scene.scene.code_sha256
    assert not calls.agents and not calls.records


def add_candidate(scene, code, verification, *, inherited=None):
    scene.scene.candidates.append(
        CodeCandidate(
            code_file=ObjectStore(scene.root).source(1, code),
            code_sha256=sha256_text(code),
            class_name="Pause",
            verification=verification,
            inherited_elements_sha256=inherited or sha256_text(scene.scene.inherited_elements_code),
        )
    )


def test_budget_exhaustion_ranks_proven_candidates_but_does_not_restore_receipts(tmp_path):
    scene = scene_input(tmp_path)
    scene.scene.fix_attempts = 4
    add_candidate(scene, FIXED, "smoke")
    best = SOURCE.replace("self.wait(1)", "self.wait(3)")
    add_candidate(scene, best, "rendered")
    stage, calls = pipeline()
    result = stage.repair(scene, context())
    assert result.strategy == "best_rollback" and result.repair_route == "review"
    assert result.source.sha256 == sha256_text(best) and result.fix_attempts == 4
    assert not result.scene.reviewed and not result.scene.rendered
    assert result.scene.execution_verification.status == "not_run"
    assert not calls.agents


def test_missing_log_rolls_back_only_matching_untampered_candidate(tmp_path):
    scene = scene_input(tmp_path, error="")
    add_candidate(
        scene, SOURCE.replace("self.wait(1)", "self.wait(3)"), "rendered", inherited="0" * 64
    )
    add_candidate(scene, SOURCE.replace("self.wait(1)", "self.wait(4)"), "rendered")
    (tmp_path / scene.scene.candidates[-1].code_file).write_text("tampered")
    add_candidate(scene, FIXED, "smoke")
    stage, calls = pipeline()
    result = stage.repair(scene, context())
    assert result.source.sha256 == sha256_text(FIXED) and result.strategy == "best_rollback"
    assert not calls.agents


@pytest.mark.parametrize("mode", ["strict", "relaxed"])
def test_persisted_budget_is_honored_in_every_mode(tmp_path, mode):
    scene = scene_input(tmp_path, GENERATION_MODE=mode)
    scene.scene.fix_attempts = 4
    stage, calls = pipeline()
    result = stage.repair(scene, context())
    assert result.repair_route == "exhausted" and result.fix_attempts == 4
    assert not calls.agents and result.source.sha256 == scene.scene.code_sha256


def test_unchanged_fix_preserves_snapshot_across_resume(tmp_path):
    scene = scene_input(tmp_path)
    stage, calls = pipeline(candidate=SOURCE)
    first = stage.repair(scene, context())
    assert first.repair_route == "coder" and first.fix_attempts == 1
    resumed = replace(scene, scene=first.scene)
    second = stage.repair(resumed, context())
    assert second.stagnant_repair_count == 1 and second.identical_error_count == 2
    assert second.fix_attempts == 2 and len(calls.agents) == 2
    assert second.scene.last_repair_code_sha256 == scene.scene.code_sha256


def stagnant(scene):
    fp = extract_render_error(ERROR, source_code=SOURCE, renderer="cairo").fingerprint
    scene.scene.last_repair_code_sha256 = scene.scene.code_sha256
    scene.scene.last_repair_error_fp = fp
    scene.scene.stagnant_repair_count = 1
    return scene


@pytest.mark.parametrize("mode", ["strict", "relaxed"])
def test_stagnation_uses_inspected_safe_candidate_without_llm(tmp_path, monkeypatch, mode):
    scene = stagnant(scene_input(tmp_path, GENERATION_MODE=mode, MAX_STAGNANT_ATTEMPTS=2))
    monkeypatch.setattr("kd1_anime.pipeline.repair.build_safe_scene_code", lambda *args: FIXED)
    stage, calls = pipeline()
    result = stage.repair(scene, context())
    assert result.strategy == "safe_template" and result.repair_route == "review"
    assert result.scene.stagnant_repair_count == 0 and not result.scene.reviewed
    assert result.fix_attempts == 1 and not calls.agents


def test_plan_changing_teaching_fallback_is_explicit_proposal_not_installation(
    tmp_path, monkeypatch
):
    scene = stagnant(scene_input(tmp_path, MAX_STAGNANT_ATTEMPTS=2, SAFE_FALLBACK_ENABLED=True))
    monkeypatch.setattr(
        "kd1_anime.pipeline.repair.build_safe_scene_code",
        lambda *args: "import os\nos.system('bad')",
    )
    monkeypatch.setattr(
        "kd1_anime.pipeline.repair.is_high_confidence_geometry_conflict", lambda *args: True
    )
    stage, calls = pipeline()
    result = stage.repair(scene, context())
    assert result.repair_route == "plan" and result.strategy == "safe_teaching_plan"
    assert result.proposed_plan is not None and result.continuity_rebuild_required
    assert result.scene.plan == scene.scene.plan and result.source.sha256 == scene.scene.code_sha256
    assert not result.scene.safe_fallback_used and not calls.agents
    assert report(scene, result)["proposed_plan"] == result.proposed_plan.model_dump(mode="json")


def test_invalid_fixer_candidate_never_installs_unsafe_code(tmp_path):
    scene = scene_input(tmp_path)
    stage, calls = pipeline(candidate="import os\nos.system('unsafe')")
    result = stage.repair(scene, context())
    assert result.repair_route == "coder" and result.fix_attempts == 1
    assert result.source.sha256 == scene.scene.code_sha256
    assert not calls.records and result.feedback
    assert not report(scene, result)["inspections"][-1]["accepted"]


def test_agent_cannot_mutate_shared_teaching_contracts(tmp_path):
    scene = scene_input(tmp_path)
    before = scene.lesson_spec.model_dump(mode="json")

    def mutate(original, error, options):
        options["lesson_spec"].misconceptions.append("Worker-owned mutation")

    stage, _ = pipeline(callback=mutate)
    stage.repair(scene, context())
    assert scene.lesson_spec.model_dump(mode="json") == before


def test_cancellation_before_or_after_fixer_never_publishes_candidate(tmp_path):
    scene = scene_input(tmp_path)
    stage, calls = pipeline()
    with pytest.raises(ToolCancelled):
        stage.repair(scene, context(cancelled=lambda: True))
    assert not calls.agents
    cancelled = []
    stage, calls = pipeline(callback=lambda *args: cancelled.append(True))
    with pytest.raises(ToolCancelled):
        stage.repair(scene, context(cancelled=lambda: bool(cancelled)))
    assert len(calls.fixes) == 1 and not calls.records
    assert scene.scene.code_sha256 == sha256_text(SOURCE)


def test_logs_and_corpus_context_are_redacted(tmp_path, monkeypatch):
    from kd1_anime.config import settings

    monkeypatch.setattr(settings, "LLM_API_KEY", "fixture-private-key")
    scene = scene_input(tmp_path, error="ValueError: fixture-private-key failed")
    stage, calls = pipeline()
    result = stage.repair(scene, context())
    assert "fixture-private-key" not in (tmp_path / result.report.path).read_text()
    assert "fixture-private-key" not in calls.fixes[0][1]
    assert "fixture-private-key" not in calls.records[0].message


def test_render_service_reads_exact_restored_job_and_direct_render_never_calls_agent(tmp_path):
    scene = scene_input(tmp_path)
    jobs = []
    service = SimpleNamespace(get_error_log=lambda *, job: jobs.append(job) or ERROR)
    stage, calls = pipeline(service=service)
    result = stage.repair(scene, context())
    assert jobs[0].job_id == "42" and jobs[0].log_err == tmp_path / "error.log"
    assert result.repair_route == "review"
    with pytest.raises(ValueError, match="Direct render"):
        stage.repair(replace(scene, direct_render=True), context())
    assert len(calls.agents) == 1


def test_fixer_failure_rolls_back_but_cancellation_is_never_swallowed(tmp_path):
    scene = scene_input(tmp_path)
    add_candidate(scene, FIXED, "smoke")

    def fail(*args):
        raise RuntimeError("Provider unavailable")

    stage, calls = pipeline(callback=fail)
    result = stage.repair(scene, context())
    assert result.strategy == "best_rollback" and result.repair_route == "review"
    assert result.fix_attempts == 1 and len(calls.agents) == 1

    def cancel(*args):
        raise ToolCancelled("Stop")

    stage, calls = pipeline(callback=cancel)
    with pytest.raises(ToolCancelled):
        stage.repair(scene, context())
    assert not calls.records


def test_each_repair_constructs_independent_fixer_and_uses_nonstreaming_transport(
    tmp_path, monkeypatch
):
    from kd1_anime.agents.auto_fixer import AutoFixerAgent

    scene = scene_input(tmp_path)
    stages, calls = [], []

    def completion(self, **kwargs):
        calls.append(kwargs)
        return FIXED

    def factory():
        agent = AutoFixerAgent()
        stages.append(agent)
        return agent

    monkeypatch.setattr(AutoFixerAgent, "call_llm", completion)
    stage, recorded = pipeline()
    stage.fixer_factory = factory
    stage.repair(scene, context())
    stage.repair(scene, context())
    assert len(stages) == 2 and stages[0] is not stages[1]
    assert len(calls) == 2 and all(call["stream"] is False for call in calls)
    assert len(recorded.records) == 2


def test_real_safe_template_is_static_only_and_requires_review(tmp_path):
    scene = stagnant(scene_input(tmp_path, MAX_STAGNANT_ATTEMPTS=2, CODEGEN_MODE="python"))
    stage, calls = pipeline()
    result = stage.repair(scene, context())
    assert result.repair_route == "review" and result.strategy == "safe_template"
    assert result.scene.static_verification.status == "passed"
    assert result.scene.execution_verification.status == "not_run"
    assert not result.scene.reviewed and not calls.agents


def test_stagnation_counter_survives_changed_but_still_failing_fixes(tmp_path):
    scene = scene_input(tmp_path, MAX_STAGNANT_ATTEMPTS=4)
    stage, _ = pipeline()
    first = stage.repair(scene, context())
    failed = first.scene
    failed.slurm_job = scene.scene.slurm_job.model_copy(update={"code_sha256": first.source.sha256})
    failed.last_error_fp = first.error_fingerprint
    stage, _ = pipeline(candidate=SOURCE.replace("self.wait(1)", "self.wait(3)"))
    second = stage.repair(replace(scene, scene=failed), context())
    assert second.repair_route == "review" and second.stagnant_repair_count == 1
    assert second.fix_attempts == 2


def test_technical_spec_hash_and_owned_agent_copy_are_bound_to_result(tmp_path):
    from kd1_anime.domain.technical import TechnicalSpec

    scene = scene_input(tmp_path)
    scene.scene.technical_spec = TechnicalSpec(scene_id=1)
    scene.scene.technical_spec_sha256 = sha256_text(scene.scene.technical_spec.model_dump_json())
    bound_hash = scene.scene.technical_spec_sha256

    def mutate(original, error, options):
        options["technical_spec"].scene_id = 2

    stage, _ = pipeline(callback=mutate)
    result = stage.repair(scene, context())
    assert result.repair_route == "review"
    assert result.technical_spec_sha256 == bound_hash
    assert result.scene.static_verification.technical_spec_sha256 == bound_hash
    assert scene.scene.technical_spec.scene_id == result.scene.technical_spec.scene_id == 1
    assert report(scene, result)["technical_spec_sha256"] == bound_hash


def test_advanced_scene_revision_keeps_exact_dispatch_identity_valid(tmp_path):
    scene = scene_input(tmp_path)
    scene.scene.revision = 12
    stage, calls = pipeline()
    result = stage.repair(scene, context())
    assert result.repair_route == "review"
    assert result.identity == WorkIdentity(12, scene.scene.input_sha256)
    assert scene.scene.submission.scene_revision == scene.submission_identity.revision == 9
    assert len(calls.agents) == 1


@pytest.mark.parametrize("dispatch", [None, WorkIdentity(8, "b" * 64), WorkIdentity(9, "c" * 64)])
def test_missing_or_stale_dispatch_identity_rejects_repair(tmp_path, dispatch):
    scene = replace(scene_input(tmp_path), submission_identity=dispatch)
    stage, calls = pipeline()
    with pytest.raises(ValueError, match="submission identity"):
        stage.repair(scene, context())
    assert not calls.agents and not calls.records


def test_zero_infrastructure_retry_budget_stops_without_spending_attempt(tmp_path):
    scene = scene_input(tmp_path, error="Permission denied", MAX_INFRA_RETRIES=0)
    stage, calls = pipeline()
    result = stage.repair(scene, context())
    assert result.repair_route == "exhausted" and result.strategy == "infrastructure_budget"
    assert result.infra_retries == result.scene.infra_retries == 0
    assert result.fix_attempts == 0 and not calls.agents and not calls.records
    assert result.source.sha256 == scene.scene.code_sha256


def test_infrastructure_retry_count_survives_resume_and_enforces_exact_limit(tmp_path):
    scene = scene_input(tmp_path, error="Permission denied", MAX_INFRA_RETRIES=2)
    scene.scene.infra_retries = 1
    stage, calls = pipeline()
    first = stage.repair(scene, context())
    assert first.repair_route == "infrastructure"
    assert first.infra_retries == first.scene.infra_retries == 2
    assert scene.scene.infra_retries == 1
    second = stage.repair(replace(scene, scene=first.scene), context())
    assert second.repair_route == "exhausted"
    assert second.infra_retries == second.scene.infra_retries == 2
    assert second.fix_attempts == 0 and not calls.agents
    assert report(scene, second)["infra_retries"] == 2
