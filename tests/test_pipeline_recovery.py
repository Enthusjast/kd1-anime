"""Recovery uses deterministic scheduler observations, never starts render code."""

import json
from dataclasses import replace

import pytest
from test_application import ArtifactBackend
from test_pipeline_render import stage as stage

from kd1_anime.agent_runtime.state import AgentRunState
from kd1_anime.cluster.jobs import PollObservation, RenderJob
from kd1_anime.domain.dependencies import WorkIdentity
from kd1_anime.domain.execution import StageAttempts
from kd1_anime.pipeline.recovery import (
    RecoveryOutput,
    RecoveryPipeline,
    plan_resume,
    plan_retry,
    validate_completed_run,
)
from kd1_anime.prompt_identity import capture_generation_identity
from kd1_anime.rag.models import RagReceipt, RagRuntimeProfile
from kd1_anime.run_store import RunManifest


def persist(
    inputs,
    attempt,
    *,
    kind="reconcile_submission",
    external_id=None,
    cancellation_status="not_requested",
    state=None,
    committed=False,
):
    original = inputs.submission_identity
    current = WorkIdentity(inputs.scene.revision, inputs.scene.input_sha256)

    def action(kind, identity):
        value = {
            "kind": kind,
            "run_id": inputs.run_id,
            "scene_id": 1,
            "resource_id": f"run:{inputs.run_id}/scene:1",
            "work_identity": {"revision": identity.revision, "input_sha256": identity.input_sha256},
            "input_sha256": identity.input_sha256,
        }
        if kind == "submit_render":
            value.update(
                backend=attempt.backend,
                class_name=inputs.scene.class_name,
                source={
                    "kind": "code",
                    "path": inputs.scene.code_file,
                    "sha256": inputs.scene.code_sha256,
                },
            )
        return value

    raw = {
        "agent": {
            "submissions": [
                {
                    "scene_id": 1,
                    "attempt": attempt.model_dump(mode="json"),
                    "cancellation_status": cancellation_status,
                }
            ],
            "intents": [
                {
                    "intent_id": "dispatch",
                    "status": "completed" if committed else "unknown",
                    "action": action("submit_render", original),
                },
                {"intent_id": "reconcile", "status": "dispatched", "action": action(kind, current)},
            ],
            "invocations": [
                {
                    "invocation_id": "dispatch-invocation",
                    "intent_id": "dispatch",
                    "action": "submit_render",
                    "status": "completed" if committed else "unknown",
                    "external_job_id": external_id,
                    "work_identity": action("submit_render", original)["work_identity"],
                },
                {
                    "intent_id": "reconcile",
                    "action": kind,
                    "status": "running",
                    "work_identity": action(kind, current)["work_identity"],
                },
            ],
        }
    }
    if state is not None:
        raw["state"] = state
    raw["agent"]["committed_invocations"] = ["dispatch-invocation"] if committed else []
    (inputs.root / "manifest.json").write_text(json.dumps(raw), encoding="utf-8")
    return raw


class RecoveryBackend(ArtifactBackend):
    def __init__(self):
        super().__init__()
        self.recover_calls = []
        self.poll_calls = []
        self.status = "PENDING"
        self.recovered_id = "41"
        self.recovery_error = None
        self.gone_outcome = None
        self.valid = True

    def recover_submission(self, attempt, *, scene_id, source, root, render_profile):
        self.recover_calls.append(attempt.token)
        if self.recovery_error:
            raise self.recovery_error
        if self.recovered_id is None:
            return None
        identifier = self.recovered_id
        return RenderJob(
            job_id=identifier,
            scene_id=scene_id,
            script_path=root / "scenes" / f"render_{scene_id}.sh",
            log_out=root / "logs" / f"scene_{scene_id}_{identifier}.out",
            log_err=root / "logs" / f"scene_{scene_id}_{identifier}.err",
            media_dir=root / "videos" / f"scene_{scene_id}" / f"attempt_{attempt.token}",
            scene_class_name="",
            submitted_at=attempt.created_at.timestamp(),
            execution_sha256=self.execution_spec.digest(),
            code_sha256=attempt.code_sha256,
            render_profile=render_profile,
        )

    def poll(self, job_ids):
        self.poll_calls.append(tuple(job_ids))
        return PollObservation({identifier: self.status for identifier in job_ids}, {})

    def classify_gone(self, job):
        return self.gone_outcome

    def validate_completed_job(self, job):
        return super().validate_completed_job(job) if self.valid else False

    def cancel_job(self, identifier):
        raise AssertionError(
            "Recovery must return a cancellation prerequisite, not cancel secretly"
        )

    def submit_scene(self, *args, **kwargs):
        raise AssertionError("Recovery cannot submit a replacement render")


@pytest.fixture
def recovery(stage):
    inputs, attempt, context, identifiers, _, _ = stage
    inputs = replace(
        inputs,
        scene=inputs.scene.model_copy(
            update={
                "revision": attempt.scene_revision + 3,
                "input_sha256": "b" * 64,
                "submission": attempt,
                "plan_ready": True,
            }
        ),
        submission_identity=WorkIdentity(attempt.scene_revision, "b" * 64),
    )
    persist(inputs, attempt)
    backend = RecoveryBackend()
    return inputs, attempt, context, identifiers, backend, RecoveryPipeline(backend)


def adopted(recovery):
    inputs, attempt, context, _, backend, pipeline = recovery
    output = pipeline.reconcile_submission(inputs, attempt, context)
    inputs = replace(
        inputs,
        scene=inputs.scene.model_copy(
            update={
                "slurm_job": output.job,
                "submission": output.submission,
            }
        ),
    )
    persist(inputs, output.submission, kind="reconcile_render")
    return inputs, context, backend, pipeline


def manifest_for(inputs):
    return RunManifest(
        schema_version=11,
        agent=AgentRunState(
            session_id="session-one",
            generation_identity=capture_generation_identity(),
            run_input_sha256="a" * 64,
        ),
        run_id=inputs.run_id,
        user_prompt="Pause",
        output_path="final.mp4",
        run_config=inputs.config,
        execution_spec=inputs.execution_spec,
        render_profile=inputs.render_profile,
        scenes={1: inputs.scene.model_copy(deep=True)},
    )


@pytest.mark.parametrize("status", ["pending", "unknown"])
def test_unique_record_recovers_original_dispatch_not_current_poll_revision(recovery, status):
    inputs, attempt, context, identifiers, backend, pipeline = recovery
    attempt = attempt.model_copy(update={"status": status})
    inputs = replace(inputs, scene=inputs.scene.model_copy(update={"submission": attempt}))
    persist(inputs, attempt)
    output = pipeline.reconcile_submission(inputs, attempt, context)
    assert output.status == "adopted"
    assert output.job.scene_class_name == "Pause"
    assert output.submission.scene_revision != inputs.scene.revision
    assert output.submission.job_id == "41" and identifiers == ["41"]
    assert backend.recover_calls == [attempt.token]
    assert inputs.scene.slurm_job is None and attempt.status == status
    assert RecoveryOutput.model_validate_json(output.model_dump_json()) == output


@pytest.mark.parametrize("saved_in", ["attempt", "invocation"])
def test_crash_after_returned_id_never_searches_or_resubmits(recovery, saved_in):
    inputs, attempt, context, identifiers, backend, pipeline = recovery
    if saved_in == "attempt":
        attempt = attempt.model_copy(update={"status": "submitted", "job_id": "73"})
        inputs = replace(inputs, scene=inputs.scene.model_copy(update={"submission": attempt}))
    persist(inputs, attempt, external_id="73")
    output = pipeline.reconcile_submission(inputs, attempt, context)
    assert output.status == "adopted" and output.job.job_id == "73"
    assert output.submission.job_id == "73" and identifiers == ["73"]
    assert not backend.recover_calls


@pytest.mark.parametrize(
    "failure", [None, RuntimeError("multiple matching jobs"), RuntimeError("query unavailable")]
)
def test_crash_before_id_absent_ambiguous_or_unknown_blocks(recovery, failure):
    inputs, attempt, context, _, backend, pipeline = recovery
    backend.recovered_id = None
    backend.recovery_error = failure
    output = pipeline.reconcile_submission(inputs, attempt, context)
    assert output.status == "blocked" and output.job is None
    assert output.submission == attempt


@pytest.mark.parametrize(
    "mutation",
    ["intent", "reservation", "dispatch", "code", "identity", "private_path", "returned_id"],
)
def test_invalid_binding_rejects_before_scheduler_query(recovery, mutation):
    inputs, attempt, context, _, backend, pipeline = recovery
    raw = persist(inputs, attempt)
    if mutation == "intent":
        raw["agent"]["intents"] = raw["agent"]["intents"][:1]
    elif mutation == "reservation":
        raw["agent"]["submissions"] = []
    elif mutation == "dispatch":
        raw["agent"]["intents"][0]["action"]["work_identity"]["input_sha256"] = "c" * 64
    elif mutation == "code":
        (inputs.root / inputs.scene.code_file).write_text("changed", encoding="utf-8")
    elif mutation == "identity":
        inputs = replace(inputs, submission_identity=WorkIdentity(inputs.scene.revision, "a" * 64))
    elif mutation == "private_path":
        inputs = replace(
            inputs, scene=inputs.scene.model_copy(update={"code_file": "../outside.py"})
        )
    else:
        attempt = attempt.model_copy(update={"status": "submitted", "job_id": "11"})
        inputs = replace(inputs, scene=inputs.scene.model_copy(update={"submission": attempt}))
        raw = persist(inputs, attempt, external_id="12")
    (inputs.root / "manifest.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        pipeline.reconcile_submission(inputs, attempt, context)
    assert not backend.recover_calls and not backend.poll_calls


def test_local_pending_attempt_is_never_adopted(recovery):
    inputs, attempt, context, _, backend, pipeline = recovery
    backend.name = "local"
    attempt = attempt.model_copy(update={"backend": "local"})
    inputs = replace(inputs, scene=inputs.scene.model_copy(update={"submission": attempt}))
    persist(inputs, attempt)
    assert pipeline.reconcile_submission(inputs, attempt, context).status == "blocked"
    assert not backend.recover_calls


@pytest.mark.parametrize("status", ["RUNNING", "UNKNOWN", "GONE", "CANCEL_FAILED"])
def test_restored_local_nonterminal_cannot_restart_or_poll(recovery, status):
    inputs, context, backend, pipeline = adopted(recovery)
    backend.name = "local"
    job = inputs.scene.slurm_job.model_copy(
        update={
            "job_id": "local-123456abcdef",
            "backend": "local",
            "status": status,
        }
    )
    attempt = inputs.scene.submission.model_copy(update={"backend": "local", "job_id": job.job_id})
    inputs = replace(
        inputs, scene=inputs.scene.model_copy(update={"slurm_job": job, "submission": attempt})
    )
    persist(inputs, attempt, kind="reconcile_render")
    assert pipeline.reconcile_scene(inputs, context).status == "blocked"
    assert not backend.poll_calls
    assert plan_retry(manifest_for(inputs), 1).blocked == (1,)


def test_confirmed_completed_job_restores_exact_artifact(recovery):
    inputs, context, backend, pipeline = adopted(recovery)
    backend.status = "COMPLETED"
    output = pipeline.reconcile_scene(inputs, context)
    assert output.status == "rendered"
    assert output.artifact.video_sha256 == output.job.output_sha256
    assert output.artifact.execution_sha256 == inputs.execution_spec.digest()
    assert output.artifact.scene_class_name == "Pause"
    assert inputs.scene.artifact is None


def test_completed_missing_artifact_keeps_job_during_propagation(recovery):
    inputs, context, backend, pipeline = adopted(recovery)
    backend.status = "COMPLETED"
    backend.valid = False
    output = pipeline.reconcile_scene(inputs, context)
    assert output.status == "pending" and output.job.status == "COMPLETED"
    assert output.artifact is None


@pytest.mark.parametrize(
    "status,expected",
    [("UNKNOWN", "blocked"), ("GONE", "restart"), ("FAILED", "failed"), ("RUNNING", "pending")],
)
def test_restored_jobs_require_confirmed_observations(recovery, status, expected):
    inputs, context, backend, pipeline = adopted(recovery)
    backend.status = status
    output = pipeline.reconcile_scene(inputs, context)
    assert output.status == expected
    assert inputs.scene.slurm_job.status == "PENDING"


def test_failed_cancellation_cannot_be_erased_by_gone_observation(recovery):
    inputs, context, backend, pipeline = adopted(recovery)
    inputs = replace(
        inputs,
        scene=inputs.scene.model_copy(
            update={
                "slurm_job": inputs.scene.slurm_job.model_copy(update={"status": "CANCEL_FAILED"}),
            }
        ),
    )
    persist(inputs, inputs.scene.submission, kind="reconcile_render")
    backend.status = "GONE"
    assert pipeline.reconcile_scene(inputs, context).status == "blocked"
    assert plan_retry(manifest_for(inputs), 1).blocked == (1,)


@pytest.mark.parametrize("mutation", ["missing", "corrupt", "missing_hash"])
def test_completed_run_rejects_missing_or_corrupt_final(recovery, mutation):
    inputs, *_ = recovery
    manifest = manifest_for(inputs)
    final = inputs.root / "final.mp4"
    final.write_bytes(b"complete")
    from kd1_anime.rendering import sha256_file

    manifest.status = "completed"
    manifest.final_video = "final.mp4"
    manifest.final_video_sha256 = sha256_file(final)
    if mutation == "missing":
        final.unlink()
    elif mutation == "corrupt":
        final.write_bytes(b"changed")
    else:
        manifest.final_video_sha256 = ""
    with pytest.raises(RuntimeError):
        validate_completed_run(manifest, inputs.root)
    with pytest.raises(RuntimeError):
        plan_retry(manifest, 1)


def test_retry_requires_parent_cancellation_then_preserves_all_counters(recovery):
    inputs, _, _, _ = adopted(recovery)
    manifest = manifest_for(inputs)
    previous = manifest.scenes[1]
    previous.fix_attempts = 3
    previous.infra_retries = 2
    previous.visual_fix_attempts = 4
    previous.review_round = 5
    previous.plan_review_round = 2
    previous.stagnant_repair_count = 2
    previous.stage_attempts = [StageAttempts(operation="generation", input_sha256="a" * 64, used=3)]
    before = manifest.model_dump(mode="json")
    pending = plan_retry(manifest, 1)
    assert pending.cancellations[0].job_id == "41" and not pending.scenes
    assert manifest.model_dump(mode="json") == before
    previous.slurm_job.cancelled = True
    previous.slurm_job.status = "CANCELLED"
    ready = plan_retry(manifest, 1)
    scene = ready.scenes[0].replacement
    assert ready.scenes[0].identity == WorkIdentity(previous.revision, previous.input_sha256)
    assert ready.clear_final and scene.slurm_job is None and scene.submission is None
    for name in (
        "fix_attempts",
        "infra_retries",
        "visual_fix_attempts",
        "review_round",
        "plan_review_round",
        "stagnant_repair_count",
        "stage_attempts",
        "submission_attempts",
    ):
        assert getattr(scene, name) == getattr(previous, name)
    assert manifest.request_budget.model_dump() == before["request_budget"]


def test_resume_failure_retains_attempts_and_budget(recovery):
    inputs, *_ = recovery
    inputs = replace(
        inputs,
        scene=inputs.scene.model_copy(
            update={
                "submission": None,
                "failed": True,
                "give_up": True,
                "fix_attempts": 4,
            }
        ),
    )
    manifest = manifest_for(inputs)
    before = manifest.model_dump(mode="json")
    plan = plan_resume(manifest)
    scene = plan.scenes[0].replacement
    assert not scene.failed and not scene.give_up and scene.fix_attempts == 4
    assert scene.code_file == inputs.scene.code_file
    assert manifest.model_dump(mode="json") == before


def test_rag_drift_invalidates_receipt_dependencies_without_budget_reset(recovery):
    inputs, *_ = recovery
    inputs = replace(
        inputs, scene=inputs.scene.model_copy(update={"submission": None, "fix_attempts": 3})
    )
    manifest = manifest_for(inputs)
    manifest.rag_receipts = {
        "propose_code_candidate:1:0": RagReceipt(
            stage="code",
            query_sha256="d" * 64,
            status="active",
            index_sha256="e" * 64,
        )
    }
    before = manifest.model_dump(mode="json")
    profile = RagRuntimeProfile(enabled=True, status="active", index_sha256="f" * 64)
    plan = plan_resume(manifest, current_rag_profile=profile)
    assert plan.rag_profile == profile
    assert plan.drop_rag_receipt_keys == ("propose_code_candidate:1:0",)
    scene = plan.scenes[0].replacement
    assert not scene.code_file and not scene.reviewed and scene.fix_attempts == 3
    assert scene.plan_ready and not plan.invalidate_planning
    assert manifest.model_dump(mode="json") == before


def test_rag_drift_does_not_drop_receipts_before_required_cancellation(recovery):
    inputs, _, _, _ = adopted(recovery)
    manifest = manifest_for(inputs)
    manifest.rag_receipts = {
        "propose_plan:0:0": RagReceipt(
            stage="plan",
            query_sha256="d" * 64,
            status="disabled",
        )
    }
    plan = plan_resume(manifest, current_rag_profile=RagRuntimeProfile(enabled=True))
    assert plan.cancellations and not plan.scenes
    assert not plan.rag_profile and not plan.drop_rag_receipt_keys
    assert not plan.invalidate_planning


@pytest.mark.parametrize("mutation", ["missing", "corrupt"])
def test_previously_verified_artifact_cannot_be_silently_reblessed(recovery, mutation):
    inputs, context, backend, pipeline = adopted(recovery)
    backend.status = "COMPLETED"
    rendered = pipeline.reconcile_scene(inputs, context)
    inputs = replace(
        inputs,
        scene=inputs.scene.model_copy(
            update={
                "slurm_job": rendered.job,
                "artifact": rendered.artifact,
                "rendered": True,
            }
        ),
    )
    persist(inputs, inputs.scene.submission, kind="reconcile_render")
    path = inputs.root / rendered.artifact.video_path
    if mutation == "missing":
        path.unlink()
    else:
        path.write_bytes(b"corrupt")
    output = pipeline.reconcile_scene(inputs, context)
    assert output.status == "restart" and output.invalidate_artifact
    assert output.artifact is None
    assert not path.exists() if mutation == "missing" else path.read_bytes() == b"corrupt"


def test_recovery_restores_reserved_resource_metadata(recovery):
    from kd1_anime.cluster.resource_estimator import RenderResourceProfile

    inputs, attempt, context, _, _, pipeline = recovery
    resources = RenderResourceProfile(cpus_per_task=4, mem_gb=8, time_limit="00:10:00")
    inputs = replace(inputs, scene=inputs.scene.model_copy(update={"resource_profile": resources}))
    persist(inputs, attempt)
    output = pipeline.reconcile_submission(inputs, attempt, context)
    assert output.job.resource_profile == resources
    assert output.job.scene_class_name == inputs.scene.class_name


def test_recovered_scheduler_record_must_use_current_private_attempt_paths(recovery):
    inputs, attempt, context, _, backend, pipeline = recovery
    original = backend.recover_submission

    def wrong_path(*args, **kwargs):
        job = original(*args, **kwargs)
        job.media_dir = inputs.root / "videos" / "scene_1" / "attempt_000000000000"
        return job

    backend.recover_submission = wrong_path
    with pytest.raises(ValueError):
        pipeline.reconcile_submission(inputs, attempt, context)


def test_completed_run_checks_public_output_and_immutable_video(recovery):
    inputs, *_ = recovery
    manifest = manifest_for(inputs)
    final = inputs.root / "final.mp4"
    final.write_bytes(b"complete")
    from kd1_anime.rendering import sha256_file

    manifest.status = "completed"
    manifest.output_path = manifest.final_video = str(final)
    manifest.final_video_sha256 = sha256_file(final)
    private = inputs.root / "artifacts" / "video" / (manifest.final_video_sha256 + ".mp4")
    private.parent.mkdir(parents=True)
    private.write_bytes(b"complete")
    manifest.artifact_refs = {"final_video": private.relative_to(inputs.root).as_posix()}
    assert validate_completed_run(manifest, inputs.root) == final
    private.write_bytes(b"corrupt")
    with pytest.raises(RuntimeError):
        validate_completed_run(manifest, inputs.root)


def test_rag_invalidation_follows_dependencies_not_unrelated_scenes(recovery):
    from kd1_anime.domain.dependencies import SceneDependency
    from kd1_anime.services.dependencies import DependencyService

    inputs, *_ = recovery
    inputs = replace(inputs, scene=inputs.scene.model_copy(update={"submission": None}))
    manifest = manifest_for(inputs)
    source = manifest.scenes[1]
    dependent_plan = source.plan.model_copy(
        update={
            "scene_id": 2,
            "dependencies_complete": True,
            "dependencies": [SceneDependency(source_scene_id=1, kind="teaching")],
        }
    )
    unrelated_plan = source.plan.model_copy(
        update={
            "scene_id": 3,
            "dependencies_complete": True,
            "dependencies": [],
        }
    )
    manifest.scenes[2] = source.model_copy(
        deep=True, update={"plan": dependent_plan, "fix_attempts": 2}
    )
    manifest.scenes[3] = source.model_copy(deep=True, update={"plan": unrelated_plan})
    manifest.dependency_graph.bind(
        2,
        dependent_plan.dependencies,
        {
            1: {
                "code": source.code_sha256,
                "contract": DependencyService.contract(source.plan, source.technical_spec),
                "teaching": DependencyService.teaching(source.plan),
            },
        },
    )
    manifest.rag_receipts = {
        "propose_code_candidate:1:0": RagReceipt(
            stage="code",
            query_sha256="d" * 64,
            status="disabled",
        )
    }
    plan = plan_resume(manifest, current_rag_profile=RagRuntimeProfile(enabled=True))
    assert {entry.scene_id for entry in plan.scenes} == {1, 2}
    assert plan.scenes[1].replacement.fix_attempts == 2
    assert manifest.scenes[3].code_file == source.code_file


@pytest.mark.parametrize("status", ["FAILED", "CANCELLED", "COMPLETED"])
def test_confirmed_terminal_local_job_can_detach_without_adoption(recovery, status):
    inputs, context, backend, pipeline = adopted(recovery)
    backend.name = "local"
    job = inputs.scene.slurm_job.model_copy(
        update={
            "job_id": "local-123456abcdef",
            "backend": "local",
            "status": status,
        }
    )
    attempt = inputs.scene.submission.model_copy(update={"backend": "local", "job_id": job.job_id})
    inputs = replace(
        inputs, scene=inputs.scene.model_copy(update={"slurm_job": job, "submission": attempt})
    )
    persist(inputs, attempt, kind="reconcile_render")
    assert pipeline.reconcile_scene(inputs, context).status == "restart"
    assert not backend.poll_calls


def test_recovery_query_requires_running_persisted_invocation(recovery):
    inputs, attempt, context, _, backend, pipeline = recovery
    raw = persist(inputs, attempt)
    raw["agent"]["invocations"][1]["status"] = "reserved"
    (inputs.root / "manifest.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="persisted reconciliation"):
        pipeline.reconcile_submission(inputs, attempt, context)
    assert not backend.recover_calls


def test_detail_rag_receipt_invalidates_that_plan_not_retry_counters(recovery):
    inputs, *_ = recovery
    inputs = replace(
        inputs,
        scene=inputs.scene.model_copy(
            update={
                "submission": None,
                "plan_reviewed": True,
                "plan_review_round": 3,
            }
        ),
    )
    manifest = manifest_for(inputs)
    manifest.rag_receipts = {
        "detail_scene:1:0": RagReceipt(
            stage="detail",
            query_sha256="d" * 64,
            status="disabled",
        )
    }
    plan = plan_resume(manifest, current_rag_profile=RagRuntimeProfile(enabled=True))
    replacement = plan.scenes[0].replacement
    assert not replacement.plan_ready and not replacement.plan_reviewed
    assert replacement.plan_review_round == 3
    assert plan.scenes[0].next_action == "detail_scene"


def test_original_dispatch_semantics_cannot_change_while_job_is_uncertain(recovery):
    inputs, attempt, context, _, backend, pipeline = recovery
    inputs = replace(inputs, scene=inputs.scene.model_copy(update={"input_sha256": "c" * 64}))
    persist(inputs, attempt)
    with pytest.raises(ValueError, match="original dispatch identity"):
        pipeline.reconcile_submission(inputs, attempt, context)
    assert not backend.recover_calls


@pytest.mark.parametrize("associated", ["dispatch", "ghost", "missing", None])
def test_reservation_association_disambiguates_identical_ghost_dispatch(recovery, associated):
    from copy import deepcopy

    inputs, attempt, context, identifiers, backend, pipeline = recovery
    attempt = attempt.model_copy(update={"status": "unknown"})
    inputs = replace(inputs, scene=inputs.scene.model_copy(update={"submission": attempt}))
    raw = persist(inputs, attempt, external_id="73")
    ghost = deepcopy(raw["agent"]["intents"][0])
    ghost["intent_id"] = "ghost"
    ghost["status"] = "dispatched"
    if associated == "ghost":
        ghost["action"]["class_name"] = "DifferentScene"
    raw["agent"]["intents"].insert(0, ghost)
    ghost_invocation = deepcopy(raw["agent"]["invocations"][0])
    ghost_invocation.update(intent_id="ghost", status="running", external_job_id=None)
    raw["agent"]["invocations"].insert(0, ghost_invocation)
    raw["agent"]["submissions"][0]["intent_id"] = associated
    (inputs.root / "manifest.json").write_text(json.dumps(raw), encoding="utf-8")
    if associated == "dispatch":
        output = pipeline.reconcile_submission(inputs, attempt, context)
        assert output.status == "adopted" and output.job.job_id == "73"
        assert output.submission.job_id == "73" and identifiers == ["73"]
    else:
        with pytest.raises(ValueError):
            pipeline.reconcile_submission(inputs, attempt, context)
        assert not identifiers
    assert not backend.recover_calls and not backend.poll_calls


@pytest.mark.parametrize("field", ["backend", "class_name", "source", "work_identity"])
def test_associated_dispatch_still_requires_exact_submission_inputs(recovery, field):
    inputs, attempt, context, _, backend, pipeline = recovery
    raw = persist(inputs, attempt)
    raw["agent"]["submissions"][0]["intent_id"] = "dispatch"
    action = raw["agent"]["intents"][0]["action"]
    if field == "backend":
        action[field] = "local"
    elif field == "class_name":
        action[field] = "DifferentScene"
    elif field == "source":
        action[field]["sha256"] = "0" * 64
    else:
        action[field]["revision"] += 1
    (inputs.root / "manifest.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="exact original dispatch intent"):
        pipeline.reconcile_submission(inputs, attempt, context)
    assert not backend.recover_calls and not backend.poll_calls


@pytest.mark.parametrize("cancellation", ["failed", "unknown", "not_requested"])
@pytest.mark.parametrize("observation", ["GONE", "UNKNOWN", "COMPLETED"])
def test_uncommitted_returned_id_retains_cancellation_until_confirmed_terminal(
    recovery,
    cancellation,
    observation,
):
    inputs, attempt, context, identifiers, backend, pipeline = recovery
    persist(inputs, attempt, external_id="73", cancellation_status=cancellation, state="ERROR")
    output = pipeline.reconcile_submission(inputs, attempt, context)
    assert output.status == "adopted" and output.job.job_id == "73"
    assert output.job.status == "CANCEL_FAILED" and not output.job.cancelled
    assert identifiers == ["73"] and not backend.recover_calls
    inputs = replace(
        inputs,
        scene=inputs.scene.model_copy(
            update={
                "slurm_job": output.job,
                "submission": output.submission,
            }
        ),
    )
    persist(
        inputs,
        output.submission,
        kind="reconcile_render",
        external_id="73",
        cancellation_status=cancellation,
        state="ERROR",
    )
    backend.status = observation
    observed = pipeline.reconcile_scene(inputs, context)
    if observation == "COMPLETED":
        assert observed.status == "rendered" and observed.job.status == "COMPLETED"
        assert observed.artifact is not None
    else:
        assert observed.status == "blocked" and observed.job.status == "CANCEL_FAILED"
        assert observed.artifact is None
        retry_manifest = manifest_for(inputs)
        assert plan_retry(retry_manifest, 1).blocked == (1,)


def test_confirmed_submit_cancellation_is_preserved_in_recovered_descriptor(recovery):
    inputs, attempt, context, _, backend, pipeline = recovery
    persist(inputs, attempt, external_id="73", cancellation_status="succeeded", state="ERROR")
    output = pipeline.reconcile_submission(inputs, attempt, context)
    assert output.job.status == "CANCELLED" and output.job.cancelled
    assert not backend.recover_calls and not backend.poll_calls


def test_later_run_error_does_not_invent_cancellation_for_committed_submit(recovery):
    inputs, attempt, context, _, backend, pipeline = recovery
    persist(inputs, attempt, external_id="73", state="ERROR", committed=True)
    output = pipeline.reconcile_submission(inputs, attempt, context)
    assert output.job.status == "PENDING" and not output.job.cancelled
    assert not backend.recover_calls


@pytest.mark.parametrize("cancellation", ["failed", "unknown"])
def test_restored_saved_job_also_respects_durable_cancellation_metadata(recovery, cancellation):
    inputs, context, backend, pipeline = adopted(recovery)
    persist(
        inputs,
        inputs.scene.submission,
        kind="reconcile_render",
        cancellation_status=cancellation,
        state="ERROR",
    )
    backend.status = "GONE"
    output = pipeline.reconcile_scene(inputs, context)
    assert output.status == "blocked" and output.job.status == "CANCEL_FAILED"
    assert not inputs.scene.slurm_job.cancelled


def test_confirmed_terminal_observation_outweighs_old_cancellation_uncertainty(recovery):
    inputs, context, backend, pipeline = adopted(recovery)
    backend.status = "COMPLETED"
    completed = pipeline.reconcile_scene(inputs, context)
    inputs = replace(inputs, scene=inputs.scene.model_copy(update={"slurm_job": completed.job}))
    persist(inputs, inputs.scene.submission, cancellation_status="failed", state="ERROR")
    output = pipeline.reconcile_submission(inputs, inputs.scene.submission, context)
    assert output.job.status == "COMPLETED" and not output.job.cancelled
