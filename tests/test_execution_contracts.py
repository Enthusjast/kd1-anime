"""Regression tests for durable execution, bounded requests and isolated execution."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import pytest
from openai import BadRequestError

from kd1_anime.agents.base import BaseAgent
from kd1_anime.api_catalog import lookup_symbol
from kd1_anime.cluster.execution import ExecutionPolicy
from kd1_anime.cluster.render_backend import LocalRenderBackend
from kd1_anime.cluster.slurm import SlurmDispatcher
from kd1_anime.config import settings
from kd1_anime.domain.dependencies import DependencyGraph, SceneDependency, StaleWorkError
from kd1_anime.domain.execution import (
    StageAttempts,
    SubmissionAttempt,
    transition_run,
    validate_scene_transition,
)
from kd1_anime.domain.planning import ScenePlan
from kd1_anime.eval.benchmark import fixed_tasks, offline_baseline
from kd1_anime.orchestrator import Orchestrator, PipelineContext, RunPaths, SceneState, State
from kd1_anime.request_budget import RequestBudget, RequestBudgetExceeded, RequestReservation
from kd1_anime.run_store import RunRepository
from kd1_anime.services.dependencies import DependencyService
from kd1_anime.services.program_compiler import ProgramCompileError, expression_source
from kd1_anime.stats import summarize_manifest
from kd1_anime.verification import ExecutionVerification, VisualVerification


def plan(scene_id=1, **updates):
    return ScenePlan(
        scene_id=scene_id,
        title="example",
        duration_seconds=4,
        purpose="show exact inputs",
        math_concept="circle",
        visual_design="fixed",
        camera_movement="fixed",
        visual_flow=["reveal"],
        key_moments=["ending"],
        computation="r=2",
        **updates,
    )


def workflow(tmp_path, *, scene_limit=64, run_limit=512):
    run_id = "20261019-120000-1234abcd"
    root = tmp_path / "runs" / run_id
    for directory in (root, root / "scenes", root / "logs", root / "videos"):
        directory.mkdir(parents=True, exist_ok=True)
    paths = RunPaths(
        run_id, root, root / "scenes", root / "logs", root / "videos", root / "output.mp4"
    )
    ctx = PipelineContext(
        "prompt",
        paths=paths,
        scene_states={1: SceneState(plan())},
        request_budget=RequestBudget(scene_limit=scene_limit, run_limit=run_limit),
    )
    orchestrator = Orchestrator()
    orchestrator._ctx = ctx
    return orchestrator, ctx


def agent_for(orchestrator, scene_id=1, create=None):
    profile = replace(
        settings.main_llm_profile(),
        api_key="test-key",
        model="fake-model",
        max_retries=1,
        silent_stream=False,
    )
    agent = BaseAgent(profile=profile)
    agent.request_context = orchestrator._request_context(scene_id)
    agent._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    return agent


def response(content="OK"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")],
        usage=None,
    )


def test_checkpoint_normalizes_before_constructing_snapshot(tmp_path):
    orchestrator, ctx = workflow(tmp_path)
    state = ctx.scene_states[1]
    state.rendered = True
    state.execution_verification = ExecutionVerification(
        status="passed", scope="formal_video", code_sha256="a" * 64
    )
    orchestrator._checkpoint(ctx, State.CODING)
    saved = RunRepository(tmp_path).load(ctx.paths.run_id).scenes[1]
    assert saved.rendered is state.rendered is False
    assert saved.artifact is state.artifact is None
    assert saved.execution_verification.status == state.execution_verification.status == "not_run"


def test_failed_checkpoint_does_not_advance_revision_or_publish_success(monkeypatch, tmp_path):
    orchestrator, ctx = workflow(tmp_path)
    orchestrator._checkpoint(ctx, State.INIT)
    before = (ctx.paths.root / "manifest.json").read_bytes()
    revision = ctx.manifest_revision
    monkeypatch.setattr(
        "kd1_anime.orchestrator.write_manifest",
        lambda *args: (_ for _ in ()).throw(OSError("disk full")),
    )
    with pytest.raises(OSError, match="disk full"):
        orchestrator._checkpoint(ctx, State.CODING)
    assert ctx.manifest_revision == revision
    assert orchestrator._stop_event.is_set()
    assert (ctx.paths.root / "manifest.json").read_bytes() == before


def test_concurrent_requests_are_reserved_on_disk_before_dispatch(tmp_path):
    orchestrator, ctx = workflow(tmp_path, scene_limit=3, run_limit=5)
    calls = []

    def create(**kwargs):
        manifest = RunRepository(tmp_path).load(ctx.paths.run_id)
        assert 1 <= len(manifest.request_budget.reservations) <= 3
        calls.append(kwargs)
        return response()

    def request(_):
        agent = agent_for(orchestrator, create=create)
        try:
            agent._create_completion(agent.client, model="fake-model")
        except RequestBudgetExceeded:
            return False
        return True

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(request, range(20)))
    assert sum(outcomes) == len(calls) == 3
    assert len(RunRepository(tmp_path).load(ctx.paths.run_id).request_budget.reservations) == 3


def test_run_budget_includes_run_wide_and_visual_profiles(tmp_path):
    orchestrator, ctx = workflow(tmp_path, scene_limit=10, run_limit=2)
    orchestrator._request_context().reserve("main", "fake")
    orchestrator._request_context(1).reserve("visual", "fake-vision")
    with pytest.raises(RequestBudgetExceeded, match="运行"):
        orchestrator._request_context(1).reserve("main", "fake")
    assert [item.profile for item in ctx.request_budget.reservations] == ["main", "visual"]


def test_failed_reservation_persistence_prevents_network_call(monkeypatch, tmp_path):
    orchestrator, _ = workflow(tmp_path)
    calls = []
    agent = agent_for(orchestrator, create=lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(
        "kd1_anime.orchestrator.write_manifest",
        lambda *args: (_ for _ in ()).throw(OSError("disk full")),
    )
    with pytest.raises(OSError):
        agent._create_completion(agent.client, model="fake-model")
    assert calls == []
    assert orchestrator._stop_event.is_set()


def test_failed_network_request_remains_consumed_after_restore(tmp_path):
    orchestrator, ctx = workflow(tmp_path, scene_limit=1)

    def fail(**kwargs):
        raise TimeoutError("unknown remote outcome")

    agent = agent_for(orchestrator, create=fail)
    with pytest.raises(TimeoutError):
        agent._create_completion(agent.client, model="fake")
    manifest = RunRepository(tmp_path).load_for_resume(ctx.paths.run_id)
    restored = Orchestrator._context_from_manifest(manifest, ctx.paths.root)
    assert len(restored.request_budget.reservations) == 1
    orchestrator._ctx = restored
    with pytest.raises(RequestBudgetExceeded):
        orchestrator._request_context(1).reserve("main", "fake")


def test_compatibility_fallback_consumes_another_actual_request(tmp_path):
    import httpx

    orchestrator, ctx = workflow(tmp_path)
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        if "max_tokens" in kwargs:
            raise BadRequestError(
                "max_tokens unsupported",
                response=httpx.Response(400, request=httpx.Request("POST", "https://invalid.test")),
                body=None,
            )
        return response()

    agent = agent_for(orchestrator, create=create)
    agent.check_api_available()
    assert len(calls) == len(ctx.request_budget.reservations) == 2


@pytest.mark.parametrize("version", [4, 5, 6, 7, 8, 9])
def test_legacy_resume_rejects_before_lock_or_any_external_action(monkeypatch, tmp_path, version):
    orchestrator, ctx = workflow(tmp_path)
    orchestrator._checkpoint(ctx, State.INIT)
    path = ctx.paths.root / "manifest.json"
    import json

    payload = json.loads(path.read_text())
    payload["schema_version"] = version
    path.write_text(json.dumps(payload))
    ctx.paths.root.chmod(0o755)
    before = path.read_bytes()
    monkeypatch.setattr(settings, "WORKSPACE_DIR", tmp_path)
    monkeypatch.setattr(
        "kd1_anime.orchestrator.lock_run", lambda *args: pytest.fail("legacy run was locked")
    )
    with pytest.raises(ValueError, match="只读"):
        orchestrator.resume(ctx.paths.run_id)
    assert path.read_bytes() == before
    assert ctx.paths.root.stat().st_mode & 0o777 == 0o755


def test_dependency_invalidation_distinguishes_code_from_export_contract():
    graph = DependencyGraph()
    initial = {
        1: {"code": "a" * 64, "contract": "b" * 64},
        2: {"code": "c" * 64, "contract": "d" * 64},
        3: {"code": "e" * 64, "contract": "f" * 64},
    }
    graph.bind(2, [SceneDependency(source_scene_id=1, kind="handoff")], initial)
    graph.bind(3, [SceneDependency(source_scene_id=1, kind="code_context")], initial)
    changed_code = {**initial, 1: {"code": "9" * 64, "contract": "b" * 64}}
    assert graph.affected(1, changed_code) == {3}
    changed_contract = {**initial, 1: {"code": "9" * 64, "contract": "8" * 64}}
    assert graph.affected(1, changed_contract) == {2, 3}


def test_dependency_graph_preserves_independent_scene_and_propagates_transitively():
    graph = DependencyGraph()
    nodes = {i: {"code": str(i) * 64, "contract": str(i) * 64} for i in range(1, 5)}
    graph.bind(2, [SceneDependency(source_scene_id=1, kind="code_context")], nodes)
    graph.bind(3, [SceneDependency(source_scene_id=2, kind="handoff")], nodes)
    graph.bind(4, [], nodes)
    assert graph.affected(1, nodes) == {2, 3}
    with pytest.raises(ValueError):
        graph.bind(1, [SceneDependency(source_scene_id=3, kind="handoff")], nodes)


def test_explicit_independence_does_not_inherit_numeric_previous_scene():
    independent = plan(2, dependencies_complete=True)
    assert DependencyService.dependencies(independent, [plan(), independent]) == []


def test_stale_worker_identity_changes_with_plan_or_source(tmp_path):
    orchestrator, ctx = workflow(tmp_path)
    state = ctx.scene_states[1]
    identity = orchestrator._work_identity(ctx, state)
    state.code = "changed without a checkpoint"
    with pytest.raises(StaleWorkError):
        orchestrator._accept_work(ctx, state, identity)


def test_execution_fsm_rejects_unapproved_stage_changes():
    with pytest.raises(ValueError):
        transition_run("finished", "executing")
    with pytest.raises(ValueError):
        validate_scene_transition("coded", "monitoring", revised=False)
    validate_scene_transition("visual_accepted", "coded", revised=True)


def test_pending_submission_cannot_be_automatically_resubmitted(tmp_path):
    orchestrator, ctx = workflow(tmp_path)
    state = ctx.scene_states[1]
    state.submission = SubmissionAttempt(backend="slurm", scene_revision=0, code_sha256="a" * 64)
    with pytest.raises(RuntimeError, match="禁止自动重提"):
        orchestrator._scene_submit(ctx, 1, state)


def test_execution_environment_never_carries_provider_secrets():
    source = {
        "PATH": "/bin",
        "LANG": "zh_CN.UTF-8",
        "LLM_API_KEY": "hidden",
        "VISUAL_LLM_API_KEY": "hidden",
        "RAG_EMBEDDING_API_KEY": "hidden",
        "PYTHONPATH": "/untrusted",
        "APPTAINERENV_LLM_API_KEY": "hidden",
        "SSH_AUTH_SOCK": "hidden",
    }
    env = ExecutionPolicy.environment(source, renderer="opengl", platform="egl")
    assert env["PATH"] == "/bin"
    assert env["PYOPENGL_PLATFORM"] == "egl"
    assert all("KEY" not in key for key in env)
    assert "PYTHONPATH" not in env
    assert "SSH_AUTH_SOCK" not in env
    assert not issubclass(LocalRenderBackend, SlurmDispatcher)


def test_required_container_is_not_bypassed_on_local_backend(monkeypatch):
    from kd1_anime.rendering import RenderProfile
    from kd1_anime.services.rendering import RenderingService

    monkeypatch.setattr(settings, "SLURM_REQUIRE_CONTAINER", True)
    monkeypatch.setattr(settings, "SLURM_CONTAINER_IMAGE", None)
    with pytest.raises(RuntimeError, match="SLURM_REQUIRE_CONTAINER"):
        RenderingService(LocalRenderBackend(), "local").preflight(RenderProfile.current())


def test_api_catalog_distinguishes_missing_from_unknown_version():
    assert lookup_symbol("Circle", "0.20.1").status == "found"
    assert "VMobject" in lookup_symbol("Circle", "0.20.1").bases
    assert lookup_symbol("InventedShape", "0.20.1").status == "missing"
    assert lookup_symbol("InventedShape", "99.0").status == "unknown"


@pytest.mark.parametrize(
    "expression", ["__import__('os')", "x.__class__", "[x for x in (1,2)]", "sin(x, x)", "f(x)"]
)
def test_function_grammar_rejects_dynamic_python(expression):
    with pytest.raises(ProgramCompileError):
        expression_source(expression)


def test_fixed_benchmark_has_24_independent_contract_cases():
    tasks = fixed_tasks()
    assert len(tasks) == len({task.task_id for task in tasks}) == 24
    report = offline_baseline()
    assert all(row["contract_check_passed"] for row in report["tasks"]), report
    assert report["rendered"] is False
    assert all(row["model_requests"] == 0 for row in report["tasks"])


def test_visual_unknown_is_not_verified(tmp_path):
    orchestrator, ctx = workflow(tmp_path)
    ctx.scene_states[1].visual_verification = VisualVerification(status="unknown")
    orchestrator._checkpoint(ctx, State.CODING)
    report = summarize_manifest(orchestrator._manifest)
    assert report["scenes"]["visual_unknown"] == 1
    assert report["scenes"]["visual_verified"] == 0
    assert report["scenes"]["visual_accepted"] == 0


# Captured before the per-test fixture replaces network probes.
_run_endpoint_probe = Orchestrator._probe_run_endpoints
CODE = "from manim import *\nclass Demo(Scene):\n    def construct(self): self.wait()\n"


def prepare_submission(tmp_path):
    orchestrator, ctx = workflow(tmp_path)
    state = ctx.scene_states[1]
    state.code = CODE
    state.class_name = "Demo"
    state.reviewed = True
    orchestrator._write_private(ctx.paths.scenes / "scene_1.py", CODE)
    return orchestrator, ctx, state


def submitted_job(ctx, state):
    from kd1_anime.cluster.jobs import RenderJob
    from kd1_anime.run_store import sha256_text

    return RenderJob(
        job_id="123",
        scene_id=1,
        submitted_at=1.0,
        script_path=ctx.paths.scenes / "render_1.sh",
        log_out=ctx.paths.logs / "out",
        log_err=ctx.paths.logs / "err",
        media_dir=ctx.paths.videos / "attempt",
        scene_class_name="Demo",
        code_sha256=sha256_text(state.code),
        render_profile=ctx.render_profile,
    )


def test_snapshot_has_no_mutable_aliases_to_uncommitted_state(tmp_path):
    orchestrator, ctx = workflow(tmp_path)
    orchestrator._checkpoint(ctx, State.INIT)
    snapshot = orchestrator._manifest
    ctx.scene_states[1].plan.title = "uncommitted"
    ctx.request_budget.reserve(RequestReservation(stage="code", profile="main", model="fake"))
    assert snapshot.scenes[1].plan.title == "example"
    assert snapshot.request_budget.reservations == []


def test_invalid_checkpoint_never_overwrites_valid_snapshot(tmp_path):
    orchestrator, ctx = workflow(tmp_path)
    orchestrator._checkpoint(ctx, State.INIT)
    before = (ctx.paths.root / "manifest.json").read_bytes()
    ctx.scene_states[1].reviewed = True
    with pytest.raises(ValueError, match="不变量"):
        orchestrator._checkpoint(ctx, State.REVIEWING)
    assert (ctx.paths.root / "manifest.json").read_bytes() == before
    with pytest.raises(ValueError, match="不变量"):
        orchestrator._checkpoint(ctx, State.DONE)


def test_run_probe_is_reserved_before_contacting_endpoint(monkeypatch, tmp_path):
    orchestrator, ctx = workflow(tmp_path)
    monkeypatch.setattr(settings, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(settings, "LLM_MODEL", "fake")
    calls = []

    def create(**kwargs):
        assert len(RunRepository(tmp_path).load(ctx.paths.run_id).request_budget.reservations) == 1
        calls.append(kwargs)
        return response()

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(BaseAgent, "client", property(lambda self: client))
    _run_endpoint_probe(orchestrator, ctx)
    assert len(calls) == 1


@pytest.mark.parametrize("cancelled", [True, False])
def test_submission_persistence_failure_cancels_and_never_resubmits(
    monkeypatch, tmp_path, cancelled
):
    from kd1_anime.run_store import write_manifest

    orchestrator, ctx, state = prepare_submission(tmp_path)
    calls = []
    cancellations = []

    def submit(*args, **kwargs):
        saved = RunRepository(tmp_path).load(ctx.paths.run_id)
        assert saved.scenes[1].submission.token == kwargs["attempt_token"]
        assert saved.scenes[1].submission.scene_revision == saved.scenes[1].revision
        calls.append(kwargs)
        return submitted_job(ctx, state)

    def save(path, manifest):
        if manifest.scenes[1].slurm_job is not None:
            raise OSError("disk full after external submission")
        return write_manifest(path, manifest)

    monkeypatch.setattr(orchestrator.slurm, "submit_scene", submit)
    monkeypatch.setattr(
        orchestrator.slurm, "cancel_job", lambda job_id: cancellations.append(job_id) or cancelled
    )
    monkeypatch.setattr("kd1_anime.orchestrator.write_manifest", save)
    events = []
    orchestrator._callback = lambda event, data: events.append(event)
    orchestrator._scene_submit(ctx, 1, state)
    assert cancellations == ["123"]
    assert len(calls) == 1
    assert "scene_submitted" not in events
    assert state.slurm_job is None
    assert orchestrator._uncommitted_jobs["123"].status == (
        "CANCELLED" if cancelled else "CANCEL_FAILED"
    )
    saved = RunRepository(tmp_path).load(ctx.paths.run_id)
    assert saved.scenes[1].submission.status == "pending"
    with pytest.raises(RuntimeError, match="禁止自动重提"):
        orchestrator._scene_submit(ctx, 1, state)


def test_uncertain_submission_is_persisted_and_blocks_repetition(monkeypatch, tmp_path):
    orchestrator, ctx, state = prepare_submission(tmp_path)
    monkeypatch.setattr(
        orchestrator.slurm,
        "submit_scene",
        lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError("no acknowledged job id")),
    )
    orchestrator._scene_submit(ctx, 1, state)
    saved = RunRepository(tmp_path).load(ctx.paths.run_id)
    assert saved.scenes[1].submission.status == "unknown"
    with pytest.raises(RuntimeError, match="禁止自动重提"):
        orchestrator._scene_submit(ctx, 1, state)


def test_submission_rejects_unreviewed_code_before_external_action(monkeypatch, tmp_path):
    orchestrator, ctx, state = prepare_submission(tmp_path)
    state.reviewed = False
    monkeypatch.setattr(
        orchestrator.slurm,
        "submit_scene",
        lambda *args, **kwargs: pytest.fail("unreviewed source was submitted"),
    )
    with pytest.raises(ValueError, match="尚未通过审查"):
        orchestrator._scene_submit(ctx, 1, state)


@pytest.mark.parametrize(
    "outputs, error",
    [
        (["123|{name}", "123|{name}"], False),
        (["123|{name}", "124|{name}"], True),
        (["", ""], False),
    ],
)
def test_submission_reconciliation_requires_a_unique_exact_job(
    monkeypatch, tmp_path, outputs, error
):
    from kd1_anime.cluster.render_backend import SlurmRenderBackend
    from kd1_anime.rendering import RenderProfile

    root = tmp_path / "run"
    attempt = SubmissionAttempt(backend="slurm", scene_revision=0, code_sha256="a" * 64)
    name = f"kd1-run-s1-{attempt.token}"
    rows = iter(outputs)
    monkeypatch.setattr(
        "kd1_anime.cluster.render_backend.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=next(rows).format(name=name)),
    )
    backend = SlurmRenderBackend()
    if error:
        with pytest.raises(RuntimeError, match="多个作业"):
            backend.recover_submission(
                attempt,
                scene_id=1,
                source=root / "scenes/scene_1.py",
                root=root,
                render_profile=RenderProfile.current(),
            )
    else:
        job = backend.recover_submission(
            attempt,
            scene_id=1,
            source=root / "scenes/scene_1.py",
            root=root,
            render_profile=RenderProfile.current(),
        )
        assert (job.job_id if job else None) == ("123" if outputs[0] else None)
        if job:
            assert job.media_dir == root / "videos/scene_1" / f"attempt_{attempt.token}"


@pytest.mark.parametrize("mode", ["strict", "relaxed"])
def test_zero_fix_budget_disables_all_repairs(monkeypatch, tmp_path, mode):
    orchestrator, ctx, state = prepare_submission(tmp_path)
    ctx.generation_mode = mode
    state.slurm_job = submitted_job(ctx, state)
    monkeypatch.setattr(settings, "MAX_FIX_ATTEMPTS", 0)
    monkeypatch.setattr(
        "kd1_anime.orchestrator.AutoFixerAgent", lambda: pytest.fail("zero budget created a fixer")
    )
    orchestrator._scene_fix(ctx, 1, state)
    assert state.fix_attempts == 0
    assert state.give_up
    assert ctx.request_budget.reservations == []


def test_retry_limits_and_counters_survive_restore_and_settings_changes(monkeypatch, tmp_path):
    orchestrator, ctx = workflow(tmp_path)
    ctx.retry_limits = {"MAX_FIX_ATTEMPTS": 2}
    ctx.scene_states[1].fix_attempts = 2
    ctx.scene_states[1].plan_replan_attempts = 3
    orchestrator._checkpoint(ctx, State.CODING)
    restored = Orchestrator._context_from_manifest(
        RunRepository(tmp_path).load(ctx.paths.run_id), ctx.paths.root
    )
    monkeypatch.setattr(settings, "MAX_FIX_ATTEMPTS", 20)
    assert orchestrator._limit(restored, "MAX_FIX_ATTEMPTS") == 2
    assert restored.scene_states[1].fix_attempts == 2
    assert restored.scene_states[1].plan_replan_attempts == 3


def test_generation_attempts_are_reserved_before_call_and_retained_on_resume(monkeypatch, tmp_path):
    from kd1_anime.exceptions import ValidationError

    orchestrator, ctx = workflow(tmp_path)
    ctx.retry_limits = {"CODE_VALIDATION_ATTEMPTS": 2}
    calls = []

    class InterruptedCoder:
        def generate_code(self, *args, **kwargs):
            saved = RunRepository(tmp_path).load(ctx.paths.run_id)
            calls.append(saved.scenes[1].stage_attempts[0].used)
            raise RuntimeError("interrupted before result")

    monkeypatch.setattr("kd1_anime.orchestrator.CoderAgent", InterruptedCoder)
    with pytest.raises(RuntimeError, match="interrupted"):
        orchestrator._generate_validated_code(ctx.scene_states[1].plan)
    restored = Orchestrator._context_from_manifest(
        RunRepository(tmp_path).load(ctx.paths.run_id), ctx.paths.root
    )
    resumed = Orchestrator()
    resumed._ctx = restored
    monkeypatch.setattr(settings, "CODE_VALIDATION_ATTEMPTS", 10)
    with pytest.raises(RuntimeError, match="interrupted"):
        resumed._generate_validated_code(restored.scene_states[1].plan)
    with pytest.raises(ValidationError, match="预算已耗尽"):
        resumed._generate_validated_code(restored.scene_states[1].plan)
    assert calls == [1, 2]
    assert RunRepository(tmp_path).load(ctx.paths.run_id).scenes[1].stage_attempts[0].used == 2


def test_new_review_round_has_distinct_generation_attempts(monkeypatch, tmp_path):
    orchestrator, ctx = workflow(tmp_path)
    ctx.retry_limits = {"CODE_VALIDATION_ATTEMPTS": 1}

    class Coder:
        def generate_code(self, *args, **kwargs):
            return CODE

    monkeypatch.setattr("kd1_anime.orchestrator.CoderAgent", Coder)
    state = ctx.scene_states[1]
    assert orchestrator._generate_validated_code(state.plan)[0] == CODE
    state.review_round = 1
    assert orchestrator._generate_validated_code(state.plan)[0] == CODE
    assert [item.used for item in state.stage_attempts] == [1, 1]
    assert len({item.input_sha256 for item in state.stage_attempts}) == 2


def test_generation_checkpoint_failure_prevents_coder_call(monkeypatch, tmp_path):
    orchestrator, ctx = workflow(tmp_path)

    class Coder:
        def generate_code(self, *args, **kwargs):
            pytest.fail("model call after failed stage reservation")

    monkeypatch.setattr("kd1_anime.orchestrator.CoderAgent", Coder)
    monkeypatch.setattr(
        "kd1_anime.orchestrator.write_manifest",
        lambda *args: (_ for _ in ()).throw(OSError("disk full")),
    )
    with pytest.raises(OSError, match="disk full"):
        orchestrator._generate_validated_code(ctx.scene_states[1].plan)
    assert orchestrator._stop_event.is_set()


def test_failed_reservation_cannot_enter_codegen_fallback(monkeypatch, tmp_path):
    import threading

    orchestrator, ctx = workflow(tmp_path)
    ctx.generation_mode = "relaxed"
    orchestrator._llm_sem = threading.Semaphore(1)
    monkeypatch.setattr(settings, "CODEGEN_MODE", "hybrid")
    monkeypatch.setattr(
        "kd1_anime.orchestrator.write_manifest",
        lambda *args: (_ for _ in ()).throw(OSError("disk full")),
    )
    monkeypatch.setattr(
        "kd1_anime.orchestrator.build_scene_program_from_contract",
        lambda *args: pytest.fail("compiler fallback after persistence failure"),
    )
    monkeypatch.setattr(
        "kd1_anime.orchestrator.build_safe_scene_code",
        lambda *args: pytest.fail("safe fallback after persistence failure"),
    )
    events = []
    orchestrator._callback = lambda event, data: events.append(event)
    with pytest.raises(OSError, match="disk full"):
        orchestrator._scene_code(ctx, 1, ctx.scene_states[1])
    assert ctx.scene_states[1].code == ""
    assert not (ctx.paths.scenes / "scene_1.py").exists()
    assert "scene_program_compiled" not in events
    assert "scene_coded" not in events


def test_late_worker_result_cannot_commit_after_another_checkpoint_failed(tmp_path):
    orchestrator, ctx, state = prepare_submission(tmp_path)
    identity = orchestrator._work_identity(ctx, state)
    source = ctx.paths.scenes / "scene_1.py"
    before = source.read_bytes()
    orchestrator._record_checkpoint_failure(OSError("disk full"))
    with pytest.raises(OSError, match="disk full"):
        orchestrator._install_repair_candidate(
            ctx, 1, state, CODE + "# late result\n", "Demo", error_fingerprint="", identity=identity
        )
    assert source.read_bytes() == before
    assert state.code == CODE


def test_technical_attempts_do_not_restart_after_interruption(monkeypatch, tmp_path):
    from kd1_anime.exceptions import ValidationError

    orchestrator, ctx = workflow(tmp_path)
    ctx.retry_limits = {"MAX_TECHNICAL_SPEC_ATTEMPTS": 1}
    calls = []

    class Planner:
        def plan(self, *args, **kwargs):
            saved = RunRepository(tmp_path).load(ctx.paths.run_id)
            calls.append(saved.scenes[1].stage_attempts[0].used)
            raise RuntimeError("interrupted before result")

    monkeypatch.setattr("kd1_anime.orchestrator.TechnicalPlannerAgent", Planner)
    with pytest.raises(RuntimeError, match="interrupted"):
        orchestrator._ensure_technical_spec(ctx, ctx.scene_states[1])
    restored = Orchestrator._context_from_manifest(
        RunRepository(tmp_path).load(ctx.paths.run_id), ctx.paths.root
    )
    resumed = Orchestrator()
    resumed._ctx = restored
    with pytest.raises(ValidationError, match="未生成有效结果"):
        resumed._ensure_technical_spec(restored, restored.scene_states[1])
    assert calls == [1]
    assert RunRepository(tmp_path).load(ctx.paths.run_id).scenes[1].stage_attempts[0].used == 1


def test_concurrent_stage_attempts_cannot_exceed_persisted_limit(tmp_path):
    from kd1_anime.exceptions import ValidationError

    orchestrator, ctx = workflow(tmp_path)
    ctx.retry_limits = {"CODE_VALIDATION_ATTEMPTS": 3}
    state = ctx.scene_states[1]
    identity = orchestrator._work_identity(ctx, state)

    def reserve():
        try:
            return orchestrator._reserve_stage_attempt(ctx, state, "generation", "a" * 64, identity)
        except ValidationError:
            return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        attempts = list(pool.map(lambda _: reserve(), range(12)))
    assert sorted(item for item in attempts if item is not None) == [1, 2, 3]
    assert RunRepository(tmp_path).load(ctx.paths.run_id).scenes[1].stage_attempts[0].used == 3


@pytest.mark.parametrize("duplicate", [True, False])
def test_manifest_rejects_duplicate_or_excessive_stage_attempts(tmp_path, duplicate):
    orchestrator, ctx = workflow(tmp_path)
    counter = StageAttempts(
        operation="technical", input_sha256="a" * 64, used=1 if duplicate else 4
    )
    ctx.retry_limits = {"MAX_TECHNICAL_SPEC_ATTEMPTS": 3}
    ctx.scene_states[1].stage_attempts = [counter, counter] if duplicate else [counter]
    with pytest.raises(ValueError, match=r"阶段尝试计数重复|尝试超过原限额"):
        orchestrator._checkpoint(ctx, State.CODING)


def test_v10_missing_stage_counter_is_rejected_instead_of_resetting(tmp_path):
    import json

    orchestrator, ctx = workflow(tmp_path)
    orchestrator._checkpoint(ctx, State.CODING)
    path = ctx.paths.root / "manifest.json"
    data = json.loads(path.read_text())
    del data["scenes"]["1"]["stage_attempts"]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="v10 Scene 1 缺少必需字段: stage_attempts"):
        RunRepository(tmp_path).load(ctx.paths.run_id)


def test_global_contract_change_invalidates_captured_consumers(tmp_path):
    from kd1_anime.run_store import sha256_text

    orchestrator, ctx, state = prepare_submission(tmp_path)
    independent = SceneState(plan(2, dependencies_complete=True))
    ctx.scene_states[2] = independent
    ctx.dependency_graph.bind(
        1,
        [],
        orchestrator._dependency_nodes(ctx),
        global_sha256=orchestrator._global_input_hash(ctx),
    )
    ctx.lesson_spec = ctx.lesson_spec.model_copy(update={"audience": "new audience"})
    original = sha256_text(state.code)
    orchestrator._invalidate_global_consumers(ctx)
    assert original != sha256_text(state.code)
    assert not state.reviewed
    assert ctx.scene_states[2] is independent


def test_review_rejects_a_completion_for_replaced_source(monkeypatch, tmp_path):
    import threading

    from kd1_anime.domain.review import ReviewResult

    orchestrator, ctx, state = prepare_submission(tmp_path)
    state.reviewed = False
    orchestrator._llm_sem = threading.Semaphore(1)

    class Reviewer:
        def review(self, *args, **kwargs):
            state.code = CODE + "# changed during request\n"
            return ReviewResult(is_valid=True)

    monkeypatch.setattr("kd1_anime.orchestrator.ReviewerAgent", Reviewer)
    with pytest.raises(StaleWorkError):
        orchestrator._scene_review(ctx, 1, state)
    assert not state.reviewed


def test_repair_install_rejects_stale_result_before_writing_source(tmp_path):
    orchestrator, ctx, state = prepare_submission(tmp_path)
    identity = orchestrator._work_identity(ctx, state)
    state.revision += 1
    before = (ctx.paths.scenes / "scene_1.py").read_bytes()
    with pytest.raises(StaleWorkError):
        orchestrator._install_repair_candidate(
            ctx, 1, state, CODE, "Demo", error_fingerprint="", identity=identity
        )
    assert (ctx.paths.scenes / "scene_1.py").read_bytes() == before


def test_star_import_catalog_accepts_explicit_import_alias():
    from kd1_anime.agents.api_linter import lint_manim_api

    result = lint_manim_api(
        "from manim import *\nfrom manim import Circle as C\na = C()", manim_version="0.20.1"
    )
    assert result.is_valid, result.errors


@pytest.mark.parametrize("key", ["MAX_SCENE_LLM_REQUESTS", "MAX_RUN_LLM_REQUESTS"])
def test_request_limits_reject_zero_on_assignment(key):
    with pytest.raises(ValueError):
        setattr(settings, key, 0)


def test_streamed_request_records_available_usage_once(tmp_path):
    from kd1_anime.request_budget import ObservedStream

    orchestrator, ctx = workflow(tmp_path)
    agent = agent_for(
        orchestrator,
        create=lambda **kwargs: iter(
            [
                SimpleNamespace(
                    usage=SimpleNamespace(prompt_tokens=8, completion_tokens=2, total_tokens=10)
                )
            ]
        ),
    )
    observed = agent._create_completion(agent.client, model="fake", stream=True)
    assert isinstance(observed, ObservedStream)
    list(observed)
    observed.close()
    import json

    rows = [json.loads(line) for line in (ctx.paths.root / "events.jsonl").read_text().splitlines()]
    completed = [row["data"] for row in rows if row["event"] == "model_request_finished"]
    assert len(completed) == 1
    assert completed[0]["usage"]["total_tokens"] == 10
    assert completed[0]["latency_ms"] >= 0


@pytest.mark.parametrize("command", ["resume", "retry", "incremental"])
@pytest.mark.parametrize("version", [4, 5, 6, 7, 8, 9])
def test_cli_refuses_legacy_execution_before_api_probe(monkeypatch, tmp_path, command, version):
    import json

    from typer.testing import CliRunner

    from kd1_anime.cli import app

    orchestrator, ctx = workflow(tmp_path)
    orchestrator._checkpoint(ctx, State.INIT)
    manifest_path = ctx.paths.root / "manifest.json"
    raw = json.loads(manifest_path.read_text())
    raw["schema_version"] = version
    manifest_path.write_text(json.dumps(raw))
    before = manifest_path.read_bytes()
    monkeypatch.setattr(settings, "WORKSPACE_DIR", tmp_path)
    monkeypatch.setattr(
        "kd1_anime.cli._ensure_llm_api_available", lambda: pytest.fail("legacy API probe")
    )
    args = (
        ["generate", "new prompt", "--incremental", ctx.paths.run_id]
        if command == "incremental"
        else [command, ctx.paths.run_id]
    )
    if command == "retry":
        args += ["--scene-id", "1"]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 1
    assert "只读" in result.output
    assert manifest_path.read_bytes() == before


def test_repair_source_change_invalidates_consumers_but_preserves_independence(tmp_path):
    orchestrator, ctx, state = prepare_submission(tmp_path)
    consumer = SceneState(
        plan(
            2,
            dependencies_complete=True,
            dependencies=[SceneDependency(source_scene_id=1, kind="code_context")],
        ),
        code=CODE,
        class_name="Demo",
        reviewed=True,
    )
    independent = SceneState(
        plan(3, dependencies_complete=True), code=CODE, class_name="Demo", reviewed=True
    )
    ctx.scene_states.update({2: consumer, 3: independent})
    for sid in (2, 3):
        orchestrator._write_private(ctx.paths.scenes / f"scene_{sid}.py", CODE)
    ctx.dependency_graph.bind(2, consumer.plan.dependencies, orchestrator._dependency_nodes(ctx))
    orchestrator._checkpoint(ctx, State.REVIEWING)
    original_revision = consumer.revision
    orchestrator._install_repair_candidate(
        ctx, 1, state, CODE.replace("self.wait()", "self.wait(2)"), "Demo", error_fingerprint=""
    )
    saved = RunRepository(tmp_path).load_for_resume(ctx.paths.run_id)
    assert consumer.code == "" and not consumer.reviewed
    assert consumer.revision > original_revision
    assert saved.scenes[2].code_sha256 == ""
    assert saved.scenes[3].reviewed and independent.code == CODE
    assert ctx.continuity_rebuild_required


def test_teaching_changes_do_not_invalidate_pure_export_consumers():
    graph = DependencyGraph()
    source = plan()
    initial = {
        1: {
            "code": "a" * 64,
            "contract": DependencyService.contract(source, None),
            "teaching": DependencyService.teaching(source),
        }
    }
    graph.bind(2, [SceneDependency(source_scene_id=1, kind="teaching")], initial)
    graph.bind(3, [SceneDependency(source_scene_id=1, kind="handoff")], initial)
    changed = source.model_copy(update={"math_concept": "new concept"})
    nodes = {1: {**initial[1], "teaching": DependencyService.teaching(changed)}}
    assert graph.affected(1, nodes) == {2}


def test_precise_patch_rejects_a_replaced_input_before_writing(monkeypatch, tmp_path):
    from kd1_anime.domain.review import FixSuggestion, ReviewResult

    orchestrator, ctx, state = prepare_submission(tmp_path)
    inspect_candidate = orchestrator.candidate_acceptor.inspect
    before = (ctx.paths.scenes / "scene_1.py").read_bytes()

    def inspect_then_replace(*args, **kwargs):
        accepted = inspect_candidate(*args, **kwargs)
        state.revision += 1
        return accepted

    monkeypatch.setattr(orchestrator.candidate_acceptor, "inspect", inspect_then_replace)
    review = ReviewResult(
        is_valid=False,
        feedback="correct duration",
        fixes=[FixSuggestion(find="self.wait()", replace="self.wait(2)")],
    )
    with pytest.raises(StaleWorkError):
        orchestrator._apply_precise_review_fixes(ctx, 1, state, review)
    assert state.code == CODE
    assert (ctx.paths.scenes / "scene_1.py").read_bytes() == before


def test_budget_exhaustion_never_soft_passes_relaxed_review(monkeypatch, tmp_path):
    import threading

    orchestrator, ctx, state = prepare_submission(tmp_path)
    ctx.generation_mode = "relaxed"
    state.reviewed = False
    orchestrator._llm_sem = threading.Semaphore(1)

    class Reviewer:
        def review(self, *args, **kwargs):
            raise RequestBudgetExceeded("exhausted")

    monkeypatch.setattr("kd1_anime.orchestrator.ReviewerAgent", Reviewer)
    with pytest.raises(RequestBudgetExceeded):
        orchestrator._scene_review(ctx, 1, state)
    assert not state.reviewed


def test_unknown_submission_cannot_be_discarded_for_replanning(tmp_path):
    orchestrator, _ctx, state = prepare_submission(tmp_path)
    state.submission = SubmissionAttempt(
        backend="slurm", scene_revision=0, code_sha256="a" * 64, status="unknown"
    )
    with pytest.raises(RuntimeError, match="禁止清空"):
        orchestrator._cancel_unfinished_scene_job(state, reason="replan")


@pytest.mark.parametrize(
    "status, can_restart",
    [
        ("RUNNING", False),
        ("UNKNOWN", False),
        ("CANCEL_FAILED", False),
        ("FAILED", True),
        ("CANCELLED", True),
    ],
)
def test_local_recovery_only_restarts_confirmed_terminal_attempts(status, can_restart):
    from kd1_anime.services.recovery import RecoveryInput, RecoveryService

    (decision,) = RecoveryService.detach_unresumable_local_jobs(
        [RecoveryInput(1, "local-abcdef123456", False, status)]
    )
    assert decision.can_restart is can_restart


@pytest.mark.parametrize("kind", ["run_limit", "scene_limit", "duplicate"])
def test_manifest_rejects_corrupt_request_history(tmp_path, kind):
    orchestrator, ctx = workflow(tmp_path)
    receipt = RequestReservation(stage="coding", scene_id=1, model="fake", profile="main")
    ctx.request_budget.reservations = [receipt, receipt.model_copy(update={"request_id": "b" * 32})]
    if kind == "duplicate":
        ctx.request_budget.reservations[1] = receipt
    else:
        setattr(ctx.request_budget, kind, 1)
    with pytest.raises(ValueError, match="请求"):
        orchestrator._checkpoint(ctx, State.CODING)
    assert not (ctx.paths.root / "manifest.json").exists()


@pytest.mark.parametrize(
    "limits",
    [
        {"MAX_FIX_ATTEMPTS": -1},
        {"MAX_FIX_ATTEMPTS": True},
        {"UNKNOWN": 2},
        {"MAX_REVIEW_ROUNDS": 0},
    ],
)
def test_manifest_rejects_invalid_retry_limits(tmp_path, limits):
    orchestrator, ctx = workflow(tmp_path)
    ctx.retry_limits = limits
    with pytest.raises(ValueError):
        orchestrator._checkpoint(ctx, State.CODING)


def test_candidate_catalog_uses_target_version_and_keeps_unknown_as_warning():
    from kd1_anime.candidate_acceptor import CandidateAcceptor, CandidateRejected

    code = CODE.replace("self.wait()", "self.add(InventedShape())")
    with pytest.raises(CandidateRejected, match="不提供公开 API"):
        CandidateAcceptor().inspect(code, plan(), manim_version="0.20.1")
    accepted = CandidateAcceptor().inspect(code, plan(), manim_version="99.0")
    assert accepted.api_warnings


def test_container_profile_does_not_claim_host_package_versions(monkeypatch, tmp_path):
    from kd1_anime.rendering import RenderProfile

    monkeypatch.setattr(settings, "SLURM_CONTAINER_IMAGE", str(tmp_path / "image.sif"))
    monkeypatch.setattr(
        RenderProfile, "_environment_versions", lambda: ("0.20.1", "host ffmpeg", "host tex")
    )
    profile = RenderProfile.current()
    assert profile.manim_version == profile.ffmpeg_version == profile.xelatex_version == ""


@pytest.mark.parametrize("failure", ["query", "name"])
def test_submission_reconciliation_cannot_adopt_failed_or_unmatched_queries(
    monkeypatch, tmp_path, failure
):
    from kd1_anime.cluster.render_backend import SlurmRenderBackend
    from kd1_anime.rendering import RenderProfile

    attempt = SubmissionAttempt(backend="slurm", scene_revision=0, code_sha256="a" * 64)
    monkeypatch.setattr(
        "kd1_anime.cluster.render_backend.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1 if failure == "query" else 0, stdout="123|different-attempt"
        ),
    )
    kwargs = dict(
        scene_id=1,
        source=tmp_path / "scene.py",
        root=tmp_path,
        render_profile=RenderProfile.current(),
    )
    if failure == "query":
        with pytest.raises(RuntimeError, match="禁止自动重提"):
            SlurmRenderBackend().recover_submission(attempt, **kwargs)
    else:
        assert SlurmRenderBackend().recover_submission(attempt, **kwargs) is None


def test_absent_usage_remains_unknown_in_request_summary(tmp_path):
    orchestrator, ctx = workflow(tmp_path)
    agent = agent_for(orchestrator, create=lambda **kwargs: response())
    agent._create_completion(agent.client, model="fake")
    report = summarize_manifest(orchestrator._manifest, root=ctx.paths.root)
    assert report["model_request_details"][0]["usage"] is None
    assert report["model_token_usage"]["total_tokens"] == {
        "known_total": None,
        "measured_requests": 0,
        "unknown_requests": 1,
    }


@pytest.mark.parametrize("renderer", ["cairo", "opengl"])
def test_local_slurm_and_smoke_use_the_same_container_and_environment_policy(
    monkeypatch, tmp_path, renderer
):
    from pathlib import Path

    from kd1_anime.rendering import RenderProfile

    orchestrator, ctx, state = prepare_submission(tmp_path)
    image = tmp_path / "render.sif"
    image.touch()
    monkeypatch.setattr(settings, "SLURM_CONTAINER_IMAGE", str(image))
    monkeypatch.setattr(settings, "SLURM_REQUIRE_CONTAINER", True)
    monkeypatch.setattr(settings, "SLURM_GPU_TYPE", "A100")
    monkeypatch.setattr(settings, "ADAPTIVE_SMOKE_RENDER", False)
    monkeypatch.setattr(settings, "LOCAL_SMOKE_RENDER_MODE", "frame")
    monkeypatch.setenv("LLM_API_KEY", "private-test-value")
    ctx.render_profile = RenderProfile.current().model_copy(update={"renderer": renderer})
    commands = []
    environments = []
    wrap = ExecutionPolicy.wrap

    def record_wrap(policy, command, root, **kwargs):
        result = wrap(policy, command, root, **kwargs)
        commands.append(result)
        assert root.resolve() == ctx.paths.root.resolve()
        return result

    def fake_popen(command, **kwargs):
        environments.append(kwargs["env"])
        return SimpleNamespace(poll=lambda: 0)

    def fake_smoke(command, **kwargs):
        environments.append(kwargs["env"])
        if "--media_dir" in command:
            media = Path(command[command.index("--media_dir") + 1])
            (media / "Demo.png").write_bytes(b"test-frame")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(ExecutionPolicy, "wrap", record_wrap)
    monkeypatch.setattr("kd1_anime.cluster.render_backend.subprocess.Popen", fake_popen)
    monkeypatch.setattr("kd1_anime.cluster.render_backend.shutil.which", lambda name: None)
    monkeypatch.setattr("kd1_anime.orchestrator._run_limited_process", fake_smoke)
    backend = LocalRenderBackend()
    job = backend.submit_scene(
        1,
        ctx.paths.scenes / "scene_1.py",
        "Demo",
        scenes_dir=ctx.paths.scenes,
        logs_dir=ctx.paths.logs,
        videos_dir=ctx.paths.videos,
        render_profile=ctx.render_profile,
    )
    backend.poll_all_statuses([job.job_id])
    orchestrator._local_smoke_render_impl(ctx, state)
    script = SlurmDispatcher()._build_script(
        1,
        ctx.paths.scenes / "scene_1.py",
        "Demo",
        ctx.paths.videos / "formal",
        ctx.paths.logs / "out",
        ctx.paths.logs / "err",
        run_root=ctx.paths.root,
        render_profile=ctx.render_profile,
    )
    assert len(commands) >= 4
    for command in commands:
        assert command[:5] == ["apptainer", "exec", "--containall", "--cleanenv", "--no-home"]
        assert ("--nv" in command) is (renderer == "opengl")
        assert command.count("--bind") == 1
        assert f"{ctx.paths.root}:{ctx.paths.root}" in command
    for environment in environments:
        assert "LLM_API_KEY" not in environment
        assert environment["PYTHONNOUSERSITE"] == "1"
        assert environment["MANIM_RENDERER"] == renderer
    assert "env -i" in script and "export PYTHONNOUSERSITE=1" in script
    assert "private-test-value" not in script


def test_plan_review_rejects_completion_for_changed_plan(monkeypatch, tmp_path):
    import threading

    from kd1_anime.domain.plan_review import PlanReviewResult

    orchestrator, ctx = workflow(tmp_path)
    state = ctx.scene_states[1]
    state.plan_ready = True
    orchestrator._llm_sem = threading.Semaphore(1)

    class Reviewer:
        def review(self, *args, **kwargs):
            state.plan.title = "replaced plan"
            return PlanReviewResult(is_valid=True)

    monkeypatch.setattr("kd1_anime.orchestrator.PlanReviewerAgent", Reviewer)
    with pytest.raises(StaleWorkError):
        orchestrator._run_plan_review_barrier(ctx)
    assert not state.plan_reviewed


def test_domain_models_keep_legacy_import_identity():
    from kd1_anime.agents.capability import CapabilityContract as LegacyCapability
    from kd1_anime.agents.plan_reviewer import PlanReviewResult as LegacyReview
    from kd1_anime.agents.state_ledger import StateLedger as LegacyLedger
    from kd1_anime.domain.capability import CapabilityContract
    from kd1_anime.domain.plan_review import PlanReviewResult
    from kd1_anime.domain.state_ledger import StateLedger

    assert LegacyCapability is CapabilityContract
    assert LegacyReview is PlanReviewResult
    assert LegacyLedger is StateLedger


def test_structured_output_repair_is_reserved_and_reported_as_retry(tmp_path):
    from pydantic import BaseModel, ConfigDict

    class Payload(BaseModel):
        model_config = ConfigDict(extra="forbid")
        count: int

    orchestrator, ctx = workflow(tmp_path)
    responses = iter([response('{"count":"invalid"}'), response('{"count":2}')])
    agent = agent_for(orchestrator, create=lambda **kwargs: next(responses))
    agent.profile = replace(agent.profile, json_repair_attempts=1)
    assert agent.call_llm_json("system", "user", Payload).count == 2
    assert [item.json_repair_attempt for item in ctx.request_budget.reservations] == [0, 1]
    assert summarize_manifest(orchestrator._manifest)["model_retry_requests"] == 1
    assert agent._json_repair_attempt == 0


def test_benchmark_does_not_equate_static_acceptance_with_content_preservation(tmp_path):
    from kd1_anime.domain.program import SceneProgram
    from kd1_anime.eval.benchmark import compare_modes, task_plan
    from kd1_anime.services.program_compiler import compile_program

    orchestrator, ctx, state = prepare_submission(tmp_path)
    task = next(item for item in fixed_tasks() if item.task_id == "geometry_circle")
    state.plan = task_plan(task)
    state.code = compile_program(SceneProgram.model_validate(task.program), state.plan)
    state.class_name = "BenchmarkScene"
    ctx.scene_codegen_mode = "compiler_first"
    orchestrator._write_private(ctx.paths.scenes / "scene_1.py", state.code)
    orchestrator._mark_static_verification(state, status="passed")
    orchestrator._checkpoint(ctx, State.CODING)
    bindings = {"compiler_first": {task.task_id: ctx.paths.root}}
    result = compare_modes(bindings)["groups"]["compiler_first"]
    assert result["contract_measured_tasks"] == 1
    assert result["tasks"][0]["recovery_verified"] is None
    state.code = CODE
    state.class_name = "Demo"
    orchestrator._write_private(ctx.paths.scenes / "scene_1.py", CODE)
    orchestrator._mark_static_verification(state, status="passed")
    orchestrator._checkpoint(ctx, State.CODING)
    result = compare_modes(bindings)["groups"]["compiler_first"]
    assert result["tasks"][0]["static_contract_accepted"] is True
    assert result["contract_preservation_rate"] is None
    assert result["contract_unknown_tasks"] == 1


def test_recovery_benchmark_requires_fault_evidence(tmp_path):
    from kd1_anime.eval.benchmark import recovery_evidence

    orchestrator, ctx, state = prepare_submission(tmp_path)
    orchestrator._checkpoint(ctx, State.CODING)
    task = next(item for item in fixed_tasks() if item.task_id == "recovery_cancel")
    assert recovery_evidence(task, orchestrator._manifest) is None
    state.slurm_job = submitted_job(ctx, state)
    state.slurm_job.status = "CANCEL_FAILED"
    state.give_up = True
    state.submission = SubmissionAttempt(
        backend="slurm",
        scene_revision=state.revision,
        code_sha256=state.slurm_job.code_sha256,
        status="submitted",
        job_id="123",
    )
    orchestrator._checkpoint(ctx, State.MONITORING)
    assert recovery_evidence(task, orchestrator._manifest) is True


def test_local_backend_cannot_confirm_cancellation_without_process_handle():
    backend = LocalRenderBackend()
    assert backend.cancel_job("local-abcdef123456") is False


def test_local_backend_does_not_claim_success_when_killed_process_still_runs(monkeypatch):
    import signal
    import subprocess

    def wait(**kwargs):
        raise subprocess.TimeoutExpired("trusted-test-process", 5)

    signals = []
    backend = LocalRenderBackend()
    backend._processes["local-abcdef123456"] = SimpleNamespace(
        pid=4321, poll=lambda: None, wait=wait
    )
    monkeypatch.setattr(
        "kd1_anime.cluster.render_backend.os.killpg", lambda pid, sig: signals.append(sig)
    )
    assert backend.cancel_job("local-abcdef123456") is False
    assert signals == [signal.SIGTERM, signal.SIGKILL]
