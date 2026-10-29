"""Durable public request contexts and standalone execution/security contracts."""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import pytest
from openai import BadRequestError

from kd1_anime.agent_runtime.session import SessionStore
from kd1_anime.agents.base import BaseAgent
from kd1_anime.api_catalog import lookup_symbol
from kd1_anime.application import Application
from kd1_anime.cluster.execution import ExecutionPolicy
from kd1_anime.cluster.render_backend import LocalRenderBackend
from kd1_anime.cluster.slurm import SlurmDispatcher
from kd1_anime.config import settings
from kd1_anime.domain.dependencies import DependencyGraph, SceneDependency
from kd1_anime.domain.execution import SubmissionAttempt, transition_run, validate_scene_transition
from kd1_anime.domain.planning import ScenePlan
from kd1_anime.eval.benchmark import fixed_tasks, offline_baseline
from kd1_anime.request_budget import RequestBudget, RequestBudgetExceeded, RequestReservation
from kd1_anime.services.dependencies import DependencyService
from kd1_anime.services.program_compiler import ProgramCompileError, expression_source

CODE = "from manim import *\nclass Demo(Scene):\n    def construct(self): self.wait()\n"
RUN_ID = "20261019-120000-1234abcd"


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


def budget_session(monkeypatch, tmp_path, *, run_limit=512):
    monkeypatch.setattr(settings, "WORKSPACE_DIR", tmp_path)
    monkeypatch.setattr(settings, "MAX_RUN_LLM_REQUESTS", run_limit)
    app = Application()
    session_id = app.create_session("A deterministic request budget")
    return app, session_id, SessionStore(tmp_path / "sessions")


def agent_for(context, create):
    profile = replace(
        settings.main_llm_profile(),
        api_key="test-key",
        model="fake-model",
        max_retries=1,
        silent_stream=False,
    )
    agent = BaseAgent(profile=profile)
    agent.request_context = context
    agent._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    return agent


def response(content="OK"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")],
        usage=None,
    )


def test_concurrent_requests_are_reserved_on_disk_before_dispatch(monkeypatch, tmp_path):
    app, session_id, store = budget_session(monkeypatch, tmp_path, run_limit=3)
    calls = []

    def create(**kwargs):
        assert 1 <= len(store.load(session_id).draft_budget.reservations) <= 3
        calls.append(kwargs)
        return response()

    def request(_):
        agent = agent_for(app.session_request_context(session_id), create)
        try:
            agent._create_completion(agent.client, model="fake-model")
        except RequestBudgetExceeded:
            return False
        return True

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(request, range(20)))
    assert sum(outcomes) == len(calls) == 3
    assert len(store.load(session_id).draft_budget.reservations) == 3


def test_run_budget_includes_run_wide_and_visual_profiles():
    budget = RequestBudget(scene_limit=10, run_limit=2)
    budget.reserve(RequestReservation(stage="planning", profile="main", model="fake"))
    budget.reserve(RequestReservation(stage="visual", scene_id=1, profile="visual", model="vision"))
    with pytest.raises(RequestBudgetExceeded, match="运行"):
        budget.reserve(RequestReservation(stage="coding", scene_id=1, profile="main", model="fake"))
    assert [item.profile for item in budget.reservations] == ["main", "visual"]


def test_scene_budget_leaves_independent_scenes_available():
    budget = RequestBudget(scene_limit=1, run_limit=3)
    budget.reserve(RequestReservation(stage="coding", scene_id=1, profile="main", model="fake"))
    with pytest.raises(RequestBudgetExceeded, match="Scene 1"):
        budget.reserve(RequestReservation(stage="review", scene_id=1, profile="main", model="fake"))
    budget.reserve(RequestReservation(stage="coding", scene_id=2, profile="main", model="fake"))
    assert len(budget.reservations) == 2


@pytest.mark.parametrize("kind", ["run_limit", "scene_limit", "duplicate"])
def test_request_history_integrity_rejects_corrupt_reservations(kind):
    first = RequestReservation(stage="coding", scene_id=1, model="fake", profile="main")
    second = first if kind == "duplicate" else first.model_copy(update={"request_id": "b" * 32})
    budget = RequestBudget(
        scene_limit=1 if kind == "scene_limit" else 3,
        run_limit=1 if kind == "run_limit" else 3,
        reservations=[first, second],
    )
    expected = {"run_limit": "运行限额", "scene_limit": "场景限额", "duplicate": "标识重复"}
    assert any(expected[kind] in error for error in budget.integrity_errors())


def test_failed_reservation_persistence_prevents_network_call(monkeypatch, tmp_path):
    app, session_id, store = budget_session(monkeypatch, tmp_path)
    calls = []
    agent = agent_for(
        app.session_request_context(session_id), lambda **kwargs: calls.append(kwargs)
    )
    monkeypatch.setattr(
        "kd1_anime.agent_runtime.session.os.replace",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )
    with pytest.raises(OSError, match="disk full"):
        agent._create_completion(agent.client, model="fake")
    assert calls == []
    assert store.load(session_id).draft_budget.reservations == []


def test_failed_network_request_remains_consumed_after_restore(monkeypatch, tmp_path):
    app, session_id, store = budget_session(monkeypatch, tmp_path, run_limit=1)

    def fail(**kwargs):
        raise TimeoutError("unknown remote outcome")

    agent = agent_for(app.session_request_context(session_id), fail)
    with pytest.raises(TimeoutError):
        agent._create_completion(agent.client, model="fake")
    assert len(store.load(session_id).draft_budget.reservations) == 1
    with pytest.raises(RequestBudgetExceeded):
        Application().session_request_context(session_id).reserve("main", "fake")


def test_compatibility_fallback_consumes_another_actual_request(monkeypatch, tmp_path):
    import httpx

    app, session_id, store = budget_session(monkeypatch, tmp_path)
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

    agent_for(app.session_request_context(session_id), create).check_api_available()
    assert len(calls) == len(store.load(session_id).draft_budget.reservations) == 2


def test_streamed_request_records_available_usage_once(monkeypatch, tmp_path):
    from kd1_anime.request_budget import ObservedStream

    app, session_id, store = budget_session(monkeypatch, tmp_path)
    agent = agent_for(
        app.session_request_context(session_id),
        lambda **kwargs: iter(
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
    usage = store.load(session_id).draft_usage
    assert len(usage) == 1 and usage[0].total_tokens == 10
    assert usage[0].usage_source == "provider_reported"


def test_absent_usage_remains_unknown(monkeypatch, tmp_path):
    app, session_id, store = budget_session(monkeypatch, tmp_path)
    agent = agent_for(app.session_request_context(session_id), lambda **kwargs: response(None))
    agent._create_completion(agent.client, model="fake")
    receipt = store.load(session_id).draft_usage[0]
    assert receipt.usage_source == "unknown"
    assert receipt.total_tokens is None and receipt.cost is None


def test_structured_output_repair_is_reserved_as_another_request(monkeypatch, tmp_path):
    from pydantic import BaseModel, ConfigDict

    class Payload(BaseModel):
        model_config = ConfigDict(extra="forbid")
        count: int

    app, session_id, store = budget_session(monkeypatch, tmp_path)
    responses = iter([response('{"count":"invalid"}'), response('{"count":2}')])
    agent = agent_for(app.session_request_context(session_id), lambda **kwargs: next(responses))
    agent.profile = replace(agent.profile, json_repair_attempts=1)
    assert agent.call_llm_json("system", "user", Payload).count == 2
    assert [
        item.json_repair_attempt for item in store.load(session_id).draft_budget.reservations
    ] == [0, 1]
    assert agent._json_repair_attempt == 0


@pytest.mark.parametrize("command", ["resume", "retry", "incremental"])
@pytest.mark.parametrize("version", range(4, 11))
def test_cli_refuses_legacy_execution_before_api_probe(monkeypatch, tmp_path, command, version):
    from typer.testing import CliRunner

    from kd1_anime.cli import app

    monkeypatch.setattr(settings, "WORKSPACE_DIR", tmp_path)
    root = tmp_path / "runs" / RUN_ID
    root.mkdir(parents=True)
    path = root / "manifest.json"
    path.write_text(json.dumps({"schema_version": version, "run_id": RUN_ID}))
    root.chmod(0o755)
    before = path.read_bytes()

    def forbidden(*args, **kwargs):
        pytest.fail("legacy execution reached a probe or runtime")

    monkeypatch.setattr("kd1_anime.cli._ensure_llm_api_available", forbidden)
    monkeypatch.setattr("kd1_anime.cli._ensure_rag_apis_available", forbidden)
    monkeypatch.setattr("kd1_anime.application._agent_runtime", forbidden)
    arguments = {
        "resume": ["resume", RUN_ID],
        "retry": ["retry", RUN_ID, "--scene-id", "1"],
        "incremental": ["generate", "new goal", "--incremental", RUN_ID],
    }[command]
    result = CliRunner().invoke(app, arguments)
    assert result.exit_code != 0
    assert "只读" in result.output
    assert path.read_bytes() == before
    assert root.stat().st_mode & 0o777 == 0o755
    assert {item.name for item in root.iterdir()} == {"manifest.json"}


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


def test_execution_fsm_rejects_unapproved_stage_changes():
    with pytest.raises(ValueError):
        transition_run("finished", "executing")
    with pytest.raises(ValueError):
        validate_scene_transition("coded", "monitoring", revised=False)
    validate_scene_transition("visual_accepted", "coded", revised=True)
    with pytest.raises(ValueError):
        validate_scene_transition("reviewed", "rendered", revised=False)
    validate_scene_transition("reviewed", "rendered", revised=False, reused=True)
    with pytest.raises(ValueError):
        validate_scene_transition("coded", "rendered", revised=False, reused=True)


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


def test_candidate_catalog_uses_target_version_and_keeps_unknown_as_warning():
    from kd1_anime.candidate_acceptor import CandidateAcceptor, CandidateRejected

    code = CODE.replace("self.wait()", "self.add(InventedShape())")
    with pytest.raises(CandidateRejected, match="不提供公开 API"):
        CandidateAcceptor().accept_candidate(code, plan(), manim_version="0.20.1")
    accepted = CandidateAcceptor().accept_candidate(code, plan(), manim_version="99.0")
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
