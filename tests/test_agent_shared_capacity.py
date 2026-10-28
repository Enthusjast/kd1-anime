"""Shared capacity across independent v11 runtimes, without external execution."""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from threading import Event
from types import SimpleNamespace

import pytest
from test_agent_vertical_slice import agent_boundary as agent_boundary
from test_application import SOURCE

from kd1_anime.agent_runtime.runtime import AgentRuntime
from kd1_anime.agents.base import BaseAgent
from kd1_anime.config import settings
from kd1_anime.control import RunCancelled
from kd1_anime.resources import ResourceCoordinator
from kd1_anime.run_store import RunRepository


def shared_resources():
    return ResourceCoordinator(
        llm_limit=1, visual_llm_limit=1, rag_limit=1, slurm_limit=1, local_limit=1
    )


def bound_runtime(dependencies, resources):
    runtime = AgentRuntime(dependencies=dependencies, resource_coordinator=resources)
    assert runtime.generate("Show a pause", dry_run=True) is None
    return runtime


def model_agent(runtime, *, visual=False):
    profile = settings.visual_llm_profile() if visual else settings.main_llm_profile()
    agent = BaseAgent(profile=profile)
    agent.control = runtime.control
    agent.request_context = runtime._request_context(None, "capacity_test")
    return agent


def client_for(create):
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def completion(agent, create, *, stream=False):
    return agent._create_completion(
        client_for(create),
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=1,
        stream=stream,
    )


@pytest.mark.parametrize("visual", [False, True])
def test_independent_runtimes_hold_model_capacity_through_transport(agent_boundary, visual):
    _, dependencies = agent_boundary
    resources = shared_resources()
    first = bound_runtime(dependencies, resources)
    second = bound_runtime(dependencies, resources)
    entered, release, second_entered = Event(), Event(), Event()

    def first_request(**kwargs):
        entered.set()
        assert release.wait(5)
        return SimpleNamespace(choices=[])

    def second_request(**kwargs):
        second_entered.set()
        return SimpleNamespace(choices=[])

    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(completion, model_agent(first, visual=visual), first_request)
        assert entered.wait(5)
        b = pool.submit(completion, model_agent(second, visual=visual), second_request)
        try:
            assert not second_entered.wait(0.05)
            # The waiting request has a durable reservation, but has not contacted its endpoint.
            deadline = time.monotonic() + 5
            while not any(
                r.stage == "capacity_test" for r in second.manifest.request_budget.reservations
            ):
                assert time.monotonic() < deadline
                time.sleep(0.01)
            raw = json.loads((second.root / "manifest.json").read_text())
            assert any(r["stage"] == "capacity_test" for r in raw["request_budget"]["reservations"])
        finally:
            release.set()
        a.result(timeout=5)
        b.result(timeout=5)
    assert second_entered.is_set()


def test_visual_endpoint_does_not_consume_main_endpoint_capacity(agent_boundary):
    _, dependencies = agent_boundary
    resources = shared_resources()
    first = bound_runtime(dependencies, resources)
    second = bound_runtime(dependencies, resources)
    assert resources.llm.acquire(blocking=False)
    try:
        response = completion(
            model_agent(second, visual=True), lambda **kw: SimpleNamespace(choices=[])
        )
        assert response.choices == []
        assert first.resources is second.resources is resources
    finally:
        resources.llm.release()


@pytest.mark.parametrize("ending", ["exhausted", "error", "close", "iterator_close"])
def test_stream_retains_slot_until_its_actual_end(agent_boundary, ending):
    _, dependencies = agent_boundary
    resources = shared_resources()
    runtime = bound_runtime(dependencies, resources)
    closed = Event()

    class Stream:
        def __iter__(self):
            yield SimpleNamespace(choices=[])
            if ending == "error":
                raise ConnectionError("isolated stream failure")

        def close(self):
            closed.set()

    stream = completion(model_agent(runtime), lambda **kw: Stream(), stream=True)
    assert not resources.llm.acquire(blocking=False)
    if ending == "close":
        stream.close()
        assert closed.is_set()
    elif ending == "iterator_close":
        iterator = iter(stream)
        next(iterator)
        iterator.close()
        assert closed.is_set()
    elif ending == "error":
        with pytest.raises(ConnectionError):
            list(stream)
        assert closed.is_set()
    else:
        assert len(list(stream)) == 1
    assert resources.llm.acquire(blocking=False)
    resources.llm.release()
    stream.close()
    assert resources.llm.acquire(blocking=False)
    assert not resources.llm.acquire(blocking=False)
    resources.llm.release()
    reservation = runtime.manifest.request_budget.reservations[-1]
    assert any(u.request_id == reservation.request_id for u in runtime.manifest.agent.usage)


def test_transport_failure_releases_capacity_but_preserves_request_reservation(agent_boundary):
    _, dependencies = agent_boundary
    resources = shared_resources()
    runtime = bound_runtime(dependencies, resources)

    def fail(**kwargs):
        raise ConnectionError("isolated endpoint failure")

    before = len(runtime.manifest.request_budget.reservations)
    with pytest.raises(ConnectionError):
        completion(model_agent(runtime), fail)
    assert len(runtime.manifest.request_budget.reservations) == before + 1
    assert resources.llm.acquire(blocking=False)
    resources.llm.release()


def test_cancelling_a_waiting_model_keeps_reservation_and_never_dispatches(agent_boundary):
    _, dependencies = agent_boundary
    resources = shared_resources()
    runtime = bound_runtime(dependencies, resources)
    entered = Event()
    assert resources.llm.acquire(blocking=False)
    before = len(runtime.manifest.request_budget.reservations)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(completion, model_agent(runtime), lambda **kw: entered.set())
        deadline = time.monotonic() + 5
        try:
            while len(runtime.manifest.request_budget.reservations) == before:
                assert time.monotonic() < deadline
                time.sleep(0.01)
            runtime.cancel_all()
            with pytest.raises(RunCancelled):
                future.result(timeout=5)
        finally:
            resources.llm.release()
    assert not entered.is_set()
    assert len(runtime.manifest.request_budget.reservations) == before + 1
    assert resources.llm.acquire(blocking=False)
    resources.llm.release()


@pytest.mark.parametrize("cancelled", [True, False])
def test_live_submission_keeps_capacity_until_authorized_durable_cancel(
    agent_boundary, monkeypatch, cancelled
):
    boundary, dependencies = agent_boundary
    resources = shared_resources()
    runtime = AgentRuntime(dependencies=dependencies, resource_coordinator=resources)
    runtime.submit_existing_scene(SOURCE, "Pause", backend="slurm")
    assert not resources.try_acquire_slurm()
    calls = []
    monkeypatch.setattr(
        boundary.backend, "cancel_job", lambda job_id: calls.append(job_id) or cancelled
    )
    runtime.cancel_all()
    manifest = RunRepository(boundary.workspace).load(runtime.manifest.run_id)
    job = manifest.scenes[1].slurm_job
    assert calls == [job.job_id]
    intent = next(i for i in manifest.agent.intents if i.action.kind == "cancel_render")
    invocation = next(i for i in manifest.agent.invocations if i.intent_id == intent.intent_id)
    assert intent.status == invocation.status == "completed"
    result = json.loads((runtime.root / invocation.result_ref.path).read_text())
    assert result["job"]["status"] == ("CANCELLED" if cancelled else "CANCEL_FAILED")
    assert resources.try_acquire_slurm() is cancelled
    if cancelled:
        resources.release_slurm()
    else:
        assert runtime._pending_submission(1) is not None
    runtime.cancel_all()
    assert calls == [job.job_id]


def test_second_runtime_waits_for_live_job_and_then_uses_released_capacity(agent_boundary):
    _, dependencies = agent_boundary
    resources = shared_resources()
    first = AgentRuntime(dependencies=dependencies, resource_coordinator=resources)
    second = AgentRuntime(dependencies=dependencies, resource_coordinator=resources)
    first.submit_existing_scene(SOURCE, "Pause", backend="slurm")
    with ThreadPoolExecutor(max_workers=1) as pool:
        waiting = pool.submit(second.submit_existing_scene, SOURCE, "Pause", backend="slurm")
        try:
            assert not waiting.done()
            first.cancel_all()
            waiting.result(timeout=5)
        finally:
            second.cancel_all()
    assert resources.try_acquire_slurm()
    resources.release_slurm()


@pytest.mark.parametrize("backend", ["slurm", "local"])
def test_cancel_during_submission_commits_job_identity_before_authorized_cleanup(
    agent_boundary, monkeypatch, backend
):
    boundary, dependencies = agent_boundary
    resources = shared_resources()
    runtime = AgentRuntime(dependencies=dependencies, resource_coordinator=resources)
    entered, release = Event(), Event()
    submit = boundary.backend.submit_scene
    monkeypatch.setattr(boundary.backend, "name", backend)
    acquire = resources.try_acquire_slurm if backend == "slurm" else resources.try_acquire_local
    release_slot = resources.release_slurm if backend == "slurm" else resources.release_local

    def delayed(*args, **kwargs):
        job = submit(*args, **kwargs)
        if backend == "local":
            boundary.backend.jobs.pop(job.job_id)
            job.job_id = f"local-{int(job.job_id):012x}"
            job.backend = "local"
            boundary.backend.jobs[job.job_id] = job
        entered.set()
        assert release.wait(5)
        return job

    monkeypatch.setattr(boundary.backend, "submit_scene", delayed)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(runtime.generate, "Show a pause", approve_plan=True, backend=backend)
        try:
            assert entered.wait(5)
            runtime.cancel_all()
            assert not acquire()
        finally:
            release.set()
        assert future.result(timeout=5) is None
    job = runtime.manifest.scenes[1].slurm_job
    assert job.job_id and job.status == "CANCELLED"
    assert any(
        i.action.kind == "cancel_render" and i.status == "completed"
        for i in runtime.manifest.agent.intents
    )
    assert acquire()
    release_slot()


def test_uncertain_submission_and_resume_registration_keep_one_shared_claim(
    agent_boundary, monkeypatch
):
    boundary, dependencies = agent_boundary
    resources = shared_resources()
    runtime = AgentRuntime(dependencies=dependencies, resource_coordinator=resources)

    def uncertain(*args, **kwargs):
        raise ConnectionError("submission acknowledgement lost")

    monkeypatch.setattr(boundary.backend, "submit_scene", uncertain)
    with pytest.raises(RuntimeError):
        runtime.submit_existing_scene(SOURCE, "Pause", backend="slurm")
    assert not resources.try_acquire_slurm()
    attempt = runtime.manifest.agent.submissions[-1].attempt
    assert attempt.status == "pending" and not attempt.job_id
    restored = AgentRuntime(dependencies=dependencies, resource_coordinator=resources)
    restored._bind(runtime.root, runtime.manifest, None)
    assert not resources.try_acquire_slurm()
    runtime.cancel_all()
    assert not resources.try_acquire_slurm()
    assert runtime._pending_submission(1).token == attempt.token
    # Repeated restoration must not double-count the same external uncertainty.
    resources.release_render((runtime.manifest.run_id, attempt.token))
    assert resources.try_acquire_slurm()
    resources.release_slurm()


@pytest.mark.parametrize("backend", ["slurm", "local"])
def test_render_identity_claims_are_shared_and_idempotent(backend):
    resources = shared_resources()
    assert resources.claim_render(("run-a", "attempt-a"), backend)
    assert resources.claim_render(("run-a", "attempt-a"), backend, existing=True)
    assert not resources.claim_render(("run-b", "attempt-b"), backend)
    resources.release_render(("run-a", "attempt-a"))
    resources.release_render(("run-a", "attempt-a"))
    assert resources.claim_render(("run-b", "attempt-b"), backend)
    resources.release_render(("run-b", "attempt-b"))


def test_rag_probes_share_capacity_across_services(tmp_path, monkeypatch):
    from kd1_anime.rag.service import RagService

    monkeypatch.setattr(settings, "RAG_INDEX_PATH", tmp_path / "index.sqlite3")
    resources = shared_resources()
    first = RagService(config=settings, rag_semaphore=resources.rag)
    second = RagService(config=settings, rag_semaphore=resources.rag)
    entered, release, second_entered = Event(), Event(), Event()

    def blocking_embed(_):
        entered.set()
        assert release.wait(5)
        return [[1.0]]

    first.embedding = SimpleNamespace(require=lambda: None, embed=blocking_embed)
    second.embedding = SimpleNamespace(
        require=lambda: None, embed=lambda _: second_entered.set() or [[1.0]]
    )
    first.reranker = second.reranker = SimpleNamespace(
        require=lambda: None, rerank=lambda *a, **k: [object()]
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        initial = pool.submit(first.probe)
        assert entered.wait(5)
        waiting = pool.submit(second.probe)
        try:
            assert not second_entered.wait(0.05)
        finally:
            release.set()
        initial.result(timeout=5)
        waiting.result(timeout=5)
    assert second_entered.is_set()


def test_request_event_persistence_failure_still_releases_model_slot(agent_boundary, monkeypatch):
    _, dependencies = agent_boundary
    resources = shared_resources()
    runtime = bound_runtime(dependencies, resources)
    context = runtime._request_context(None, "capacity_test")
    receipt = context.reserve(settings.main_llm_profile().label, "deterministic-model")
    assert not resources.llm.acquire(blocking=False)

    def fail(*args, **kwargs):
        raise OSError("isolated usage checkpoint failure")

    monkeypatch.setattr(runtime, "_patch", fail)
    with pytest.raises(OSError):
        context.complete(receipt, time.monotonic())
    assert resources.llm.acquire(blocking=False)
    resources.llm.release()


def test_cancel_at_operation_exit_still_cancels_submitted_job(agent_boundary, monkeypatch):
    import threading

    boundary, dependencies = agent_boundary
    runtime = AgentRuntime(dependencies=dependencies, resource_coordinator=shared_resources())
    checked, release = Event(), Event()
    original = runtime.control.cancelled.is_set

    def controlled_is_set():
        result = original()
        if (
            not result
            and not runtime._operation_active
            and threading.current_thread() is not threading.main_thread()
            and hasattr(runtime, "committer")
            and runtime.manifest.scenes
            and runtime.manifest.scenes[1].slurm_job is not None
            and not checked.is_set()
        ):
            checked.set()
            assert release.wait(5)
        return result

    monkeypatch.setattr(runtime.control.cancelled, "is_set", controlled_is_set)
    with ThreadPoolExecutor(max_workers=2) as pool:
        submission = pool.submit(runtime.submit_existing_scene, SOURCE, "Pause", backend="slurm")
        try:
            assert checked.wait(5)
            cancellation = pool.submit(runtime.cancel_all)
            assert runtime.control.cancelled.wait(5)
        finally:
            release.set()
        with suppress(RuntimeError):
            submission.result(timeout=5)
        cancellation.result(timeout=5)
    manifest = RunRepository(boundary.workspace).load(runtime.manifest.run_id)
    assert manifest.scenes[1].slurm_job.status == "CANCELLED"
    assert any(
        item.action.kind == "cancel_render" and item.status == "completed"
        for item in manifest.agent.intents
    )
