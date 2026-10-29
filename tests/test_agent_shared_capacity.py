"""Shared capacity across independent v11 runtimes, without external execution."""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event
from types import SimpleNamespace

import pytest
from test_agent_vertical_slice import agent_boundary as agent_boundary
from test_application import SOURCE

from kd1_anime.agent_runtime.runtime import AgentRuntime
from kd1_anime.commits import RunCommitter
from kd1_anime.config import settings
from kd1_anime.resources import ResourceCoordinator
from kd1_anime.run_store import RunRepository


def shared_resources():
    return ResourceCoordinator(
        llm_limit=1, visual_llm_limit=1, rag_limit=1, slurm_limit=1, local_limit=1
    )


def stored_manifest(runtime):
    return RunRepository(settings.WORKSPACE_DIR).load(runtime.root.name)


def probing_runtime(dependencies, resources, exercise, *, visual=False):
    """Exercise transport only through the runtime's injected public probe boundary."""

    def probe(agent):
        if visual:
            agent.profile = settings.visual_llm_profile()
            agent.model = agent.profile.model
        exercise(agent)

    return AgentRuntime(
        dependencies=replace(dependencies, probe=probe),
        resource_coordinator=resources,
    )


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
    first = probing_runtime(
        dependencies, resources, lambda agent: completion(agent, first_request), visual=visual
    )
    second = probing_runtime(
        dependencies, resources, lambda agent: completion(agent, second_request), visual=visual
    )
    entered, release, second_entered = Event(), Event(), Event()

    def first_request(**kwargs):
        entered.set()
        assert release.wait(5)
        return SimpleNamespace(choices=[])

    def second_request(**kwargs):
        second_entered.set()
        return SimpleNamespace(choices=[])

    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(first.generate, "First pause")
        assert entered.wait(5)
        b = pool.submit(second.generate, "Second pause")
        try:
            assert not second_entered.wait(0.05)
            # The waiting request has a durable reservation, but has not contacted its endpoint.
            deadline = time.monotonic() + 5
            while not hasattr(second, "root") or not any(
                r.stage == "endpoint_probe"
                for r in stored_manifest(second).request_budget.reservations
            ):
                assert time.monotonic() < deadline
                time.sleep(0.01)
            raw = json.loads((second.root / "manifest.json").read_text())
            assert any(
                r["stage"] == "endpoint_probe" for r in raw["request_budget"]["reservations"]
            )
        finally:
            release.set()
        a.result(timeout=5)
        b.result(timeout=5)
    assert second_entered.is_set()


def test_visual_endpoint_does_not_consume_main_endpoint_capacity(agent_boundary):
    _, dependencies = agent_boundary
    resources = shared_resources()
    responses = []

    def exercise(agent):
        try:
            responses.append(completion(agent, lambda **kw: SimpleNamespace(choices=[])))
            assert not resources.llm.acquire(blocking=False)
        finally:
            resources.llm.release()

    runtime = probing_runtime(dependencies, resources, exercise, visual=True)
    assert resources.llm.acquire(blocking=False)
    assert runtime.generate("Show a pause") is None
    assert responses[0].choices == []
    reservation = stored_manifest(runtime).request_budget.reservations[0]
    assert reservation.stage == "endpoint_probe"
    assert reservation.profile == settings.visual_llm_profile().label


@pytest.mark.parametrize("ending", ["exhausted", "error", "close", "iterator_close"])
def test_stream_retains_slot_until_its_actual_end(agent_boundary, ending):
    _, dependencies = agent_boundary
    resources = shared_resources()
    observed = []
    closed = Event()

    class Stream:
        def __iter__(self):
            yield SimpleNamespace(choices=[])
            if ending == "error":
                raise ConnectionError("isolated stream failure")

        def close(self):
            closed.set()

    def exercise(agent):
        stream = completion(agent, lambda **kw: Stream(), stream=True)
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
        observed.append(ending)

    runtime = probing_runtime(dependencies, resources, exercise)
    assert runtime.generate("Show a pause") is None
    assert observed == [ending]
    manifest = stored_manifest(runtime)
    reservation = next(
        r for r in manifest.request_budget.reservations if r.stage == "endpoint_probe"
    )
    assert any(u.request_id == reservation.request_id for u in manifest.agent.usage)


def test_transport_failure_releases_capacity_but_preserves_request_reservation(agent_boundary):
    _, dependencies = agent_boundary
    resources = shared_resources()
    dispatched = []

    def fail(**kwargs):
        dispatched.append(True)
        raise ConnectionError("isolated endpoint failure")

    runtime = probing_runtime(dependencies, resources, lambda agent: completion(agent, fail))
    assert runtime.generate("Show a pause") is None
    manifest = stored_manifest(runtime)
    assert dispatched == [True]
    assert len(manifest.request_budget.reservations) == 1
    assert manifest.request_budget.reservations[0].stage == "endpoint_probe"
    assert manifest.status == "failed"
    assert resources.llm.acquire(blocking=False)
    resources.llm.release()


def test_cancelling_a_waiting_model_keeps_reservation_and_never_dispatches(agent_boundary):
    _, dependencies = agent_boundary
    resources = shared_resources()
    entered = Event()
    runtime = probing_runtime(
        dependencies, resources, lambda agent: completion(agent, lambda **kw: entered.set())
    )
    assert resources.llm.acquire(blocking=False)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(runtime.generate, "Show a pause")
        deadline = time.monotonic() + 5
        try:
            while (
                not hasattr(runtime, "root")
                or not stored_manifest(runtime).request_budget.reservations
            ):
                assert time.monotonic() < deadline
                time.sleep(0.01)
            runtime.cancel_all()
            assert future.result(timeout=5) is None
        finally:
            resources.llm.release()
    assert not entered.is_set()
    manifest = stored_manifest(runtime)
    assert len(manifest.request_budget.reservations) == 1
    assert manifest.request_budget.reservations[0].stage == "endpoint_probe"
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
    manifest = stored_manifest(runtime)
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
        reservation = manifest.agent.submissions[-1]
        assert reservation.attempt.job_id == job.job_id
        assert reservation.attempt.token not in manifest.agent.retired_submissions
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
    manifest = stored_manifest(runtime)
    job = manifest.scenes[1].slurm_job
    assert job.job_id and job.status == "CANCELLED"
    assert any(
        i.action.kind == "cancel_render" and i.status == "completed" for i in manifest.agent.intents
    )
    assert acquire()
    release_slot()


def test_idle_cancel_rejects_replaced_legacy_manifest_before_lock(agent_boundary, monkeypatch):
    boundary, dependencies = agent_boundary
    runtime = AgentRuntime(dependencies=dependencies)
    _, _, run_id = runtime.submit_existing_scene(SOURCE, "Pause", backend="slurm")
    root = boundary.workspace / "runs" / run_id
    path = root / "manifest.json"
    raw = json.loads(path.read_text())
    raw["schema_version"] = 10
    raw.pop("agent")
    path.write_text(json.dumps(raw))
    before = path.read_bytes()
    monkeypatch.setattr(
        "kd1_anime.agent_runtime.runtime.lock_run",
        lambda *_: pytest.fail("legacy cancellation acquired the run lock"),
    )
    with pytest.raises(ValueError, match="只读"):
        runtime.cancel_all()
    assert path.read_bytes() == before


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
    manifest = stored_manifest(runtime)
    attempt = manifest.agent.submissions[-1].attempt
    assert attempt.status == "pending" and not attempt.job_id
    for _ in range(2):
        restored = AgentRuntime(dependencies=dependencies, resource_coordinator=resources)
        started = []

        def cancel_on_start(name, data, *, started=started, restored=restored):
            if name == "run_started":
                started.append(data["run_id"])
                assert not resources.try_acquire_slurm()
                restored.cancel_all()

        assert restored.resume(manifest.run_id, callback=cancel_on_start) is None
        assert started == [manifest.run_id]
        assert not resources.try_acquire_slurm()
        persisted = stored_manifest(restored)
        assert persisted.agent.submissions[-1].attempt.token == attempt.token
        assert attempt.token not in persisted.agent.retired_submissions
    runtime.cancel_all()
    assert not resources.try_acquire_slurm()
    # Repeated public recovery must not double-count the same external uncertainty.
    resources.release_render((manifest.run_id, attempt.token))
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
    commit = RunCommitter.commit
    failed = []

    def fail_usage(self, command):
        proposed = command.patch.agent
        if proposed is not None and proposed.usage:
            failed.append(True)
            raise OSError("isolated usage checkpoint failure")
        return commit(self, command)

    monkeypatch.setattr(RunCommitter, "commit", fail_usage)
    runtime = probing_runtime(
        dependencies,
        resources,
        lambda agent: completion(agent, lambda **kw: SimpleNamespace(choices=[])),
    )
    assert runtime.generate("Show a pause") is None
    manifest = stored_manifest(runtime)
    assert failed
    assert len(manifest.request_budget.reservations) == 1
    assert not manifest.agent.usage
    assert manifest.status == "failed"
    assert resources.llm.acquire(blocking=False)
    resources.llm.release()


def test_cancel_after_submission_checkpoint_still_cancels_submitted_job(agent_boundary):
    _, dependencies = agent_boundary
    checked, release = Event(), Event()

    def after_commit(manifest):
        job = manifest.scenes.get(1)
        if job is not None and job.slurm_job is not None and not checked.is_set():
            checked.set()
            assert release.wait(5)

    resources = shared_resources()
    runtime = AgentRuntime(
        dependencies=replace(dependencies, after_commit=after_commit),
        resource_coordinator=resources,
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        submission = pool.submit(runtime.submit_existing_scene, SOURCE, "Pause", backend="slurm")
        try:
            assert checked.wait(5)
            assert stored_manifest(runtime).scenes[1].slurm_job.job_id
            runtime.cancel_all()
        finally:
            release.set()
        with pytest.raises(RuntimeError):
            submission.result(timeout=5)
    manifest = stored_manifest(runtime)
    assert manifest.scenes[1].slurm_job.status == "CANCELLED"
    assert any(
        item.action.kind == "cancel_render" and item.status == "completed"
        for item in manifest.agent.intents
    )
    assert resources.try_acquire_slurm()
    resources.release_slurm()
