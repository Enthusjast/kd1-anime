import pytest
from test_agent_vertical_slice import agent_boundary as agent_boundary

from kd1_anime.agent_runtime.runtime import AgentRuntime
from kd1_anime.commits import RunCommitter
from kd1_anime.resources import ResourceCoordinator
from kd1_anime.run_store import RunRepository


@pytest.mark.parametrize("cancelled", [False, True])
@pytest.mark.parametrize("checkpoint_writable", [False, True])
def test_submission_result_checkpoint_failure_cancels_and_preserves_attempt_identity(
    agent_boundary, monkeypatch, cancelled, checkpoint_writable
):
    boundary, dependencies = agent_boundary
    commit = RunCommitter.commit
    cancellations = []

    def fail_result(self, command):
        if command.phase == "submit_render":
            raise OSError("isolated checkpoint failure")
        proposed = command.patch.agent
        if (
            not checkpoint_writable
            and proposed is not None
            and any(item.cancellation_status != "not_requested" for item in proposed.submissions)
        ):
            raise OSError("isolated compensation checkpoint failure")
        return commit(self, command)

    def cancel(job_id):
        cancellations.append(job_id)
        return cancelled

    monkeypatch.setattr(RunCommitter, "commit", fail_result)
    monkeypatch.setattr(boundary.backend, "cancel_job", cancel)
    resources = ResourceCoordinator(llm_limit=1, slurm_limit=1)
    runtime = AgentRuntime(dependencies=dependencies, resource_coordinator=resources)
    assert runtime.generate("Show a pause", approve_plan=True) is None
    manifest = RunRepository(boundary.workspace).load(runtime.manifest.run_id)
    attempt = manifest.agent.submissions[-1].attempt
    assert len(boundary.backend.sources) == 1 and cancellations == [attempt.job_id]
    assert attempt.job_id and attempt.status == "submitted"
    assert not manifest.scenes[1].rendered and manifest.scenes[1].artifact is None
    assert manifest.agent.intents[-1].status == "unknown"
    assert manifest.status == "failed" and manifest.state == "ERROR"
    reservation = manifest.agent.submissions[-1]
    expected = ("succeeded" if cancelled else "failed") if checkpoint_writable else "not_requested"
    assert reservation.cancellation_status == expected
    # Backend success without its durable confirmation must still retain capacity.
    released = cancelled and checkpoint_writable
    assert resources.try_acquire_slurm() is released
    if released:
        resources.release_slurm()
    restored = AgentRuntime(dependencies=dependencies, resource_coordinator=resources)
    started = []

    def cancel_on_start(name, data):
        if name == "run_started":
            started.append(data["run_id"])
            assert resources.try_acquire_slurm() is released
            if released:
                resources.release_slurm()
            restored.cancel_all()

    assert restored.resume(manifest.run_id, callback=cancel_on_start) is None
    assert started == [manifest.run_id]
    persisted = RunRepository(boundary.workspace).load(manifest.run_id)
    assert persisted.agent.submissions[-1].attempt.token == attempt.token
    assert attempt.token not in persisted.agent.retired_submissions
    assert len(boundary.backend.sources) == 1
    assert resources.try_acquire_slurm() is released
    if released:
        resources.release_slurm()


def test_failed_attempt_reservation_releases_capacity_before_dispatch(agent_boundary, monkeypatch):
    boundary, dependencies = agent_boundary
    commit = RunCommitter.commit
    failed = []

    def fail_attempt(self, command):
        proposed = command.patch.agent
        if proposed is not None and proposed.submissions and not failed:
            failed.append(True)
            raise OSError("attempt checkpoint failed")
        return commit(self, command)

    monkeypatch.setattr(RunCommitter, "commit", fail_attempt)
    resources = ResourceCoordinator(llm_limit=1, slurm_limit=1)
    runtime = AgentRuntime(dependencies=dependencies, resource_coordinator=resources)
    assert runtime.generate("Show a pause", approve_plan=True) is None
    assert failed and not boundary.backend.sources
    assert resources.try_acquire_slurm()
    resources.release_slurm()
