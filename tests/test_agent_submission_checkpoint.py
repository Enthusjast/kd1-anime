import pytest
from test_agent_vertical_slice import agent_boundary as agent_boundary

from kd1_anime.agent_runtime.runtime import AgentRuntime
from kd1_anime.commits import RunCommitter
from kd1_anime.run_store import RunRepository


@pytest.mark.parametrize("cancelled", [False, True])
def test_submission_result_checkpoint_failure_cancels_and_preserves_attempt_identity(
    agent_boundary, monkeypatch, cancelled
):
    boundary, dependencies = agent_boundary
    commit = RunCommitter.commit
    cancellations = []

    def fail_result(self, command):
        if command.phase == "submit_render":
            raise OSError("isolated checkpoint failure")
        return commit(self, command)

    def cancel(job_id):
        cancellations.append(job_id)
        return cancelled

    monkeypatch.setattr(RunCommitter, "commit", fail_result)
    monkeypatch.setattr(boundary.backend, "cancel_job", cancel)
    runtime = AgentRuntime(dependencies=dependencies)
    assert runtime.generate("Show a pause", approve_plan=True) is None
    manifest = RunRepository(boundary.workspace).load(runtime.manifest.run_id)
    attempt = manifest.agent.submissions[-1].attempt
    assert len(boundary.backend.sources) == 1 and cancellations == [attempt.job_id]
    assert attempt.job_id and attempt.status == "submitted"
    assert not manifest.scenes[1].rendered and manifest.scenes[1].artifact is None
    assert manifest.agent.intents[-1].status == "unknown"
    assert manifest.status == "failed" and manifest.state == "ERROR"
