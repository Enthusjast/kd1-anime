import time

import pytest
import test_application
from test_agent_vertical_slice import agent_boundary as agent_boundary

from kd1_anime.agent_runtime.runtime import AgentRuntime
from kd1_anime.cluster.jobs import PollObservation
from kd1_anime.config import settings
from kd1_anime.run_store import RunRepository


def failure_backend(boundary, monkeypatch, status):
    submit = boundary.backend.submit_scene

    def submit_source(scene_id, source, *args, **kwargs):
        monkeypatch.setattr(test_application, "SOURCE", source.read_text())
        return submit(scene_id, source, *args, **kwargs)

    monkeypatch.setattr(boundary.backend, "submit_scene", submit_source)
    monkeypatch.setattr(
        boundary.backend,
        "poll",
        lambda ids: PollObservation(
            {job_id: status if job_id == "1" else "COMPLETED" for job_id in ids}, {}
        ),
    )


def test_render_fix_uses_dispatch_identity_and_reviews_new_source_before_resubmission(
    agent_boundary, monkeypatch
):
    boundary, dependencies = agent_boundary
    failure_backend(boundary, monkeypatch, "FAILED")
    monkeypatch.setattr(
        boundary.backend,
        "get_error_log",
        lambda *a, **k: (
            "Traceback (most recent call last):\nNameError: name 'duration' is not defined"
        ),
    )
    fixes = []

    class Fixer:
        def fix(self, code, *args, **kwargs):
            receipt = self.request_context.reserve("main", "isolated-render-fixer")
            fixes.append(code)
            self.request_context.complete(receipt, time.monotonic())
            return code.replace("self.wait(1)", "self.wait(2)")

    dependencies.fixer_factory = Fixer
    runtime = AgentRuntime(dependencies=dependencies)
    output = runtime.generate("Show a pause", approve_plan=True)
    manifest = RunRepository(boundary.workspace).load(runtime.manifest.run_id)
    assert manifest.status == "completed" and output.is_file()
    scene = manifest.scenes[1]
    assert scene.fix_attempts == 1 and len(fixes) == 1 and scene.reviewed and scene.rendered
    first, second = boundary.backend.sources
    assert first.read_text() != second.read_text()
    assert (
        scene.code_sha256 == scene.artifact.code_sha256 == scene.execution_verification.code_sha256
    )
    assert scene.submission.job_id == "2" and len(manifest.agent.submissions) == 2
    actions = [item.action for item in manifest.agent.invocations]
    repaired = actions.index("propose_render_fix")
    assert "review_code" in actions[repaired + 1 : actions.index("submit_render", repaired)]
    assert boundary.calls.count("code_review") == 2
    assert AgentRuntime(dependencies=dependencies).resume(manifest.run_id) == output
    assert len(boundary.backend.sources) == 2


@pytest.mark.parametrize("limit", [0, 2])
def test_infrastructure_retries_obey_persisted_budget_without_code_rewrite(
    agent_boundary, monkeypatch, limit
):
    boundary, dependencies = agent_boundary
    monkeypatch.setattr(settings, "MAX_INFRA_RETRIES", limit)
    failure_backend(boundary, monkeypatch, "NODE_FAIL")
    monkeypatch.setattr(
        boundary.backend,
        "get_error_log",
        lambda *a, **k: "node failure: infrastructure unavailable",
    )
    runtime = AgentRuntime(dependencies=dependencies)
    output = runtime.generate("Show a pause", approve_plan=True)
    manifest = RunRepository(boundary.workspace).load(runtime.manifest.run_id)
    scene = manifest.scenes[1]
    assert scene.fix_attempts == 0
    if limit:
        assert manifest.status == "completed" and output.is_file()
        assert scene.infra_retries == 1 and len(boundary.backend.sources) == 2
        assert boundary.backend.sources[0].read_bytes() == boundary.backend.sources[1].read_bytes()
    else:
        assert output is None and manifest.status == "failed"
        assert scene.infra_retries == 0 and len(boundary.backend.sources) == 1
    assert boundary.calls.count("code_review") == 1
