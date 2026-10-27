import pytest
from test_agent_vertical_slice import agent_boundary as agent_boundary

from kd1_anime.agent_runtime.runtime import AgentRuntime
from kd1_anime.agent_runtime.session import UserCommand
from kd1_anime.domain.dependencies import WorkIdentity
from kd1_anime.domain.review import ReviewResult
from kd1_anime.queries import QueryService
from kd1_anime.run_store import RunRepository, restore_run_path


def test_plan_flag_never_approves_diff_and_approved_install_is_reviewed_separately(agent_boundary):
    boundary, dependencies = agent_boundary
    original = dependencies.reviewer_factory

    class Reviewer(original):
        def review(self, code, plan, **kwargs):
            accepted = super().review(code, plan, **kwargs)
            if "self.wait(1.0)" in code:
                return accepted
            return ReviewResult(
                is_valid=False,
                severity="major",
                feedback="Use explicit floating duration",
                fixes=[{"find": "self.wait(1)", "replace": "self.wait(1.0)"}],
            )

    dependencies.reviewer_factory = Reviewer
    runtime = AgentRuntime(dependencies=dependencies)
    assert runtime.generate("Show a pause", dry_run=True, approve_plan=True) is None
    pending = runtime.manifest.agent.pending_approval
    assert pending.kind == "inspect_candidate" and pending.proposal is not None
    scene = runtime.manifest.scenes[1]
    assert pending.work_identity == WorkIdentity(scene.revision, scene.input_sha256)
    assert pending.input_sha256 == pending.proposal.digest() != scene.input_sha256
    assert "self.wait(1.0)" not in restore_run_path(runtime.root, scene.code_file).read_text()
    assert not boundary.backend.sources and boundary.calls.count("code_review") == 1
    requests = QueryService(RunRepository(boundary.workspace), runtime.sessions).approvals(
        runtime.manifest.run_id
    )
    diff_request = next(item for item in requests if item.scene_id == 1)
    assert diff_request.status == "pending" and diff_request.proposal == pending.proposal
    assert diff_request.input_sha256 == pending.proposal.digest()
    assert diff_request.proposal.base_source.sha256 == scene.code_sha256
    assert {
        grant.action for grant in runtime.sessions.load(runtime.manifest.agent.session_id).grants
    } == {"review_plan"}
    runtime.sessions.issue_grant(
        runtime.manifest.agent.session_id,
        UserCommand(session_id=runtime.manifest.agent.session_id, kind="approve", action=pending),
    )

    def interrupt_after_install(manifest):
        source = restore_run_path(runtime.root, manifest.scenes[1].code_file).read_text()
        if "self.wait(1.0)" in source and not manifest.scenes[1].reviewed:
            assert boundary.calls.count("code_review") == 1
            raise OSError("after pure inspected installation")

    dependencies.after_commit = interrupt_after_install
    with pytest.raises(OSError, match="pure inspected installation"):
        AgentRuntime(dependencies=dependencies).resume(runtime.manifest.run_id)
    dependencies.after_commit = None
    assert AgentRuntime(dependencies=dependencies).resume(runtime.manifest.run_id) is None
    manifest = RunRepository(boundary.workspace).load(runtime.manifest.run_id)
    assert manifest.state == "DONE" and manifest.scenes[1].reviewed
    assert boundary.calls.count("code_review") == 2 and not boundary.backend.sources
    assert not manifest.agent.pending_candidates
    assert manifest.agent.scene_review_counters[1].review_round == 2
    assert (
        manifest.agent.integrity_errors(
            manifest.run_id, {item.request_id for item in manifest.request_budget.reservations}
        )
        == []
    )
