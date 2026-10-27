from test_agent_vertical_slice import agent_boundary as agent_boundary

from kd1_anime.agent_runtime.runtime import AgentRuntime
from kd1_anime.agent_runtime.session import SessionStore, UserCommand
from kd1_anime.queries import QueryService
from kd1_anime.run_store import RunRepository


def test_runtime_plan_request_is_queryable_and_consumed_status_survives_transport_commits(
    agent_boundary,
):
    boundary, dependencies = agent_boundary
    assert AgentRuntime(dependencies=dependencies).generate("Show a pause", dry_run=True) is None
    root = next((boundary.workspace / "runs").iterdir())
    repository = RunRepository(boundary.workspace)
    manifest = repository.load(root.name)
    sessions = SessionStore(boundary.workspace / "sessions")
    queries = QueryService(repository, sessions)
    requests = queries.approvals(root.name)
    assert len(requests) == 1 and requests[0].status == "pending"
    assert requests[0].session_id == manifest.agent.session_id
    action = manifest.agent.pending_approval
    assert requests[0].resource_id == action.resource_id
    assert requests[0].input_sha256 == action.input_sha256
    sessions.issue_grant(
        manifest.agent.session_id,
        UserCommand(
            session_id=manifest.agent.session_id,
            kind="approve",
            action=action,
        ),
    )
    assert queries.approvals(root.name)[0].status == "approved"
    assert AgentRuntime(dependencies=dependencies).resume(root.name) is None
    assert repository.load(root.name).status == "dry_run_complete"
    assert queries.approvals(root.name)[0].status == "consumed"
