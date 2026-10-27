"""Exercise real orchestration without a provider or external services."""

from types import SimpleNamespace

from test_agent_tools import HASH, IDENTITY, RUN, fixture, result

from kd1_anime.agent_runtime.contracts import PermissionScope, ProposePlan
from kd1_anime.agent_runtime.loop import AgentLoop
from kd1_anime.agent_runtime.policy import Policy, PolicyBudget, PolicySnapshot


def setup(*, waiting=None, allow=True, progress=False, next_action=None, requires_approval=False):
    snapshots = [
        PolicySnapshot(
            run_id=RUN,
            state="INIT",
            work_identity=IDENTITY,
            waiting_for_approval=waiting == "approval",
            waiting_for_job=waiting == "job",
        )
    ]
    model_calls = []
    executions = []

    def execute(action, context):
        executions.append(action)
        if progress == "revision":
            snapshots[0] = snapshots[0].model_copy(update={"revision": snapshots[0].revision + 1})
        elif progress:
            snapshots[0] = snapshots[0].model_copy(update={"state": "DONE", "revision": 2})
        return result()

    executor, persisted = fixture(
        execute, side_effect="commit", requires_approval=requires_approval
    )
    session = SimpleNamespace(
        session_id="session-one",
        run_ids=(RUN,),
        grants=(),
        permissions=(PermissionScope(permission="write", resource_id=f"run:{RUN}"),)
        if allow
        else (),
    )
    action = ProposePlan(
        run_id=RUN, resource_id=f"run:{RUN}", input_sha256=HASH, work_identity=IDENTITY
    )

    def propose(snapshot, available):
        model_calls.append(True)
        return action

    loop = AgentLoop(
        policy=Policy(),
        executor=executor,
        snapshot=lambda: snapshots[0],
        session=lambda: session,
        budget=PolicyBudget,
        next_action=next_action or (lambda snapshot: action),
        propose_action=propose,
    )
    return loop, model_calls, executions, persisted


def test_waiting_approval_or_job_returns_without_action_or_model_request():
    for waiting in ("approval", "job"):
        loop, model, executions, persisted = setup(waiting=waiting)
        outcome = loop.run(max_steps=10)
        assert outcome.status == "NEED_USER"
        assert outcome.reason == f"waiting_for_{waiting}"
        assert outcome.steps == 0
        assert model == executions == persisted == []


def test_real_loop_progress_reaches_done_without_unnecessary_provider():
    loop, model, executions, _ = setup(progress=True)
    outcome = loop.run(max_steps=10)
    assert outcome.status == "DONE"
    assert len(executions) == 1
    assert model == []


def test_denial_stops_before_tool_or_provider_and_duplicate_never_reexecutes():
    loop, model, executions, persisted = setup(allow=False)
    assert loop.run(max_steps=10).status == "NEED_USER"
    assert executions == persisted == model == []
    loop, model, executions, persisted = setup()
    outcome = loop.run(max_steps=10)
    assert outcome.status == "FAILED"
    assert outcome.reason == "no_progress"
    assert len(executions) == 1


def test_cancelled_and_exhausted_bounds_make_no_requests():
    loop, model, executions, persisted = setup()
    assert loop.run(max_steps=0).status == "BUDGET_EXHAUSTED"
    assert loop.run(max_steps=10, cancelled=lambda: True).reason == "cancelled"
    assert executions == persisted == model == []


def test_new_approval_request_stops_without_persistence_or_provider():
    loop, model, executions, persisted = setup(requires_approval=True)
    outcome = loop.run(max_steps=10)
    assert outcome.status == "NEED_USER"
    assert outcome.reason == "approval_required"
    assert executions == persisted == model == []


def test_provider_called_only_for_explicit_proposal_and_never_when_no_tools_available():
    loop, model, executions, persisted = setup(progress=True, next_action=lambda snapshot: None)
    assert loop.run(max_steps=5).status == "DONE"
    assert len(model) == len(executions) == 1
    loop, model, executions, persisted = setup(allow=False, next_action=lambda snapshot: None)
    assert loop.run(max_steps=5).status == "NEED_USER"
    assert model == executions == persisted == []
    loop, model, executions, persisted = setup(
        requires_approval=True, next_action=lambda snapshot: None
    )
    assert loop.run(max_steps=5).status == "NEED_USER"
    assert model == executions == persisted == []


def test_repeated_signature_is_bounded_even_if_manifest_revision_advances():
    loop, _, executions, _ = setup(progress="revision")
    outcome = loop.run(max_steps=100)
    assert outcome.status == "FAILED"
    assert outcome.reason == "repeated_action"
    assert len(executions) == 1


def test_request_budget_exhaustion_does_not_call_proposer():
    loop, model, executions, persisted = setup(next_action=lambda snapshot: None)
    loop.budget = lambda: PolicyBudget(requests_remaining=0)
    assert loop.run(max_steps=5).status == "BUDGET_EXHAUSTED"
    assert model == executions == persisted == []
