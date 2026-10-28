import pytest
from test_agent_vertical_slice import agent_boundary as agent_boundary

from kd1_anime.agent_runtime.session import SessionStore
from kd1_anime.application import Application
from kd1_anime.run_store import RunRepository


def test_confirmed_clarification_session_is_linked_to_new_run(agent_boundary, monkeypatch):
    boundary, dependencies = agent_boundary
    from kd1_anime.agent_runtime.runtime import AgentRuntime

    monkeypatch.setattr(
        "kd1_anime.application._agent_runtime", lambda: AgentRuntime(dependencies=dependencies)
    )
    app = Application()
    session_id = app.create_session("Original mathematical question")
    app.session_command(session_id, "clarification", "Show the limiting step")
    app.session_command(session_id, "ready", "Show a pause")
    app.session_command(session_id, "confirm", "Show a pause")
    assert (
        app.generate("Show a pause", dry_run=True, approve_plan=True, session_id=session_id) is None
    )
    root = next((boundary.workspace / "runs").iterdir())
    manifest = RunRepository(boundary.workspace).load(root.name)
    session = SessionStore(boundary.workspace / "sessions").load(session_id)
    assert manifest.agent.session_id == session_id and session.run_ids == (root.name,)
    assert [item.kind for item in session.commands[:4]] == [
        "goal",
        "clarification",
        "ready",
        "confirm",
    ]
    assert session.goal == "Original mathematical question"
    assert {item.permission for item in session.permissions} == {"read", "write", "exec", "cluster"}


def test_clarifier_reservation_and_provider_usage_transfer_into_run(agent_boundary, monkeypatch):
    import time
    from types import SimpleNamespace

    from kd1_anime.agent_runtime.runtime import AgentRuntime

    boundary, dependencies = agent_boundary
    monkeypatch.setattr(
        "kd1_anime.application._agent_runtime", lambda: AgentRuntime(dependencies=dependencies)
    )
    app = Application()
    session_id = app.create_session("Original goal")
    context = app.session_request_context(session_id)
    receipt = context.reserve(
        "main",
        "model",
        prompt_sha256="a" * 64,
        estimated_prompt_tokens=10,
        reserved_output_tokens=20,
    )
    response = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=7, completion_tokens=4, total_tokens=11)
    )
    context.complete(receipt, time.monotonic(), response)
    app.session_command(session_id, "confirm", "Show a pause")
    app.generate("Show a pause", dry_run=True, approve_plan=True, session_id=session_id)
    root = next((boundary.workspace / "runs").iterdir())
    manifest = RunRepository(boundary.workspace).load(root.name)
    assert any(
        item.request_id == receipt.request_id for item in manifest.request_budget.reservations
    )
    usage = next(item for item in manifest.agent.usage if item.request_id == receipt.request_id)
    assert usage.usage_source == "provider_reported" and usage.total_tokens == 11
    assert usage.prompt_sha256 == "a" * 64 and usage.cost_status == "unknown" and usage.cost is None


def test_unconfirmed_session_cannot_start_generation(agent_boundary, monkeypatch):
    boundary, dependencies = agent_boundary
    from kd1_anime.agent_runtime.runtime import AgentRuntime

    monkeypatch.setattr(
        "kd1_anime.application._agent_runtime", lambda: AgentRuntime(dependencies=dependencies)
    )
    app = Application()
    session_id = app.create_session("Original goal")
    app.session_command(session_id, "ready", "Different goal")
    with pytest.raises(ValueError):
        app.generate("Different goal", session_id=session_id)
    assert not boundary.backend.sources


def test_approval_decision_rejects_wrong_hash_before_issuing_grant(agent_boundary, monkeypatch):
    from kd1_anime.agent_runtime.runtime import AgentRuntime

    boundary, dependencies = agent_boundary
    monkeypatch.setattr(
        "kd1_anime.application._agent_runtime", lambda: AgentRuntime(dependencies=dependencies)
    )
    app = Application()
    assert app.generate("Show a pause", dry_run=True) is None
    root = next((boundary.workspace / "runs").iterdir())
    request = app.approval_requests(root.name)[-1]
    with pytest.raises(ValueError):
        app.decide_approval(
            root.name,
            True,
            input_sha256="0" * 64,
            resource_id=request.resource_id,
            work_identity=request.work_identity,
        )
    assert app.approval_requests(root.name)[-1].status == "pending"
    app.decide_approval(
        root.name,
        True,
        input_sha256=request.input_sha256,
        resource_id=request.resource_id,
        work_identity=request.work_identity,
    )
    assert app.resume(root.name) is None
    assert RunRepository(boundary.workspace).load(root.name).status == "dry_run_complete"
    assert app.approval_requests(root.name)[-1].status == "consumed"
