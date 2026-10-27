import pytest
from pydantic import ValidationError

from kd1_anime.agent_runtime.contracts import (
    ACTION_ADAPTER,
    AgentEvent,
    AgentMessage,
    ApprovalGrant,
    EventPayload,
    ObjectRef,
    ProposePlan,
    UsageReceipt,
)

RUN = "20261027-120000-1234abcd"
SESSION = "session-1234abcd"
ACTION = {
    "kind": "propose_plan",
    "run_id": RUN,
    "resource_id": f"run:{RUN}",
    "input_sha256": "a" * 64,
}


def test_runtime_actions_are_closed_and_cannot_authorize_or_execute_arbitrary_commands():
    assert isinstance(ACTION_ADAPTER.validate_python(ACTION), ProposePlan)
    for kind in ("approve_plan", "shell", "python", "write_file", "spawn_agent"):
        with pytest.raises(ValidationError):
            ACTION_ADAPTER.validate_python({**ACTION, "kind": kind})
    with pytest.raises(ValidationError):
        ACTION_ADAPTER.validate_python({**ACTION, "command": "rm -rf /"})


def test_visible_messages_reject_hidden_reasoning_and_redact_credentials():
    message = AgentMessage(
        session_id=SESSION,
        role="user",
        kind="goal",
        content="Show a circle; api_key=secret-value <think>hidden chain</think>",
    )
    dumped = message.model_dump_json()
    assert "hidden chain" not in dumped and "secret-value" not in dumped
    assert "Show a circle" in dumped
    with pytest.raises(ValidationError):
        AgentMessage(
            session_id=SESSION, role="assistant", kind="action", reasoning_content="private"
        )


def test_messages_cannot_relabel_assistant_output_as_user_command():
    with pytest.raises(ValidationError):
        AgentMessage(session_id=SESSION, role="assistant", kind="command", content="approve")
    with pytest.raises(ValidationError):
        AgentMessage(session_id=SESSION, role="user", kind="action", action=ACTION)


@pytest.mark.parametrize("path", ["/tmp/source.py", "../source.py", "objects/../../source.py"])
def test_runtime_references_cannot_escape_run(path):
    with pytest.raises(ValidationError):
        ObjectRef(path=path, sha256="a" * 64, kind="code")


def test_events_bind_durability_sequence_revision_and_typed_payload():
    event = AgentEvent(
        run_id=RUN,
        stream_id=RUN,
        sequence=1,
        manifest_revision=2,
        durability="committed",
        name="snapshot_committed",
        payload=EventPayload(state="PLANNING"),
    )
    assert event.sequence == 1 and event.payload.state == "PLANNING"
    with pytest.raises(ValidationError):
        AgentEvent.model_validate({**event.model_dump(), "sequence": 0})
    with pytest.raises(ValidationError):
        EventPayload(reasoning_content="private")


def test_approval_contract_cannot_be_issued_by_model_and_binds_run():
    grant = ApprovalGrant(
        action="propose_code_candidate",
        resource_id=f"run:{RUN}",
        input_sha256="a" * 64,
        session_id=SESSION,
        run_id=RUN,
        issued_by="local_user",
    )
    assert grant.run_id == RUN and grant.consumed_by_intent is None
    with pytest.raises(ValidationError):
        ApprovalGrant.model_validate({**grant.model_dump(), "issued_by": "assistant"})


def test_unknown_usage_is_not_reported_as_zero_cost():
    receipt = UsageReceipt(request_id="request-1")
    assert receipt.usage_source == receipt.cost_status == "unknown"
    assert receipt.total_tokens is None and receipt.cost is None
    with pytest.raises(ValidationError):
        UsageReceipt(request_id="request-1", cost_status="unknown", cost=0)
