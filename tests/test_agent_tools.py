"""The real executor reserves before dispatch and never retries unknown effects."""

import pytest

from kd1_anime.agent_runtime.contracts import (
    InspectCandidate,
    ProposePlan,
    RuntimeRecord,
    SubmitRender,
)
from kd1_anime.agent_runtime.policy import PolicyDecision, ToolPolicySpec, required_permission
from kd1_anime.agent_runtime.tools import (
    ExecutionAdapters,
    ToolDefinition,
    ToolExecutor,
    ToolRegistry,
    ToolResult,
    action_signature,
)
from kd1_anime.commits import RunPatch
from kd1_anime.domain.dependencies import WorkIdentity
from kd1_anime.services.results import StageResult

RUN = "20261019-120000-1234abcd"
HASH = "a" * 64
IDENTITY = WorkIdentity(1, HASH)


class Output(RuntimeRecord):
    message: str


def action():
    return ProposePlan(
        run_id=RUN, resource_id=f"run:{RUN}", input_sha256=HASH, work_identity=IDENTITY
    )


def fixture(
    execute,
    *,
    side_effect="external",
    attempts=1,
    crash=None,
    domain_action=None,
    requires_approval=False,
):
    calls = []
    intents = {}
    selected = domain_action or action()

    def save(name, value):
        calls.append((name, value))
        if crash == name:
            raise OSError("disk unavailable")
        if name == "intent":
            intents[(value.session_id, action_signature(value.action))] = value

    adapters = ExecutionAdapters(
        latest_intent=lambda action, session_id: intents.get(
            (session_id, action_signature(action))
        ),
        persist_intent=lambda value: save("intent", value),
        consume_grant=lambda grant, intent: save("grant", intent),
        persist_invocation=lambda value: save("invocation", value),
        commit=lambda intent, invocation, result: (save("commit", result), 2)[1],
        emit_committed=lambda intent, invocation, revision: save("event", revision),
    )
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            policy=ToolPolicySpec(
                action=selected.kind,
                permission=required_permission(selected),
                side_effect=side_effect,
                max_attempts=attempts,
                requires_approval=requires_approval,
            ),
            input_model=type(selected),
            output_model=Output,
            execute=execute,
        )
    )
    return ToolExecutor(registry, adapters), calls


def result():
    return ToolResult(StageResult(IDENTITY, Output(message="complete")), RunPatch(state="PLANNING"))


def test_intent_grant_dispatch_external_identity_commit_event_order():
    def execute(action, context):
        context.checkpoint()
        context.record_external_id("12345")
        return result()

    executor, calls = fixture(execute)
    outcome = executor.execute(
        action(), "session-one", PolicyDecision(effect="allow", reason="approved", grant_id="grant")
    )
    names = [name for name, _ in calls]
    assert names[:3] == ["intent", "grant", "invocation"]
    external = next(
        value for name, value in calls if name == "invocation" and value.external_job_id
    )
    assert external.external_job_id == "12345"
    assert names.index("commit") < names.index("event")
    assert outcome.status == "completed"
    assert calls[0][1].status == "reserved"


@pytest.mark.parametrize("point", ["intent", "grant", "invocation"])
def test_reservation_crashes_never_dispatch(point):
    dispatched = []
    executor, calls = fixture(lambda *args: dispatched.append(True), crash=point)
    with pytest.raises(OSError):
        executor.execute(
            action(),
            "session-one",
            PolicyDecision(effect="allow", reason="approved", grant_id="grant"),
        )
    assert dispatched == []
    assert not any(name == "commit" for name, _ in calls)


def test_unknown_external_effect_is_not_retried_or_committed():
    dispatched = []

    def execute(action, context):
        dispatched.append(True)
        context.record_external_id("42")
        raise TimeoutError("secret exception must not be persisted")

    executor, calls = fixture(execute)
    outcome = executor.execute(
        action(), "session-one", PolicyDecision(effect="allow", reason="allowed")
    )
    assert dispatched == [True]
    assert outcome.status == "unknown"
    assert calls[-1][1].status == "unknown"
    assert calls[-1][1].external_job_id == "42"
    assert not any(name in {"commit", "event"} for name, _ in calls)
    assert "secret exception" not in repr(calls)


def test_queries_retry_only_declared_transient_failures():
    attempts = []

    def execute(action, context):
        attempts.append(True)
        if len(attempts) == 1:
            raise ConnectionError("temporary")
        return result()

    query = InspectCandidate(
        run_id=RUN,
        resource_id=f"run:{RUN}/scene:1",
        scene_id=1,
        input_sha256=HASH,
        work_identity=IDENTITY,
        candidate={"path": "scenes/1.py", "sha256": HASH, "kind": "code"},
    )
    executor, calls = fixture(execute, side_effect="none", attempts=2, domain_action=query)
    outcome = executor.execute(
        query, "session-one", PolicyDecision(effect="allow", reason="allowed")
    )
    assert outcome.status == "completed"
    assert len(attempts) == 2
    assert [
        value.attempt for name, value in calls if name == "invocation" and value.status == "running"
    ] == [1, 2]


def test_cancellation_after_dispatch_keeps_unknown_not_fake_stopped_thread():
    cancelled = []

    def execute(action, context):
        cancelled.append(True)
        context.checkpoint()
        return result()

    executor, calls = fixture(execute)
    outcome = executor.execute(
        action(),
        "session-one",
        PolicyDecision(effect="allow", reason="allowed"),
        cancelled=lambda: bool(cancelled),
    )
    assert outcome.status == "unknown"
    assert len(cancelled) == 1
    assert not any(name == "commit" for name, _ in calls)


def test_commit_failure_preserves_unknown_intent_and_never_emits_success():
    executor, calls = fixture(lambda action, context: result(), crash="commit")
    with pytest.raises(OSError):
        executor.execute(action(), "session-one", PolicyDecision(effect="allow", reason="allowed"))
    assert not any(name == "event" for name, _ in calls)
    assert calls[-1][0] == "intent"
    assert calls[-1][1].status == "unknown"


def test_tool_declarations_reject_retrying_external_side_effects():
    with pytest.raises(ValueError):
        fixture(lambda action, context: result(), attempts=2)


def test_completed_or_unknown_intent_never_redispatches_or_reconsumes_grant():
    for fail in (False, True):
        dispatched = []

        def execute(action, context, *, dispatched=dispatched, fail=fail):
            dispatched.append(True)
            if fail:
                raise ConnectionError("unknown network outcome")
            return result()

        executor, calls = fixture(execute)
        decision = PolicyDecision(effect="allow", reason="approved", grant_id="grant")
        first = executor.execute(action(), "session-one", decision)
        before = len(calls)
        second = executor.execute(action(), "session-one", decision)
        assert len(dispatched) == 1
        assert len(calls) == before
        assert second.intent.intent_id == first.intent.intent_id
        assert second.status == ("pending" if fail else "completed")
        assert len([name for name, _ in calls if name == "grant"]) == 1


def test_cooperative_deadline_never_commits_or_retries_external_work():
    def execute(action, context):
        context.record_external_id("123")
        context.deadline = 0
        context.checkpoint()
        return result()

    executor, calls = fixture(execute)
    outcome = executor.execute(
        action(), "session-one", PolicyDecision(effect="allow", reason="allowed")
    )
    assert outcome.status == "unknown"
    assert outcome.invocation.external_job_id == "123"
    assert not any(name == "commit" for name, _ in calls)


@pytest.mark.parametrize("name", ["submit_render", "cancel_render", "merge_video"])
def test_external_action_cannot_claim_to_be_a_retryable_query(name):
    with pytest.raises(ValueError):
        ToolPolicySpec(action=name, permission="exec", side_effect="none", max_attempts=2)
    with pytest.raises(ValueError):
        ToolPolicySpec(action=name, permission="exec", side_effect="none")


def test_constructed_output_is_sanitized_before_commit_not_merely_checked():
    raw = Output.model_construct(message="<think>private secret reasoning</think>visible")

    def execute(action, context):
        return ToolResult(StageResult(IDENTITY, raw), RunPatch(state="PLANNING"))

    executor, calls = fixture(execute)
    outcome = executor.execute(
        action(), "session-one", PolicyDecision(effect="allow", reason="allowed")
    )
    committed = next(value for name, value in calls if name == "commit")
    assert "private secret reasoning" not in committed.stage.value.message
    assert committed.stage.value.message.endswith("visible")
    assert outcome.result.stage.value is not raw
    assert "private secret reasoning" in raw.message


def test_missing_notification_keeps_committed_success_without_redispatch():
    dispatched = []

    def execute(action, context):
        dispatched.append(True)
        return result()

    executor, calls = fixture(execute, crash="event")
    decision = PolicyDecision(effect="allow", reason="allowed")
    first = executor.execute(action(), "session-one", decision)
    assert first.status == "completed"
    assert first.reason == "notification_missing"
    assert first.intent.status == "completed"
    second = executor.execute(action(), "session-one", decision)
    assert second.status == "completed"
    assert dispatched == [True]
    assert len([name for name, _ in calls if name == "commit"]) == 1
    assert len([name for name, _ in calls if name == "event"]) == 1


def test_required_grant_cannot_be_omitted_by_a_bare_allow_decision():
    executor, calls = fixture(lambda action, context: result(), requires_approval=True)
    with pytest.raises(PermissionError):
        executor.execute(action(), "session-one", PolicyDecision(effect="allow", reason="allowed"))
    assert calls == []


@pytest.mark.parametrize("record_id", [True, False])
def test_submission_cannot_commit_before_its_external_id_is_durable(record_id):
    submission = SubmitRender(
        run_id=RUN,
        resource_id=f"run:{RUN}/scene:1",
        scene_id=1,
        input_sha256=HASH,
        work_identity=IDENTITY,
        backend="slurm",
        class_name="Scene",
        source={"path": "scenes/1.py", "sha256": HASH, "kind": "code"},
    )

    def execute(action, context):
        if record_id:
            context.record_external_id("123")
        return result()

    executor, calls = fixture(execute, domain_action=submission)
    outcome = executor.execute(
        submission, "session-one", PolicyDecision(effect="allow", reason="allowed")
    )
    assert outcome.status == ("completed" if record_id else "unknown")
    if record_id:
        names = [name for name, _ in calls]
        saved_id_index = next(
            index
            for index, (name, value) in enumerate(calls)
            if name == "invocation" and value.external_job_id == "123"
        )
        assert saved_id_index < names.index("commit")
    else:
        assert not any(name in {"commit", "event"} for name, _ in calls)
