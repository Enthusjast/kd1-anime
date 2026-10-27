import json
import multiprocessing
import os
import stat
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

import kd1_anime.agent_runtime.session as session_module
from kd1_anime.agent_runtime.contracts import (
    AgentMessage,
    ExecutionIntent,
    PermissionScope,
    ProposePlan,
    ReviewPlan,
)
from kd1_anime.agent_runtime.session import Session, SessionStore, UserCommand
from kd1_anime.domain.dependencies import WorkIdentity

RUN = "20261027-120000-1234abcd"
OTHER_RUN = "20261027-120000-1234abce"
NOW = datetime(2026, 10, 27, 12, tzinfo=timezone.utc)


def action(**changes):
    return ReviewPlan(
        **(
            dict(
                run_id=RUN,
                resource_id=f"run:{RUN}",
                input_sha256="a" * 64,
                work_identity=WorkIdentity(3, "b" * 64),
            )
            | changes
        ),
    )


def session_store(tmp_path, **kwargs):
    store = SessionStore(tmp_path / "sessions", **kwargs)
    session = store.create("Explain a circle", session_id="lesson", run_ids=(RUN, OTHER_RUN))
    return store, session


def approve(store, session, target=None, **kwargs):
    command = UserCommand(session_id=session.session_id, kind="approve", action=target or action())
    return store.issue_grant(session.session_id, command, **kwargs)


def test_user_commands_and_messages_are_durable_queryable_and_immutable(tmp_path):
    store, session = session_store(tmp_path)
    for kind, content in (
        ("goal", "A new goal"),
        ("clarification", "Use geometry"),
        ("ready", "Proceed"),
        ("reject", "Not that plan"),
        ("cancel", "Stop"),
    ):
        store.command(
            session.session_id,
            UserCommand(
                session_id=session.session_id,
                kind=kind,
                content=content,
                action=action() if kind == "reject" else None,
            ),
        )
    store.append_message(
        session.session_id,
        AgentMessage(
            session_id=session.session_id,
            role="assistant",
            kind="action",
            action=action(),
            content="Which radius?",
        ),
    )
    loaded = SessionStore(store.root).load(session.session_id)
    assert loaded.goal == "A new goal"
    assert [command.kind for command in loaded.commands] == [
        "goal",
        "goal",
        "clarification",
        "ready",
        "reject",
        "cancel",
    ]
    assert loaded.messages[-1].content == "Which radius?"
    assert loaded.revision == session.revision + 6
    with pytest.raises(ValidationError):
        loaded.goal = "mutable"
    with pytest.raises(ValidationError):
        UserCommand(session_id="lesson", kind="approve", role="assistant", action=action())
    with pytest.raises(ValidationError):
        UserCommand(session_id="lesson", kind="approve")
    with pytest.raises(ValidationError):
        UserCommand(session_id="lesson", kind="shell", content="echo unsafe")
    with pytest.raises(ValidationError):
        Session(**loaded.model_dump(), unknown="untrusted")


def test_disk_privacy_permissions_and_independent_sessions(tmp_path, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *args: pytest.fail("Session must not read stdin"))
    store, session = session_store(tmp_path, secrets=("configured-secret-value",))
    store.command(
        session.session_id,
        UserCommand(
            session_id="lesson",
            kind="clarification",
            content="Visible configured-secret-value <think>private chain</think> Bearer abc-token",
        ),
    )
    other = store.create("Another goal", session_id="other")
    disk = (store.root / "lesson" / "session.json").read_text()
    assert "configured-secret-value" not in disk
    assert "private chain" not in disk
    assert "abc-token" not in disk
    assert "Visible" in disk
    assert store.load(other.session_id).goal == "Another goal"
    for path, mode in (
        (store.root, 0o700),
        (store.root / "lesson", 0o700),
        (store.root / "lesson" / "session.json", 0o600),
        (store.root / "lesson" / ".session.lock", 0o600),
    ):
        assert path.stat().st_mode & 0o777 == mode
    with pytest.raises(ValidationError):
        UserCommand(session_id="lesson", kind="clarification", reasoning="hidden")
    with pytest.raises(ValidationError):
        UserCommand(session_id="lesson", kind="clarification", content=RuntimeError("secret"))


def test_parallel_updates_and_reads_never_lose_messages_or_publish_partial_json(tmp_path):
    store, session = session_store(tmp_path)

    def update(index):
        independent = SessionStore(store.root)
        independent.append_message(
            session.session_id,
            AgentMessage(
                message_id=f"message-{index}",
                session_id="lesson",
                role="user",
                kind="clarification",
                content=f"message {index}",
            ),
        )
        return json.loads((store.root / "lesson" / "session.json").read_text())["session_id"]

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert set(pool.map(update, range(48))) == {"lesson"}
    loaded = store.load("lesson")
    assert len(loaded.messages) == 49
    assert len({message.message_id for message in loaded.messages}) == 49
    assert loaded.revision == session.revision + 48


def test_interrupted_replace_preserves_previous_snapshot_and_removes_tempfiles(
    tmp_path, monkeypatch
):
    store, _ = session_store(tmp_path)
    before = (store.root / "lesson" / "session.json").read_bytes()

    def fail_replace(*args, **kwargs):
        raise OSError("interrupted publication")

    monkeypatch.setattr(session_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="interrupted publication"):
        store.link_run("lesson", "20261027-120000-1234abcf")
    assert (store.root / "lesson" / "session.json").read_bytes() == before
    assert {path.name for path in (store.root / "lesson").iterdir()} == {
        "session.json",
        ".session.lock",
    }


@pytest.mark.parametrize("part", ["root", "session", "snapshot", "lock", "ancestor"])
def test_symlinks_are_rejected_without_touching_targets(tmp_path, part):
    store, _ = session_store(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "target"
    target.write_text("untouched")
    selected_store = store
    selected_id = "lesson"
    if part == "root":
        root_link = tmp_path / "linked-root"
        root_link.symlink_to(store.root, target_is_directory=True)
        selected_store = SessionStore(root_link)
    elif part == "ancestor":
        ancestor = tmp_path / "linked-parent"
        ancestor.symlink_to(tmp_path, target_is_directory=True)
        selected_store = SessionStore(ancestor / "sessions")
    elif part == "session":
        (store.root / "linked-session").symlink_to(outside, target_is_directory=True)
        selected_id = "linked-session"
    else:
        entry = store.root / "lesson" / ("session.json" if part == "snapshot" else ".session.lock")
        entry.unlink()
        entry.symlink_to(target)
    with pytest.raises((ValueError, OSError)):
        selected_store.load(selected_id)
    assert target.read_text() == "untouched"


def test_hardlinks_and_traversal_are_rejected(tmp_path):
    store, _ = session_store(tmp_path)
    for session_id in ("..", "../outside", "/tmp/outside", "a/b", "a\\b", "."):
        with pytest.raises(ValueError):
            store.load(session_id)
    os.link(store.root / "lesson" / "session.json", tmp_path / "shared.json")
    with pytest.raises(ValueError, match="private"):
        store.load("lesson")


def test_grant_exact_binding_consumes_once_and_replays_same_intent(tmp_path):
    store, session = session_store(tmp_path)
    grant = approve(store, session)
    intent = ExecutionIntent(session_id="lesson", action=action())
    consumed = store.consume_grant("lesson", grant.grant_id, intent)
    assert consumed.consumed_by_intent == intent.intent_id
    revision = store.load("lesson").revision
    assert SessionStore(store.root).consume_grant("lesson", grant.grant_id, intent) == consumed
    assert store.load("lesson").revision == revision
    with pytest.raises(ValueError, match="consumed"):
        store.consume_grant(
            "lesson", grant.grant_id, ExecutionIntent(session_id="lesson", action=action())
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"input_sha256": "c" * 64},
        {"run_id": OTHER_RUN, "resource_id": f"run:{OTHER_RUN}"},
        {"resource_id": f"run:{RUN}/scene:1", "scene_id": 1},
        {"work_identity": WorkIdentity(4, "b" * 64)},
        {"work_identity": WorkIdentity(3, "c" * 64)},
        {"work_identity": None},
    ],
)
def test_grant_rejects_changed_input_resource_run_or_work_revision(tmp_path, changes):
    store, session = session_store(tmp_path)
    grant = approve(store, session)
    intent = ExecutionIntent(session_id="lesson", action=action(**changes))
    with pytest.raises(ValueError, match="binding"):
        store.consume_grant("lesson", grant.grant_id, intent)
    assert store.load("lesson").grants[0].consumed_by_intent is None


def test_expired_grant_and_cross_session_intent_cannot_be_used(tmp_path):
    store, _ = session_store(tmp_path)
    command = UserCommand(session_id="lesson", kind="approve", action=action(), created_at=NOW)
    grant = store.issue_grant("lesson", command, expires_at=NOW + timedelta(minutes=1))
    with pytest.raises(ValueError, match="expired"):
        store.consume_grant(
            "lesson",
            grant.grant_id,
            ExecutionIntent(session_id="lesson", action=action()),
            now=NOW + timedelta(minutes=2),
        )
    with pytest.raises(ValueError, match="binding"):
        store.consume_grant(
            "lesson",
            grant.grant_id,
            ExecutionIntent(session_id="different", action=action()),
            now=NOW,
        )


def test_grant_issue_requires_approve_command_and_valid_resource_and_linked_run(tmp_path):
    store, session = session_store(tmp_path)
    with pytest.raises(ValueError, match="approve"):
        store.issue_grant("lesson", UserCommand(session_id="lesson", kind="ready"))
    with pytest.raises(ValueError, match="resource"):
        approve(store, session, action(resource_id=f"run:{RUN}/scene:2"))
    with pytest.raises(ValueError, match="linked"):
        approve(
            store,
            session,
            action(run_id="20261027-120000-1234abcf", resource_id="run:20261027-120000-1234abcf"),
        )
    with pytest.raises(ValueError, match="scope"):
        store.create("Goal", permissions=(PermissionScope(permission="write", resource_id="*"),))
    with pytest.raises(ValueError, match="approve"):
        store.command("lesson", UserCommand(session_id="lesson", kind="approve", action=action()))


@pytest.mark.parametrize("kind", ["reject", "cancel", "goal", "clarification"])
def test_user_rejection_cancellation_or_changed_input_invalidates_grant(tmp_path, kind):
    store, session = session_store(tmp_path)
    grant = approve(store, session)
    store.command(
        "lesson",
        UserCommand(
            session_id="lesson",
            kind=kind,
            content="Changed instruction",
            action=action() if kind == "reject" else None,
        ),
    )
    with pytest.raises(ValueError, match="superseded"):
        store.consume_grant(
            "lesson", grant.grant_id, ExecutionIntent(session_id="lesson", action=action())
        )


def test_concurrent_grant_consumers_have_one_winner(tmp_path):
    store, session = session_store(tmp_path)
    grant = approve(store, session)

    def consume(_):
        intent = ExecutionIntent(session_id="lesson", action=action())
        try:
            SessionStore(store.root).consume_grant("lesson", grant.grant_id, intent)
        except ValueError:
            return None
        return intent.intent_id

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(consume, range(16)))
    winners = [result for result in results if result is not None]
    assert len(winners) == 1
    assert store.load("lesson").grants[0].consumed_by_intent == winners[0]


def test_repeated_command_and_message_are_idempotent_but_conflicts_fail(tmp_path):
    store, _ = session_store(tmp_path)
    command = UserCommand(session_id="lesson", kind="ready", command_id="ready-command")
    first = store.command("lesson", command)
    assert store.command("lesson", command) == first
    with pytest.raises(ValueError, match="conflict"):
        store.command("lesson", command.model_copy(update={"content": "different"}))
    message = AgentMessage(
        message_id="same-message",
        session_id="lesson",
        role="user",
        kind="clarification",
        content="Visible",
    )
    first = store.append_message("lesson", message)
    assert store.append_message("lesson", message) == first
    with pytest.raises(ValueError, match="conflict"):
        store.append_message("lesson", message.model_copy(update={"content": "different"}))


def _append_from_process(root, index):
    SessionStore(root).append_message(
        "lesson",
        AgentMessage(
            message_id=f"process-message-{index}",
            session_id="lesson",
            role="user",
            kind="clarification",
            content=f"process {index}",
        ),
    )


def test_independent_processes_serialize_session_updates(tmp_path):
    store, session = session_store(tmp_path)
    with ProcessPoolExecutor(
        max_workers=3, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = [pool.submit(_append_from_process, store.root, index) for index in range(12)]
        for future in futures:
            future.result(timeout=30)
    loaded = store.load("lesson")
    assert loaded.revision == session.revision + 12
    assert len(loaded.messages) == 13


def test_killed_writer_artifacts_recover_without_reusing_partial_snapshot(tmp_path):
    store, session = session_store(tmp_path)
    directory = store.root / "lesson"
    os.link(directory / "session.json", directory / (".session-" + "a" * 32 + ".previous"))
    (directory / (".session-" + "b" * 32 + ".tmp")).write_text('{"partial":')
    assert SessionStore(store.root).load("lesson") == session
    assert {path.name for path in directory.iterdir()} == {"session.json", ".session.lock"}


def test_directory_sync_failure_restores_previous_committed_snapshot(tmp_path, monkeypatch):
    store, session = session_store(tmp_path)
    before = (store.root / "lesson" / "session.json").read_bytes()
    original_fsync = session_module.os.fsync
    failed = False

    def fail_directory_sync_once(descriptor):
        nonlocal failed
        if stat.S_ISDIR(os.fstat(descriptor).st_mode) and not failed:
            failed = True
            raise OSError("directory sync interrupted")
        return original_fsync(descriptor)

    monkeypatch.setattr(session_module.os, "fsync", fail_directory_sync_once)
    with pytest.raises(OSError, match="directory sync interrupted"):
        store.link_run("lesson", "20261027-120000-1234abcf")
    assert (store.root / "lesson" / "session.json").read_bytes() == before
    assert store.load("lesson") == session


def test_approval_action_kind_and_forged_model_role_are_rejected(tmp_path):
    store, session = session_store(tmp_path)
    grant = approve(store, session)
    different_action = ProposePlan(**action().model_dump(exclude={"kind"}))
    with pytest.raises(ValueError, match="binding"):
        store.consume_grant(
            "lesson", grant.grant_id, ExecutionIntent(session_id="lesson", action=different_action)
        )
    forged = UserCommand.model_construct(
        session_id="lesson", kind="approve", action=action(), role="assistant"
    )
    with pytest.raises(ValueError):
        store.issue_grant("lesson", forged)


def test_approval_command_retry_is_idempotent_and_replay_survives_expiration(tmp_path):
    store, _ = session_store(tmp_path)
    command = UserCommand(session_id="lesson", kind="approve", action=action(), created_at=NOW)
    expires_at = NOW + timedelta(minutes=1)
    grant = store.issue_grant("lesson", command, expires_at=expires_at)
    revision = store.load("lesson").revision
    assert store.issue_grant("lesson", command, expires_at=expires_at) == grant
    assert store.load("lesson").revision == revision
    intent = ExecutionIntent(session_id="lesson", action=action())
    consumed = store.consume_grant("lesson", grant.grant_id, intent, now=NOW)
    assert store.consume_grant("lesson", grant.grant_id, intent, now=expires_at) == consumed
