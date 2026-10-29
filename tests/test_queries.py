"""Queries expose frozen committed facts, not in-flight execution context."""

import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from kd1_anime.agent_runtime.contracts import EventPayload, ReviewPlan
from kd1_anime.agent_runtime.session import SessionStore, UserCommand
from kd1_anime.agent_runtime.state import AgentRunState
from kd1_anime.domain.dependencies import WorkIdentity
from kd1_anime.domain.planning import ScenePlan
from kd1_anime.event_store import EventStore
from kd1_anime.prompt_identity import capture_generation_identity
from kd1_anime.queries import QueryService
from kd1_anime.run_store import RunManifest, RunRepository, StoredSceneState, write_manifest

RUN_ID = "20261027-120000-1234abcd"


def saved_run(tmp_path):
    root = tmp_path / "runs" / RUN_ID
    root.mkdir(parents=True)
    plan = ScenePlan(
        scene_id=1,
        title="Committed title",
        duration_seconds=10,
        purpose="test",
        math_concept="circle",
        visual_design="dark",
        camera_movement="fixed",
        visual_flow=["show"],
        key_moments=["pause"],
        computation="radius=1",
        dependencies_complete=True,
    )
    manifest = RunManifest(
        schema_version=11,
        agent=AgentRunState(
            session_id="query-session",
            generation_identity=capture_generation_identity(),
            run_input_sha256="a" * 64,
        ),
        run_id=RUN_ID,
        user_prompt="private-credential visible goal",
        output_path="output.mp4",
        scenes={1: StoredSceneState(plan=plan)},
    )
    write_manifest(root / "manifest.json", manifest)
    return root, manifest


def test_run_and_scene_views_use_only_committed_manifest_and_replay_cursor(tmp_path):
    root, manifest = saved_run(tmp_path)
    store = EventStore(root, RUN_ID)
    notice = store.reconcile(manifest)
    queries = QueryService(RunRepository(tmp_path), secrets=("private-credential",))
    manifest.state = "PLANNING"
    manifest.scenes[1].plan.title = "Private in-flight title"
    view = queries.run(RUN_ID)
    assert view.state == "INIT"
    assert view.revision == 0
    assert view.high_watermark == notice.sequence
    assert view.scenes[0].title == "Committed title"
    assert queries.scenes(RUN_ID) == view.scenes
    assert "private-credential" not in view.model_dump_json()
    with pytest.raises(ValidationError):
        view.state = "FAILED"
    with pytest.raises(ValidationError):
        view.scenes[0].title = "mutated"
    assert store.replay(after=view.high_watermark) == ()
    write_manifest(root / "manifest.json", manifest)
    assert queries.run(RUN_ID).state == "PLANNING"
    assert view.state == "INIT"


def test_queries_reject_semantically_corrupt_committed_manifest(tmp_path):
    root, _ = saved_run(tmp_path)
    path = root / "manifest.json"
    raw = json.loads(path.read_text())
    raw["scenes"]["1"]["rendered"] = True
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        QueryService(RunRepository(tmp_path)).run(RUN_ID)


def test_session_view_is_loaded_from_independent_session_store(tmp_path):
    sessions = SessionStore(tmp_path / "sessions", secrets=("private-credential",))
    session = sessions.create("private-credential visible goal", run_ids=(RUN_ID,))
    queries = QueryService(RunRepository(tmp_path), sessions)
    view = queries.session(session.session_id)
    assert view.run_ids == (RUN_ID,)
    assert view.revision == session.revision
    assert "private-credential" not in view.model_dump_json()
    assert "visible goal" in view.goal
    with pytest.raises(ValidationError):
        view.goal = "changed"


def test_approval_view_matches_exact_committed_request_to_user_grant(tmp_path):
    root, manifest = saved_run(tmp_path)
    sessions = SessionStore(tmp_path / "sessions")
    session = sessions.create("goal", run_ids=(RUN_ID,))
    manifest.agent = manifest.agent.model_copy(update={"session_id": session.session_id})
    write_manifest(root / "manifest.json", manifest)
    identity = WorkIdentity(0, "a" * 64)
    action = ReviewPlan(
        run_id=RUN_ID,
        resource_id=f"run:{RUN_ID}",
        input_sha256=identity.input_sha256,
        work_identity=identity,
    )
    store = EventStore(root, RUN_ID)
    request = store.append(
        name="approval_requested",
        durability="committed",
        manifest_revision=0,
        payload=EventPayload(
            action="review_plan",
            metadata={
                "session_id": session.session_id,
                "resource_id": action.resource_id,
                "input_sha256": action.input_sha256,
                "work_identity": {"revision": 0, "input_sha256": "a" * 64},
            },
        ),
    )
    queries = QueryService(RunRepository(tmp_path), sessions)
    pending = queries.approvals(RUN_ID, session.session_id)
    assert len(pending) == 1
    assert pending[0].event_id == request.event_id
    assert pending[0].status == "pending"
    command = UserCommand(
        session_id=session.session_id,
        kind="approve",
        action=action,
        created_at=datetime.now(timezone.utc),
    )
    grant = sessions.issue_grant(session.session_id, command)
    approved = queries.approvals(RUN_ID, session.session_id)
    assert approved[0].status == "approved"
    assert approved[0].grant_id == grant.grant_id
    assert pending[0].status == "pending"


def test_approval_query_does_not_accept_grant_for_different_input(tmp_path):
    root, manifest = saved_run(tmp_path)
    sessions = SessionStore(tmp_path / "sessions")
    session = sessions.create("goal", run_ids=(RUN_ID,))
    manifest.agent = manifest.agent.model_copy(update={"session_id": session.session_id})
    write_manifest(root / "manifest.json", manifest)
    action = ReviewPlan(
        run_id=RUN_ID,
        resource_id=f"run:{RUN_ID}",
        input_sha256="b" * 64,
        work_identity=WorkIdentity(0, "b" * 64),
    )
    sessions.issue_grant(
        session.session_id,
        UserCommand(session_id=session.session_id, kind="approve", action=action),
    )
    EventStore(root, RUN_ID).append(
        name="approval_requested",
        durability="committed",
        manifest_revision=0,
        payload=EventPayload(
            action="review_plan",
            metadata={
                "session_id": session.session_id,
                "resource_id": action.resource_id,
                "input_sha256": "a" * 64,
                "work_identity": {"revision": 0, "input_sha256": "a" * 64},
            },
        ),
    )
    view = QueryService(RunRepository(tmp_path), sessions).approvals(RUN_ID, session.session_id)
    assert view[0].status == "pending"
    assert view[0].grant_id is None


def test_query_views_accept_an_existing_application_run_lock(tmp_path):
    from kd1_anime.run_store import lock_run

    root, _ = saved_run(tmp_path)
    queries = QueryService(RunRepository(tmp_path))
    with lock_run(root):
        view = queries.run(RUN_ID, lock_held=True)
        assert queries.scenes(RUN_ID, lock_held=True) == view.scenes
        assert queries.approvals(RUN_ID, lock_held=True) == ()


def test_snapshot_resync_cursor_skips_ephemeral_payloads_then_replays_new_facts(tmp_path):
    root, manifest = saved_run(tmp_path)
    store = EventStore(root, RUN_ID)
    store.reconcile(manifest)
    transient = store.append(
        name="progress",
        durability="ephemeral",
        manifest_revision=0,
        payload=EventPayload(message="in-flight progress"),
    )
    snapshot = QueryService(RunRepository(tmp_path)).run(RUN_ID)
    assert snapshot.high_watermark == transient.sequence
    assert "in-flight progress" not in snapshot.model_dump_json()
    committed = store.append(
        name="diagnostic",
        durability="committed",
        manifest_revision=0,
        payload=EventPayload(message="new committed notice"),
    )
    assert store.replay(after=snapshot.high_watermark) == (committed,)


def test_scene_approval_view_becomes_stale_after_new_committed_work_identity(tmp_path):
    from kd1_anime.agent_runtime.contracts import ReviewCode
    from kd1_anime.domain.dependencies import WorkIdentity

    root, manifest = saved_run(tmp_path)
    manifest.scenes[1].input_sha256 = "a" * 64
    write_manifest(root / "manifest.json", manifest)
    sessions = SessionStore(tmp_path / "sessions")
    session = sessions.create("goal", run_ids=(RUN_ID,))
    action = ReviewCode(
        run_id=RUN_ID,
        resource_id=f"run:{RUN_ID}/scene:1",
        scene_id=1,
        input_sha256="a" * 64,
        work_identity=WorkIdentity(0, "a" * 64),
    )
    sessions.issue_grant(
        session.session_id,
        UserCommand(session_id=session.session_id, kind="approve", action=action),
    )
    EventStore(root, RUN_ID).append(
        name="approval_requested",
        durability="committed",
        manifest_revision=0,
        scene_revision=0,
        payload=EventPayload(
            action="review_code",
            scene_id=1,
            metadata={
                "session_id": session.session_id,
                "resource_id": action.resource_id,
                "input_sha256": action.input_sha256,
                "work_identity": {"revision": 0, "input_sha256": "a" * 64},
            },
        ),
    )
    queries = QueryService(RunRepository(tmp_path), sessions)
    assert queries.approvals(RUN_ID, session.session_id)[0].status == "approved"
    manifest.revision = 1
    manifest.scenes[1].revision = 1
    write_manifest(root / "manifest.json", manifest)
    assert queries.approvals(RUN_ID, session.session_id)[0].status == "stale"
