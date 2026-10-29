"""Exercise v3 JSONL ordering, recovery and privacy without external services."""

import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from kd1_anime.agent_runtime.contracts import EventPayload
from kd1_anime.agent_runtime.state import AgentRunState
from kd1_anime.event_store import EventStore, EventStoreError, LegacyEvent
from kd1_anime.prompt_identity import capture_generation_identity
from kd1_anime.run_store import RunManifest, lock_run, write_manifest

RUN_ID = "20261027-120000-1234abcd"


def durable_run(tmp_path):
    root = tmp_path / "runs" / RUN_ID
    root.mkdir(parents=True)
    manifest = RunManifest(
        run_id=RUN_ID,
        user_prompt="private goal",
        output_path="out.mp4",
        agent=AgentRunState(
            session_id="session-one",
            generation_identity=capture_generation_identity(),
            run_input_sha256="a" * 64,
        ),
    )
    write_manifest(root / "manifest.json", manifest)
    return root, manifest


def append(store, **changes):
    fields = dict(
        name="diagnostic",
        durability="committed",
        manifest_revision=0,
        payload=EventPayload(message="visible"),
    )
    fields.update(changes)
    return store.append(**fields)


def test_racing_store_instances_allocate_one_ordered_stream(tmp_path):
    root, _ = durable_run(tmp_path)
    barrier = Barrier(8)

    def publish(index):
        barrier.wait()
        return append(EventStore(root, RUN_ID), event_id=f"event-{index}")

    with ThreadPoolExecutor(max_workers=8) as workers:
        events = list(workers.map(publish, range(8)))
    store = EventStore(root, RUN_ID)
    assert sorted(event.sequence for event in events) == list(range(1, 9))
    assert [event.sequence for event in store.replay()] == list(range(1, 9))
    assert [event.sequence for event in store.replay(after=5)] == [6, 7, 8]
    assert store.high_watermark == 8
    assert root.stat().st_mode & 0o777 == 0o700
    assert (root / "events.jsonl").stat().st_mode & 0o777 == 0o600


def test_ephemeral_delivery_consumes_sequence_but_not_disk_or_replay(tmp_path):
    root, _ = durable_run(tmp_path)
    store = EventStore(root, RUN_ID)
    first = append(store)
    transient = append(
        store,
        name="stream_chunk",
        durability="ephemeral",
        payload=EventPayload(message="not recoverable"),
    )
    final = append(EventStore(root, RUN_ID))
    assert [first.sequence, transient.sequence, final.sequence] == [1, 2, 3]
    assert [event.sequence for event in store.replay()] == [1, 3]
    assert "not recoverable" not in (root / "events.jsonl").read_text()
    assert store.high_watermark == 3


def test_only_ephemeral_stream_creates_no_event_log(tmp_path):
    root, _ = durable_run(tmp_path)
    store = EventStore(root, RUN_ID)
    event = append(store, name="progress", durability="ephemeral")
    assert event.sequence == 1
    assert store.replay() == ()
    assert not (root / "events.jsonl").exists()


def test_trailing_interruption_is_ignored_and_removed_before_next_append(tmp_path):
    root, _ = durable_run(tmp_path)
    store = EventStore(root, RUN_ID)
    first = append(store)
    path = root / "events.jsonl"
    with path.open("ab") as handle:
        handle.write(b'{"schema_version":3,"sequence":')
    assert store.replay() == (first,)
    second = append(store)
    assert store.replay() == (first, second)
    assert len(path.read_text().splitlines()) == 2
    for line in path.read_text().splitlines():
        assert json.loads(line)["schema_version"] == 3


@pytest.mark.parametrize(
    "damage",
    [
        "invalid",
        "duplicate_id",
        "duplicate_sequence",
        "foreign_run",
        "foreign_stream",
        "future_revision",
        "ephemeral_on_disk",
    ],
)
def test_middle_corruption_and_identity_conflicts_fail_closed(tmp_path, damage):
    root, _ = durable_run(tmp_path)
    store = EventStore(root, RUN_ID)
    first = append(store)
    second = append(store)
    raw = second.model_dump(mode="json")
    if damage == "invalid":
        damaged = "not-json"
    else:
        if damage == "duplicate_id":
            raw["event_id"] = first.event_id
        elif damage == "duplicate_sequence":
            raw["sequence"] = first.sequence
        elif damage == "foreign_run":
            raw["run_id"] = "20261027-120000-ffffffff"
        elif damage == "foreign_stream":
            raw["stream_id"] = "other-stream"
        elif damage == "future_revision":
            raw["manifest_revision"] = 99
        elif damage == "ephemeral_on_disk":
            raw["durability"] = "ephemeral"
        damaged = json.dumps(raw)
    path = root / "events.jsonl"
    path.write_text(
        first.model_dump_json() + "\n" + damaged + "\n" + second.model_dump_json() + "\n"
    )
    before = path.read_bytes()
    with pytest.raises(EventStoreError):
        store.replay()
    with pytest.raises(EventStoreError):
        append(store)
    assert path.read_bytes() == before


def test_duplicate_append_and_uncommitted_revision_are_rejected(tmp_path):
    root, manifest = durable_run(tmp_path)
    store = EventStore(root, RUN_ID)
    append(store, event_id="once")
    with pytest.raises(EventStoreError):
        append(store, event_id="once")
    with pytest.raises(EventStoreError):
        append(store, manifest_revision=1)
    manifest.revision = 1
    write_manifest(root / "manifest.json", manifest)
    assert append(store, manifest_revision=1).manifest_revision == 1


def test_reconcile_recovers_missing_notification_and_is_idempotent(tmp_path):
    root, manifest = durable_run(tmp_path)
    manifest.revision = 1
    manifest.state = "PLANNING"
    write_manifest(root / "manifest.json", manifest)
    store = EventStore(root, RUN_ID)
    recovered = store.reconcile(manifest)
    assert recovered.name == "snapshot_committed"
    assert recovered.manifest_revision == 1
    assert recovered.payload.state == "PLANNING"
    assert store.reconcile(manifest) == recovered
    assert EventStore(root, RUN_ID).reconcile(manifest) == recovered
    assert len(store.replay()) == 1
    assert "private goal" not in (root / "events.jsonl").read_text()
    manifest.revision = 2
    with pytest.raises(EventStoreError):
        store.reconcile(manifest)


def test_caller_owned_run_lock_does_not_reacquire_flock(tmp_path):
    root, manifest = durable_run(tmp_path)
    store = EventStore(root, RUN_ID)
    with lock_run(root):
        event = store.reconcile(manifest, lock_held=True)
        assert store.replay(lock_held=True) == (event,)
        assert store.watermark(lock_held=True) == event.sequence


def test_configured_secrets_and_hidden_thought_are_not_persisted(tmp_path):
    root, _ = durable_run(tmp_path)
    store = EventStore(root, RUN_ID, secrets=("private-credential",))
    event = append(
        store,
        payload=EventPayload(message="private-credential <think>private thought</think> visible"),
    )
    text = (root / "events.jsonl").read_text()
    assert "private-credential" not in text
    assert "private thought" not in text
    assert "visible" in event.payload.message
    with pytest.raises(ValueError):
        store.append(
            name="diagnostic",
            durability="committed",
            manifest_revision=0,
            payload={"metadata": {"reasoning_content": "secret"}},
        )
    with pytest.raises(ValueError):
        store.append(
            name="diagnostic",
            durability="committed",
            manifest_revision=0,
            payload={"metadata": {"error": RuntimeError("private-credential")}},
        )


def test_v2_is_read_only_and_never_rewritten(tmp_path):
    root, _ = durable_run(tmp_path)
    path = root / "events.jsonl"
    legacy = dict(
        schema_version=2,
        manifest_revision=0,
        scene_id=None,
        scene_revision=None,
        event_id="old",
        timestamp="2026-10-27T12:00:00+00:00",
        run_id=RUN_ID,
        event="checkpoint",
        data={"message": "old notification"},
    )
    path.write_text(json.dumps(legacy) + "\n")
    before = path.read_bytes()
    store = EventStore(root, RUN_ID)
    assert isinstance(store.replay()[0], LegacyEvent)
    with pytest.raises(EventStoreError):
        append(store)
    assert path.read_bytes() == before


def test_symlinked_event_log_is_never_followed(tmp_path):
    root, _ = durable_run(tmp_path)
    outside = tmp_path / "outside"
    outside.write_text("do not touch")
    (root / "events.jsonl").symlink_to(outside)
    with pytest.raises((EventStoreError, ValueError)):
        append(EventStore(root, RUN_ID))
    assert outside.read_text() == "do not touch"


def test_ephemeral_sequence_reservation_survives_a_fresh_store(tmp_path):
    root, _ = durable_run(tmp_path)
    first = append(EventStore(root, RUN_ID), durability="ephemeral", name="progress")
    reservation = json.loads((root / ".events.sequence.json").read_text())
    assert reservation["high_watermark"] == first.sequence
    assert (root / ".events.sequence.json").stat().st_mode & 0o777 == 0o600
    recovered = EventStore(root, RUN_ID)
    assert recovered.high_watermark == first.sequence
    assert recovered.replay() == ()
    assert append(recovered).sequence == first.sequence + 1


@pytest.mark.parametrize(
    "damage", ["invalid", "empty", "missing", "foreign", "backward", "boolean"]
)
def test_corrupt_sequence_reservation_never_reuses_an_allocation(tmp_path, damage):
    root, _ = durable_run(tmp_path)
    store = EventStore(root, RUN_ID)
    append(store)
    path = root / ".events.sequence.json"
    if damage == "invalid":
        path.write_text("not-json")
    elif damage == "empty":
        path.write_text("")
    elif damage == "missing":
        path.unlink()
    else:
        data = json.loads(path.read_text())
        if damage == "foreign":
            data["run_id"] = "20261027-120000-ffffffff"
        elif damage == "backward":
            data["high_watermark"] = 0
        elif damage == "boolean":
            data["high_watermark"] = True
        path.write_text(json.dumps(data))
    before = (root / "events.jsonl").read_bytes()
    with pytest.raises(EventStoreError):
        append(store)
    with pytest.raises(EventStoreError):
        store.replay()
    assert (root / "events.jsonl").read_bytes() == before


def test_failed_log_append_leaves_a_gap_not_a_reused_sequence(monkeypatch, tmp_path):
    import kd1_anime.event_store as module

    root, _ = durable_run(tmp_path)
    store = EventStore(root, RUN_ID)
    real_open = module.os.open

    def interrupted_open(path, flags, *args, **kwargs):
        if path == root / "events.jsonl" and flags & module.os.O_CREAT:
            raise OSError("interrupted log publication")
        return real_open(path, flags, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(module.os, "open", interrupted_open)
        with pytest.raises(EventStoreError):
            append(store)
    assert store.high_watermark == 1
    assert store.replay() == ()
    assert append(store).sequence == 2


def test_v2_read_does_not_create_sequence_reservation(tmp_path):
    root, _ = durable_run(tmp_path)
    path = root / "events.jsonl"
    legacy = dict(
        schema_version=2,
        manifest_revision=0,
        event_id="old",
        timestamp="2026-10-27T12:00:00+00:00",
        run_id=RUN_ID,
        event="checkpoint",
        data={},
    )
    path.write_text(json.dumps(legacy) + "\n")
    store = EventStore(root, RUN_ID)
    assert len(store.replay()) == 1
    assert store.high_watermark == 0
    assert not (root / ".events.sequence.json").exists()


@pytest.mark.parametrize("tail", ['{"schema_version":2,', '{"schema_version":3,'])
def test_unidentified_first_interrupted_record_is_not_silently_replaced(tmp_path, tail):
    root, _ = durable_run(tmp_path)
    path = root / "events.jsonl"
    path.write_text(tail)
    store = EventStore(root, RUN_ID)
    assert store.replay() == ()
    with pytest.raises(EventStoreError):
        append(store)
    assert path.read_text() == tail
    assert not (root / ".events.sequence.json").exists()


def test_first_v3_interrupted_record_recovers_from_durable_reservation(tmp_path):
    root, _ = durable_run(tmp_path)
    store = EventStore(root, RUN_ID)
    append(store, name="progress", durability="ephemeral")
    (root / "events.jsonl").write_text('{"schema_version":3,')
    event = append(store)
    assert event.sequence == 2
    assert store.replay() == (event,)


def test_nonexistent_scene_revision_is_not_a_committed_fact(tmp_path):
    root, _ = durable_run(tmp_path)
    with pytest.raises(EventStoreError):
        append(EventStore(root, RUN_ID), scene_revision=0, payload=EventPayload(scene_id=1))


def test_snapshot_notification_identity_conflict_is_explicit(tmp_path):
    root, manifest = durable_run(tmp_path)
    store = EventStore(root, RUN_ID)
    append(store, name="snapshot_committed", event_id="wrong-snapshot-identity")
    with pytest.raises(EventStoreError):
        store.reconcile(manifest)
