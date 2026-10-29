"""Ordered notifications of committed facts, never an execution/recovery engine.

The manifest remains authoritative. The private sequence reservation survives a
crash, including allocations for ephemeral payloads that are never written.
Callers already holding ``lock_run`` must pass ``lock_held=True``.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Literal
from uuid import NAMESPACE_URL, uuid5

from pydantic import JsonValue

from kd1_anime.agent_runtime.contracts import (
    AgentEvent,
    EventPayload,
    RunId,
    RuntimeRecord,
    sanitize_record,
)
from kd1_anime.run_store import (
    MANIFEST_SCHEMA_VERSION,
    RUN_ID_PATTERN,
    RunManifest,
    atomic_write_json,
    migrate_manifest_data,
)


class EventStoreError(ValueError):
    """The notification log or its committed identity is inconsistent."""


class LegacyEvent(RuntimeRecord):
    """Read-only v2 envelope; it deliberately has no invented v3 sequence."""

    schema_version: Literal[2]
    manifest_revision: int
    scene_id: int | None = None
    scene_revision: int | None = None
    event_id: str
    timestamp: datetime
    run_id: RunId
    event: str
    data: JsonValue


class EventStore:
    def __init__(self, run_root: Path, run_id: str, *, secrets: tuple[str, ...] = ()):
        if not RUN_ID_PATTERN.fullmatch(run_id):
            raise EventStoreError("Invalid run identity")
        self.root = Path(run_root).absolute()
        self.run_id = run_id
        self.stream_id = f"run:{run_id}"
        self.path = self.root / "events.jsonl"
        self.sequence_path = self.root / ".events.sequence.json"
        self.secrets = secrets

    def _check_root(self) -> None:
        if any(path.is_symlink() for path in (self.root, *self.root.parents)):
            raise EventStoreError("Run directory cannot contain symbolic links")
        if not self.root.is_dir():
            raise EventStoreError("Committed run directory does not exist")
        self.root.chmod(0o700)

    def _manifest_before_lock(self) -> RunManifest:
        if any(path.is_symlink() for path in (self.root, *self.root.parents)):
            raise EventStoreError("Run directory cannot contain symbolic links")
        if not self.root.is_dir():
            raise EventStoreError("Committed run directory does not exist")
        return self._manifest()

    def _require_writable(self) -> None:
        if self._manifest_before_lock().schema_version != MANIFEST_SCHEMA_VERSION:
            raise EventStoreError("Historical run events are read-only")

    @contextmanager
    def locked(self, *, lock_held: bool = False) -> Iterator[None]:
        """Serialize writers using the existing process/thread-safe run flock.

        No in-process mutex precedes flock: that would invert locks when a
        caller already holding lock_run publishes with lock_held=True.
        """
        self._check_root()
        if lock_held:
            yield
            return
        try:
            descriptor = os.open(
                self.root / ".run.lock",
                os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
        except OSError as exc:
            raise EventStoreError("Cannot safely open run lock") from exc
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise EventStoreError("Run lock must be a regular file")
            os.fchmod(descriptor, 0o600)
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    @staticmethod
    def _read_bytes(path: Path, *, missing: bool = False) -> bytes:
        try:
            descriptor = os.open(
                path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
            )
        except FileNotFoundError:
            if missing:
                return b""
            raise EventStoreError(f"Missing committed file: {path.name}") from None
        except OSError as exc:
            raise EventStoreError(f"Cannot safely open {path.name}") from exc
        with os.fdopen(descriptor, "rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise EventStoreError(f"{path.name} must be a regular file")
            return handle.read()

    def _manifest(self) -> RunManifest:
        try:
            raw = json.loads(self._read_bytes(self.root / "manifest.json"))
            if not isinstance(raw, dict):
                raise ValueError("Manifest must be an object")
            manifest = RunManifest.model_validate(migrate_manifest_data(raw, self.root))
            if manifest.run_id != self.run_id:
                raise ValueError("Run and manifest identities disagree")
            errors = manifest.integrity_errors()
            if errors:
                raise ValueError("Manifest integrity: " + "; ".join(errors))
            return manifest
        except (ValueError, TypeError) as exc:
            raise EventStoreError("Invalid committed manifest") from exc

    def _validate_identity(self, event: AgentEvent | LegacyEvent, manifest: RunManifest) -> None:
        if event.run_id != self.run_id:
            raise EventStoreError("Foreign run identity in event log")
        if event.manifest_revision < 0 or event.manifest_revision > manifest.revision:
            raise EventStoreError("Event refers to an uncommitted manifest revision")
        if isinstance(event, AgentEvent):
            if event.stream_id != self.stream_id:
                raise EventStoreError("Foreign stream identity in event log")
            scene_id = event.payload.scene_id
        else:
            scene_id = event.scene_id
        if event.scene_revision is not None:
            scene = manifest.scenes.get(scene_id)
            if scene is None or not 0 <= event.scene_revision <= scene.revision:
                raise EventStoreError("Event refers to an uncommitted scene revision")

    def _scan(self, manifest: RunManifest) -> tuple[tuple[AgentEvent | LegacyEvent, ...], int, int]:
        raw = self._read_bytes(self.path, missing=True)
        # A record is committed to this log only after its terminating newline.
        boundary = raw.rfind(b"\n") + 1
        records: list[AgentEvent | LegacyEvent] = []
        ids: set[str] = set()
        snapshots: set[int] = set()
        previous = 0
        version = 0
        for number, line in enumerate(raw[:boundary].splitlines(), start=1):
            try:
                item = json.loads(line)
                current_version = item.get("schema_version") if isinstance(item, dict) else None
                if current_version not in (2, 3) or version not in (0, current_version):
                    raise ValueError("Mixed or unsupported event versions")
                version = current_version
                event = (AgentEvent if version == 3 else LegacyEvent).model_validate(item)
                self._validate_identity(event, manifest)
                if event.event_id in ids:
                    raise ValueError("Duplicate event ID")
                ids.add(event.event_id)
                if isinstance(event, AgentEvent):
                    if event.durability != "committed":
                        raise ValueError("Ephemeral payload in durable log")
                    if event.sequence <= previous:
                        raise ValueError("Duplicate or inverted sequence")
                    previous = event.sequence
                    if event.name == "snapshot_committed":
                        if event.manifest_revision in snapshots:
                            raise ValueError("Conflicting snapshot notification identity")
                        snapshots.add(event.manifest_revision)
                records.append(event)
            except (ValueError, TypeError) as exc:
                raise EventStoreError(f"Corrupt event record at line {number}") from exc
        return tuple(records), boundary, version

    def _watermark(self, events: tuple[AgentEvent | LegacyEvent, ...]) -> int:
        maximum = max(
            (event.sequence for event in events if isinstance(event, AgentEvent)), default=0
        )
        raw = self._read_bytes(self.sequence_path, missing=True)
        if not raw:
            if self.sequence_path.exists() or maximum:
                # A lost reservation may have covered ephemeral allocations
                # beyond the final durable line. Never reconstruct a lower one.
                raise EventStoreError("Missing or empty sequence reservation")
            return 0
        try:
            reservation = json.loads(raw)
            if not isinstance(reservation, dict) or set(reservation) != {
                "schema_version",
                "run_id",
                "high_watermark",
            }:
                raise ValueError("Invalid sequence reservation fields")
            value = reservation["high_watermark"]
            if (
                type(reservation["schema_version"]) is not int
                or reservation["schema_version"] != 1
                or reservation["run_id"] != self.run_id
                or type(value) is not int
                or value < maximum
                or value < 0
            ):
                raise ValueError("Invalid sequence reservation identity or value")
            return value
        except (ValueError, TypeError) as exc:
            raise EventStoreError("Corrupt sequence reservation") from exc

    def watermark(self, *, lock_held: bool = False) -> int:
        manifest = self._manifest_before_lock()
        if manifest.schema_version != MANIFEST_SCHEMA_VERSION:
            events, _, version = self._scan(manifest)
            return 0 if version == 2 else self._watermark(events)
        with self.locked(lock_held=lock_held):
            events, _, version = self._scan(self._manifest())
            return 0 if version == 2 else self._watermark(events)

    @property
    def high_watermark(self) -> int:
        return self.watermark()

    def replay(
        self, after: int | None = None, *, lock_held: bool = False
    ) -> tuple[AgentEvent | LegacyEvent, ...]:
        if after is not None and (type(after) is not int or after < 0):
            raise EventStoreError("Replay cursor must be a nonnegative integer")
        manifest = self._manifest_before_lock()
        if manifest.schema_version != MANIFEST_SCHEMA_VERSION:
            return self._replay(manifest, after)
        with self.locked(lock_held=lock_held):
            return self._replay(self._manifest(), after)

    def _replay(self, manifest: RunManifest, after: int | None):
        events, _, version = self._scan(manifest)
        if version == 2:
            if after is not None:
                raise EventStoreError("v2 events do not support sequence cursors")
            return events
        self._watermark(events)
        return tuple(event for event in events if after is None or event.sequence > after)

    def append(
        self, event: AgentEvent | Mapping | None = None, *, lock_held: bool = False, **draft_fields
    ) -> AgentEvent:
        if event is not None and draft_fields:
            raise EventStoreError("Supply either an envelope or draft fields")
        self._require_writable()
        fields = (
            event.model_dump(mode="json")
            if isinstance(event, AgentEvent)
            else dict(event or draft_fields)
        )
        with self.locked(lock_held=lock_held):
            manifest = self._manifest()
            events, boundary, version = self._scan(manifest)
            if version == 2:
                raise EventStoreError("v2 event logs are read-only")
            if (
                version == 0
                and self.path.exists()
                and self.path.stat().st_size > boundary
                and not self.sequence_path.exists()
            ):
                # Our writer reserves sequence durably before creating its first
                # line. Without that evidence this may be an interrupted v2 log.
                raise EventStoreError("Cannot replace an unidentified interrupted event log")
            sequence = self._watermark(events) + 1
            if "sequence" in fields and fields["sequence"] != sequence:
                raise EventStoreError("Event sequence must equal the next allocation")
            fields = {
                "run_id": self.run_id,
                "stream_id": self.stream_id,
                **fields,
                "sequence": sequence,
            }
            validated = AgentEvent.model_validate(sanitize_record(fields, self.secrets))
            self._validate_identity(validated, manifest)
            if any(item.event_id == validated.event_id for item in events):
                raise EventStoreError("Duplicate event ID")
            if validated.name == "snapshot_committed" and any(
                isinstance(item, AgentEvent)
                and item.name == validated.name
                and item.manifest_revision == validated.manifest_revision
                for item in events
            ):
                raise EventStoreError("Duplicate snapshot notification")
            atomic_write_json(
                self.sequence_path,
                {"schema_version": 1, "run_id": self.run_id, "high_watermark": sequence},
            )
            if validated.durability == "committed":
                encoded = (validated.model_dump_json() + "\n").encode("utf-8")
                try:
                    descriptor = os.open(
                        self.path,
                        os.O_CREAT
                        | os.O_WRONLY
                        | getattr(os, "O_NOFOLLOW", 0)
                        | getattr(os, "O_NONBLOCK", 0),
                        0o600,
                    )
                except OSError as exc:
                    raise EventStoreError("Cannot safely append event log") from exc
                with os.fdopen(descriptor, "wb") as handle:
                    if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                        raise EventStoreError("Event log must be a regular file")
                    os.fchmod(handle.fileno(), 0o600)
                    handle.truncate(boundary)
                    handle.seek(boundary)
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
                directory = os.open(self.root, os.O_RDONLY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            return validated

    def reconcile(self, manifest: RunManifest, *, lock_held: bool = False) -> AgentEvent:
        """Repair only a missing notification of the current on-disk snapshot."""
        self._require_writable()
        with self.locked(lock_held=lock_held):
            committed = self._manifest()
            if committed.model_dump(mode="json") != manifest.model_dump(mode="json"):
                raise EventStoreError("Reconciliation requires the committed manifest snapshot")
            events, _, version = self._scan(committed)
            if version == 2:
                raise EventStoreError("v2 event logs are read-only")
            event_id = uuid5(
                NAMESPACE_URL, f"kd1-anime:{self.run_id}:snapshot:{manifest.revision}"
            ).hex
            digest = hashlib.sha256(committed.model_dump_json().encode("utf-8")).hexdigest()
            payload = EventPayload(
                state=committed.state, status=committed.status, metadata={"manifest_sha256": digest}
            )
            for event in events:
                if event.event_id == event_id or (
                    event.name == "snapshot_committed"
                    and event.manifest_revision == manifest.revision
                ):
                    if (
                        event.event_id != event_id
                        or event.name != "snapshot_committed"
                        or event.payload != payload
                        or event.manifest_revision != manifest.revision
                    ):
                        raise EventStoreError("Conflicting reconciled snapshot identity")
                    return event
            return self.append(
                event_id=event_id,
                name="snapshot_committed",
                durability="committed",
                manifest_revision=manifest.revision,
                timestamp=committed.updated_at,
                payload=payload,
                lock_held=True,
            )
