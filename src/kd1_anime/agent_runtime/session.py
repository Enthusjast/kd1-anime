"""User-owned sessions, visible command history, and one-use approval grants."""

from __future__ import annotations

import fcntl
import os
import re
import stat
from contextlib import contextmanager, suppress
from datetime import datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import Field, model_validator

from kd1_anime.agent_runtime.contracts import (
    AgentAction,
    AgentMessage,
    ApprovalGrant,
    ExecutionIntent,
    PermissionScope,
    RunId,
    RuntimeRecord,
    sanitize_record,
    utc_now,
)

_SESSION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
_RUN_RESOURCE = re.compile(r"run:(\d{8}-\d{6}-[0-9a-f]{8})(?:/scene:([1-9]\d*))?\Z")
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def _session_id(value: str) -> str:
    if not isinstance(value, str) or not _SESSION_ID.fullmatch(value):
        raise ValueError("Invalid session identifier")
    return value


def _check_action(action: AgentAction, run_ids: tuple[str, ...]) -> None:
    resource = f"run:{action.run_id}"
    if action.scene_id is not None:
        resource += f"/scene:{action.scene_id}"
    if action.resource_id != resource:
        raise ValueError("Action resource does not match its run and scene")
    if action.run_id not in run_ids:
        raise ValueError("Action run must be linked to the session")


def _binding(value) -> tuple:
    return (
        value.action if isinstance(value, ApprovalGrant) else value.kind,
        value.resource_id,
        value.input_sha256,
        value.run_id,
        value.work_identity,
    )


class UserCommand(RuntimeRecord):
    command_id: str = Field(default_factory=lambda: uuid4().hex, min_length=1)
    session_id: str = Field(min_length=1)
    kind: Literal["goal", "clarification", "ready", "approve", "reject", "cancel"]
    role: Literal["user"] = "user"
    content: str = ""
    action: AgentAction | None = None
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_command(self):
        _session_id(self.session_id)
        if self.kind in {"approve", "reject"} and self.action is None:
            raise ValueError("Approval and rejection require an exact action")
        if self.kind not in {"approve", "reject"} and self.action is not None:
            raise ValueError("Only approval and rejection commands bind actions")
        if self.kind in {"goal", "clarification"} and not self.content:
            raise ValueError("Goals and clarifications require visible content")
        return self


class Session(RuntimeRecord):
    session_id: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    messages: tuple[AgentMessage, ...] = ()
    commands: tuple[UserCommand, ...] = ()
    grants: tuple[ApprovalGrant, ...] = ()
    run_ids: tuple[RunId, ...] = ()
    permissions: tuple[PermissionScope, ...] = ()
    revision: int = Field(default=0, ge=0, strict=True)

    @model_validator(mode="after")
    def validate_session(self):
        _session_id(self.session_id)
        if len(set(self.run_ids)) != len(self.run_ids):
            raise ValueError("Duplicate linked run")
        for records, field in (
            (self.messages, "message_id"),
            (self.commands, "command_id"),
            (self.grants, "grant_id"),
        ):
            identifiers = [getattr(record, field) for record in records]
            if len(identifiers) != len(set(identifiers)):
                raise ValueError("Duplicate session record identifier")
            for record in records:
                if record.session_id != self.session_id:
                    raise ValueError("Session record binding mismatch")
        for message in self.messages:
            if message.run_id is not None and message.run_id not in self.run_ids:
                raise ValueError("Message run must be linked to the session")
            if message.action is not None:
                _check_action(message.action, self.run_ids)
        for command in self.commands:
            if command.action is not None:
                _check_action(command.action, self.run_ids)
        for grant in self.grants:
            resource = _RUN_RESOURCE.fullmatch(grant.resource_id)
            if resource is None or resource.group(1) != grant.run_id:
                raise ValueError("Grant resource binding mismatch")
            if grant.run_id not in self.run_ids:
                raise ValueError("Grant run must be linked to the session")
            if not any(
                command.kind == "approve"
                and command.action is not None
                and _binding(command.action) == _binding(grant)
                and command.created_at == grant.issued_at
                for command in self.commands
            ):
                raise ValueError("Grant requires a trusted user approve command")
        for permission in self.permissions:
            if permission.resource_id == f"session:{self.session_id}":
                continue
            resource = _RUN_RESOURCE.fullmatch(permission.resource_id)
            if resource is None or resource.group(1) not in self.run_ids:
                raise ValueError("Permission scope must name this session or a linked run resource")
        return self


class SessionStore:
    """Serialize independent sessions under per-session process-safe private locks.

    Approval commands are a trusted local-user API, never a model tool. The
    executor must durably reserve its ExecutionIntent before consuming a grant.
    A successful replay of the same reservation is idempotent, not permission
    to dispatch that reservation a second time.
    """

    def __init__(self, root: str | Path, *, secrets: tuple[str, ...] = ()):
        self.root = Path(root).absolute()
        self.secrets = tuple(secrets)

    @contextmanager
    def _root_directory(self, *, create: bool = False):
        # Open every component without following links, including ancestors.
        descriptor = os.open(self.root.anchor, _DIRECTORY_FLAGS)
        try:
            for component in self.root.parts[1:]:
                if component in {".", ".."}:
                    raise ValueError("Session root must not contain traversal")
                if create:
                    with suppress(FileExistsError):
                        os.mkdir(component, 0o700, dir_fd=descriptor)
                following = os.open(component, _DIRECTORY_FLAGS, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = following
            if os.fstat(descriptor).st_uid != os.geteuid():
                raise ValueError("Session root must be owned by the local user")
            os.fchmod(descriptor, 0o700)
            yield descriptor
        finally:
            os.close(descriptor)

    @contextmanager
    def _locked_directory(self, session_id: str, *, create: bool = False):
        _session_id(session_id)
        with self._root_directory(create=create) as root_fd:
            if create:
                with suppress(FileExistsError):
                    os.mkdir(session_id, 0o700, dir_fd=root_fd)
                os.fsync(root_fd)
            directory_fd = os.open(session_id, _DIRECTORY_FLAGS, dir_fd=root_fd)
            try:
                if os.fstat(directory_fd).st_uid != os.geteuid():
                    raise ValueError("Session directory must be owned by the local user")
                os.fchmod(directory_fd, 0o700)
                lock_fd = os.open(
                    ".session.lock",
                    os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK,
                    0o600,
                    dir_fd=directory_fd,
                )
                try:
                    self._private_file(lock_fd)
                    os.fchmod(lock_fd, 0o600)
                    fcntl.flock(lock_fd, fcntl.LOCK_EX)
                    self._recover(directory_fd)
                    yield directory_fd
                finally:
                    os.close(lock_fd)
            finally:
                os.close(directory_fd)

    @staticmethod
    def _private_file(descriptor: int) -> None:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_uid != os.geteuid():
            raise ValueError("Session data and locks must be private regular files")

    @staticmethod
    def _recover(directory_fd: int) -> None:
        # A killed writer can leave its old inode hard-linked to a backup.
        # The atomically published session.json is always authoritative.
        recovered = False
        for name in os.listdir(directory_fd):
            if re.fullmatch(r"\.session-[0-9a-f]{32}\.(?:tmp|previous)", name):
                os.unlink(name, dir_fd=directory_fd)
                recovered = True
        if recovered:
            os.fsync(directory_fd)

    def _load(self, directory_fd: int, session_id: str) -> Session:
        descriptor = os.open(
            "session.json", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd
        )
        try:
            self._private_file(descriptor)
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "r", encoding="utf-8", closefd=False) as handle:
                session = Session.model_validate_json(handle.read())
        finally:
            os.close(descriptor)
        if session.session_id != session_id:
            raise ValueError("Session snapshot identifier binding mismatch")
        return self._sanitize(session)

    def _sanitize(self, record):
        return type(record).model_validate(
            sanitize_record(record.model_dump(mode="json"), self.secrets)
        )

    def _save(self, directory_fd: int, session: Session) -> Session:
        session = self._sanitize(session)
        payload = (session.model_dump_json(indent=2) + "\n").encode("utf-8")
        temporary = f".session-{uuid4().hex}.tmp"
        backup = f".session-{uuid4().hex}.previous"
        descriptor = os.open(
            temporary,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_fd,
        )
        published = False
        backed_up = False
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(descriptor)
            try:
                existing = os.stat("session.json", dir_fd=directory_fd, follow_symlinks=False)
            except FileNotFoundError:
                existing = None
            if existing is not None:
                if not stat.S_ISREG(existing.st_mode) or existing.st_nlink != 1:
                    raise ValueError("Session destination must be a private regular file")
                os.link(
                    "session.json",
                    backup,
                    src_dir_fd=directory_fd,
                    dst_dir_fd=directory_fd,
                    follow_symlinks=False,
                )
                backed_up = True
            os.replace(temporary, "session.json", src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
            published = True
            os.fsync(directory_fd)
        except BaseException:
            if published:
                if backed_up:
                    os.replace(
                        backup, "session.json", src_dir_fd=directory_fd, dst_dir_fd=directory_fd
                    )
                else:
                    os.unlink("session.json", dir_fd=directory_fd)
                with suppress(OSError):
                    os.fsync(directory_fd)
            raise
        finally:
            os.close(descriptor)
            for name in (temporary, backup):
                with suppress(FileNotFoundError):
                    os.unlink(name, dir_fd=directory_fd)
        return session

    def _replace(self, directory_fd: int, session: Session, **changes) -> Session:
        updated = Session.model_validate(
            session.model_dump() | changes | {"revision": session.revision + 1}
        )
        return self._save(directory_fd, updated)

    def create(
        self,
        goal: str,
        *,
        session_id: str | None = None,
        permissions: tuple[PermissionScope, ...] = (),
        run_ids: tuple[str, ...] = (),
    ) -> Session:
        session_id = _session_id(session_id or uuid4().hex)
        command = self._sanitize(UserCommand(session_id=session_id, kind="goal", content=goal))
        message = AgentMessage(
            session_id=session_id,
            role="user",
            kind="goal",
            content=command.content,
            message_id=command.command_id,
            created_at=command.created_at,
        )
        session = Session(
            session_id=session_id,
            goal=command.content,
            commands=(command,),
            messages=(message,),
            permissions=permissions,
            run_ids=run_ids,
        )
        with self._locked_directory(session_id, create=True) as directory_fd:
            try:
                os.stat("session.json", dir_fd=directory_fd, follow_symlinks=False)
            except FileNotFoundError:
                return self._save(directory_fd, session)
            raise ValueError("Session already exists")

    def load(self, session_id: str) -> Session:
        with self._locked_directory(session_id) as directory_fd:
            return self._load(directory_fd, session_id)

    def append_message(self, session_id: str, message: AgentMessage) -> Session:
        message = self._sanitize(AgentMessage.model_validate(message.model_dump()))
        with self._locked_directory(session_id) as directory_fd:
            session = self._load(directory_fd, session_id)
            for previous in session.messages:
                if previous.message_id == message.message_id:
                    if previous != message:
                        raise ValueError("Message identifier conflict")
                    return session
            return self._replace(directory_fd, session, messages=(*session.messages, message))

    def _add_command(
        self,
        directory_fd: int,
        session: Session,
        command: UserCommand,
        *,
        grant: ApprovalGrant | None = None,
    ) -> Session:
        for previous in session.commands:
            if previous.command_id == command.command_id:
                if previous != command:
                    raise ValueError("Command identifier conflict")
                return session
        message = AgentMessage(
            message_id=command.command_id,
            session_id=command.session_id,
            role="user",
            kind=command.kind if command.kind in {"goal", "clarification"} else "command",
            content=command.content,
            created_at=command.created_at,
        )
        if any(previous.message_id == message.message_id for previous in session.messages):
            raise ValueError("Command message identifier conflict")
        changes = {
            "commands": (*session.commands, command),
            "messages": (*session.messages, message),
        }
        if command.kind == "goal":
            changes["goal"] = command.content
        if grant is not None:
            changes["grants"] = (*session.grants, grant)
        return self._replace(directory_fd, session, **changes)

    def command(self, session_id: str, command: UserCommand) -> Session:
        command = self._sanitize(UserCommand.model_validate(command.model_dump()))
        if command.kind == "approve":
            raise ValueError("Use issue_grant for an approve command")
        with self._locked_directory(session_id) as directory_fd:
            session = self._load(directory_fd, session_id)
            return self._add_command(directory_fd, session, command)

    def link_run(self, session_id: str, run_id: str) -> Session:
        with self._locked_directory(session_id) as directory_fd:
            session = self._load(directory_fd, session_id)
            if run_id in session.run_ids:
                return session
            return self._replace(directory_fd, session, run_ids=(*session.run_ids, run_id))

    def issue_grant(
        self, session_id: str, command: UserCommand, *, expires_at: datetime | None = None
    ) -> ApprovalGrant:
        if not isinstance(command, UserCommand):
            raise ValueError("A trusted user approve command is required")
        command = self._sanitize(UserCommand.model_validate(command.model_dump()))
        if command.kind != "approve" or command.action is None:
            raise ValueError("A trusted user approve command is required")
        with self._locked_directory(session_id) as directory_fd:
            session = self._load(directory_fd, session_id)
            if command.session_id != session_id:
                raise ValueError("Approval session binding mismatch")
            _check_action(command.action, session.run_ids)
            for previous in session.commands:
                if previous.command_id == command.command_id:
                    if previous != command:
                        raise ValueError("Command identifier conflict")
                    for grant in session.grants:
                        if grant.issued_at == command.created_at and _binding(grant) == _binding(
                            command.action
                        ):
                            if grant.expires_at != expires_at:
                                raise ValueError("Approval expiration conflict")
                            return grant
                    raise ValueError("Approval command has no matching grant")
            grant = ApprovalGrant(
                action=command.action.kind,
                resource_id=command.action.resource_id,
                input_sha256=command.action.input_sha256,
                session_id=session_id,
                run_id=command.action.run_id,
                work_identity=command.action.work_identity,
                issued_at=command.created_at,
                expires_at=expires_at,
            )
            updated = self._add_command(directory_fd, session, command, grant=grant)
            return updated.grants[-1]

    def consume_grant(
        self,
        session_id: str,
        grant_id: str,
        intent: ExecutionIntent,
        *,
        now: datetime | None = None,
    ) -> ApprovalGrant:
        intent = ExecutionIntent.model_validate(intent.model_dump())
        now = now or utc_now()
        if now.utcoffset() is None:
            raise ValueError("Grant consumption requires a timezone")
        with self._locked_directory(session_id) as directory_fd:
            session = self._load(directory_fd, session_id)
            index = next(
                (
                    position
                    for position, item in enumerate(session.grants)
                    if item.grant_id == grant_id
                ),
                None,
            )
            if index is None:
                raise ValueError("Unknown approval grant")
            grant = session.grants[index]
            if intent.session_id != session_id or _binding(intent.action) != _binding(grant):
                raise ValueError("Approval grant binding mismatch")
            _check_action(intent.action, session.run_ids)
            if grant.consumed_by_intent is not None:
                if grant.consumed_by_intent != intent.intent_id:
                    raise ValueError("Approval grant already consumed")
                return grant
            if intent.status != "reserved":
                raise ValueError("Only a reserved execution intent can consume a grant")
            if now < grant.issued_at:
                raise ValueError("Approval grant has not been issued yet")
            if grant.expires_at is not None and now >= grant.expires_at:
                raise ValueError("Approval grant expired")
            approval_index = next(
                position
                for position, command in enumerate(session.commands)
                if command.kind == "approve"
                and command.created_at == grant.issued_at
                and command.action is not None
                and _binding(command.action) == _binding(grant)
            )
            for command in session.commands[approval_index + 1 :]:
                if command.kind in {"cancel", "goal", "clarification"} or (
                    command.kind == "reject"
                    and command.action is not None
                    and _binding(command.action) == _binding(grant)
                ):
                    raise ValueError("Approval grant superseded by a user command")
            consumed = ApprovalGrant.model_validate(
                grant.model_dump() | {"consumed_by_intent": intent.intent_id}
            )
            grants = (*session.grants[:index], consumed, *session.grants[index + 1 :])
            updated = self._replace(directory_fd, session, grants=grants)
            return updated.grants[index]
