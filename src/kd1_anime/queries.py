"""Public read-only views assembled from committed manifests and sessions.

No worker Context, provider, process, or remote job is consulted. Run locks are
acquired before session locks; a view and its replay cursor share one snapshot.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, TypeAdapter

from kd1_anime.agent_runtime.contracts import (
    ActionName,
    AgentMessage,
    ApprovalGrant,
    DiffBinding,
    PermissionScope,
    RunId,
    RuntimeRecord,
    Sha256,
    sanitize_record,
    utc_now,
)
from kd1_anime.agent_runtime.policy import grant_is_current
from kd1_anime.agent_runtime.session import SessionStore, UserCommand
from kd1_anime.domain.dependencies import WorkIdentity
from kd1_anime.domain.execution import FSMState, RunPhase
from kd1_anime.event_store import EventStore
from kd1_anime.run_store import RunRepository, RunStatus, ScenePhase, VisualStatus


class SceneView(RuntimeRecord):
    scene_id: int = Field(ge=1)
    revision: int = Field(ge=0)
    input_sha256: str
    title: str
    phase: ScenePhase
    reviewed: bool
    rendered: bool
    failed: bool
    visual_status: VisualStatus
    visual_score: float | None = None
    job_id: str | None = None
    job_status: str | None = None


class RunView(RuntimeRecord):
    run_id: RunId
    schema_version: int
    revision: int = Field(ge=0)
    high_watermark: int = Field(ge=0)
    state: FSMState
    status: RunStatus
    run_phase: RunPhase
    goal: str
    created_at: datetime
    updated_at: datetime
    scenes: tuple[SceneView, ...]
    stop_reason: str = ""


class SessionView(RuntimeRecord):
    session_id: str
    goal: str
    revision: int = Field(ge=0)
    messages: tuple[AgentMessage, ...]
    commands: tuple[UserCommand, ...]
    grants: tuple[ApprovalGrant, ...]
    run_ids: tuple[RunId, ...]
    permissions: tuple[PermissionScope, ...]


class ApprovalRequestView(RuntimeRecord):
    event_id: str
    run_id: RunId
    session_id: str
    manifest_revision: int = Field(ge=0)
    scene_id: int | None = Field(default=None, ge=1)
    scene_revision: int | None = Field(default=None, ge=0)
    action: ActionName
    resource_id: str
    input_sha256: Sha256
    work_identity: WorkIdentity | None = None
    status: Literal["pending", "approved", "consumed", "expired", "stale"]
    grant_id: str | None = None
    proposal: DiffBinding | None = None


_IDENTITY = TypeAdapter(WorkIdentity)


class QueryService:
    def __init__(
        self,
        repository: RunRepository,
        sessions: SessionStore | None = None,
        *,
        secrets: tuple[str, ...] = (),
    ):
        self.repository = repository
        self.sessions = sessions
        self.secrets = secrets

    def _events(self, run_id: str) -> EventStore:
        return EventStore(self.repository.run_root(run_id), run_id, secrets=self.secrets)

    def _manifest(self, run_id):
        manifest = self.repository.load(run_id)
        errors = manifest.integrity_errors()
        if errors:
            raise ValueError("Manifest integrity: " + "; ".join(errors))
        return manifest

    def run(self, run_id: str, *, lock_held: bool = False) -> RunView:
        events = self._events(run_id)
        with events.locked(lock_held=lock_held):
            manifest = self._manifest(run_id)
            scenes = tuple(
                SceneView(
                    scene_id=scene_id,
                    revision=scene.revision,
                    input_sha256=scene.input_sha256,
                    title=scene.plan.title,
                    phase=scene.phase,
                    reviewed=scene.reviewed,
                    rendered=scene.rendered,
                    failed=scene.failed,
                    visual_status=scene.visual_status,
                    visual_score=scene.visual_score,
                    job_id=scene.slurm_job.job_id if scene.slurm_job is not None else None,
                    job_status=scene.slurm_job.status if scene.slurm_job is not None else None,
                )
                for scene_id, scene in sorted(manifest.scenes.items())
            )
            return RunView.model_validate(
                sanitize_record(
                    dict(
                        run_id=manifest.run_id,
                        schema_version=manifest.schema_version,
                        revision=manifest.revision,
                        high_watermark=events.watermark(lock_held=True),
                        state=manifest.state,
                        status=manifest.status,
                        run_phase=manifest.run_phase,
                        goal=manifest.user_prompt,
                        stop_reason=manifest.agent.stop_reason
                        if manifest.agent is not None
                        else manifest.error,
                        created_at=manifest.created_at,
                        updated_at=manifest.updated_at,
                        scenes=scenes,
                    ),
                    self.secrets,
                )
            )

    def scenes(self, run_id: str, *, lock_held: bool = False) -> tuple[SceneView, ...]:
        return self.run(run_id, lock_held=lock_held).scenes

    def session(self, session_id: str) -> SessionView:
        if self.sessions is None:
            raise ValueError("Session queries require a SessionStore")
        session = self.sessions.load(session_id)
        return SessionView.model_validate(
            sanitize_record(session.model_dump(mode="json"), self.secrets)
        )

    def approvals(
        self,
        run_id: str,
        session_id: str | None = None,
        *,
        now: datetime | None = None,
        lock_held: bool = False,
    ) -> tuple[ApprovalRequestView, ...]:
        if now is not None and now.utcoffset() is None:
            raise ValueError("Approval query time requires a timezone")
        instant = now or utc_now()
        events = self._events(run_id)
        with events.locked(lock_held=lock_held):
            manifest = self._manifest(run_id)
            requests = [
                event
                for event in events.replay(lock_held=True)
                if event.schema_version == 3 and event.name == "approval_requested"
            ]
            sessions = {}
            views = []
            for event in requests:
                metadata = event.payload.metadata
                requested_session = metadata.get("session_id")
                if not isinstance(requested_session, str) or not requested_session:
                    raise ValueError("Approval notification requires a session identity")
                if session_id is not None and requested_session != session_id:
                    continue
                action = event.payload.action
                if action is None:
                    raise ValueError("Approval notification requires a domain action")
                resource = metadata.get("resource_id")
                input_hash = metadata.get("input_sha256")
                scene_id = event.payload.scene_id
                expected_resource = f"run:{run_id}" + (
                    f"/scene:{scene_id}" if scene_id is not None else ""
                )
                if resource != expected_resource:
                    raise ValueError("Approval notification resource identity is inconsistent")
                identity = (
                    _IDENTITY.validate_python(metadata["work_identity"])
                    if metadata.get("work_identity") is not None
                    else None
                )
                proposal = (
                    DiffBinding.model_validate(metadata["proposal"])
                    if metadata.get("proposal") is not None
                    else None
                )
                if proposal is not None and (
                    proposal.work_identity != identity or proposal.digest() != input_hash
                ):
                    raise ValueError("Approval notification diff binding is inconsistent")
                if requested_session not in sessions and self.sessions is not None:
                    sessions[requested_session] = self.sessions.load(requested_session)
                session = sessions.get(requested_session)
                matches = (
                    [
                        grant
                        for grant in session.grants
                        if (
                            grant.session_id == requested_session
                            and grant.run_id == run_id
                            and grant.action == action
                            and grant.resource_id == resource
                            and grant.input_sha256 == input_hash
                            and grant.work_identity == identity
                        )
                    ]
                    if session is not None
                    else []
                )
                current = manifest.scenes.get(scene_id) if scene_id is not None else None
                stale = (
                    (
                        current is None
                        or identity != WorkIdentity(current.revision, current.input_sha256)
                    )
                    if scene_id is not None
                    else event.manifest_revision != manifest.revision
                )
                if scene_id is None and manifest.agent is not None:
                    stale = identity != WorkIdentity(
                        manifest.agent.run_revision, manifest.agent.run_input_sha256
                    )
                if scene_id is not None and identity is None:
                    stale = True
                grant = next(
                    (item for item in reversed(matches) if item.consumed_by_intent is not None),
                    None,
                )
                if grant is None:
                    grant = next(
                        (
                            item
                            for item in reversed(matches)
                            if item.expires_at is None or item.expires_at > instant
                        ),
                        None,
                    )
                if grant is None and matches:
                    grant = matches[-1]
                if (
                    grant is not None
                    and grant.consumed_by_intent is None
                    and not grant_is_current(grant, session)
                ):
                    stale = True
                status = "stale" if stale else "pending"
                if not stale and grant is not None:
                    if grant.consumed_by_intent is not None:
                        status = "consumed"
                    elif grant.expires_at is not None and grant.expires_at <= instant:
                        status = "expired"
                    else:
                        status = "approved"
                views.append(
                    ApprovalRequestView(
                        event_id=event.event_id,
                        run_id=run_id,
                        session_id=requested_session,
                        manifest_revision=event.manifest_revision,
                        scene_id=scene_id,
                        scene_revision=event.scene_revision,
                        action=action,
                        resource_id=resource,
                        input_sha256=input_hash,
                        work_identity=identity,
                        status=status,
                        grant_id=grant.grant_id if grant is not None else None,
                        proposal=proposal,
                    )
                )
            return tuple(views)
