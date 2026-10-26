"""Commands and snapshots at the single durable state boundary."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, fields

from kd1_anime.domain.dependencies import StaleWorkError, WorkIdentity


@dataclass(frozen=True, slots=True)
class CommitCommand:
    expected_revision: int
    phase: str
    status: str = "running"
    error: str = ""
    scene_id: int | None = None
    identity: WorkIdentity | None = None
    apply: Callable | None = None
    queued_at: float | None = None

    def check(self, context, current_identity=None):
        if self.expected_revision != context.manifest_revision:
            raise StaleWorkError("Commit was built from a superseded manifest")
        if self.identity is not None and self.identity != current_identity:
            raise StaleWorkError("Commit was built from superseded scene inputs")


def restore_snapshot(target, snapshot):
    """Preserve worker-held scene identities while replacing all proposed values."""
    existing = target.scene_states
    restored = {}
    for scene_id, stored in snapshot.scene_states.items():
        scene = existing.get(scene_id)
        if scene is None:
            restored[scene_id] = deepcopy(stored)
        else:
            values = deepcopy(stored.__dict__)
            previous_job = scene.slurm_job
            if (
                previous_job is not None
                and stored.slurm_job is not None
                and previous_job.job_id == stored.slurm_job.job_id
            ):
                previous_job.__dict__.update(deepcopy(stored.slurm_job.__dict__))
                values["execution"].slurm_job = previous_job
            scene.__dict__.clear()
            scene.__dict__.update(values)
            restored[scene_id] = scene
    for item in fields(target):
        if item.name != "scene_states":
            setattr(target, item.name, deepcopy(getattr(snapshot, item.name)))
    target.scene_states = restored
