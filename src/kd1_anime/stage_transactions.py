"""Isolate legacy stage mutations at the orchestration boundary.

Services receive explicit snapshots. This adapter lets staged orchestration code
propose state without exposing it to concurrent workers before durable commit.
"""

from __future__ import annotations

from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, fields

from kd1_anime.domain.dependencies import StaleWorkError, WorkIdentity

CURRENT_TRANSACTION: ContextVar[StageTransaction | None] = ContextVar(
    "kd1_stage_transaction", default=None
)


@dataclass
class StageTransaction:
    context: object
    base: object
    scene_id: int
    identity: WorkIdentity

    @classmethod
    def capture(cls, context, scene_id, identity):
        return cls(deepcopy(context), deepcopy(context), scene_id, identity)

    def apply_to(self, current):
        """Merge only proposed fields; request reservations stay authoritative."""
        for sid, proposed in self.context.scene_states.items():
            baseline = self.base.scene_states.get(sid)
            if proposed == baseline:
                continue
            live = current.scene_states.get(sid)
            if sid != self.scene_id and live != baseline:
                raise StaleWorkError("A concurrent scene changed before dependent invalidation")
            replacement = deepcopy(proposed)
            if live is not None:
                # Reservations can be checkpointed by the transport while this
                # stage computes; they cannot be replaced by an older snapshot.
                counters = {
                    (item.operation, item.input_sha256): deepcopy(item)
                    for item in live.stage_attempts
                }
                for item in replacement.stage_attempts:
                    key = (item.operation, item.input_sha256)
                    if key not in counters or item.used > counters[key].used:
                        counters[key] = item
                replacement.stage_attempts = list(counters.values())
            current.scene_states[sid] = replacement
        for definition in fields(self.context):
            name = definition.name
            if name in {"scene_states", "request_budget", "manifest_revision", "run_phase"}:
                continue
            before = getattr(self.base, name)
            proposed = getattr(self.context, name)
            if proposed == before:
                continue
            live = getattr(current, name)
            if live == before:
                setattr(current, name, deepcopy(proposed))
            elif (
                isinstance(before, list)
                and isinstance(proposed, list)
                and proposed[: len(before)] == before
            ):
                setattr(current, name, [*live, *deepcopy(proposed[len(before) :])])
            elif isinstance(before, dict) and isinstance(proposed, dict):
                merged = deepcopy(live)
                for key in before.keys() | proposed.keys():
                    if before.get(key) == proposed.get(key):
                        continue
                    if live.get(key) != before.get(key) and live.get(key) != proposed.get(key):
                        raise StaleWorkError(f"Concurrent stage field changed: {name}")
                    if key in proposed:
                        merged[key] = deepcopy(proposed[key])
                    else:
                        merged.pop(key, None)
                setattr(current, name, merged)
            else:
                raise StaleWorkError(f"Concurrent stage field changed: {name}")


def uses_legacy_stage_double(workflow):
    # Existing integrations that override the commit hook own their state model.
    # Keep this compatibility decision outside scheduling and domain services.
    return "_checkpoint" in vars(workflow)
