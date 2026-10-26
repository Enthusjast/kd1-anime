"""Explicit scene inputs and conservative transitive invalidation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import ConfigDict, Field

from kd1_anime.models import ValidatedModel as BaseModel


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


class SceneDependency(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_scene_id: int = Field(ge=1)
    kind: Literal["teaching", "handoff", "code_context"]
    element_ids: list[str] = Field(default_factory=list, max_length=100)


class DependencyBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_scene_id: int = Field(ge=1)
    consumer_scene_id: int = Field(ge=1)
    kind: Literal["teaching", "handoff", "code_context"]
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_revision: int = Field(default=0, ge=0)


class DependencyGraph(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    bindings: list[DependencyBinding] = Field(default_factory=list)
    global_inputs: dict[int, str] = Field(default_factory=dict)

    def bind(
        self,
        consumer: int,
        dependencies: Sequence[SceneDependency],
        nodes: Mapping[int, dict],
        *,
        global_sha256: str | None = None,
    ) -> None:
        bindings = [item for item in self.bindings if item.consumer_scene_id != consumer]
        for dependency in dependencies:
            source = dependency.source_scene_id
            if source not in nodes or source >= consumer:
                raise ValueError(f"Scene {consumer} 依赖无效或非先行场景 {source}")
            node = nodes[source]
            bindings.append(
                DependencyBinding(
                    source_scene_id=source,
                    consumer_scene_id=consumer,
                    kind=dependency.kind,
                    source_sha256=node["code"]
                    if dependency.kind == "code_context"
                    else node.get("teaching", node["contract"])
                    if dependency.kind == "teaching"
                    else node["contract"],
                    source_revision=node.get("revision", 0),
                )
            )
        self.bindings = bindings
        if global_sha256 is not None:
            self.global_inputs[consumer] = global_sha256

    def changed_globals(self, current_sha256: str) -> set[int]:
        return {
            consumer
            for consumer, captured in self.global_inputs.items()
            if captured != current_sha256
        }

    def affected(
        self,
        source: int,
        nodes: Mapping[int, dict],
        fallback_edges: Sequence[DependencyBinding] = (),
    ) -> set[int]:
        edges = [*self.bindings, *fallback_edges]
        affected: set[int] = set()
        frontier = {source}
        while frontier:
            current = frontier.pop()
            for edge in edges:
                if edge.source_scene_id != current or edge.consumer_scene_id in affected:
                    continue
                # After invalidating an intermediate producer its future contract is unknown.
                changed = current != source or edge.kind == "code_context"
                if not changed:
                    key = (
                        "teaching"
                        if edge.kind == "teaching" and "teaching" in nodes[current]
                        else "contract"
                    )
                    changed = edge.source_sha256 != nodes[current][key]
                if changed:
                    affected.add(edge.consumer_scene_id)
                    frontier.add(edge.consumer_scene_id)
        return affected


@dataclass(frozen=True)
class WorkIdentity:
    revision: int
    input_sha256: str


class StaleWorkError(RuntimeError):
    """A worker returned a result for superseded scene inputs."""
