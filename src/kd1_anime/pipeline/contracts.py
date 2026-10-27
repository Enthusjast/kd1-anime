"""Explicit immutable stage inputs; workers never receive a mutable run Context."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from kd1_anime.agent_runtime.contracts import ObjectRef, RuntimeRecord
from kd1_anime.domain.diagnostics import Diagnostic
from kd1_anime.domain.planning import (
    ContinuityBible,
    LessonSpec,
    SceneOutline,
    ScenePlan,
    TeachingGraph,
)
from kd1_anime.domain.runtime import ExecutionSpec, RunConfig
from kd1_anime.domain.technical import TechnicalSpec
from kd1_anime.rendering import MergeProfile, RenderProfile, SceneArtifact
from kd1_anime.run_store import StoredSceneState, StoredSlurmJob


@dataclass(frozen=True, slots=True)
class SceneInputs:
    run_id: str
    root: Path
    config: RunConfig
    render_profile: RenderProfile
    execution_spec: ExecutionSpec
    scene: StoredSceneState
    plans: tuple[ScenePlan, ...]
    lesson_spec: LessonSpec
    teaching_graph: TeachingGraph
    continuity_bible: ContinuityBible | None


@dataclass(frozen=True, slots=True)
class FinalizationInputs:
    run_id: str
    root: Path
    config: RunConfig
    render_profile: RenderProfile
    merge_profile: MergeProfile
    scenes: tuple[StoredSceneState, ...]
    output_path: Path


class PlanningOutput(RuntimeRecord):
    outlines: tuple[SceneOutline, ...]
    plans: tuple[ScenePlan, ...]
    lesson_spec: LessonSpec
    teaching_graph: TeachingGraph
    continuity_bible: ContinuityBible | None = None
    diagnostics: tuple[Diagnostic, ...] = ()


class TechnicalOutput(RuntimeRecord):
    spec: TechnicalSpec
    diagnostics: tuple[Diagnostic, ...] = ()


class SceneOutput(RuntimeRecord):
    source: ObjectRef
    class_name: str
    diagnostics: tuple[Diagnostic, ...] = ()


class RenderingOutput(RuntimeRecord):
    status: Literal["pending", "rendered", "failed", "cancelled", "unknown"]
    job: StoredSlurmJob | None = None
    artifact: SceneArtifact | None = None
    diagnostics: tuple[Diagnostic, ...] = ()


class FinalizationOutput(RuntimeRecord):
    video: ObjectRef
    output_path: str
    diagnostics: tuple[Diagnostic, ...] = ()
