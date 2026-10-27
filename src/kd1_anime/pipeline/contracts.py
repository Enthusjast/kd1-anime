"""Explicit immutable stage inputs; workers never receive a mutable run Context."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from kd1_anime.agent_runtime.contracts import ObjectRef, RuntimeRecord
from kd1_anime.domain.diagnostics import Diagnostic
from kd1_anime.domain.planning import (
    ContinuityBible,
    ElementManifest,
    LessonSpec,
    SceneOutline,
    ScenePlan,
    TeachingGraph,
)
from kd1_anime.domain.runtime import ExecutionSpec, RunConfig
from kd1_anime.domain.state_ledger import StateLedger
from kd1_anime.domain.technical import TechnicalHandoff, TechnicalSpec
from kd1_anime.rag.models import RagReceipt, RagRuntimeProfile
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
    element_manifest: ElementManifest = field(default_factory=ElementManifest)
    state_ledger: StateLedger = field(default_factory=StateLedger)
    previous_technical_handoff: TechnicalHandoff | None = None
    rag_context: str = ""
    direct_render: bool = False


@dataclass(frozen=True, slots=True)
class DetailInputs:
    run_id: str
    root: Path
    config: RunConfig
    outline: SceneOutline
    outlines: tuple[SceneOutline, ...]
    lesson_spec: LessonSpec
    teaching_graph: TeachingGraph
    continuity_bible: ContinuityBible | None
    user_prompt: str
    renderer: Literal["cairo", "opengl"]
    continuity_context: str = ""
    rag_context: str = ""


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
    rag_profile: RagRuntimeProfile | None = None
    rag_receipts: tuple[RagReceipt, ...] = ()


class DetailOutput(RuntimeRecord):
    plan: ScenePlan
    diagnostics: tuple[Diagnostic, ...] = ()
    rag_receipts: tuple[RagReceipt, ...] = ()


class TechnicalOutput(RuntimeRecord):
    spec: TechnicalSpec
    diagnostics: tuple[Diagnostic, ...] = ()
    rag_receipts: tuple[RagReceipt, ...] = ()


class SceneOutput(RuntimeRecord):
    source: ObjectRef
    class_name: str
    diagnostics: tuple[Diagnostic, ...] = ()
    rag_receipts: tuple[RagReceipt, ...] = ()


class RenderingOutput(RuntimeRecord):
    status: Literal["pending", "rendered", "failed", "cancelled", "unknown"]
    job: StoredSlurmJob | None = None
    artifact: SceneArtifact | None = None
    diagnostics: tuple[Diagnostic, ...] = ()


class FinalizationOutput(RuntimeRecord):
    video: ObjectRef
    output_path: str
    diagnostics: tuple[Diagnostic, ...] = ()
