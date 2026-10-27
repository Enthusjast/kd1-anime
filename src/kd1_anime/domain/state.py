"""Runtime aggregates; compatibility attributes delegate to bounded state records."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from kd1_anime.cluster.jobs import RenderJob as SlurmJob
from kd1_anime.cluster.render_backend import RenderBackendName
from kd1_anime.cluster.resource_estimator import RenderResourceProfile
from kd1_anime.config import GenerationMode, resolve_runtime_path, settings
from kd1_anime.domain.candidates import CodeCandidate
from kd1_anime.domain.capability import CapabilityContract
from kd1_anime.domain.dependencies import DependencyGraph
from kd1_anime.domain.execution import RunPhase, StageAttempts, SubmissionAttempt
from kd1_anime.domain.plan_review import PlanReviewIssue
from kd1_anime.domain.planning import (
    ContinuityBible,
    ElementManifest,
    ExtractedElement,
    LessonSpec,
    SceneOutline,
    ScenePlan,
    TeachingGraph,
)
from kd1_anime.domain.runtime import ExecutionSpec, RunConfig, VisualEvalProfile
from kd1_anime.domain.scene_fields import SCENE_FIELDS, SCENE_GROUPS
from kd1_anime.domain.state_ledger import StateLedger
from kd1_anime.domain.technical import TechnicalSpec
from kd1_anime.rag.models import RagReceipt, RagRuntimeProfile
from kd1_anime.rendering import MergeProfile, RenderProfile, SceneArtifact
from kd1_anime.request_budget import RequestBudget
from kd1_anime.verification import ExecutionVerification, StaticVerification, VisualVerification


@dataclass(frozen=True, slots=True)
class RunPaths:
    run_id: str
    root: Path
    scenes: Path
    logs: Path
    videos: Path
    output: Path

    @classmethod
    def create(cls, output_path: Path | None = None) -> RunPaths:
        stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
        run_id = f"{stamp}-{uuid4().hex[:8]}"
        root = resolve_runtime_path(settings.WORKSPACE_DIR) / "runs" / run_id
        configured_output = (output_path or settings.OUTPUT_FILE).expanduser()
        if output_path is None and configured_output == Path("output_final.mp4"):
            output = root / "output_final.mp4"
        else:
            output = resolve_runtime_path(configured_output)
        return cls(
            run_id=run_id,
            root=root,
            scenes=root / "scenes",
            logs=root / "logs",
            videos=root / "videos",
            output=output,
        )


@dataclass
class VisualCandidate:
    score: float
    has_major_issue: bool
    passed: bool
    inherited_elements_sha256: str
    code: str
    class_name: str
    slurm_job: SlurmJob | None
    artifact: SceneArtifact
    exported_elements_code: str
    exported_elements: list[ExtractedElement]
    report_file: str
    report_sha256: str


@dataclass
class PlanningState:
    plan: ScenePlan
    plan_ready: bool = False
    plan_replan_attempts: int = 0
    plan_review_round: int = 0
    plan_reviewed: bool = False
    plan_review_feedback: str = ""
    plan_review_signature: str = ""
    technical_spec: TechnicalSpec | None = None
    technical_contract_stale: bool = False
    technical_spec_sha256: str = ""
    technical_input_sha256: str = ""
    technical_status: str = "pending"
    technical_error: str = ""
    capability_contract: CapabilityContract | None = None
    capability_status: str = "pending"


@dataclass
class CodeRevisionState:
    code: str = ""
    class_name: str = ""
    review_round: int = 0
    safe_fallback_used: bool = False
    safe_fallback_reason: str = ""
    identical_plan_review_count: int = 0
    unknown_animation_detected: bool = False
    unknown_animation_details: list[str] = field(default_factory=list)
    rewrite_feedback: str = ""
    review_signature: str = ""
    identical_review_count: int = 0
    reviewed: bool = False
    inherited_elements_code: str = ""
    exported_elements_code: str = ""
    exported_elements: list[ExtractedElement] = field(default_factory=list)
    candidates: list[CodeCandidate] = field(default_factory=list)


@dataclass
class ExecutionState:
    submission: SubmissionAttempt | None = None
    submission_attempts: list[SubmissionAttempt] = field(default_factory=list)
    stage_attempts: list[StageAttempts] = field(default_factory=list)
    fix_attempts: int = 0
    infra_retries: int = 0
    slurm_job: SlurmJob | None = None
    artifact: SceneArtifact | None = None
    rendered: bool = False
    resource_profile: RenderResourceProfile | None = None
    last_error_fp: str = ""
    identical_error_count: int = 0
    last_repair_code_sha256: str = ""
    last_repair_error_fp: str = ""
    stagnant_repair_count: int = 0


@dataclass
class VerificationState:
    static_verification: StaticVerification = field(default_factory=StaticVerification)
    execution_verification: ExecutionVerification = field(default_factory=ExecutionVerification)
    visual_verification: VisualVerification = field(default_factory=VisualVerification)
    local_smoke_status: str = "pending"
    visual_status: str = "skipped"
    visual_fix_attempts: int = 0
    visual_score: float | None = None
    visual_report_file: str = ""
    visual_report_sha256: str = ""
    visual_artifact_sha256: str = ""
    visual_feedback: str = ""
    visual_best_candidate: VisualCandidate | None = None


@dataclass
class LifecycleState:
    revision: int = 0
    input_sha256: str = ""
    give_up: bool = False
    failed: bool = False
    failure_reason: str = ""
    failure_category: str = ""
    phase: str = "pending"


@dataclass(init=False)
class SceneState:
    planning: PlanningState
    code_revision: CodeRevisionState
    execution: ExecutionState
    verification: VerificationState
    lifecycle: LifecycleState

    def __init__(self, plan, **values):
        object.__setattr__(self, "planning", PlanningState(plan))
        object.__setattr__(self, "code_revision", CodeRevisionState())
        object.__setattr__(self, "execution", ExecutionState())
        object.__setattr__(self, "verification", VerificationState())
        object.__setattr__(self, "lifecycle", LifecycleState())
        for name, value in values.items():
            if name not in SCENE_FIELDS:
                raise TypeError(f"Unknown scene state field {name}")
            setattr(self, name, value)

    def __getattr__(self, name):
        metadata = SCENE_FIELDS.get(name)
        if metadata is None:
            raise AttributeError(name)
        return getattr(object.__getattribute__(self, metadata.domain_group), name)

    def __setattr__(self, name, value):
        metadata = SCENE_FIELDS.get(name)
        if metadata is not None:
            setattr(object.__getattribute__(self, metadata.domain_group), name, value)
        elif name in SCENE_GROUPS:
            object.__setattr__(self, name, value)
        else:
            raise AttributeError(name)


@dataclass
class PipelineContext:
    user_prompt: str
    run_config: RunConfig = field(default_factory=RunConfig.capture)
    execution_spec: ExecutionSpec | None = None
    artifact_refs: dict[str, str] = field(default_factory=dict)
    original_prompt: str | None = None  # 原始用户提示，用于评估
    paths: RunPaths = field(default_factory=RunPaths.create)
    dry_run: bool = False
    interactive: bool = False
    auto_fix: bool = True
    # 直接构造 PipelineContext 的库调用/旧测试保持严格兼容；正式新运行
    # 会在入口处显式写入 settings.GENERATION_MODE。
    generation_mode: GenerationMode = "strict"
    # 运行时标记：代码/审查 worker 是否需要把共享连续性账本延迟到
    # 所有 worker 完成后再按 Scene ID 发布。不会写入 manifest。
    parallel_generation: bool = False
    # 本次运行固定使用的渲染后端；恢复时只能使用 manifest 中的值。
    backend: RenderBackendName = field(default_factory=lambda: settings.RENDER_BACKEND)
    # 显式 --smoke 可让 dry-run 执行一次本地低质量预检；该开关写入
    # manifest，恢复时继续沿用，不能因 CLI 省略参数而悄悄跳过。
    local_smoke_enabled: bool = False
    # Direct ``render`` runs already contain user-reviewed code.  They must
    # never enter the planning/TechnicalSpec/Reviewer LLM stages, including
    # after a no-wait run is resumed.
    direct_render: bool = False
    approve_plan: bool = False
    plan_approved: bool = False
    outlines: list[SceneOutline] = field(default_factory=list)
    scenes: list[ScenePlan] = field(default_factory=list)
    scene_states: dict[int, SceneState] = field(default_factory=dict)
    lesson_spec: LessonSpec = field(default_factory=LessonSpec)
    teaching_graph: TeachingGraph = field(default_factory=TeachingGraph)
    continuity_bible: ContinuityBible | None = None
    element_manifest: ElementManifest = field(default_factory=ElementManifest)
    state_ledger: StateLedger = field(default_factory=StateLedger)
    expected_final_duration: float | None = None
    plan_review_status: str = "skipped"
    continuity_review_status: str = "passed"
    continuity_review_round: int = 0
    # 保留旧字段供只读诊断；v9 恢复不补发连续性修正预算。
    continuity_resume_recheck_used: bool = False
    continuity_warnings: list[str] = field(default_factory=list)
    fsm_warnings: list[str] = field(default_factory=list)
    # Plan Compiler 的确定性发现按场景缓存，交给 Plan Reviewer 一起处理，
    # 避免每个场景重复计算或把跨场景错误丢失。
    plan_compile_issues: dict[int, list[PlanReviewIssue]] = field(default_factory=dict)
    plan_review_cycle_signature: str = ""
    plan_review_cycle_count: int = 0
    final_video: Path | None = None
    final_video_sha256: str = ""
    render_profile: RenderProfile = field(default_factory=RenderProfile.current)
    merge_profile: MergeProfile = field(default_factory=MergeProfile.current)
    run_phase: RunPhase = "initializing"
    manifest_revision: int = 0
    request_budget: RequestBudget = field(
        default_factory=lambda: RequestBudget(
            scene_limit=settings.MAX_SCENE_LLM_REQUESTS, run_limit=settings.MAX_RUN_LLM_REQUESTS
        )
    )
    dependency_graph: DependencyGraph = field(default_factory=DependencyGraph)
    scene_codegen_mode: str = field(default_factory=lambda: settings.SCENE_CODEGEN_MODE)
    retry_limits: dict[str, int] = field(default_factory=dict)

    # 增量渲染支持
    incremental: bool = False
    base_run_id: str | None = None
    scenes_to_render: list[int] = field(default_factory=list)
    scenes_to_reuse: list[int] = field(default_factory=list)

    # 评估-改进循环支持
    eval_round: int = 0
    scenes_to_improve: list[int] = field(default_factory=list)
    # 渲染阶段上游 AutoFix 改变代码后，需要停止当前渲染批次并重建下游交接。
    continuity_rebuild_required: bool = False
    visual_eval_profile: VisualEvalProfile = field(default_factory=VisualEvalProfile)
    rag_profile: RagRuntimeProfile = field(default_factory=RagRuntimeProfile)
    rag_receipts: dict[str, RagReceipt] = field(default_factory=dict)
    rag_warnings: list[str] = field(default_factory=list)
