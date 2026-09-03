"""有限状态机：串联规划、代码生成、审查、Slurm 渲染、修复与拼接。"""

from __future__ import annotations

import inspect
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from uuid import uuid4

from rich.console import Console
from rich.prompt import Confirm

from kd1_anime.agents.api_linter import lint_manim_api
from kd1_anime.agents.auto_fixer import AutoFixerAgent
from kd1_anime.agents.coder import CoderAgent
from kd1_anime.agents.continuity import (
    ContinuityIssue,
    ContinuityReviewerAgent,
    apply_deterministic_continuity_repairs,
    deterministic_continuity_issues,
    extract_scene_continuity_elements,
    normalize_scene_plan_contract,
    strip_redundant_optional_export_block,
)
from kd1_anime.agents.failure_router import classify_failure
from kd1_anime.agents.lifecycle import (
    repair_required_export_alias_lifecycle,
    validate_animation_lifecycle,
)
from kd1_anime.agents.plan_compiler import PlanCompiler, normalize_scene_timeline_contract
from kd1_anime.agents.plan_reviewer import (
    PlanReviewerAgent,
    PlanReviewIssue,
    PlanReviewResult,
    classify_plan_review_issues,
    dedupe_plan_review_issues,
    deterministic_plan_issues,
)
from kd1_anime.agents.planner import (
    ContinuityBible,
    ElementManifest,
    ExtractedElement,
    LessonSpec,
    PlannerAgent,
    PlanningDraft,
    SceneHandoff,
    SceneOutline,
    ScenePlan,
    TeachingGraph,
    VisualElementState,
    normalize_transition_claim_assignments,
    repair_obvious_math_contradictions,
)
from kd1_anime.agents.reviewer import ReviewerAgent, ReviewFinding, ReviewResult
from kd1_anime.agents.safe_fallback import (
    build_safe_fallback_plan,
    fallback_reason_summary,
    is_high_confidence_geometry_conflict,
)
from kd1_anime.agents.scene_ir import (
    SceneProgramCompileError,
    build_scene_program_from_contract,
    compile_scene_program,
)
from kd1_anime.agents.scene_templates import build_safe_scene_code
from kd1_anime.agents.state_ledger import LedgerElement, StateLedger
from kd1_anime.agents.technical_planner import (
    TechnicalPlannerAgent,
    TechnicalSpec,
    compile_technical_spec,
    normalize_technical_spec_contract,
)
from kd1_anime.agents.validator import CodeValidationResult, validate_manim_code
from kd1_anime.cluster.slurm import (
    FAILURE_STATES,
    JobMonitor,
    SlurmDispatcher,
    SlurmJob,
    SlurmMonitorCoordinator,
)
from kd1_anime.config import resolve_runtime_path, settings
from kd1_anime.exceptions import (
    LLMError,
    LLMResponseError,
    PipelineError,
    RunError,
    RunNotFoundError,
    ValidationError,
)
from kd1_anime.logging import get_logger
from kd1_anime.media.merger import VideoMerger
from kd1_anime.rag.models import RagReceipt, RagRuntimeProfile
from kd1_anime.rag.service import RagService
from kd1_anime.rendering import (
    MergeProfile,
    RenderProfile,
    SceneArtifact,
    effective_transition_duration,
    probe_video,
    sha256_file,
)
from kd1_anime.resources import ResourceCoordinator
from kd1_anime.run_store import (
    MANIFEST_NAME,
    RunManifest,
    RunRepository,
    StoredSceneState,
    StoredVisualCandidate,
    VisualEvalProfile,
    atomic_write_json,
    atomic_write_text,
    get_reusable_video_path,
    is_valid_fsm_transition,
    lock_run,
    restore_run_path,
    restore_slurm_job,
    sha256_text,
    store_slurm_job,
    write_manifest,
)
from kd1_anime.security import redact_jsonable, redact_text, redact_value

logger = get_logger(__name__)
console = Console()
Callback = Callable[[str, dict], None]
MAX_EVENT_DATA_CHARS = 50_000


def _run_limited_process(
    command: list[str],
    *,
    timeout: float,
    memory_mb: int,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    """运行本地预检并在超时后清理整个进程组。

    生成代码可能启动额外子进程；``subprocess.run(timeout=...)`` 只保证
    主进程被终止，容易留下 Manim/FFmpeg 子进程。Linux 的 ``prlimit`` 可
    进一步限制地址空间和 CPU 时间，缺少时仍保留进程组清理兜底。
    """

    wrapped = list(command)
    prlimit = shutil.which("prlimit")
    if prlimit:
        wrapped = [
            prlimit,
            f"--as={int(memory_mb) * 1024 * 1024}",
            f"--cpu={max(1, int(timeout) + 5)}",
            "--",
            *wrapped,
        ]
    try:
        process = subprocess.Popen(
            wrapped,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            start_new_session=True,
        )
    except OSError:
        raise
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        with suppress(OSError, ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                process.kill()
            stdout, stderr = process.communicate()
        raise subprocess.TimeoutExpired(
            wrapped,
            timeout,
            output=stdout,
            stderr=stderr,
        ) from exc
    return subprocess.CompletedProcess(wrapped, process.returncode, stdout, stderr)


# 这些状态通常是代码之外的暂时性基础设施问题；优先重新排队，不要把
# 节点故障/抢占交给 LLM 当成业务代码错误修改。
RETRYABLE_INFRA_STATES = {
    "PREEMPTED",
    "NODE_FAIL",
    "BOOT_FAIL",
    "DEADLINE",
    "REVOKED",
    "SPECIAL_EXIT",
}
FIXABLE_RENDER_STATES = {"FAILED", "TIMEOUT", "OUT_OF_MEMORY", "RUN_TIMEOUT"}


class State(Enum):
    INIT = auto()
    PLANNING = auto()
    DETAILING = auto()
    PLAN_REVIEWING = auto()
    CODING = auto()
    REVIEWING = auto()
    DISPATCHING = auto()
    MONITORING = auto()
    FIXING = auto()
    VISUAL_EVALUATING = auto()
    MERGING = auto()
    EVALUATING = auto()  # 新增：评估状态
    DONE = auto()
    ERROR = auto()


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
class SceneState:
    plan: ScenePlan
    code: str = ""
    class_name: str = ""
    review_round: int = 0
    fix_attempts: int = 0
    infra_retries: int = 0
    slurm_job: SlurmJob | None = None
    artifact: SceneArtifact | None = None
    rendered: bool = False
    give_up: bool = False
    failed: bool = False
    failure_reason: str = ""
    failure_category: str = ""
    safe_fallback_used: bool = False
    safe_fallback_reason: str = ""
    # 导演分镜是否已完成(区别于占位 plan)。场景独立推进时据此判断下一步。
    plan_ready: bool = False
    # Plan Review 与 Code Review 分离：只有前者通过才允许进入 Coder。
    plan_review_round: int = 0
    plan_reviewed: bool = False
    plan_review_feedback: str = ""
    plan_review_signature: str = ""
    identical_plan_review_count: int = 0
    technical_spec: TechnicalSpec | None = None
    technical_spec_sha256: str = ""
    technical_input_sha256: str = ""
    technical_status: str = "pending"
    technical_error: str = ""
    # 本地 Smoke Render 的恢复凭据；未通过或未完成时，恢复/AutoFix 后
    # 必须在 Reviewer 前重新执行，不能只依赖旧的内存状态。
    local_smoke_status: str = "pending"
    # Reviewer major 反馈待重写时暂存; 调度器据此把场景重新排入编码阶段。
    rewrite_feedback: str = ""
    # 代码和 Reviewer 反馈的联合指纹；相同组合重复出现时提前收敛。
    review_signature: str = ""
    identical_review_count: int = 0
    # 渲染错误指纹: 连续相同错误 → 环境问题, 提前放弃
    last_error_fp: str = ""
    identical_error_count: int = 0

    reviewed: bool = False
    # Scene N 的最终导出状态，作为 Scene N+1 的唯一代码级交接来源。
    inherited_elements_code: str = ""
    exported_elements_code: str = ""
    exported_elements: list[ExtractedElement] = field(default_factory=list)
    visual_status: str = "skipped"
    visual_fix_attempts: int = 0
    visual_score: float | None = None
    visual_report_file: str = ""
    visual_report_sha256: str = ""
    visual_artifact_sha256: str = ""
    visual_feedback: str = ""
    visual_best_candidate: VisualCandidate | None = None


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
class PipelineContext:
    user_prompt: str
    original_prompt: str | None = None  # 原始用户提示，用于评估
    paths: RunPaths = field(default_factory=RunPaths.create)
    dry_run: bool = False
    interactive: bool = False
    auto_fix: bool = True
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
    # 连续性 warning 恢复时只自动重新检查一次，避免每次 resume 都重新进入死循环。
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
    manifest_revision: int = 0

    # 增量渲染支持
    incremental: bool = False
    base_run_id: str | None = None
    base_manifest: RunManifest | None = None
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


class Orchestrator:
    def __init__(self, resource_coordinator: ResourceCoordinator | None = None) -> None:
        self.planner = PlannerAgent()
        self.slurm = SlurmDispatcher()
        self.merger = VideoMerger()
        self._callback: Callback | None = None
        self._ctx: PipelineContext | None = None
        self._manifest: RunManifest | None = None
        # 场景级并行调度时多个工作线程会并发写 manifest, 需要串行化
        self._state_lock = threading.RLock()
        self._stop_event = threading.Event()
        self._cancel_requested = threading.Event()
        self._checkpoint_error: Exception | None = None
        self._phase_lock = threading.Lock()
        # 渲染线程在产物完成后即可触发视觉评估；同一时刻只允许一个视觉
        # 门修改连续性状态，避免两个场景同时请求修复造成下游竞态。
        self._visual_eval_lock = threading.RLock()
        self._slurm_monitor: SlurmMonitorCoordinator | None = None
        self._emitted_phases: set[str] = set()
        self._event_log_lock = threading.Lock()
        self._event_log_warning_emitted = False
        self._resource_coordinator = resource_coordinator
        self.rag = RagService(
            rag_semaphore=(resource_coordinator.rag if resource_coordinator is not None else None)
        )

    @staticmethod
    def _configured_visual_profile(*, enabled: bool | None = None) -> VisualEvalProfile:
        use_visual = settings.ENABLE_VISUAL_EVAL if enabled is None else enabled
        model = settings.VISUAL_LLM_MODEL or settings.EVAL_VISUAL_MODEL or ""
        return VisualEvalProfile(
            enabled=use_visual,
            model=model if use_visual else "",
            frame_count=settings.VISUAL_EVAL_FRAME_COUNT,
            threshold=settings.VISUAL_EVAL_THRESHOLD,
            max_fix_attempts=settings.MAX_VISUAL_FIX_ATTEMPTS,
        )

    @staticmethod
    def _reset_visual_receipt(
        ctx: PipelineContext,
        state: SceneState,
        *,
        clear_candidate: bool = False,
        reset_attempts: bool = False,
    ) -> None:
        """使当前视频的视觉收据失效，同时可保留跨修复的最佳候选。"""

        state.visual_status = "pending" if ctx.visual_eval_profile.enabled else "skipped"
        state.visual_score = None
        state.visual_report_file = ""
        state.visual_report_sha256 = ""
        state.visual_artifact_sha256 = ""
        state.visual_feedback = ""
        if clear_candidate:
            state.visual_best_candidate = None
        if reset_attempts:
            state.visual_fix_attempts = 0

    @staticmethod
    def _reset_technical_spec(state: SceneState) -> None:
        """计划或继承上下文变化后使旧 TechnicalSpec 失效。"""

        state.technical_spec = None
        state.technical_spec_sha256 = ""
        state.technical_input_sha256 = ""
        state.technical_status = "pending"
        state.technical_error = ""

    def _cancel_unfinished_scene_job(self, state: SceneState, *, reason: str) -> None:
        """在丢弃场景代码/计划前取消仍可能执行的旧 Job。

        计划重规划、技术合同失效和连续性重建都会清空旧状态。若只把
        ``state.slurm_job`` 设为 ``None``，远端作业仍可能继续写入媒体目录，
        形成孤儿作业或污染下一次提交。终态 Job 不再调用 ``scancel``；未知
        或运行中的 Job 必须确认取消成功，否则抛错并拒绝继续重建。
        """

        with self._state_lock:
            job = state.slurm_job
            if job is None:
                return
            terminal = job.cancelled or job.status in {
                "COMPLETED",
                "CANCELLED",
                *FAILURE_STATES,
            }
            job_id = job.job_id
        if terminal:
            return
        if not self.slurm.cancel_job(job_id):
            raise RuntimeError(
                f"Scene {state.plan.scene_id} 的旧 Job {job_id} {reason}，"
                "取消失败，禁止清空状态并重复提交"
            )
        with self._state_lock:
            if state.slurm_job is job:
                job.cancelled = True
                job.status = "CANCELLED"
                job.failure_reason = f"{reason}，已取消旧作业"

    def _emit(self, event: str, **data) -> None:
        if event in {"scene_failed", "scene_give_up"} and "category" not in data:
            scene_id = data.get("scene_id")
            if self._ctx is not None and scene_id in self._ctx.scene_states:
                category = self._ctx.scene_states[scene_id].failure_category
                if category:
                    data["category"] = category
        safe_data = redact_value(data, self._event_secrets())
        self._append_event(event, safe_data)
        if self._callback:
            self._callback(event, safe_data)

    @staticmethod
    def _event_secrets() -> tuple[str, ...]:
        """返回可能出现在外部服务异常中的凭据值。"""

        return tuple(
            value
            for value in (
                settings.LLM_API_KEY,
                settings.VISUAL_LLM_API_KEY,
                settings.RAG_EMBEDDING_API_KEY,
                settings.RAG_RERANK_API_KEY,
            )
            if value
        )

    def _append_event(self, event: str, data: object, *, durable: bool = False) -> None:
        """将运行事件追加到私有 JSONL 日志；诊断日志失败不改变流水线结果。"""

        ctx = self._ctx
        if ctx is None:
            return
        path = ctx.paths.root / "events.jsonl"
        safe_data = redact_jsonable(data, self._event_secrets())
        encoded_data = json.dumps(safe_data, ensure_ascii=False, separators=(",", ":"))
        if len(encoded_data) > MAX_EVENT_DATA_CHARS:
            safe_data = {
                "truncated": True,
                "sha256": sha256_text(encoded_data),
                "preview": encoded_data[:2_000],
            }
        record = {
            "schema_version": 1,
            "event_id": uuid4().hex,
            "timestamp": datetime.now().astimezone().isoformat(),
            "run_id": ctx.paths.run_id,
            "event": event,
            "data": safe_data,
        }
        try:
            with self._event_log_lock:
                if path.is_symlink():
                    raise OSError(f"事件日志不能是符号链接: {path}")
                path.parent.mkdir(parents=True, exist_ok=True)
                descriptor = os.open(
                    path,
                    os.O_CREAT | os.O_WRONLY | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                )
                try:
                    os.fchmod(descriptor, 0o600)
                    with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
                        descriptor = -1
                        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
                        handle.write("\n")
                        handle.flush()
                        if durable:
                            os.fsync(handle.fileno())
                finally:
                    if descriptor >= 0:
                        os.close(descriptor)
        except (OSError, TypeError, ValueError) as exc:
            with self._event_log_lock:
                should_warn = not self._event_log_warning_emitted
                self._event_log_warning_emitted = True
            if should_warn:
                logger.warning(
                    "运行事件日志写入失败: %s",
                    redact_text(exc, self._event_secrets()),
                )

    @staticmethod
    def _supports_keyword(callable_obj: object, name: str) -> bool:
        """判断可选能力，兼容外部集成和旧测试替身。"""

        try:
            parameters = inspect.signature(callable_obj).parameters
        except (AttributeError, TypeError, ValueError):
            return False
        return name in parameters or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
        )

    def _current_rag_profile(self) -> RagRuntimeProfile:
        if not self.rag.enabled:
            return RagRuntimeProfile()
        runtime = self.rag.runtime_status()
        index = runtime.get("index") or {}
        return RagRuntimeProfile(
            enabled=bool(runtime["enabled"]),
            status=runtime["status"],
            index_sha256=index.get("index_sha256", ""),
            embedding_model=str(runtime.get("embedding_model", "")),
            reranker_model=str(runtime.get("reranker_model", "")),
            top_k=self.rag.config.RAG_TOP_K,
            rerank_top_n=self.rag.config.RAG_RERANK_TOP_N,
            max_context_chars=self.rag.config.RAG_MAX_CONTEXT_CHARS,
            chunker_version=str(index.get("chunker_version", "")),
            chunk_size=int(index.get("chunk_size", 0) or 0),
            chunk_overlap=int(index.get("chunk_overlap", 0) or 0),
        )

    def _retrieve_rag(
        self,
        ctx: PipelineContext,
        query: str,
        *,
        receipt_key: str,
        stage: str,
        source_kinds: set[str] | None = None,
        preferred_source_kinds: set[str] | None = None,
        exclude_frameworks: set[str] | None = None,
        code_sha256: str = "",
        inherited_elements_sha256: str = "",
    ) -> str:
        """获取 RAG 上下文并保存不含密钥的检索收据。"""

        try:
            result = self.rag.search(
                query[:50_000],
                stage=stage,
                source_kinds=source_kinds,
                preferred_source_kinds=preferred_source_kinds,
                exclude_frameworks=exclude_frameworks,
                code_sha256=code_sha256,
                inherited_elements_sha256=inherited_elements_sha256,
            )
        except Exception as exc:
            result = None
            receipt = RagReceipt(
                stage=stage,
                query_sha256=sha256_text(query[:50_000]),
                code_sha256=code_sha256,
                inherited_elements_sha256=inherited_elements_sha256,
                status="degraded",
                warning=f"RAG 检索异常，已跳过: {redact_text(exc, self._event_secrets())}",
            )
        else:
            receipt = result.receipt

        with self._state_lock:
            if ctx.rag_profile.index_sha256 != receipt.index_sha256:
                ctx.rag_receipts.clear()
            ctx.rag_receipts[receipt_key] = receipt
            if len(ctx.rag_receipts) > 256:
                oldest_key = next(iter(ctx.rag_receipts))
                del ctx.rag_receipts[oldest_key]
            ctx.rag_profile = RagRuntimeProfile(
                enabled=self.rag.enabled,
                status=receipt.status,
                index_sha256=receipt.index_sha256,
                embedding_model=self.rag.config.RAG_EMBEDDING_MODEL,
                reranker_model=self.rag.config.RAG_RERANK_MODEL,
                top_k=self.rag.config.RAG_TOP_K,
                rerank_top_n=self.rag.config.RAG_RERANK_TOP_N,
                max_context_chars=self.rag.config.RAG_MAX_CONTEXT_CHARS,
                chunker_version=ctx.rag_profile.chunker_version,
                chunk_size=ctx.rag_profile.chunk_size,
                chunk_overlap=ctx.rag_profile.chunk_overlap,
            )
            if receipt.warning and receipt.warning not in ctx.rag_warnings:
                ctx.rag_warnings.append(f"[{stage}] {receipt.warning}")
        self._emit(
            "rag_status",
            stage=stage,
            status=receipt.status,
            warning=receipt.warning,
            embedding_model=self.rag.config.RAG_EMBEDDING_MODEL,
            reranker_model=self.rag.config.RAG_RERANK_MODEL,
        )
        return result.context if result is not None else ""

    def _reconcile_rag_context(self, ctx: PipelineContext) -> None:
        """恢复时发现索引/模型变化就丢弃旧收据，后续阶段重新检索。"""

        current = self._current_rag_profile()
        previous = ctx.rag_profile
        changed = (
            previous.enabled != current.enabled
            or previous.index_sha256 != current.index_sha256
            or previous.embedding_model != current.embedding_model
            or previous.reranker_model != current.reranker_model
            or previous.chunker_version != current.chunker_version
            or previous.chunk_size != current.chunk_size
            or previous.chunk_overlap != current.chunk_overlap
        )
        if changed:
            ctx.rag_receipts.clear()
            ctx.rag_warnings.append("恢复运行：RAG 索引或 Embedding 模型已变化，将重新检索")
        ctx.rag_profile = current

    def cancel_all(self) -> None:
        self._cancel_requested.set()
        self._stop_event.set()
        if not self._ctx:
            return
        with self._state_lock:
            states = list(self._ctx.scene_states.values())
        for state in states:
            job = state.slurm_job
            if (
                job
                and not state.rendered
                and not job.cancelled
                and job.status not in {"COMPLETED", "CANCELLED", *FAILURE_STATES}
                and self.slurm.cancel_job(job.job_id)
            ):
                with self._state_lock:
                    job.cancelled = True
                    job.status = "CANCELLED"
                    job.failure_reason = "本地流水线停止时已取消远端任务"

    def _ask_retry_or_skip(self, scene_id: int, error: str) -> bool:
        if not self._ctx or not self._ctx.interactive:
            return False
        # 交互式询问前暂停 Live 仪表盘，避免输入冲突
        from kd1_anime.dashboard import suspend_all

        with suspend_all():
            console.print(
                f"Scene {scene_id} 操作失败：\n{error}",
                markup=False,
                style="yellow",
            )
            try:
                return Confirm.ask("再重试一次？", default=False, console=console)
            except (EOFError, KeyboardInterrupt):
                return False

    @staticmethod
    def _mark_failed(state: SceneState, reason: str, category: str = "system") -> None:
        state.failed = True
        state.failure_reason = reason
        state.failure_category = category

    @staticmethod
    def _validate(code: str, *, renderer: str | None = None) -> CodeValidationResult:
        return validate_manim_code(code, renderer=renderer)

    @staticmethod
    def _write_private(path: Path, content: str) -> None:
        atomic_write_text(path, content, mode=0o600)

    def _write_stage_artifact(
        self,
        ctx: PipelineContext,
        name: str,
        payload: object,
    ) -> None:
        """写入阶段诊断快照；诊断文件失败不能改变流水线语义。"""

        try:
            artifacts_dir = ctx.paths.root / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            artifacts_dir.chmod(0o700)
            atomic_write_json(artifacts_dir / name, payload)
        except (OSError, TypeError, ValueError) as exc:
            with self._state_lock:
                ctx.continuity_warnings.append(f"阶段产物 {name} 写入失败: {exc}")

    def _store_visual_candidate(
        self,
        ctx: PipelineContext,
        candidate: VisualCandidate,
    ) -> StoredVisualCandidate:
        code_hash = sha256_text(candidate.code)
        code_path = (
            ctx.paths.root
            / "visual_candidates"
            / f"scene_{candidate.artifact.scene_id}"
            / f"code_{code_hash}.py"
        )
        if (
            not code_path.is_file()
            or sha256_text(code_path.read_text(encoding="utf-8")) != code_hash
        ):
            self._write_private(code_path, candidate.code)
        code_relative = code_path.resolve().relative_to(ctx.paths.root.resolve()).as_posix()
        return StoredVisualCandidate(
            score=candidate.score,
            has_major_issue=candidate.has_major_issue,
            passed=candidate.passed,
            inherited_elements_sha256=candidate.inherited_elements_sha256,
            code_file=code_relative,
            code_sha256=code_hash,
            class_name=candidate.class_name,
            slurm_job=(
                store_slurm_job(candidate.slurm_job, ctx.paths.root)
                if candidate.slurm_job is not None
                else None
            ),
            artifact=candidate.artifact,
            exported_elements_code=candidate.exported_elements_code,
            exported_elements=candidate.exported_elements,
            report_file=candidate.report_file,
            report_sha256=candidate.report_sha256,
        )

    def _checkpoint(
        self,
        ctx: PipelineContext,
        state: State,
        *,
        status: str = "running",
        error: str = "",
    ) -> None:
        with self._state_lock:
            previous_state = self._manifest.state if self._manifest is not None else ""
            transition_warning = ""
            if previous_state and not is_valid_fsm_transition(previous_state, state.name):
                transition_warning = (
                    f"FSM 检查点转移 {previous_state} -> {state.name} 不在已知转移表中；"
                    "考虑到并发 worker/恢复入口，仅记录诊断并继续"
                )
                if transition_warning not in ctx.fsm_warnings:
                    ctx.fsm_warnings.append(transition_warning)
            scenes: dict[int, StoredSceneState] = {}
            for scene_id, scene in sorted(ctx.scene_states.items()):
                code_file = ""
                code_hash = ""
                if scene.code:
                    code_path = ctx.paths.scenes / f"scene_{scene_id}.py"
                    code_file = code_path.resolve().relative_to(ctx.paths.root.resolve()).as_posix()
                    code_hash = sha256_text(scene.code)
                scenes[scene_id] = StoredSceneState(
                    plan=scene.plan,
                    code_file=code_file,
                    code_sha256=code_hash,
                    class_name=scene.class_name,
                    review_round=scene.review_round,
                    fix_attempts=scene.fix_attempts,
                    infra_retries=scene.infra_retries,
                    slurm_job=(
                        store_slurm_job(scene.slurm_job, ctx.paths.root)
                        if scene.slurm_job
                        else None
                    ),
                    artifact=scene.artifact,
                    phase=self._scene_phase(scene),
                    reviewed=scene.reviewed,
                    rendered=scene.rendered,
                    give_up=scene.give_up,
                    failed=scene.failed,
                    failure_reason=scene.failure_reason,
                    failure_category=scene.failure_category,
                    safe_fallback_used=scene.safe_fallback_used,
                    safe_fallback_reason=scene.safe_fallback_reason,
                    plan_ready=scene.plan_ready,
                    plan_review_round=scene.plan_review_round,
                    plan_reviewed=scene.plan_reviewed,
                    plan_review_feedback=scene.plan_review_feedback,
                    plan_review_signature=scene.plan_review_signature,
                    identical_plan_review_count=scene.identical_plan_review_count,
                    technical_spec=scene.technical_spec,
                    technical_spec_sha256=scene.technical_spec_sha256,
                    technical_input_sha256=scene.technical_input_sha256,
                    technical_status=scene.technical_status,
                    technical_error=scene.technical_error,
                    local_smoke_status=scene.local_smoke_status,
                    rewrite_feedback=scene.rewrite_feedback,
                    review_signature=scene.review_signature,
                    identical_review_count=scene.identical_review_count,
                    last_error_fp=scene.last_error_fp,
                    identical_error_count=scene.identical_error_count,
                    inherited_elements_code=scene.inherited_elements_code,
                    exported_elements_code=scene.exported_elements_code,
                    exported_elements=scene.exported_elements,
                    visual_status=scene.visual_status,
                    visual_fix_attempts=scene.visual_fix_attempts,
                    visual_score=scene.visual_score,
                    visual_report_file=scene.visual_report_file,
                    visual_report_sha256=scene.visual_report_sha256,
                    visual_artifact_sha256=scene.visual_artifact_sha256,
                    visual_feedback=scene.visual_feedback,
                    visual_best_candidate=(
                        self._store_visual_candidate(ctx, scene.visual_best_candidate)
                        if scene.visual_best_candidate is not None
                        else None
                    ),
                )
            ctx.manifest_revision = (
                max(
                    ctx.manifest_revision,
                    self._manifest.revision if self._manifest else 0,
                )
                + 1
            )
            manifest = RunManifest(
                revision=ctx.manifest_revision,
                run_id=ctx.paths.run_id,
                created_at=(
                    self._manifest.created_at if self._manifest else datetime.now().astimezone()
                ),
                status=status,
                state=state.name,
                user_prompt=ctx.user_prompt,
                dry_run=ctx.dry_run,
                interactive=ctx.interactive,
                auto_fix=ctx.auto_fix,
                direct_render=ctx.direct_render,
                approve_plan=ctx.approve_plan,
                plan_approved=ctx.plan_approved,
                output_path=str(ctx.paths.output),
                render_profile=ctx.render_profile,
                merge_profile=ctx.merge_profile,
                outlines=ctx.outlines,
                scenes=scenes,
                lesson_spec=ctx.lesson_spec,
                teaching_graph=ctx.teaching_graph,
                continuity_bible=ctx.continuity_bible,
                element_manifest=ctx.element_manifest,
                state_ledger=ctx.state_ledger,
                expected_final_duration=ctx.expected_final_duration,
                plan_review_status=ctx.plan_review_status,
                plan_review_cycle_signature=ctx.plan_review_cycle_signature,
                plan_review_cycle_count=ctx.plan_review_cycle_count,
                continuity_review_status=ctx.continuity_review_status,
                continuity_review_round=ctx.continuity_review_round,
                continuity_resume_recheck_used=ctx.continuity_resume_recheck_used,
                continuity_warnings=ctx.continuity_warnings[-100:],
                fsm_warnings=ctx.fsm_warnings[-100:],
                final_video=str(ctx.final_video) if ctx.final_video else None,
                final_video_sha256=ctx.final_video_sha256,
                error=error[-50_000:],
                incremental=ctx.incremental,
                base_run_id=ctx.base_run_id,
                eval_round=ctx.eval_round,
                continuity_rebuild_required=ctx.continuity_rebuild_required,
                visual_eval_profile=ctx.visual_eval_profile,
                rag_profile=ctx.rag_profile,
                rag_receipts=dict(ctx.rag_receipts),
                rag_warnings=ctx.rag_warnings[-100:],
            )
            write_manifest(ctx.paths.root / MANIFEST_NAME, manifest)
            self._manifest = manifest
            self._append_event(
                "fsm_checkpoint",
                {
                    "state": state.name,
                    "status": status,
                    "revision": manifest.revision,
                    "previous_state": previous_state,
                    "transition_warning": transition_warning,
                },
                durable=True,
            )

    def _record_checkpoint_failure(self, error: Exception) -> None:
        """记录持久化故障并停止其它 worker，避免继续推进未保存的状态。"""

        with self._state_lock:
            if self._checkpoint_error is None:
                self._checkpoint_error = error
        self._stop_event.set()
        self._emit("checkpoint_failed", error=str(error))

    def _checkpoint_slurm_job_update(self, ctx: PipelineContext, _job: SlurmJob) -> None:
        """集中监控器发现实际启动时间变化时安全持久化。"""

        try:
            self._checkpoint(ctx, State.MONITORING)
        except Exception as exc:
            self._record_checkpoint_failure(exc)

    @staticmethod
    def _scene_phase(scene: SceneState) -> str:
        if scene.failed or scene.give_up:
            return "failed"
        if scene.visual_status == "evaluating":
            return "visual_evaluating"
        if scene.rendered and scene.visual_status in {"passed", "warning", "unknown"}:
            return "visual_accepted"
        if scene.rendered and scene.artifact:
            return "rendered"
        if scene.slurm_job:
            return "monitoring"
        if scene.plan_reviewed:
            if scene.technical_status in {"generating", "failed"}:
                return "technical_planning" if scene.technical_status == "generating" else "failed"
            if scene.technical_status != "passed" and not scene.code:
                return "technical_planning"
            if scene.reviewed:
                return "reviewed"
            if scene.code:
                return "coded"
            return "plan_reviewed"
        if scene.plan_ready and not scene.plan_reviewed:
            return "plan_reviewing"
        if scene.reviewed:
            return "reviewed"
        if scene.code:
            return "coded"
        if scene.plan_ready:
            return "detailed"
        return "pending"

    @staticmethod
    def _restore_visual_candidate(
        stored: StoredVisualCandidate | None,
        root: Path,
    ) -> VisualCandidate | None:
        if stored is None:
            return None
        code_path = restore_run_path(root, stored.code_file)
        if code_path.is_symlink() or not code_path.is_file():
            raise ValueError("最佳视觉候选代码文件不存在或不安全")
        code = code_path.read_text(encoding="utf-8")
        if sha256_text(code) != stored.code_sha256:
            raise ValueError("最佳视觉候选代码哈希不一致")
        report_path = restore_run_path(root, stored.report_file)
        if report_path.is_symlink() or not report_path.is_file():
            raise ValueError("最佳视觉候选报告不存在或不安全")
        if sha256_file(report_path) != stored.report_sha256:
            raise ValueError("最佳视觉候选报告哈希不一致")
        return VisualCandidate(
            score=stored.score,
            has_major_issue=stored.has_major_issue,
            passed=stored.passed,
            inherited_elements_sha256=stored.inherited_elements_sha256,
            code=code,
            class_name=stored.class_name,
            slurm_job=(restore_slurm_job(stored.slurm_job, root) if stored.slurm_job else None),
            artifact=stored.artifact,
            exported_elements_code=stored.exported_elements_code,
            exported_elements=list(stored.exported_elements),
            report_file=stored.report_file,
            report_sha256=stored.report_sha256,
        )

    @staticmethod
    def _context_from_manifest(manifest: RunManifest, root: Path) -> PipelineContext:
        root = root.resolve()
        output = Path(manifest.output_path).expanduser()
        if not output.is_absolute():
            raise ValueError("manifest.output_path 必须是绝对路径")
        final_video = Path(manifest.final_video).expanduser() if manifest.final_video else None
        if final_video is not None and not final_video.is_absolute():
            raise ValueError("manifest.final_video 必须是绝对路径")
        if final_video is not None and final_video.resolve() != output.resolve():
            raise ValueError("manifest.final_video 必须与 manifest.output_path 一致")
        paths = RunPaths(
            run_id=manifest.run_id,
            root=root,
            scenes=root / "scenes",
            logs=root / "logs",
            videos=root / "videos",
            output=output,
        )
        scene_states: dict[int, SceneState] = {}
        for scene_id, stored in manifest.scenes.items():
            code = ""
            if stored.code_file:
                code_path = restore_run_path(root, stored.code_file)
                expected_code_path = (root / "scenes" / f"scene_{scene_id}.py").resolve()
                if code_path != expected_code_path:
                    raise ValueError(f"Scene {scene_id} 的代码路径不是规范场景路径")
                if not code_path.is_file() or code_path.is_symlink():
                    raise ValueError(f"Scene {scene_id} 的代码文件不存在或不安全")
                code = code_path.read_text(encoding="utf-8")
                if sha256_text(code) != stored.code_sha256:
                    raise ValueError(f"Scene {scene_id} 的代码哈希与运行清单不一致")
            job = restore_slurm_job(stored.slurm_job, root) if stored.slurm_job else None
            if job is not None:
                if not code or not stored.code_sha256:
                    raise ValueError(f"Scene {scene_id} 存在 Slurm Job 但缺少代码身份")
                if job.scene_id != scene_id or job.scene_class_name != stored.class_name:
                    raise ValueError(f"Scene {scene_id} 的 Slurm Job 场景身份不一致")
                if job.code_sha256 != stored.code_sha256:
                    raise ValueError(f"Scene {scene_id} 的 Slurm Job 代码哈希不一致")
                if job.render_profile.digest() != manifest.render_profile.digest():
                    raise ValueError(f"Scene {scene_id} 的 Slurm Job 渲染配置不一致")
            artifact = stored.artifact
            if artifact is not None:
                if (
                    artifact.scene_id != scene_id
                    or artifact.scene_class_name != stored.class_name
                    or artifact.code_sha256 != stored.code_sha256
                ):
                    raise ValueError(f"Scene {scene_id} 的渲染产物身份不一致")
                if artifact.render_profile_sha256 != manifest.render_profile.digest():
                    raise ValueError(f"Scene {scene_id} 的渲染产物配置不一致")
            if stored.rendered and artifact is None:
                raise ValueError(f"Scene {scene_id} 标记为已渲染但缺少产物凭据")
            visual_status = stored.visual_status
            visual_score = stored.visual_score
            visual_report_file = stored.visual_report_file
            visual_report_sha256 = stored.visual_report_sha256
            visual_artifact_sha256 = stored.visual_artifact_sha256
            visual_feedback = stored.visual_feedback
            if visual_status == "evaluating":
                # 本地 LLM 请求没有可恢复的远端 Job；中断后重新评估同一视频。
                visual_status = "pending"
                visual_score = None
                visual_report_file = ""
                visual_report_sha256 = ""
                visual_artifact_sha256 = ""
                visual_feedback = ""
            if visual_report_file:
                report_path = restore_run_path(root, visual_report_file)
                if report_path.is_symlink() or not report_path.is_file():
                    raise ValueError(f"Scene {scene_id} 的视觉报告不存在或不安全")
                if sha256_file(report_path) != visual_report_sha256:
                    raise ValueError(f"Scene {scene_id} 的视觉报告哈希不一致")
            if visual_status in {"passed", "warning", "unknown"}:
                if artifact is None or not visual_report_file:
                    raise ValueError(f"Scene {scene_id} 的视觉终态缺少视频或评估报告")
                if visual_artifact_sha256 != artifact.video_sha256:
                    # 收据不属于当前视频时不能复用，也不把它伪装成恢复失败；
                    # 清空当前收据，让视觉门在合并前重新评估精确产物。
                    visual_status = "pending"
                    visual_score = None
                    visual_report_file = ""
                    visual_report_sha256 = ""
                    visual_artifact_sha256 = ""
                    visual_feedback = ""
            scene_states[scene_id] = SceneState(
                plan=stored.plan,
                code=code,
                class_name=stored.class_name,
                review_round=stored.review_round,
                fix_attempts=stored.fix_attempts,
                infra_retries=getattr(stored, "infra_retries", 0),
                slurm_job=job,
                artifact=artifact,
                reviewed=stored.reviewed,
                rendered=stored.rendered,
                give_up=stored.give_up,
                failed=stored.failed,
                failure_reason=stored.failure_reason,
                failure_category=getattr(stored, "failure_category", ""),
                safe_fallback_used=getattr(stored, "safe_fallback_used", False),
                safe_fallback_reason=getattr(stored, "safe_fallback_reason", ""),
                plan_ready=getattr(stored, "plan_ready", False),
                plan_review_round=getattr(stored, "plan_review_round", 0),
                plan_reviewed=getattr(
                    stored,
                    "plan_reviewed",
                    stored.reviewed or stored.rendered,
                ),
                plan_review_feedback=getattr(stored, "plan_review_feedback", ""),
                plan_review_signature=getattr(stored, "plan_review_signature", ""),
                identical_plan_review_count=getattr(stored, "identical_plan_review_count", 0),
                technical_spec=getattr(stored, "technical_spec", None),
                technical_spec_sha256=getattr(stored, "technical_spec_sha256", ""),
                technical_input_sha256=getattr(stored, "technical_input_sha256", ""),
                technical_status=getattr(stored, "technical_status", "pending"),
                technical_error=getattr(stored, "technical_error", ""),
                local_smoke_status=getattr(stored, "local_smoke_status", "pending"),
                rewrite_feedback=getattr(stored, "rewrite_feedback", ""),
                review_signature=getattr(stored, "review_signature", ""),
                identical_review_count=getattr(stored, "identical_review_count", 0),
                last_error_fp=getattr(stored, "last_error_fp", ""),
                identical_error_count=getattr(stored, "identical_error_count", 0),
                inherited_elements_code=getattr(stored, "inherited_elements_code", ""),
                exported_elements_code=getattr(stored, "exported_elements_code", ""),
                exported_elements=list(getattr(stored, "exported_elements", [])),
                visual_status=visual_status,
                visual_fix_attempts=stored.visual_fix_attempts,
                visual_score=visual_score,
                visual_report_file=visual_report_file,
                visual_report_sha256=visual_report_sha256,
                visual_artifact_sha256=visual_artifact_sha256,
                visual_feedback=visual_feedback,
                visual_best_candidate=Orchestrator._restore_visual_candidate(
                    stored.visual_best_candidate, root
                ),
            )
        direct_render = getattr(manifest, "direct_render", False) or (
            not manifest.auto_fix
            and manifest.plan_review_status == "skipped"
            and len(manifest.scenes) == 1
            and manifest.user_prompt.startswith("Direct render of Scene class ")
        )
        context = PipelineContext(
            user_prompt=manifest.user_prompt,
            original_prompt=manifest.user_prompt,
            paths=paths,
            dry_run=manifest.dry_run,
            interactive=manifest.interactive,
            auto_fix=manifest.auto_fix,
            direct_render=direct_render,
            approve_plan=getattr(manifest, "approve_plan", False),
            plan_approved=getattr(manifest, "plan_approved", False),
            outlines=manifest.outlines,
            scenes=[scene_states[key].plan for key in sorted(scene_states)],
            scene_states=scene_states,
            lesson_spec=getattr(manifest, "lesson_spec", LessonSpec()),
            teaching_graph=getattr(manifest, "teaching_graph", TeachingGraph()),
            continuity_bible=manifest.continuity_bible,
            element_manifest=getattr(manifest, "element_manifest", ElementManifest()),
            state_ledger=getattr(manifest, "state_ledger", StateLedger()),
            expected_final_duration=getattr(manifest, "expected_final_duration", None),
            plan_review_status=(
                "pending"
                if getattr(manifest, "plan_review_status", "skipped") in {"passed", "skipped"}
                and any(
                    state.plan_ready
                    and not getattr(state, "plan_reviewed", state.reviewed or state.rendered)
                    and not state.failed
                    and not state.give_up
                    for state in manifest.scenes.values()
                )
                else getattr(manifest, "plan_review_status", "skipped")
            ),
            plan_review_cycle_signature=getattr(manifest, "plan_review_cycle_signature", ""),
            plan_review_cycle_count=getattr(manifest, "plan_review_cycle_count", 0),
            continuity_review_status=manifest.continuity_review_status,
            continuity_review_round=manifest.continuity_review_round,
            continuity_resume_recheck_used=getattr(
                manifest, "continuity_resume_recheck_used", False
            ),
            continuity_warnings=list(manifest.continuity_warnings),
            fsm_warnings=list(getattr(manifest, "fsm_warnings", [])),
            final_video=final_video,
            final_video_sha256=manifest.final_video_sha256,
            incremental=manifest.incremental,
            base_run_id=manifest.base_run_id,
            render_profile=manifest.render_profile,
            merge_profile=getattr(manifest, "merge_profile", MergeProfile.current()),
            manifest_revision=manifest.revision,
            eval_round=manifest.eval_round,
            continuity_rebuild_required=getattr(manifest, "continuity_rebuild_required", False),
            visual_eval_profile=manifest.visual_eval_profile,
            rag_profile=getattr(manifest, "rag_profile", RagRuntimeProfile()),
            rag_receipts=dict(getattr(manifest, "rag_receipts", {})),
            rag_warnings=list(getattr(manifest, "rag_warnings", [])),
        )
        if manifest.incremental and manifest.base_run_id:
            context.base_manifest = RunRepository(settings.WORKSPACE_DIR).load(manifest.base_run_id)
            context.base_manifest.validate_for_resume()
        if context.expected_final_duration is None and context.outlines:
            transition = effective_transition_duration(
                (item.duration_seconds for item in context.outlines), context.merge_profile
            )
            context.expected_final_duration = max(
                0.0,
                sum(item.duration_seconds for item in context.outlines)
                - transition * max(0, len(context.outlines) - 1),
            )
        return context

    def _generate_validated_code(
        self,
        plan: ScenePlan,
        *,
        feedback: str = "",
        previous_code: str = "",
        stream: bool = False,
        renderer: str | None = None,
        continuity_bible: ContinuityBible | None = None,
        inherited_elements_code: str = "",
        inherited_elements: list[VisualElementState] | None = None,
        elements_to_remove: list[VisualElementState] | None = None,
        element_manifest: ElementManifest | None = None,
        technical_spec: TechnicalSpec | None = None,
        rag_context: str = "",
        lesson_spec: LessonSpec | None = None,
        teaching_graph: TeachingGraph | None = None,
    ) -> tuple[str, str]:
        agent = CoderAgent()
        current_feedback = feedback
        current_previous = previous_code
        last_validation: CodeValidationResult | None = None
        last_continuity_error = ""
        last_lifecycle_error = ""
        last_api_errors: tuple[str, ...] = ()
        if technical_spec is not None:
            technical_result = compile_technical_spec(
                plan,
                technical_spec,
                renderer=renderer,
            )
            if not technical_result.is_valid:
                raise ValidationError(
                    "TechnicalSpec 未通过确定性编译：\n"
                    + "\n".join(f"- {error}" for error in technical_result.errors),
                    hint="请重新生成 TechnicalSpec，不要直接修改代码绕过技术合同",
                )
        max_validation_attempts = settings.CODE_VALIDATION_ATTEMPTS
        for attempt in range(1, max_validation_attempts + 1):
            code_kwargs = {
                "feedback": current_feedback,
                "previous_code": current_previous,
                "stream": stream,
                "renderer": renderer,
            }
            if continuity_bible is not None:
                code_kwargs["continuity_bible"] = continuity_bible
            if inherited_elements_code:
                code_kwargs["inherited_elements_code"] = inherited_elements_code
            if inherited_elements:
                code_kwargs["inherited_elements"] = inherited_elements
            if elements_to_remove:
                code_kwargs["elements_to_remove"] = elements_to_remove
            if element_manifest is not None and self._supports_keyword(
                agent.generate_code, "element_manifest"
            ):
                code_kwargs["element_manifest"] = element_manifest
            if technical_spec is not None and self._supports_keyword(
                agent.generate_code, "technical_spec"
            ):
                code_kwargs["technical_spec"] = technical_spec
            if rag_context and self._supports_keyword(agent.generate_code, "rag_context"):
                code_kwargs["rag_context"] = rag_context
            if lesson_spec is not None and self._supports_keyword(
                agent.generate_code, "lesson_spec"
            ):
                code_kwargs["lesson_spec"] = lesson_spec
            if teaching_graph is not None and self._supports_keyword(
                agent.generate_code, "teaching_graph"
            ):
                code_kwargs["teaching_graph"] = teaching_graph
            code = agent.generate_code(
                plan,
                **code_kwargs,
            )
            if technical_spec is not None:
                code, lifecycle_repairs = repair_required_export_alias_lifecycle(
                    code,
                    technical_spec,
                )
                if lifecycle_repairs:
                    log = getattr(agent, "_log", None)
                    if callable(log):
                        log(
                            "已应用确定性生命周期兼容修复: " + "；".join(lifecycle_repairs),
                            style="yellow",
                        )
            code = strip_redundant_optional_export_block(code, plan)

            validation = self._validate(code, renderer=renderer)
            api_result = lint_manim_api(code, renderer=renderer, scene_plan=plan)
            continuity_error = ""
            try:
                extract_scene_continuity_elements(code, plan)
            except ValueError as exc:
                continuity_error = str(exc)
            lifecycle_error = ""
            if technical_spec is not None and validation.is_valid and not continuity_error:
                lifecycle_result = validate_animation_lifecycle(
                    code,
                    technical_spec,
                    renderer=renderer,
                )
                if not lifecycle_result.is_valid:
                    lifecycle_error = "\n".join(lifecycle_result.errors)
            if (
                validation.is_valid
                and api_result.is_valid
                and not continuity_error
                and not lifecycle_error
            ):
                return code, validation.scene_classes[0]
            last_validation = validation
            last_api_errors = api_result.errors
            last_continuity_error = continuity_error
            last_lifecycle_error = lifecycle_error
            # 提供详细的修复指导
            feedback_parts = [
                f"上一候选是第 {attempt}/{max_validation_attempts} 次尝试，未通过确定性校验；"
                f"现在进行第 {min(attempt + 1, max_validation_attempts)}/{max_validation_attempts} "
                "次修复。不得原样返回上一候选代码，必须针对下面的确定性错误做最小修改：\n"
                f"{validation.feedback}"
            ]
            if api_result.errors:
                feedback_parts.append(
                    "\nManim API 静态检查未通过，必须先修复：\n- " + "\n- ".join(api_result.errors)
                )
            if current_previous and code == current_previous:
                feedback_parts.append(
                    "\n上一候选代码与本次输出完全相同，说明上一次修复没有生效；"
                    "请改变实现结构，不要再次复制同一份代码。\n"
                )
            if continuity_error:
                feedback_parts.append(
                    "\n连续性导出合同校验未通过，必须修复以下问题：\n- " + continuity_error
                )
                if "不能包含已移除元素" in continuity_error:
                    feedback_parts.append(
                        "\n移除元素的修复方式：该元素仍需在 construct() 开头定义并用于"
                        " FadeOut，但它的完整定义和 element_id 标记必须移到"
                        " KD1_CONTINUITY_EXPORT_BEGIN 之前；导出区只能保留结束时"
                        "仍存在的 required=true 元素。不要把已移除元素留在 marker 内，"
                        "也不要为了删除 marker 而删除场景中的 FadeOut。\n"
                    )
                feedback_parts.append(
                    "\n导出区修复规则：只允许导出 required=true 的边界元素；"
                    "required=false 的对象只能作为场景内部对象，不能出现在 marker 内。"
                    "如果没有 required=true 元素，保留空的导出区或删除 marker；"
                    "tex_template/COLORS/FONTS 等场景上下文必须放在 marker 之前。\n"
                )
            if lifecycle_error:
                feedback_parts.append(
                    "\n动画生命周期校验未通过，必须修复以下问题：\n- " + lifecycle_error
                )
                if "重定义仍处于 active 的对象" in lifecycle_error:
                    feedback_parts.append(
                        "\n生命周期修复规则：导出区只能有一个；继承且需要继续交接的对象只能定义一次，"
                        "并且应放在唯一的 KD1_CONTINUITY_EXPORT_BEGIN/END 区内；"
                        "不要先在 marker 外复制继承代码，再在 marker 内重新定义，"
                        "也不要在动画结束位置再次重建同名对象。需要淡出的继承元素"
                        "才定义在 marker 外，并保留唯一的 FadeOut。\n"
                    )
                if "必须导出的对象不 active" in lifecycle_error:
                    feedback_parts.append(
                        "\n导出对象激活规则：required=true 的导出变量必须在动画流程中"
                        "使用 Create、Write、FadeIn 等 introducer 实际引入，并在结尾"
                        "保持 active；不要只定义该变量或只让带 _initial、_shrunk 等"
                        "后缀的临时变量参与动画。若使用 Transform 阶段目标，需保证"
                        "最终仍由合同中的 variable_name 对象承接。\n"
                    )
                if "animate 作用于未 active 对象" in lifecycle_error:
                    feedback_parts.append(
                        "\n本次错误通常表示把 required 导出变量和 *_initial 临时变量混用了。"
                        "请把首次 FadeIn/Create/Write 和后续所有 animate/Transform 的 source"
                        "都改为同一个合同变量；例如直接 FadeIn(v1) 后再执行"
                        "v1.animate.scale(...)。不要用 `v1 = v1_initial` 代替 introducer；"
                        "若必须从临时对象交接到 v1，只能使用 ReplacementTransform(v1_initial, v1)，"
                        "并删除之后对 v1_initial 的动画。\n"
                    )
                if "Transform 的 source 未 active" in lifecycle_error:
                    feedback_parts.append(
                        "\nTransform 源对象修复规则：VGroup 本身只有在整体通过 self.add、"
                        "FadeIn/Create 等方式引入后才是 active；单独引入它的子对象不等于"
                        "引入 VGroup。请不要先 FadeIn 子对象再 Transform 一个后创建的 group，"
                        "应整体引入该 group，或改为对已经 active 的子对象分别执行动画。\n"
                    )

            # 如果是 TexTemplate 相关错误，提供正确示例
            if any("TexTemplate" in err or "tex_template" in err for err in validation.errors):
                feedback_parts.append("""
\n=== 正确的 TexTemplate 配置示例 ===
你必须在 construct() 方法开头添加以下代码：

    tex_template = TexTemplate(tex_compiler="xelatex", output_format=".xdv")
    tex_template.add_to_preamble(r"\\usepackage{ctex}")
    config.tex_template = tex_template

然后每个 Tex/MathTex 调用都必须传入 tex_template=tex_template：

    eq = MathTex(r"\\frac{a}{b}", tex_template=tex_template)
    text = Tex(r"中文文本", tex_template=tex_template)
""")

            current_feedback = "".join(feedback_parts)
            current_previous = code
        raise ValidationError(
            "生成代码未通过确定性校验：\n"
            + (
                "\n".join(
                    part
                    for part in (
                        last_validation.feedback if last_validation else "",
                        f"连续性导出合同错误: {last_continuity_error}"
                        if last_continuity_error
                        else "",
                        f"动画生命周期错误: {last_lifecycle_error}" if last_lifecycle_error else "",
                        "Manim API 静态检查错误: " + "; ".join(last_api_errors)
                        if last_api_errors
                        else "",
                    )
                    if part
                )
                or "未知错误"
            ),
            hint="尝试简化场景或调整 prompt",
        )

    def run_incremental(
        self,
        user_prompt: str,
        base_run_id: str,
        callback: Callback | None = None,
        dry_run: bool = False,
        interactive: bool = False,
        output_path: Path | None = None,
        approve_plan: bool = False,
    ) -> Path | None:
        """增量渲染：只重新渲染受 prompt 变化影响的场景。"""
        # CLI 会在进入流水线前做真实网络探测；库调用方至少也必须通过
        # 同一份配置完整性检查，避免运行几分钟后才因空 Key 失败。
        settings.require_llm_key()
        if settings.ENABLE_VISUAL_EVAL and not dry_run:
            settings.require_visual_llm()
        if self.rag.enabled:
            self.rag.require_ready()
        if len(user_prompt) > settings.MAX_PROMPT_CHARS:
            raise ValueError(
                f"用户需求过长：{len(user_prompt)} 字符，最大允许 {settings.MAX_PROMPT_CHARS} 字符\n"
                f"提示：可以将需求拆分为多个较短的动画，或使用更简洁的描述"
            )

        # 加载基础运行的 manifest
        repository = RunRepository(settings.WORKSPACE_DIR)
        try:
            base_manifest = repository.load(base_run_id)
        except (OSError, ValueError) as exc:
            raise RunNotFoundError(f"无法加载基础运行 {base_run_id}: {exc}") from exc
        base_manifest.validate_for_resume()

        if base_manifest.status not in ("completed", "dry_run_complete"):
            raise RunError(f"基础运行 {base_run_id} 未完成（状态：{base_manifest.status}）")

        self._callback = callback
        self._manifest = None
        self._cancel_requested.clear()
        self._stop_event.clear()
        ctx = PipelineContext(
            user_prompt=user_prompt,
            original_prompt=user_prompt,
            dry_run=dry_run,
            interactive=interactive,
            incremental=True,
            approve_plan=approve_plan,
            base_run_id=base_run_id,
            base_manifest=base_manifest,
            paths=RunPaths.create(output_path),
            visual_eval_profile=self._configured_visual_profile(
                enabled=settings.ENABLE_VISUAL_EVAL and not dry_run
            ),
            rag_profile=self._current_rag_profile(),
        )
        self._ctx = ctx
        ctx.paths.root.mkdir(parents=True, exist_ok=True)
        ctx.paths.root.chmod(0o700)

        with lock_run(ctx.paths.root):
            return self._execute(ctx, State.INIT)

    def run(
        self,
        user_prompt: str,
        callback: Callback | None = None,
        dry_run: bool = False,
        interactive: bool = False,
        output_path: Path | None = None,
        approve_plan: bool = False,
    ) -> Path | None:
        # 保持 programmatic API 与 CLI 的配置门槛一致。网络可用性由 CLI
        # 的启动探测负责，底层 Agent 仍会在真正调用时给出详细错误。
        settings.require_llm_key()
        if settings.ENABLE_VISUAL_EVAL and not dry_run:
            settings.require_visual_llm()
        if self.rag.enabled:
            self.rag.require_ready()
        if len(user_prompt) > settings.MAX_PROMPT_CHARS:
            raise ValueError(
                f"用户需求过长：{len(user_prompt)} 字符，最大允许 {settings.MAX_PROMPT_CHARS} 字符\n提示：可以将需求拆分为多个较短的动画，或使用更简洁的描述"
            )
        self._callback = callback
        self._manifest = None
        self._cancel_requested.clear()
        self._stop_event.clear()
        ctx = PipelineContext(
            user_prompt=user_prompt,
            original_prompt=user_prompt,
            dry_run=dry_run,
            interactive=interactive,
            paths=RunPaths.create(output_path),
            approve_plan=approve_plan,
            visual_eval_profile=self._configured_visual_profile(
                enabled=settings.ENABLE_VISUAL_EVAL and not dry_run
            ),
            rag_profile=self._current_rag_profile(),
        )
        self._ctx = ctx
        ctx.paths.root.mkdir(parents=True, exist_ok=True)
        ctx.paths.root.chmod(0o700)
        with lock_run(ctx.paths.root):
            return self._execute(ctx, State.INIT)

    def submit_existing_scene(
        self,
        source_code: str,
        class_name: str,
        *,
        scene_id: int = 1,
        wait: bool = False,
    ) -> tuple[SlurmJob, Path | None, str]:
        """提交用户已有的单 Scene 文件，并让它拥有完整的运行清单。"""

        self._cancel_requested.clear()
        self._stop_event.clear()
        profile = RenderProfile.current()
        validation = self._validate(source_code, renderer=profile.renderer)
        if not validation.is_valid or class_name not in validation.scene_classes:
            raise ValueError("直接渲染代码未通过确定性校验")
        self._preflight_environment(profile)
        paths = RunPaths.create()
        for directory in (
            paths.root,
            paths.scenes,
            paths.logs,
            paths.videos,
            paths.output.parent,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        paths.root.chmod(0o700)
        for directory in (paths.scenes, paths.logs, paths.videos):
            directory.chmod(0o700)
        code_path = paths.scenes / f"scene_{scene_id}.py"
        self._write_private(code_path, source_code)
        prompt = f"Direct render of Scene class {class_name}"
        self._write_private(paths.root / "prompt.md", prompt)
        plan = ScenePlan(
            scene_id=scene_id,
            title=f"Direct render: {class_name}",
            duration_seconds=1,
            purpose="渲染用户提供的 Manim Scene",
            math_concept="由用户提供的代码定义",
            visual_design="由用户提供的代码定义",
            camera_movement="由用户提供的代码定义",
            visual_flow=["执行用户提供的 Scene"],
            key_moments=["由用户提供的代码定义"],
            computation="由用户提供的代码定义",
        )
        scene_state = SceneState(
            plan=plan,
            code=source_code,
            class_name=class_name,
            plan_ready=True,
            reviewed=True,
        )
        ctx = PipelineContext(
            user_prompt=prompt,
            original_prompt=prompt,
            paths=paths,
            auto_fix=False,
            direct_render=True,
            scenes=[plan],
            scene_states={scene_id: scene_state},
            visual_eval_profile=self._configured_visual_profile(enabled=False),
        )
        self._ctx = ctx
        self._manifest = None

        with lock_run(paths.root):
            self._emit("run_started", run_id=paths.run_id, run_dir=str(paths.root))
            if wait:
                try:
                    final_video = self._execute(ctx, State.DISPATCHING)
                except Exception as exc:
                    job = scene_state.slurm_job
                    suffix = f"\n错误日志: {job.log_err}" if job else ""
                    raise RuntimeError(f"{exc}{suffix}") from exc
            else:
                try:
                    self._checkpoint(ctx, State.DISPATCHING)
                    next_state = self._handle_dispatching(ctx)
                    self._checkpoint(ctx, next_state)
                except KeyboardInterrupt:
                    self.cancel_all()
                    self._checkpoint(
                        ctx,
                        State.DISPATCHING,
                        status="interrupted",
                        error="用户中断",
                    )
                    raise
                except Exception as exc:
                    self.cancel_all()
                    self._checkpoint(ctx, State.DISPATCHING, status="failed", error=str(exc))
                    raise
                if scene_state.slurm_job is None:
                    message = scene_state.failure_reason or "Slurm 提交失败"
                    self._checkpoint(ctx, State.ERROR, status="failed", error=message)
                    raise RuntimeError(message)
                final_video = None
        if scene_state.slurm_job is None:
            raise RuntimeError("直接渲染结束但没有 Slurm Job")
        return scene_state.slurm_job, final_video, paths.run_id

    def resume(
        self,
        run_id: str,
        callback: Callback | None = None,
        interactive: bool = False,
        retry_scene_id: int | None = None,
    ) -> Path | None:
        """从原子清单恢复，不重新提交仍有 Job ID 的场景。"""

        repository = RunRepository(settings.WORKSPACE_DIR)
        root = repository.run_root(run_id)
        # 先确认清单存在，避免拼错 run-id 时创建空目录；真正读取放在锁内，
        # 防止原进程在 load 与 flock 之间推进状态而导致使用过期 Job ID。
        if not repository.manifest_path(run_id).is_file():
            raise FileNotFoundError(f"找不到运行清单: {run_id}")
        with lock_run(root):
            manifest = repository.load(run_id)
            manifest.validate_for_resume()
            self._callback = callback
            self._cancel_requested.clear()
            self._stop_event.clear()
            self._manifest = manifest
            ctx = self._context_from_manifest(manifest, root)
            ctx.interactive = interactive
            self._ctx = ctx
            self._reconcile_rag_context(ctx)
            if retry_scene_id is not None:
                if retry_scene_id not in ctx.scene_states:
                    raise ValueError(f"运行 {run_id} 不包含 Scene {retry_scene_id}")
                if manifest.status == "completed":
                    raise RuntimeError("已完成运行不能原地重试，请使用新的增量运行")
                retry_state = ctx.scene_states[retry_scene_id]
                if retry_state.slurm_job is not None and not retry_state.rendered:
                    job = retry_state.slurm_job
                    if job.status not in {
                        "COMPLETED",
                        "CANCELLED",
                        *FAILURE_STATES,
                    } and not self.slurm.cancel_job(job.job_id):
                        raise RuntimeError(
                            f"Scene {retry_scene_id} 的 Job {job.job_id} 取消失败，拒绝重复提交"
                        )
                    retry_state.slurm_job = None
                retry_state.rendered = False
                retry_state.artifact = None
                retry_state.failed = False
                retry_state.give_up = False
                retry_state.failure_reason = ""
                retry_state.failure_category = ""
                retry_state.visual_status = (
                    "pending" if ctx.visual_eval_profile.enabled else "skipped"
                )
                retry_state.visual_score = None
                retry_state.visual_report_file = ""
                retry_state.visual_report_sha256 = ""
                retry_state.visual_artifact_sha256 = ""
                retry_state.visual_feedback = ""
                retry_state.visual_best_candidate = None
                if not retry_state.code:
                    retry_state.reviewed = False
                ctx.final_video = None
                ctx.final_video_sha256 = ""
                state = State.CODING
                ctx.plan_review_status = "passed" if retry_state.plan_reviewed else "pending"

            if manifest.status == "completed":
                if ctx.final_video and ctx.final_video.is_file():
                    if (
                        ctx.final_video_sha256
                        and sha256_file(ctx.final_video) != ctx.final_video_sha256
                    ):
                        raise RuntimeError("运行标记为完成，但最终视频哈希与清单不一致")
                    return ctx.final_video
                raise RuntimeError("运行标记为完成，但最终视频不存在")
            incomplete_dry_run = manifest.status == "dry_run_complete" and any(
                not scene.reviewed or scene.failed or scene.give_up
                for scene in manifest.scenes.values()
            )
            if manifest.status == "dry_run_complete" and not incomplete_dry_run:
                return None
            if ctx.visual_eval_profile.enabled:
                settings.visual_llm_profile(model_override=ctx.visual_eval_profile.model).require()
            try:
                state = State[manifest.state]
            except KeyError as exc:
                raise ValueError(f"运行清单包含未知 FSM 状态: {manifest.state}") from exc
            if retry_scene_id is not None:
                state = State.CODING

            if incomplete_dry_run and retry_scene_id is None:
                # 旧版本会把含失败场景的 dry-run 误标为完成，导致 resume
                # 看到 DONE 后直接返回。显式恢复这类清单时，从代码审查屏障
                # 重新开始；已完成的场景仍会被顺序屏障安全跳过。
                state = State.CODING if ctx.scene_states else State.PLANNING

            # resume 明确表示用户愿意再次尝试；先重置放弃标记，再处理 ERROR。
            # 旧逻辑先判断 ERROR，导致“所有场景都已放弃”时无法进入这里的
            # 重试分支，仪表盘提示可恢复但实际直接报“无可用场景”。
            reset_give_up = False
            reset_failed = False
            target_states = (
                [ctx.scene_states[retry_scene_id]]
                if retry_scene_id is not None
                else list(ctx.scene_states.values())
            )
            for scene in target_states:
                if scene.give_up:
                    scene.give_up = False
                    scene.review_round = 0
                    scene.fix_attempts = 0
                    scene.failure_reason = ""
                    scene.failure_category = ""
                    reset_give_up = True
                # 失败清单也允许显式 resume。旧实现只在 ERROR 快照中清除
                # failed，若最后一次检查点是 MONITORING，调度器会永久跳过
                # 这些场景，表现为“恢复成功但没有重新开始”。
                if scene.failed and not scene.rendered:
                    scene.failed = False
                    scene.failure_reason = ""
                    scene.failure_category = ""
                    reset_failed = True

            if state is State.ERROR:
                # 允许从 ERROR 状态恢复：无场景时重跑概要规划；已有场景
                # 则让场景级 worker 根据 plan_ready/code/reviewed 自己选择
                # 分镜、编码或审查，不因某个失败快照而永久卡死。
                if not ctx.scene_states:
                    state = State.PLANNING
                else:
                    for scene in target_states:
                        if scene.failed:
                            scene.failed = False
                            scene.failure_reason = ""
                            scene.failure_category = ""
                    state = State.CODING
                self._emit("run_resuming_from_error", run_id=run_id, state=state.name)

            pending_plan_review = any(
                not scene.plan_reviewed
                and scene.plan_ready
                and not scene.failed
                and not scene.give_up
                for scene in ctx.scene_states.values()
            )
            if pending_plan_review:
                ctx.plan_review_status = "pending"
            elif ctx.plan_review_status == "failed":
                # 计划本身已经审查通过，但上次可能在代码审查阶段失败；
                # 恢复时不能把后续失败误当成计划屏障失败。
                ctx.plan_review_status = "passed"

            if retry_scene_id is not None:
                with suppress(Exception):
                    from kd1_anime.dashboard import quiet

                    if not quiet():
                        console.print(
                            f"[yellow]将只重试 Scene {retry_scene_id}，其它场景保持不变[/]"
                        )
            if reset_give_up:
                # resume 时仪表盘可能已激活, 避免直接打印破坏 Live 渲染
                with suppress(Exception):
                    from kd1_anime.dashboard import quiet

                    if not quiet():
                        console.print("[yellow]发现已放弃的场景，将重置并重试[/]")
                if state not in {
                    State.PLAN_REVIEWING,
                    State.CODING,
                    State.REVIEWING,
                    State.DISPATCHING,
                }:
                    # 回到审查阶段重新评估已生成的代码
                    state = State.REVIEWING

            if reset_failed:
                with suppress(Exception):
                    from kd1_anime.dashboard import quiet

                    if not quiet():
                        console.print("[yellow]发现失败的场景，将重置并重试[/]")

            cancelled_jobs = False
            for scene in ctx.scene_states.values():
                if scene.slurm_job and (
                    scene.slurm_job.cancelled or scene.slurm_job.status == "CANCELLED"
                ):
                    scene.slurm_job = None
                    cancelled_jobs = True
            if cancelled_jobs and state in {
                State.DISPATCHING,
                State.MONITORING,
                State.FIXING,
                State.MERGING,
                State.DONE,
            }:
                state = State.DISPATCHING

            # rendered 标记只对上一次检查点有效；用户可能已经手动清理了
            # 场景视频。先把失效产物退回渲染队列，再核对仍在集群中的 Job。
            self._reconcile_rendered_artifacts(ctx)

            # 核对恢复的 Slurm 作业: 上次会话提交的作业可能早已结束/已不存在,
            # 直接监控会得到连续 UNKNOWN → CANCEL_FAILED → 场景永久判死。
            self._reconcile_restored_jobs(ctx)

            # 上一次运行可能在连续性重规划达到上限后被中断。按恢复策略，
            # 第一次 resume 额外开启一轮有限修正；这次机会必须持久化，
            # 否则用户每次 resume 都会重新进入同一个连续性循环。若本轮
            # 再次耗尽预算，warning 将作为非阻断终态沿用已有计划。
            if (
                ctx.continuity_bible is not None
                and ctx.continuity_review_status == "warning"
                and not ctx.continuity_resume_recheck_used
                and any(
                    not scene.rendered and not scene.failed and not scene.give_up
                    for scene in ctx.scene_states.values()
                )
            ):
                with self._state_lock:
                    ctx.continuity_review_status = "pending"
                    ctx.continuity_review_round = 0
                    ctx.continuity_resume_recheck_used = True
                    ctx.continuity_warnings.append("恢复运行：重新开启连续性审查与有限修正")
                    self._checkpoint(ctx, state)
                self._emit(
                    "continuity_review_resume_recheck",
                    run_id=run_id,
                    max_rechecks=1,
                )

            # schema 2 的早期清单没有 continuity bible。恢复未完成运行时补建并
            # 持久化一份；已经渲染完成的旧场景不再触发连续性重规划。
            if ctx.continuity_bible is None and ctx.outlines and ctx.scene_states:
                self._plan_continuity_bible(ctx)
                if any(not scene.rendered for scene in ctx.scene_states.values()):
                    if ctx.continuity_review_status == "passed":
                        ctx.continuity_review_status = "pending"
                else:
                    ctx.continuity_review_status = "passed"
                self._checkpoint(ctx, state)

            self._emit("run_resumed", run_id=run_id, state=state.name)
            return self._execute(ctx, state)

    def plan_only(
        self,
        user_prompt: str,
        callback: Callback | None = None,
        interactive: bool = False,
        output_path: Path | None = None,
        approve_plan: bool = False,
        review: bool = True,
        preflight: bool = True,
    ) -> list[ScenePlan]:
        """只生成并审查完整计划，不进入 TechnicalSpec/Coder/渲染阶段。"""

        if preflight:
            settings.require_llm_key()
            if self.rag.enabled:
                self.rag.require_ready()
        if len(user_prompt) > settings.MAX_PROMPT_CHARS:
            raise ValueError(
                f"用户需求过长：{len(user_prompt)} 字符，最大允许 {settings.MAX_PROMPT_CHARS} 字符"
            )
        self._callback = callback
        self._manifest = None
        self._cancel_requested.clear()
        self._stop_event.clear()
        ctx = PipelineContext(
            user_prompt=user_prompt,
            original_prompt=user_prompt,
            paths=RunPaths.create(output_path),
            dry_run=True,
            interactive=interactive,
            approve_plan=approve_plan,
            visual_eval_profile=self._configured_visual_profile(enabled=False),
            rag_profile=self._current_rag_profile(),
        )
        self._ctx = ctx
        ctx.paths.root.mkdir(parents=True, exist_ok=True)
        ctx.paths.root.chmod(0o700)
        with lock_run(ctx.paths.root):
            try:
                self._initialize_new_run(ctx)
                if not review:
                    ctx.plan_review_status = "skipped"
                    ctx.continuity_review_status = "passed"
                    self._checkpoint(ctx, State.DETAILING)
                self._run_scheduler(ctx, planning_only=True)
                if review and ctx.plan_review_status == "failed":
                    details = "; ".join(
                        f"Scene {scene_id}: {state.failure_reason or '计划未通过'}"
                        for scene_id, state in sorted(ctx.scene_states.items())
                        if state.failed or not state.plan_reviewed
                    )
                    raise RuntimeError(f"计划审查未通过，未生成代码: {details}")
                unfinished = [
                    (scene_id, state)
                    for scene_id, state in sorted(ctx.scene_states.items())
                    if state.failed
                    or state.give_up
                    or not state.plan_ready
                    or (review and not state.plan_reviewed)
                ]
                if unfinished:
                    details = "; ".join(
                        f"Scene {scene_id}: {state.failure_reason or '计划/分镜未完成'}"
                        for scene_id, state in unfinished
                    )
                    raise RuntimeError(f"计划生成未完成，未生成代码: {details}")
                self._checkpoint(ctx, State.DONE, status="dry_run_complete")
                return [
                    state.plan
                    for state in sorted(
                        ctx.scene_states.values(), key=lambda item: item.plan.scene_id
                    )
                ]
            except Exception as exc:
                self.cancel_all()
                with suppress(Exception):
                    self._checkpoint(ctx, State.ERROR, status="failed", error=str(exc))
                raise

    def run_from_plan(
        self,
        plan_file: Path,
        *,
        callback: Callback | None = None,
        dry_run: bool = False,
        interactive: bool = False,
        output_path: Path | None = None,
        approve_plan: bool = False,
    ) -> Path | None:
        """从 ``kd1-anime plan --output`` 生成的计划文件继续执行。"""

        payload = json.loads(Path(plan_file).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
            raise ValueError("计划文件必须是包含 items 数组的 JSON 对象")
        if payload.get("schema_version") != 1:
            raise ValueError("不支持的计划文件 schema_version")
        scenes = [ScenePlan.model_validate(item) for item in payload["items"]]
        if not scenes:
            raise ValueError("计划文件没有场景")
        scenes = sorted(scenes, key=lambda item: item.scene_id)
        if [scene.scene_id for scene in scenes] != list(range(1, len(scenes) + 1)):
            raise ValueError("计划文件中的 scene_id 必须从 1 开始连续编号")
        lesson_spec = LessonSpec.model_validate(payload.get("lesson_spec") or {})
        teaching_graph = TeachingGraph.model_validate(payload.get("teaching_graph") or {})
        bible_payload = payload.get("continuity_bible")
        continuity_bible = ContinuityBible.model_validate(bible_payload or {})
        user_prompt = str(payload.get("user_prompt") or lesson_spec.topic or "")
        if not user_prompt.strip():
            raise ValueError("计划文件缺少 user_prompt")
        settings.require_llm_key()
        if settings.ENABLE_VISUAL_EVAL and not dry_run:
            settings.require_visual_llm()
        if self.rag.enabled:
            self.rag.require_ready()
        self._callback = callback
        self._manifest = None
        self._cancel_requested.clear()
        self._stop_event.clear()
        ctx = PipelineContext(
            user_prompt=user_prompt,
            original_prompt=user_prompt,
            paths=RunPaths.create(output_path),
            dry_run=dry_run,
            interactive=interactive,
            approve_plan=approve_plan,
            outlines=[
                SceneOutline(
                    scene_id=scene.scene_id,
                    title=scene.title,
                    duration_seconds=scene.duration_seconds,
                    purpose=scene.purpose,
                    math_concept=scene.math_concept,
                    claim_ids=list(scene.claim_ids),
                    visual_unit_id=scene.visual_unit_id,
                    teaching_role=scene.teaching_role,
                )
                for scene in scenes
            ],
            scenes=list(scenes),
            lesson_spec=lesson_spec,
            teaching_graph=teaching_graph,
            continuity_bible=continuity_bible,
            plan_review_status="pending",
            continuity_review_status="pending",
            visual_eval_profile=self._configured_visual_profile(
                enabled=settings.ENABLE_VISUAL_EVAL and not dry_run
            ),
            rag_profile=self._current_rag_profile(),
        )
        transition = effective_transition_duration(
            (scene.duration_seconds for scene in scenes), ctx.merge_profile
        )
        ctx.expected_final_duration = max(
            0.0,
            sum(scene.duration_seconds for scene in scenes) - transition * max(0, len(scenes) - 1),
        )
        ctx.scene_states = {
            scene.scene_id: SceneState(
                plan=scene,
                plan_ready=True,
                visual_status=("pending" if ctx.visual_eval_profile.enabled else "skipped"),
            )
            for scene in scenes
        }
        self._ctx = ctx
        ctx.paths.root.mkdir(parents=True, exist_ok=True)
        ctx.paths.root.chmod(0o700)
        with lock_run(ctx.paths.root):
            self._handle_init(ctx)
            self._write_private(
                ctx.paths.root / "plan.json",
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            )
            self._checkpoint(ctx, State.DETAILING)
            return self._execute(ctx, State.CODING)

    def _execute(self, ctx: PipelineContext, state: State) -> Path | None:
        try:
            self._emit(
                "rag_status",
                status=ctx.rag_profile.status,
                embedding_model=ctx.rag_profile.embedding_model,
                reranker_model=ctx.rag_profile.reranker_model,
                warning=ctx.rag_warnings[-1] if ctx.rag_warnings else "",
            )
            # ---- 准备: 目录 + 全局概要 (仅全新运行; resume 已从清单加载) ----
            if not ctx.scene_states:
                self._initialize_new_run(ctx)
            elif ctx.scenes:
                # resume 等已有 scene_states 的入口: 补发 plan_complete 供 TUI 表格展示
                self._emit(
                    "plan_complete",
                    scenes=[
                        state.plan
                        for state in sorted(
                            ctx.scene_states.values(), key=lambda s: s.plan.scene_id
                        )
                    ],
                    visual_enabled=ctx.visual_eval_profile.enabled,
                )

            # resume: 把已有场景的当前进度以事件补发给 TUI/仪表盘, 否则调度器
            # 会跳过 rendered/failed 场景 (不发任何事件), 仪表盘会误显示为"未开始"。
            self._emit_scene_snapshot(ctx)

            # ---- 场景级并行调度主循环 ----
            improve = True
            while improve:
                self._run_scheduler(ctx)
                if self._cancel_requested.is_set():
                    # 批量处理收到 Ctrl-C 时，worker 线程无法直接收到
                    # KeyboardInterrupt；在调度器收尾后转成同一条中断收尾路径，
                    # 确保 manifest 记录 interrupted 而不是误记为 failed。
                    raise KeyboardInterrupt
                if ctx.plan_review_status == "failed":
                    details = "; ".join(
                        f"Scene {scene_id}: {scene.failure_reason or '计划未通过'}"
                        for scene_id, scene in sorted(ctx.scene_states.items())
                        if scene.failed or not scene.plan_reviewed
                    )
                    raise RuntimeError(f"计划审查未通过，已阻止代码生成: {details}")
                planning_failures = [
                    (scene_id, scene)
                    for scene_id, scene in sorted(ctx.scene_states.items())
                    if scene.failure_category == "planning"
                    or (not scene.plan_ready and not scene.rendered)
                ]
                if planning_failures:
                    details = "; ".join(
                        f"Scene {scene_id}: {scene.failure_reason or '计划/分镜未完成'}"
                        for scene_id, scene in planning_failures
                    )
                    raise RuntimeError(f"计划/分镜阶段失败，已阻止后续生成: {details}")
                if ctx.continuity_rebuild_required:
                    # 上游 AutoFix 已经停止本轮渲染并清空下游；下一轮从
                    # 顺序编码屏障重新建立继承代码，再恢复并行渲染。
                    ctx.continuity_rebuild_required = False
                    self._stop_event.clear()
                    self._checkpoint(ctx, State.CODING)
                    continue
                if self._stop_event.is_set():
                    with self._state_lock:
                        reason = (
                            ctx.continuity_warnings[-1]
                            if ctx.continuity_warnings
                            else "流水线被停止，未能安全完成连续性重建"
                        )
                    raise RuntimeError(reason)
                if ctx.dry_run:
                    break
                if self._visual_gate(ctx):
                    if ctx.continuity_rebuild_required:
                        ctx.continuity_rebuild_required = False
                        self._stop_event.clear()
                    self._checkpoint(ctx, State.CODING)
                    continue
                if self._boundary_visual_gate(ctx):
                    if ctx.continuity_rebuild_required:
                        ctx.continuity_rebuild_required = False
                        self._stop_event.clear()
                    self._checkpoint(ctx, State.CODING)
                    continue
                try:
                    self._merge(ctx)
                except Exception as exc:
                    route = classify_failure(str(exc), phase="merge")
                    self._emit(
                        "failure_routed",
                        category=route.category,
                        handler=route.handler,
                        reason=route.reason,
                    )
                    raise RuntimeError(f"视频合并失败（{route.category}）：{exc}") from exc
                self._final_visual_report(ctx)
                improve = self._eval(ctx)

            # ---- 收尾 ----
            if ctx.dry_run:
                unfinished = [
                    (scene_id, scene)
                    for scene_id, scene in sorted(ctx.scene_states.items())
                    if not scene.reviewed
                    or scene.failed
                    or scene.give_up
                    or (ctx.plan_review_status != "skipped" and not scene.plan_reviewed)
                ]
                if unfinished:
                    # 如果上游场景已经明确失败，下游只是被顺序屏障
                    # 阻塞，不要把“缺少继承状态”再包装成第二个错误。
                    root_failures = [
                        item for item in unfinished if item[1].failed or item[1].give_up
                    ]
                    reported = root_failures or unfinished
                    details = "; ".join(
                        f"Scene {scene_id}: {scene.failure_reason or '未通过编码/审查'}"
                        for scene_id, scene in reported
                    )
                    message = f"Dry-run 未完成，存在失败场景: {details}"
                    self._checkpoint(ctx, State.ERROR, status="failed", error=message)
                    raise RuntimeError(message)
                self._checkpoint(ctx, State.DONE, status="dry_run_complete")
                self._emit("dry_run_complete", run_dir=str(ctx.paths.root))
                return None
            if ctx.final_video is None:
                message = "流水线结束但没有最终视频"
                self._checkpoint(ctx, State.ERROR, status="failed", error=message)
                raise RuntimeError(message)
            self._checkpoint(ctx, State.DONE, status="completed")
            return ctx.final_video
        except KeyboardInterrupt:
            self.cancel_all()
            try:
                self._checkpoint(
                    ctx,
                    self._latest_checkpoint_state(state),
                    status="interrupted",
                    error="用户中断",
                )
            except Exception as checkpoint_error:
                console.print(f"[yellow]写入中断清单失败: {checkpoint_error}[/]", markup=False)
            raise
        except Exception as exc:
            self.cancel_all()
            try:
                self._checkpoint(
                    ctx,
                    self._latest_checkpoint_state(state),
                    status="failed",
                    error=str(exc),
                )
            except Exception as checkpoint_error:
                console.print(f"[yellow]写入失败清单失败: {checkpoint_error}[/]", markup=False)
            raise

    def _latest_checkpoint_state(self, fallback: State) -> State:
        """异常收尾时保留最后已持久化阶段，不用 _execute 入参覆盖。"""

        if self._manifest is None:
            return fallback
        try:
            checkpoint_state = State[self._manifest.state]
            # 旧的/不完整 dry-run 可能最后写成 DONE，但异常收尾不能再
            # 生成 status=failed + state=DONE 的自相矛盾清单；DONE 只允许
            # 与 completed/dry_run_complete 配对。
            if checkpoint_state is State.DONE:
                return State.ERROR
            return checkpoint_state
        except KeyError:
            return fallback

    @staticmethod
    def _preflight_environment(profile: RenderProfile | None = None) -> None:
        """在创建/提交渲染任务前验证本地控制端和渲染配置。"""

        profile = profile or RenderProfile.current()
        missing = [name for name in ("sbatch", "ffmpeg", "ffprobe") if not shutil.which(name)]
        container = settings.SLURM_CONTAINER_IMAGE
        if settings.SLURM_REQUIRE_CONTAINER and not container:
            raise RuntimeError("SLURM_REQUIRE_CONTAINER=true，但未配置 SLURM_CONTAINER_IMAGE")
        if container:
            image = Path(container).expanduser()
            if not image.is_file():
                raise RuntimeError(f"Apptainer 镜像不存在: {image}")
            if not shutil.which("apptainer"):
                missing.append("apptainer")
        if profile.renderer == "opengl" and not settings.SLURM_GPU_TYPE:
            raise RuntimeError(
                "MANIM_RENDERER=opengl 时必须配置 SLURM_GPU_TYPE；否则无法保证 Slurm 分配 GPU 节点"
            )
        if missing:
            raise RuntimeError("运行环境缺少命令: " + ", ".join(dict.fromkeys(missing)))

    def _local_smoke_render(
        self,
        ctx: PipelineContext,
        state: SceneState,
    ) -> None:
        """运行并持久化本地 Smoke Render 状态。"""

        enabled = not ctx.dry_run and settings.LOCAL_SMOKE_RENDER_ENABLED
        with self._state_lock:
            state.local_smoke_status = "running" if enabled else "skipped"
        if not enabled:
            return
        try:
            with self._state_lock:
                self._checkpoint(ctx, State.CODING)
            self._local_smoke_render_impl(ctx, state)
        except Exception:
            with self._state_lock:
                state.local_smoke_status = "failed"
                try:
                    self._checkpoint(ctx, State.CODING)
                except Exception as checkpoint_error:
                    self._record_checkpoint_failure(checkpoint_error)
            raise
        with self._state_lock:
            state.local_smoke_status = "passed"
            self._checkpoint(ctx, State.CODING)
        self._emit("scene_smoke_rendered", scene_id=state.plan.scene_id)

    def _local_smoke_render_impl(
        self,
        ctx: PipelineContext,
        state: SceneState,
    ) -> None:
        """可选地在本地低质量渲染当前场景，尽早发现运行时错误。"""

        source = ctx.paths.scenes / f"scene_{state.plan.scene_id}.py"
        if not source.is_file() or not state.class_name:
            raise RuntimeError("本地 Smoke Render 缺少场景代码或类名")
        self._emit("scene_smoke_rendering", scene_id=state.plan.scene_id)
        with tempfile.TemporaryDirectory(
            prefix=f"kd1-smoke-{state.plan.scene_id}-",
            dir=ctx.paths.root,
        ) as directory:
            media_dir = Path(directory) / "media"
            media_dir.mkdir(parents=True, exist_ok=True)
            smoke_width = max(16, (ctx.render_profile.pixel_width // 8) // 2 * 2)
            smoke_height = max(16, (ctx.render_profile.pixel_height // 8) // 2 * 2)
            smoke_fps = min(ctx.render_profile.frame_rate, 15)
            env = os.environ.copy()
            env["MANIM_RENDERER"] = ctx.render_profile.renderer
            if ctx.render_profile.renderer == "opengl":
                env["PYOPENGL_PLATFORM"] = ctx.render_profile.opengl_platform
            image = settings.SLURM_CONTAINER_IMAGE

            def run_smoke(manim_command: list[str], *, write_video: bool = False) -> None:
                command_args = list(manim_command)
                if write_video and ctx.render_profile.renderer == "opengl":
                    command_args.insert(-2, "--write_to_movie")
                if image:
                    command = [
                        "apptainer",
                        "exec",
                        "--containall",
                        "--cleanenv",
                        "--no-home",
                    ]
                    if ctx.render_profile.renderer == "opengl":
                        command.append("--nv")
                    command.extend(["--env", f"MANIM_RENDERER={ctx.render_profile.renderer}"])
                    if ctx.render_profile.renderer == "opengl":
                        command.extend(
                            [
                                "--env",
                                f"PYOPENGL_PLATFORM={ctx.render_profile.opengl_platform}",
                            ]
                        )
                    if settings.SLURM_CONTAINER_DISABLE_NETWORK:
                        command.extend(["--net", "--network", "none"])
                    command.extend(
                        [
                            "--bind",
                            f"{ctx.paths.root.resolve()}:{ctx.paths.root.resolve()}",
                            str(Path(image).expanduser().resolve()),
                            *command_args,
                        ]
                    )
                else:
                    command = [sys.executable, "-m", *command_args]
                try:
                    result = _run_limited_process(
                        command,
                        timeout=settings.LOCAL_SMOKE_RENDER_TIMEOUT,
                        memory_mb=settings.LOCAL_SMOKE_RENDER_MEMORY_MB,
                        env=env,
                    )
                except subprocess.TimeoutExpired as exc:
                    raise RuntimeError("本地 Smoke Render 超时") from exc
                except OSError as exc:
                    raise RuntimeError(f"本地 Smoke Render 无法启动: {exc}") from exc
                if result.returncode != 0:
                    output = (result.stderr or result.stdout or "").strip()
                    detail = output[-10_000:] if output else f"退出码 {result.returncode}"
                    raise RuntimeError("本地 Smoke Render 失败:\n" + detail)

            common_args = [
                "manim",
                "render",
                f"--renderer={ctx.render_profile.renderer}",
                f"-q{settings.LOCAL_SMOKE_RENDER_QUALITY}",
                "--resolution",
                f"{smoke_width},{smoke_height}",
                "--fps",
                str(smoke_fps),
                "--disable_caching",
            ]
            if settings.LOCAL_SMOKE_RENDER_MODE in {"frame", "both"}:
                frame_dir = media_dir / "__frame_smoke__"
                frame_dir.mkdir(parents=True, exist_ok=True)
                run_smoke(
                    [
                        *common_args,
                        "--format",
                        "png",
                        "--save_last_frame",
                        "--media_dir",
                        str(frame_dir),
                        str(source),
                        state.class_name,
                    ]
                )
                frames = [
                    path
                    for path in frame_dir.rglob("*.png")
                    if path.is_file()
                    and path.stat().st_size > 0
                    and (
                        path.stem == state.class_name
                        or path.stem.startswith(f"{state.class_name}_")
                    )
                ]
                if not frames:
                    raise RuntimeError("本地 Frame Canary 完成但没有生成最后一帧 PNG")
            if settings.LOCAL_SMOKE_RENDER_MODE in {"video", "both"}:
                video_dir = media_dir / "__smoke__"
                video_dir.mkdir(parents=True, exist_ok=True)
                run_smoke(
                    [
                        *common_args,
                        "--media_dir",
                        str(video_dir),
                        str(source),
                        state.class_name,
                    ],
                    write_video=True,
                )
                videos = [
                    path
                    for path in video_dir.rglob(f"{state.class_name}.mp4")
                    if path.is_file() and path.stat().st_size > 0
                ]
                if not videos:
                    raise RuntimeError("本地 Video Canary 完成但没有生成最终 MP4")
        self._write_stage_artifact(
            ctx,
            f"smoke_scene_{state.plan.scene_id}_local.json",
            {
                "schema_version": 1,
                "scene_id": state.plan.scene_id,
                "status": "passed",
                "renderer": ctx.render_profile.renderer,
                "quality": settings.LOCAL_SMOKE_RENDER_QUALITY,
                "mode": settings.LOCAL_SMOKE_RENDER_MODE,
                "resolution": [smoke_width, smoke_height],
                "frame_rate": smoke_fps,
                "container": bool(settings.SLURM_CONTAINER_IMAGE),
            },
        )

    def _handle_init(self, ctx: PipelineContext) -> State:
        for directory in (
            ctx.paths.root,
            ctx.paths.scenes,
            ctx.paths.logs,
            ctx.paths.videos,
            ctx.paths.output.parent,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        ctx.paths.root.chmod(0o700)
        for directory in (ctx.paths.scenes, ctx.paths.logs, ctx.paths.videos):
            directory.chmod(0o700)
        self._write_private(ctx.paths.root / "prompt.md", ctx.user_prompt)
        self._emit("run_started", run_id=ctx.paths.run_id, run_dir=str(ctx.paths.root))

        if not ctx.dry_run:
            if not settings.SLURM_CONTAINER_IMAGE:
                warning = (
                    "未配置 SLURM_CONTAINER_IMAGE：LLM 生成代码将直接以当前用户身份运行。"
                    "共享集群建议启用 Apptainer，并设置 SLURM_REQUIRE_CONTAINER=true。"
                )
                if self._callback:
                    self._emit("security_warning", message=warning)
                else:
                    console.print(f"[bold yellow]安全警告:[/] {warning}")
            self._preflight_environment(ctx.render_profile)

        # 增量渲染分析
        if ctx.incremental:
            logger.info("增量渲染模式：分析场景变化...")
            self._emit("incremental_start", base_run_id=ctx.base_run_id)

        return State.PLANNING

    def _handle_dispatching(self, ctx: PipelineContext) -> State:
        """仅供 ``render --no-wait`` 使用；普通流水线走逐 Scene 调度器。"""

        self._emit("stage_start", stage="dispatching")
        active = [
            state
            for state in ctx.scene_states.values()
            if state.slurm_job and not state.rendered and not state.failed and not state.give_up
        ]
        limit = settings.SLURM_MAX_IN_FLIGHT
        available = max(0, limit - len(active)) if limit else len(ctx.scene_states)
        submitted = 0
        for scene_id, state in sorted(ctx.scene_states.items()):
            if submitted >= available:
                break
            if state.rendered or state.failed or state.give_up or state.slurm_job:
                continue
            self._scene_submit(ctx, scene_id, state)
            if state.slurm_job is not None:
                submitted += 1
        has_active_jobs = any(
            state.slurm_job and not state.rendered and not state.failed and not state.give_up
            for state in ctx.scene_states.values()
        )
        if has_active_jobs:
            return State.MONITORING
        has_unsubmitted = any(
            state.code
            and not state.slurm_job
            and not state.rendered
            and not state.failed
            and not state.give_up
            for state in ctx.scene_states.values()
        )
        return State.DISPATCHING if has_unsubmitted else State.MERGING

    def _initialize_new_run(self, ctx: PipelineContext) -> None:
        """初始化新运行的概要、教学合同、连续性圣经和占位状态。"""

        self._handle_init(ctx)
        self._checkpoint(ctx, State.PLANNING)
        self._plan_outline(ctx)
        self._plan_continuity_bible(ctx)
        ctx.scene_states = {
            outline.scene_id: SceneState(
                plan=self._placeholder_plan(outline),
                visual_status=("pending" if ctx.visual_eval_profile.enabled else "skipped"),
            )
            for outline in ctx.outlines
        }
        ctx.plan_review_status = "pending"
        # 在并行 Detail worker 启动前保存概要、教学合同和连续性圣经；
        # 进程此时中断时 resume 不会丢失全片规范或重新生成不同的合同。
        self._checkpoint(ctx, State.DETAILING)
        self._emit(
            "plan_complete",
            scenes=ctx.scenes,
            visual_enabled=ctx.visual_eval_profile.enabled,
        )

    # ------------------------------------------------------------------
    # 场景级并行调度：每个 Scene 独立推进并共享受限资源。
    # ------------------------------------------------------------------

    @staticmethod
    def _placeholder_plan(outline: SceneOutline) -> ScenePlan:
        """由概要构造占位 ScenePlan (detail 完成前使用)。"""
        return ScenePlan(
            scene_id=outline.scene_id,
            title=outline.title,
            duration_seconds=outline.duration_seconds,
            purpose=outline.purpose,
            math_concept=outline.math_concept,
            claim_ids=list(outline.claim_ids),
            visual_unit_id=outline.visual_unit_id,
            teaching_role=outline.teaching_role,
            visual_design="…",
            camera_movement="…",
            visual_flow=["…"],
            key_moments=["…"],
            computation="…",
        )

    def _plan_outline(self, ctx: PipelineContext) -> None:
        """全局场景概要 (一次); 带交互重试。"""
        while True:
            try:
                outline_kwargs: dict[str, object] = {}
                rag_context = self._retrieve_rag(
                    ctx,
                    ctx.user_prompt,
                    receipt_key="outline",
                    stage="outline",
                    source_kinds={"manim_doc", "example", "recipe"},
                )
                draft_method = getattr(self.planner, "plan_draft", None)
                if callable(draft_method):
                    if self._supports_keyword(draft_method, "rag_context"):
                        outline_kwargs["rag_context"] = rag_context
                    draft = draft_method(ctx.user_prompt, **outline_kwargs)
                    if not isinstance(draft, PlanningDraft):
                        raise ValueError("Planner.plan_draft 返回了无效的 PlanningDraft")
                    ctx.lesson_spec = draft.lesson_spec
                    ctx.lesson_spec, math_repairs = repair_obvious_math_contradictions(
                        ctx.lesson_spec,
                        ctx.user_prompt,
                    )
                    if math_repairs:
                        ctx.continuity_warnings.extend(
                            "教学事实合同：" + repair for repair in math_repairs
                        )
                        self._emit("math_contract_repaired", repairs=math_repairs)
                    ctx.teaching_graph = draft.teaching_graph
                    outlines = draft.items
                else:
                    if self._supports_keyword(self.planner.plan_outline, "rag_context"):
                        outline_kwargs["rag_context"] = rag_context
                    outlines = self.planner.plan_outline(ctx.user_prompt, **outline_kwargs)
                break
            except LLMError as exc:
                logger.error(f"LLM 调用失败: {exc}")
                if not self._ask_retry_or_skip(0, str(exc)):
                    raise LLMResponseError(
                        f"场景概要规划失败: {exc}",
                        hint="检查 LLM API 配置和网络连接",
                    ) from exc
            except Exception as exc:
                logger.error(f"场景概要规划时发生未知错误: {exc}")
                if not self._ask_retry_or_skip(0, str(exc)):
                    raise PipelineError(f"场景概要规划失败: {exc}") from exc
        ctx.outlines = outlines
        ctx.scenes = [self._placeholder_plan(o) for o in outlines]
        transition = effective_transition_duration(
            (item.duration_seconds for item in outlines), ctx.merge_profile
        )
        ctx.expected_final_duration = max(
            0.0,
            sum(item.duration_seconds for item in outlines)
            - transition * max(0, len(outlines) - 1),
        )
        if not ctx.lesson_spec.topic:
            ctx.lesson_spec = ctx.lesson_spec.model_copy(update={"topic": ctx.user_prompt[:2_000]})
        if not ctx.teaching_graph.claim_order and ctx.lesson_spec.claims:
            ctx.teaching_graph = ctx.teaching_graph.model_copy(
                update={"claim_order": [claim.claim_id for claim in ctx.lesson_spec.claims]}
            )
        if not ctx.teaching_graph.scene_claims:
            ctx.teaching_graph = ctx.teaching_graph.model_copy(
                update={
                    "scene_claims": {
                        outline.scene_id: list(outline.claim_ids)
                        for outline in outlines
                        if outline.claim_ids
                    }
                }
            )
        if len(outlines) > settings.MAX_SCENES:
            raise RuntimeError(
                f"Planner 生成了 {len(outlines)} 个场景，超过 MAX_SCENES={settings.MAX_SCENES}"
            )
        self._write_stage_artifact(
            ctx,
            "outline.json",
            {
                "schema_version": 2,
                "items": [item.model_dump(mode="json") for item in outlines],
                "lesson_spec": ctx.lesson_spec.model_dump(mode="json"),
                "teaching_graph": ctx.teaching_graph.model_dump(mode="json"),
            },
        )
        self._write_stage_artifact(
            ctx,
            "lesson_spec.json",
            ctx.lesson_spec.model_dump(mode="json"),
        )
        self._write_stage_artifact(
            ctx,
            "teaching_graph.json",
            ctx.teaching_graph.model_dump(mode="json"),
        )

    def _plan_continuity_bible(self, ctx: PipelineContext) -> None:
        """概要完成后固定全片连续性规范，再允许场景分镜并行生成。"""

        if not ctx.outlines:
            raise RuntimeError("无法建立连续性圣经：没有场景概要")
        self._emit("continuity_bible_start", scene_count=len(ctx.outlines))
        planner_method = getattr(self.planner, "plan_continuity_bible", None)
        if not callable(planner_method):
            # 兼容外部集成/测试替换的旧 Planner；正式 Planner 始终提供该方法。
            ctx.continuity_bible = ContinuityBible()
            ctx.continuity_review_status = "passed"
            ctx.continuity_warnings.append("当前 Planner 不支持连续性圣经，已沿用默认规范")
            self._emit("continuity_warning", reason=ctx.continuity_warnings[-1])
            self._write_stage_artifact(
                ctx,
                "continuity_bible.json",
                ctx.continuity_bible.model_dump(mode="json"),
            )
            return
        try:
            rag_context = self._retrieve_rag(
                ctx,
                ctx.user_prompt
                + "\n"
                + "\n".join(f"{item.title}: {item.math_concept}" for item in ctx.outlines),
                receipt_key="continuity",
                stage="continuity",
                source_kinds={"manim_doc", "example", "recipe"},
            )
            bible_kwargs: dict[str, object] = {
                "stream": False,
                "renderer": ctx.render_profile.renderer,
            }
            if self._supports_keyword(planner_method, "rag_context"):
                bible_kwargs["rag_context"] = rag_context
            if self._supports_keyword(planner_method, "lesson_spec"):
                bible_kwargs["lesson_spec"] = ctx.lesson_spec
            if self._supports_keyword(planner_method, "teaching_graph"):
                bible_kwargs["teaching_graph"] = ctx.teaching_graph
            with self._llm_slot():
                ctx.continuity_bible = planner_method(ctx.user_prompt, ctx.outlines, **bible_kwargs)
        except Exception as exc:
            # 连续性增强不能让基础流水线因一次额外 LLM 调用完全不可用；
            # 使用确定的默认圣经并保留 pending，后续仍会进行全片连续性审查。
            ctx.continuity_bible = ContinuityBible()
            ctx.continuity_review_status = "pending"
            warning = f"连续性圣经生成失败，已使用默认规范: {exc}"
            ctx.continuity_warnings.append(warning)
            self._emit("continuity_warning", reason=warning)
            self._write_stage_artifact(
                ctx,
                "continuity_bible.json",
                ctx.continuity_bible.model_dump(mode="json"),
            )
            return
        ctx.continuity_review_status = "pending"
        ctx.continuity_review_round = 0
        self._write_stage_artifact(
            ctx,
            "continuity_bible.json",
            ctx.continuity_bible.model_dump(mode="json"),
        )
        self._emit("continuity_bible_ready")

    def _llm_slot(self):
        """在调度器已初始化信号量时复用它，否则提供无锁上下文。"""

        from contextlib import nullcontext

        return self._llm_sem if hasattr(self, "_llm_sem") else nullcontext()

    def _ensure_plan_approved(self, ctx: PipelineContext) -> None:
        """可选的人机确认闸门；默认完全自动，不影响普通流水线。"""

        if not ctx.approve_plan or ctx.plan_approved:
            return
        if not sys.stdin.isatty():
            # CI/dry-run 等非 TTY 场景不能阻塞等待输入；显式传入
            # --approve-plan 已经表达了用户确认意图。
            ctx.plan_approved = True
            ctx.continuity_warnings.append("非交互环境已按 --approve-plan 自动确认计划")
            try:
                self._checkpoint(ctx, State.PLAN_REVIEWING)
            except Exception as exc:
                self._record_checkpoint_failure(exc)
                raise RuntimeError(f"运行状态持久化失败，流水线已停止: {exc}") from exc
            return
        from kd1_anime.dashboard import suspend_all

        with suspend_all():
            console.print("\n[bold]计划审查完成，待生成的场景如下：[/]")
            for scene in sorted(ctx.scene_states.values(), key=lambda item: item.plan.scene_id):
                if not scene.failed and not scene.give_up:
                    console.print(
                        f"  Scene {scene.plan.scene_id}: {scene.plan.title} "
                        f"({scene.plan.duration_seconds:g}s)"
                    )
            if not Confirm.ask("批准该计划并开始生成代码？", default=True, console=console):
                raise PipelineError("用户未批准计划，流水线已停止")
        ctx.plan_approved = True
        self._checkpoint(ctx, State.PLAN_REVIEWING)

    def _visual_llm_slot(self):
        """批处理时复用进程级视觉模型配额。"""

        from contextlib import nullcontext

        if self._resource_coordinator is not None:
            return self._resource_coordinator.visual_llm
        return nullcontext()

    # ------------------------------------------------------------------
    # 场景级并行调度 (per-scene pipeline)
    # 每个 Scene 一个工作线程, 独立推进分镜后的代码→渲染→修复流程；
    # 计划审查已在调度器屏障中完成。
    # LLM 并发受 LLM_PARALLEL_WORKERS 信号量限制; 提交受 SLURM_MAX_IN_FLIGHT 名额限制。
    # ------------------------------------------------------------------

    def _reconcile_restored_jobs(self, ctx: PipelineContext) -> None:
        """resume 时核对清单里恢复的 Slurm 作业, 避免监控失效作业导致场景判死。

        中断后再次 resume, 上次会话提交的作业可能早已结束或已从集群消失;
        若直接监控会得到连续 UNKNOWN → UNKNOWN_TIMEOUT → scancel 失败 →
        CANCEL_FAILED ("禁止自动重提"), 场景被永久判死。这里在调度前核对:
        - COMPLETED 且视频存在 → 直接标记渲染完成 (复用上次结果)
        - COMPLETED 但视频缺失 → 清空, 重跑
        - GONE (squeue 可查但无记录且 sacct 无账务记录) → 作业已消失, 清空后重新提交
        - UNKNOWN (集群查询失败/作业不可见) → 保留 Job ID；未确认终态或成功取消前
          绝不自动重提，避免临时 Slurm 故障制造重复作业
        - FAILED/CANCELLED 等终态 → 保留, 交给监控触发自动修复
        - RUNNING/PENDING → 保留, 继续监控
        """
        for scene_id, state in sorted(ctx.scene_states.items()):
            job = state.slurm_job
            if job is None or state.rendered or state.failed or state.give_up:
                continue
            try:
                status = self.slurm.poll_all_statuses([job.job_id]).get(job.job_id, "UNKNOWN")
            except Exception:
                continue  # 集群查询异常 → 保守保留, 交给监控处理
            start_time = getattr(self.slurm, "last_start_times", {}).get(job.job_id)
            if start_time is not None:
                job.started_at = start_time
            job.status = status
            if status == "COMPLETED":
                if self.slurm.validate_completed_job(job):
                    state.artifact = self._artifact_from_job(ctx, state, job)
                    state.rendered = True
                    self._reset_visual_receipt(ctx, state)
                    self._emit("scene_rendered", scene_id=scene_id)
                else:
                    # 不要在产物尚未传播时把已完成作业改写成 FAILED；
                    # JobMonitor 会在共享文件系统宽限期内继续重验。
                    job.status = "COMPLETED"
            elif status == "GONE":
                outcome = self.slurm._classify_gone(job)
                if outcome == "COMPLETED":
                    state.artifact = self._artifact_from_job(ctx, state, job)
                    state.rendered = True
                    self._reset_visual_receipt(ctx, state)
                    self._emit("scene_rendered", scene_id=scene_id)
                elif outcome is None:
                    # 已确认不在调度器且没有任何运行痕迹，可安全重新提交。
                    state.slurm_job = None
                else:
                    job.status = "FAILED"
            elif status == "UNKNOWN":
                job.status = "UNKNOWN"
            else:
                job.status = status

    def _reconcile_rendered_artifacts(self, ctx: PipelineContext) -> None:
        """恢复时校验已标记完成的场景产物，并把丢失产物退回渲染队列。

        ``rendered`` 只表示上一次检查点已经验证过视频，不保证用户没有在
        此后清理文件。若继续保留这个标记，调度器会跳过场景，最终才在合并
        阶段失败，而且无法自动重渲染。场景视频是可重建的派生物，因此这里
        只清除失效凭据并保留已校验的代码；仍在集群中的旧 Job 留给后续
        ``_reconcile_restored_jobs`` 处理，避免在状态不明时重复提交。
        """

        invalidated = False
        for scene_id, state in sorted(ctx.scene_states.items()):
            if not state.rendered or state.artifact is None:
                continue
            try:
                self._artifact_video_path(ctx, state.artifact)
            except (OSError, RuntimeError, ValueError) as exc:
                with self._state_lock:
                    state.rendered = False
                    state.artifact = None
                    state.failed = False
                    state.give_up = False
                    state.failure_reason = ""
                    state.failure_category = ""
                    self._reset_visual_receipt(ctx, state)
                    state.failure_reason = f"恢复时发现渲染产物不可用，将重新处理: {exc}"
                    state.failure_category = "render"
                invalidated = True
                self._emit("scene_artifact_invalid", scene_id=scene_id, reason=str(exc))
            else:
                # 旧版本可能在渲染成功后留下了 stale failed/give_up 标记；
                # 已验证的产物是更强的事实，清除这两个冲突状态，避免
                # 调度器跳过场景或合并阶段误报失败。
                with self._state_lock:
                    state.failed = False
                    state.give_up = False
                    state.failure_reason = ""
                    state.failure_category = ""
        if invalidated:
            with self._state_lock:
                ctx.final_video = None
                ctx.final_video_sha256 = ""
                self._checkpoint(ctx, State.MONITORING)

    def _emit_scene_snapshot(self, ctx: PipelineContext) -> None:
        """resume 后把已有场景的当前进度以事件形式补发给 TUI/仪表盘。

        调度器只对未完成场景发事件, 已 rendered / failed / give_up 的场景会被
        跳过, 导致仪表盘把它们显示为"未开始"。这里按清单记录的当前状态补发
        事件, 让仪表盘恢复后立即反映真实进度。
        """
        for state in sorted(ctx.scene_states.values(), key=lambda s: s.plan.scene_id):
            scene_id = state.plan.scene_id
            # 前置阶段按实际状态补发, 让已渲染场景也显示完整流水线 (分镜✓编码✓审查✓渲染✓)
            if state.plan_ready:
                self._emit("scene_detailed", scene_id=scene_id, title=state.plan.title)
            if state.safe_fallback_used:
                self._emit(
                    "scene_safe_fallback",
                    scene_id=scene_id,
                    reason=state.safe_fallback_reason,
                )
            if state.plan_reviewed:
                self._emit("scene_plan_review_pass", scene_id=scene_id)
            if state.technical_status == "passed":
                self._emit("scene_technical_ready", scene_id=scene_id)
            elif state.technical_status == "generating":
                self._emit("scene_technical_planning", scene_id=scene_id)
            if state.code:
                self._emit("scene_coded", scene_id=scene_id)
            if state.reviewed:
                self._emit("scene_review_pass", scene_id=scene_id)
            # 终态事件
            if state.rendered:
                self._emit("scene_rendered", scene_id=scene_id)
                if state.visual_status == "passed":
                    self._emit(
                        "scene_visual_pass",
                        scene_id=scene_id,
                        score=state.visual_score,
                    )
                elif state.visual_status == "unknown":
                    self._emit(
                        "scene_visual_unknown",
                        scene_id=scene_id,
                        reason=state.visual_feedback,
                    )
                elif state.visual_status == "warning":
                    self._emit(
                        "scene_visual_warning",
                        scene_id=scene_id,
                        score=state.visual_score,
                        reason=state.visual_feedback or "视觉问题已记录",
                    )
                elif state.visual_status == "evaluating":
                    self._emit("scene_visual_evaluating", scene_id=scene_id)
            elif state.failed:
                self._emit("scene_failed", scene_id=scene_id, reason=state.failure_reason or "")
            elif state.give_up:
                self._emit("scene_give_up", scene_id=scene_id, reason=state.failure_reason or "")
            elif state.slurm_job is not None:
                self._emit("scene_submitted", scene_id=scene_id, job_id=state.slurm_job.job_id)

    @staticmethod
    def _planning_cycle_signature(ctx: PipelineContext) -> str:
        """计算当前计划/发现的指纹，防止审查器在同一输入上来回空转。"""

        payload = {
            "lesson_spec": ctx.lesson_spec.model_dump(mode="json"),
            "teaching_graph": ctx.teaching_graph.model_dump(mode="json"),
            "plans": [
                state.plan.model_dump(mode="json")
                for state in sorted(ctx.scene_states.values(), key=lambda item: item.plan.scene_id)
                if not state.failed and not state.give_up and state.plan_ready
            ],
            "compile_issues": {
                str(scene_id): [issue.model_dump(mode="json") for issue in issues]
                for scene_id, issues in sorted(ctx.plan_compile_issues.items())
            },
            "plan_status": ctx.plan_review_status,
            "continuity_status": ctx.continuity_review_status,
        }
        return sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))

    def _run_scheduler(self, ctx: PipelineContext, *, planning_only: bool = False) -> None:
        """启动每个场景的独立流水线线程, 全部结束后返回。"""
        import threading

        self._llm_sem = (
            self._resource_coordinator.llm
            if self._resource_coordinator
            else threading.Semaphore(max(1, settings.LLM_PARALLEL_WORKERS))
        )
        self._slot_lock = threading.Lock()
        self._in_flight = 0
        self._reserved_existing_scenes: set[int] = set()
        if not self._cancel_requested.is_set():
            self._stop_event.clear()
        self._checkpoint_error = None
        with self._phase_lock:
            self._emitted_phases.clear()

        # 正式运行先完成所有场景的 Detail，再做计划正确性审查和全片连续性审查；
        # 通过后才进入编码/代码审查/渲染。编码/审查必须顺序执行，因为 Scene N 的真实
        # 最终 Mobject 定义要作为 Scene N+1 的输入；渲染和监控仍然并行。
        self._run_detail_barrier(ctx)
        if self._checkpoint_error is not None:
            raise RuntimeError(
                f"运行状态持久化失败，流水线已停止: {self._checkpoint_error}"
            ) from self._checkpoint_error
        if any(
            state.failure_category == "planning" or (not state.plan_ready and not state.rendered)
            for state in ctx.scene_states.values()
        ):
            # Detail 失败时不要拿剩余场景继续编译/编码；否则 LessonSpec 的
            # 缺失断言会被错误归因到其它场景，既浪费 LLM 预算又掩盖真正根因。
            self._checkpoint(ctx, State.DETAILING)
            return
        self._normalize_pending_scene_contracts(ctx)
        self._compile_scene_plans(ctx)
        # 计划审查与全片连续性审查可能互相触发：连续性重规划后需要重新
        # 审查计划，而计划重规划后也需要再次确认全片交接。最多往返有限次，
        # 防止两个审查器互相把同一方案推回去。
        for _ in range(4):
            cycle_signature = self._planning_cycle_signature(ctx)
            if cycle_signature == ctx.plan_review_cycle_signature:
                ctx.plan_review_cycle_count += 1
            else:
                ctx.plan_review_cycle_signature = cycle_signature
                ctx.plan_review_cycle_count = 0
            if ctx.plan_review_cycle_count >= 2 and ctx.plan_review_status in {
                "pending",
                "reviewing",
            }:
                reason = "计划/连续性审查在相同输入上重复，已冻结计划并停止空转"
                with self._state_lock:
                    ctx.plan_review_status = "failed"
                    ctx.continuity_warnings.append(reason)
                    self._checkpoint(ctx, State.PLAN_REVIEWING)
                return
            try:
                self._checkpoint(ctx, State.PLAN_REVIEWING)
            except Exception as exc:
                self._record_checkpoint_failure(exc)
                raise RuntimeError(f"运行状态持久化失败，流水线已停止: {exc}") from exc
            if ctx.plan_review_status in {"pending", "reviewing"}:
                if not self._cancel_requested.is_set():
                    self._run_plan_review_barrier(ctx)
                if self._checkpoint_error is not None:
                    raise RuntimeError(
                        f"运行状态持久化失败，流水线已停止: {self._checkpoint_error}"
                    ) from self._checkpoint_error
                if ctx.continuity_rebuild_required or self._stop_event.is_set():
                    return
            if ctx.plan_review_status == "failed":
                return
            if ctx.continuity_bible is not None and ctx.continuity_review_status in {
                "pending",
                "reviewing",
            }:
                if not self._cancel_requested.is_set():
                    self._run_continuity_review(ctx)
                if self._checkpoint_error is not None:
                    raise RuntimeError(
                        f"运行状态持久化失败，流水线已停止: {self._checkpoint_error}"
                    ) from self._checkpoint_error
                if ctx.continuity_rebuild_required or self._stop_event.is_set():
                    return
                # warning 是连续性 LLM 审查耗尽预算后的“接受并继续”终态，
                # 只保留诊断信息，不再回到 pending，也不阻断后续编码。
            if ctx.plan_review_status in {"pending", "reviewing"}:
                continue
            if ctx.continuity_bible is not None and ctx.continuity_review_status in {
                "pending",
                "reviewing",
            }:
                continue
            break

        if ctx.plan_review_status in {"pending", "reviewing"}:
            reason = "计划审查与连续性审查未能在有限轮次内收敛"
            with self._state_lock:
                ctx.plan_review_status = "failed"
                ctx.continuity_warnings.append(reason)
                self._checkpoint(ctx, State.REVIEWING)
            return

        if planning_only:
            self._checkpoint(ctx, State.PLAN_REVIEWING)
            return

        self._ensure_plan_approved(ctx)
        self._run_code_review_barrier(ctx)
        if self._checkpoint_error is not None:
            raise RuntimeError(
                f"运行状态持久化失败，流水线已停止: {self._checkpoint_error}"
            ) from self._checkpoint_error
        if ctx.continuity_rebuild_required or self._stop_event.is_set():
            return
        # 代码屏障是渲染前的硬闸门。任何场景尚未完成编码/审查时，不能
        # 让场景 worker 越过它直接提交 Slurm；尤其不能让下游在缺少上游
        # 导出状态时自行生成一份“看似连续”的代码。
        if any(state.failed or state.give_up for state in ctx.scene_states.values()) or any(
            not state.reviewed and not state.failed and not state.give_up and state.plan_ready
            for state in ctx.scene_states.values()
        ):
            return

        self._slurm_monitor = SlurmMonitorCoordinator(
            self.slurm,
            on_job_update=lambda job: self._checkpoint_slurm_job_update(ctx, job),
        )
        threads: list[threading.Thread] = []
        try:
            for scene_id, state in sorted(ctx.scene_states.items()):
                if state.rendered or state.failed or state.give_up:
                    continue
                if state.slurm_job is not None:
                    self._reserve_existing_slot(scene_id)
                thread = threading.Thread(
                    target=self._scene_worker,
                    args=(ctx, scene_id, state),
                    name=f"scene-{scene_id}",
                    daemon=True,
                )
                threads.append(thread)
                thread.start()
            for thread in threads:
                thread.join()
        finally:
            monitor = self._slurm_monitor
            if monitor is not None:
                if self._stop_event.is_set() or self._cancel_requested.is_set():
                    monitor.cancel_pending(reason="流水线停止")
                monitor.close()
            self._slurm_monitor = None
        if self._checkpoint_error is not None:
            raise RuntimeError(
                f"运行状态持久化失败，流水线已停止: {self._checkpoint_error}"
            ) from self._checkpoint_error

    def _run_code_review_barrier(self, ctx: PipelineContext) -> None:
        """按 Scene ID 顺序完成编码、审查，并固定代码级连续性上下文。"""

        if ctx.direct_render:
            return
        if ctx.plan_review_status not in {"passed", "skipped"}:
            return
        for scene_id, state in sorted(ctx.scene_states.items()):
            if self._stop_event.is_set() or state.failed or state.give_up:
                continue
            try:
                # 场景代码按 ID 顺序生成，是因为后一场景可能要消费前一
                # 场景刚刚审查通过的导出区。上游失败时继续处理下游只会
                # 把“缺少状态账本/继承代码”制造成第二个假故障，既浪费
                # LLM 调用，也让用户难以定位真正根因。下游保持 pending，
                # 待用户 resume 后从最早失败场景重新建立交接。
                if scene_id > 1:
                    previous_state = ctx.scene_states.get(scene_id - 1)
                    requires_inherited_state = bool(state.plan.inherited_elements)
                    upstream_unavailable = previous_state is None or (
                        previous_state.failed
                        or previous_state.give_up
                        or not previous_state.reviewed
                        or not previous_state.code
                    )
                    if requires_inherited_state and upstream_unavailable:
                        dependency_reason = f"等待 Scene {scene_id - 1} 编码/审查通过后建立继承状态"
                        self._emit(
                            "scene_waiting_for_dependency",
                            scene_id=scene_id,
                            dependency_scene_id=scene_id - 1,
                            reason=dependency_reason,
                        )
                        break
                if not state.rendered:
                    self._normalize_plan_contract_for_coding(ctx, state)
                previous_context = state.inherited_elements_code
                self._prepare_inherited_context(ctx, scene_id, state)
                reusable_visual = state.visual_best_candidate
                # 新生成/视觉候选场景都必须先拥有当前输入对应的技术合同。
                # 已经完成的旧清单允许直接复用，避免恢复历史结果时强制重新调用 LLM。
                if not (state.rendered and state.code):
                    self._ensure_technical_spec(ctx, state)
                if previous_context != state.inherited_elements_code and state.code:
                    # 恢复旧运行或上游重写后，不能继续使用基于旧交接状态生成的代码。
                    if state.slurm_job is not None and not state.rendered:
                        if not self.slurm.cancel_job(state.slurm_job.job_id):
                            raise RuntimeError(
                                f"Scene {scene_id} 的旧 Slurm Job {state.slurm_job.job_id} "
                                "仍在使用旧连续性上下文，取消失败，禁止重复提交"
                            )
                        state.slurm_job.cancelled = True
                        state.slurm_job.status = "CANCELLED"
                        state.slurm_job.failure_reason = "上游场景连续性状态变化，已取消旧作业"
                    state.code = ""
                    state.class_name = ""
                    state.reviewed = False
                    state.rewrite_feedback = ""
                    state.review_signature = ""
                    state.identical_review_count = 0
                    state.artifact = None
                    state.rendered = False
                    state.slurm_job = None
                    state.exported_elements_code = ""
                    state.exported_elements = []
                    self._remove_element_manifest_scene(ctx, scene_id)
                    state.safe_fallback_used = False
                    state.safe_fallback_reason = ""
                    self._reset_visual_receipt(
                        ctx,
                        state,
                        clear_candidate=True,
                        reset_attempts=True,
                    )
                    self._write_private(ctx.paths.scenes / f"scene_{scene_id}.py", "")
                if (
                    not state.code
                    and reusable_visual is not None
                    and reusable_visual.passed
                    and reusable_visual.inherited_elements_sha256
                    == sha256_text(state.inherited_elements_code)
                ):
                    try:
                        self._artifact_video_path(ctx, reusable_visual.artifact)
                        self._restore_visual_candidate_into_state(
                            ctx,
                            scene_id,
                            state,
                            reusable_visual,
                        )
                    except (OSError, RuntimeError, ValueError) as exc:
                        # 候选只是优化路径；损坏时清除并走正常编码/渲染，
                        # 不能让一个可重建场景因旧候选失效而永久失败。
                        state.visual_best_candidate = None
                        ctx.continuity_warnings.append(
                            f"Scene {scene_id} 的视觉候选无法复用，已重新生成: {exc}"
                        )
                    else:
                        state.visual_feedback = ""
                        self._checkpoint(ctx, State.REVIEWING)
                        self._emit(
                            "scene_coded",
                            scene_id=scene_id,
                            file_path=str(ctx.paths.scenes / f"scene_{scene_id}.py"),
                        )
                        self._emit("scene_review_pass", scene_id=scene_id)
                        self._emit("scene_rendered", scene_id=scene_id)
                        self._emit(
                            "scene_visual_pass",
                            scene_id=scene_id,
                            score=reusable_visual.score,
                        )
                        continue
                # 已有成功产物仍需补齐导出状态，但不能因恢复而重新调用 LLM。
                if state.rendered and state.code:
                    if not state.exported_elements_code:
                        self._refresh_scene_export(state)
                    self._update_element_manifest(ctx, state)
                    self._update_state_ledger(ctx, state)
                    self._checkpoint(ctx, State.REVIEWING)
                    continue
                while not state.reviewed:
                    if self._stop_event.is_set() or state.failed or state.give_up:
                        break
                    if not state.code or state.rewrite_feedback:
                        self._phase_emit("coding")
                        self._scene_code(ctx, scene_id, state)
                    elif (
                        not ctx.dry_run
                        and settings.LOCAL_SMOKE_RENDER_ENABLED
                        and state.local_smoke_status != "passed"
                    ):
                        self._local_smoke_render(ctx, state)
                    self._phase_emit("reviewing")
                    self._scene_review(ctx, scene_id, state)
                if state.reviewed and state.code and not state.exported_elements_code:
                    self._refresh_scene_export(state)
                    self._update_element_manifest(ctx, state)
                self._checkpoint(ctx, State.REVIEWING)
            except Exception as exc:
                if self._activate_safe_fallback(ctx, scene_id, state, str(exc)):
                    continue
                route = classify_failure(str(exc), phase="coding")
                self._emit(
                    "failure_routed",
                    scene_id=scene_id,
                    category=route.category,
                    handler=route.handler,
                    reason=route.reason,
                )
                with self._state_lock:
                    category = route.category if route.category != "unknown" else "coding"
                    self._mark_failed(
                        state,
                        f"Scene {scene_id} 编码/审查失败: {exc}",
                        category,
                    )
                    try:
                        self._checkpoint(ctx, State.REVIEWING)
                    except Exception as checkpoint_error:
                        self._record_checkpoint_failure(checkpoint_error)
                self._emit("scene_failed", scene_id=scene_id, reason=state.failure_reason)

        # 顺序代码屏障没有完成全部可运行场景时，禁止后面的渲染 worker
        # 在没有 TechnicalSpec/继承代码的状态下自行启动。失败场景本身
        # 仍由外层收尾逻辑报告，pending 场景会在显式 resume 时重试。
        if any(state.failed or state.give_up for state in ctx.scene_states.values()):
            return
        if any(
            not state.reviewed and not state.failed and not state.give_up and state.plan_ready
            for state in ctx.scene_states.values()
        ):
            return

    def _normalize_plan_contract_for_coding(
        self,
        ctx: PipelineContext,
        state: SceneState,
    ) -> None:
        """在代码屏障前修复可确定的元素边界合同漂移。

        运行恢复或旧版清单时，计划可能已经通过了旧的 Plan Review，但其
        ``new_elements``/``handoff`` 仍不一致。若直接交给 Coder，代码会在
        连续性导出合同处失败。这里只调用无创作歧义的确定性归一化，不会
        绕过数学计划审查或改写已经存在的代码。
        """

        if ctx.continuity_bible is None:
            return
        previous_state = ctx.scene_states.get(state.plan.scene_id - 1)
        previous_plan = (
            previous_state.plan
            if previous_state is not None and previous_state.plan_ready
            else None
        )
        normalized, repairs = normalize_scene_plan_contract(
            state.plan,
            ctx.continuity_bible,
            previous_plan=previous_plan,
            has_next_scene=any(
                item.plan.scene_id > state.plan.scene_id
                for item in ctx.scene_states.values()
                if item.plan_ready
            ),
        )
        if not repairs:
            return
        code_invalidated = bool(state.code or state.reviewed or state.slurm_job)
        if code_invalidated:
            self._cancel_unfinished_scene_job(state, reason="使用旧计划合同")
        with self._state_lock:
            state.plan = normalized
            if code_invalidated:
                state.code = ""
                state.class_name = ""
                state.reviewed = False
                state.rewrite_feedback = ""
                state.review_signature = ""
                state.identical_review_count = 0
                state.slurm_job = None
                state.artifact = None
                state.rendered = False
                state.exported_elements_code = ""
                state.exported_elements = []
                self._remove_element_manifest_scene(ctx, state.plan.scene_id)
                self._reset_visual_receipt(ctx, state, clear_candidate=True, reset_attempts=True)
                self._write_private(ctx.paths.scenes / f"scene_{state.plan.scene_id}.py", "")
            self._reset_technical_spec(state)
            ctx.scenes = [
                item.plan
                for item in sorted(ctx.scene_states.values(), key=lambda item: item.plan.scene_id)
            ]
            ctx.continuity_warnings.append(
                f"Scene {state.plan.scene_id} 编码前自动修复连续性合同：" + "；".join(repairs)
            )
            self._write_stage_artifact(
                ctx,
                f"scene_{state.plan.scene_id}_plan.json",
                {"schema_version": 1, "plan": normalized.model_dump(mode="json")},
            )
            self._checkpoint(ctx, State.REVIEWING)
        self._emit(
            "continuity_contract_repaired",
            scene_id=state.plan.scene_id,
            repairs=repairs,
        )

    @staticmethod
    def _technical_input_hash(ctx: PipelineContext, state: SceneState) -> str:
        inherited_ids = {item.element_id for item in state.plan.inherited_elements}
        payload = {
            "plan": state.plan.model_dump(mode="json"),
            "lesson_spec": ctx.lesson_spec.model_dump(mode="json"),
            "teaching_graph": ctx.teaching_graph.model_dump(mode="json"),
            "inherited_elements_code": state.inherited_elements_code,
            "element_manifest": [
                entry.model_dump(mode="json")
                for entry in sorted(
                    ctx.element_manifest.for_elements(inherited_ids),
                    key=lambda item: item.element_id,
                )
            ],
            "renderer": ctx.render_profile.renderer,
        }
        return sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))

    def _ensure_technical_spec(self, ctx: PipelineContext, state: SceneState) -> None:
        """确保当前场景在 Coder 前拥有与输入匹配的 TechnicalSpec。"""

        input_sha256 = self._technical_input_hash(ctx, state)
        if (
            state.technical_spec is not None
            and state.technical_status == "passed"
            and state.technical_input_sha256 == input_sha256
            and state.technical_spec_sha256 == sha256_text(state.technical_spec.model_dump_json())
        ):
            result = compile_technical_spec(
                state.plan,
                state.technical_spec,
                renderer=ctx.render_profile.renderer,
            )
            if result.is_valid:
                return

        # 计划或继承上下文发生变化时，旧代码不能继续使用旧技术合同。
        if (
            state.technical_input_sha256
            and state.technical_input_sha256 != input_sha256
            and state.code
            and not state.rendered
        ):
            self._cancel_unfinished_scene_job(state, reason="仍使用旧技术合同")
            state.code = ""
            state.class_name = ""
            state.reviewed = False
            state.rewrite_feedback = ""
            state.review_signature = ""
            state.identical_review_count = 0
            state.artifact = None
            state.rendered = False
            state.slurm_job = None
            state.exported_elements_code = ""
            state.exported_elements = []
            self._remove_element_manifest_scene(ctx, state.plan.scene_id)
            self._reset_visual_receipt(ctx, state, clear_candidate=True, reset_attempts=True)
            self._write_private(ctx.paths.scenes / f"scene_{state.plan.scene_id}.py", "")

        scene_id = state.plan.scene_id
        legacy_reviewed_code = (
            bool(state.code)
            and state.reviewed
            and state.technical_spec is None
            and state.technical_status == "pending"
            and not state.rendered
        )
        self._phase_emit("technical")
        self._emit("scene_technical_planning", scene_id=scene_id, title=state.plan.title)
        with self._state_lock:
            state.technical_status = "generating"
            state.technical_error = ""
            self._checkpoint(ctx, State.REVIEWING)

        rag_context = self._retrieve_rag(
            ctx,
            "\n".join(
                (
                    state.plan.title,
                    state.plan.math_concept,
                    state.plan.computation,
                    state.plan.camera_movement,
                    ctx.render_profile.renderer,
                )
            ),
            receipt_key=f"scene:{scene_id}:technical",
            stage="technical",
            source_kinds={"manim_doc", "example", "recipe"},
            preferred_source_kinds={"recipe"},
            inherited_elements_sha256=sha256_text(state.inherited_elements_code)
            if state.inherited_elements_code
            else "",
        )
        inherited_ids = {item.element_id for item in state.plan.inherited_elements}
        manifest = (
            ctx.element_manifest.model_copy(
                update={"entries": ctx.element_manifest.for_elements(inherited_ids)}
            )
            if inherited_ids
            else None
        )
        try:
            technical_planner = TechnicalPlannerAgent()
            planner_supports_feedback = self._supports_keyword(technical_planner.plan, "feedback")
            feedback = ""
            spec: TechnicalSpec | None = None
            result = None
            max_attempts = max(1, settings.MAX_TECHNICAL_SPEC_ATTEMPTS)
            for attempt in range(1, max_attempts + 1):
                optional_kwargs: dict[str, object] = {
                    "continuity_bible": ctx.continuity_bible,
                    "inherited_elements_code": state.inherited_elements_code,
                    "element_manifest": manifest,
                    "renderer": ctx.render_profile.renderer,
                    "rag_context": rag_context,
                    "stream": False,
                    "lesson_spec": ctx.lesson_spec,
                    "teaching_graph": ctx.teaching_graph,
                }
                planner_kwargs: dict[str, object] = {
                    key: value
                    for key, value in optional_kwargs.items()
                    if self._supports_keyword(technical_planner.plan, key)
                }
                if feedback and planner_supports_feedback:
                    planner_kwargs["feedback"] = feedback
                with self._llm_slot():
                    spec = technical_planner.plan(state.plan, **planner_kwargs)
                spec, contract_repairs = normalize_technical_spec_contract(
                    state.plan,
                    spec,
                    renderer=ctx.render_profile.renderer,
                )
                if contract_repairs:
                    ctx.continuity_warnings.extend(
                        f"Scene {scene_id} TechnicalSpec 自动对齐：{repair}"
                        for repair in contract_repairs
                    )
                result = compile_technical_spec(
                    state.plan,
                    spec,
                    renderer=ctx.render_profile.renderer,
                )
                if result.is_valid:
                    break
                feedback = "\n".join(f"- {error}" for error in result.errors)[:18_000]
                if attempt >= max_attempts or not planner_supports_feedback:
                    raise ValidationError(
                        "TechnicalSpec 未通过确定性编译：\n" + feedback,
                        hint="请修正技术计划后再生成代码",
                    )
            if spec is None or result is None or not result.is_valid:
                raise ValidationError(
                    "TechnicalSpec 未生成有效结果",
                    hint="请修正技术计划后再生成代码",
                )
        except Exception as exc:
            with self._state_lock:
                state.technical_status = "failed"
                state.technical_error = str(exc)[:50_000]
                self._checkpoint(ctx, State.REVIEWING)
            self._emit("scene_technical_failed", scene_id=scene_id, reason=state.technical_error)
            raise

        spec_sha256 = sha256_text(spec.model_dump_json())
        with self._state_lock:
            state.technical_spec = spec
            state.technical_spec_sha256 = spec_sha256
            state.technical_input_sha256 = input_sha256
            state.technical_status = "passed"
            state.technical_error = ""
            if state.code and not state.rendered and not legacy_reviewed_code:
                # 具有新技术合同的代码必须重新经过生命周期校验和 Reviewer，
                # 即使旧计划没有结构化元素，也不能绕过新的代码审查屏障。
                state.reviewed = False
            if result.warnings:
                ctx.continuity_warnings.extend(
                    f"Scene {scene_id} TechnicalSpec: {warning}" for warning in result.warnings
                )
            self._write_stage_artifact(
                ctx,
                f"scene_{scene_id}_technical_spec.json",
                {
                    "schema_version": 1,
                    "scene_id": scene_id,
                    "input_sha256": input_sha256,
                    "spec_sha256": spec_sha256,
                    "warnings": list(result.warnings),
                    "spec": spec.model_dump(mode="json"),
                },
            )
            self._checkpoint(ctx, State.REVIEWING)
        self._emit("scene_technical_ready", scene_id=scene_id)

    def _run_detail_barrier(self, ctx: PipelineContext) -> None:
        """并行完成尚未生成的场景分镜，作为连续性审查的屏障。"""

        import threading

        threads: list[threading.Thread] = []
        for scene_id, state in sorted(ctx.scene_states.items()):
            if state.plan_ready or state.rendered or state.failed or state.give_up:
                continue
            thread = threading.Thread(
                target=self._detail_worker,
                args=(ctx, scene_id, state),
                name=f"scene-detail-{scene_id}",
                daemon=True,
            )
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()

    def _detail_worker(self, ctx: PipelineContext, scene_id: int, state: SceneState) -> None:
        """连续性屏障中的单场景 Detail worker。"""

        try:
            if self._stop_event.is_set():
                return
            self._phase_emit("detailing")
            self._scene_detail(ctx, scene_id, state)
        except Exception as exc:
            with self._state_lock:
                self._mark_failed(state, f"Scene {scene_id} 分镜生成失败: {exc}", "planning")
            try:
                self._checkpoint(ctx, State.DETAILING)
            except Exception as checkpoint_error:
                self._record_checkpoint_failure(checkpoint_error)
            self._emit("scene_failed", scene_id=scene_id, reason=str(exc))

    def _normalize_pending_scene_contracts(self, ctx: PipelineContext) -> None:
        """在连续性审查前修复所有尚未编码场景的机械合同错误。"""

        if ctx.continuity_bible is None:
            return
        changed = False
        for scene_id, state in sorted(ctx.scene_states.items()):
            if state.failed or state.give_up or state.reviewed or not state.plan_ready:
                continue
            previous_plan = None
            previous_state = ctx.scene_states.get(scene_id - 1)
            if previous_state is not None and previous_state.plan_ready:
                previous_plan = previous_state.plan
            normalized, repairs = normalize_scene_plan_contract(
                state.plan,
                ctx.continuity_bible,
                previous_plan=previous_plan,
                has_next_scene=any(
                    item.plan.scene_id > state.plan.scene_id
                    for item in ctx.scene_states.values()
                    if item.plan_ready
                ),
            )
            if not repairs:
                continue
            state.plan = normalized
            self._reset_technical_spec(state)
            changed = True
            reason = "；".join(repairs)
            ctx.continuity_warnings.append(f"Scene {scene_id} 已自动修复连续性合同：{reason}")
            self._emit(
                "continuity_contract_repaired",
                scene_id=scene_id,
                repairs=repairs,
            )
        if changed:
            ctx.scenes = [
                state.plan
                for state in sorted(ctx.scene_states.values(), key=lambda item: item.plan.scene_id)
            ]
            self._checkpoint(ctx, State.DETAILING)

    def _normalize_scene_claim_contracts(self, ctx: PipelineContext) -> bool:
        """对齐 Detail 阶段的数学断言和时间线证据。

        概要阶段已经锁定了每个场景的 ``claim_ids``。Detail 模型有时会把
        前置推导拆成 ``claim_4_derive_1`` 等新 ID，或遗漏概要断言的详细
        ``math_claims``/时间线绑定，随后计划编译器会在每一轮重复报告同一个
        错误。额外断言不是可由视觉合同推断的内容，因而可以确定地删除；
        缺少的已锁定断言则从全片 LessonSpec 原样补回，并绑定到最相关的
        时间线事件。这里不创造新的数学事实，只恢复概要阶段已经批准的
        教学合同。
        """

        changed = False
        lesson_claims = {claim.claim_id: claim for claim in ctx.lesson_spec.claims}

        def claim_event_score(claim_text: str, event: object) -> int:
            event_text = str(getattr(event, "action", "")).lower()
            terms = re.findall(
                r"[a-z][a-z0-9_]{1,}|\d+(?:\.\d+)?|[\u4e00-\u9fff]{2,}",
                claim_text.lower(),
            )
            score = sum(3 for term in set(terms) if term in event_text)
            score += sum(
                1
                for marker in (
                    "公式",
                    "曲面",
                    "函数",
                    "方程",
                    "结论",
                    "展示",
                    "绘制",
                    "计算",
                    "误差",
                    "切平面",
                    "偏导",
                    "全微分",
                )
                if marker in event_text
            )
            return score

        for state in sorted(ctx.scene_states.values(), key=lambda item: item.plan.scene_id):
            if (
                state.failed
                or state.give_up
                or not state.plan_ready
                or state.code
                or state.rendered
            ):
                continue
            allowed = set(state.plan.claim_ids)
            extra_claims = [
                claim.claim_id for claim in state.plan.math_claims if claim.claim_id not in allowed
            ]
            normalized_claims = [
                claim for claim in state.plan.math_claims if claim.claim_id in allowed
            ]
            present_claims = {claim.claim_id for claim in normalized_claims}
            missing_claims = [
                claim_id
                for claim_id in state.plan.claim_ids
                if claim_id not in present_claims and claim_id in lesson_claims
            ]
            normalized_claims.extend(
                lesson_claims[claim_id].model_copy(deep=True) for claim_id in missing_claims
            )
            normalized_timeline = [
                event.model_copy(
                    update={
                        "math_claim_ids": [
                            claim_id for claim_id in event.math_claim_ids if claim_id in allowed
                        ]
                    }
                )
                if any(claim_id not in allowed for claim_id in event.math_claim_ids)
                else event
                for event in state.plan.timeline
            ]
            timeline_claims = {
                claim_id for event in normalized_timeline for claim_id in event.math_claim_ids
            }
            missing_timeline_claims = [
                claim_id
                for claim_id in state.plan.claim_ids
                if (claim_id in present_claims or claim_id in missing_claims)
                and claim_id not in timeline_claims
            ]
            if missing_timeline_claims and normalized_timeline:
                events = list(normalized_timeline)
                for claim_id in missing_timeline_claims:
                    claim = next(
                        (item for item in normalized_claims if item.claim_id == claim_id),
                        None,
                    )
                    claim_text = (
                        " ".join(
                            str(value)
                            for value in (
                                getattr(claim, "statement", ""),
                                getattr(claim, "expression_before", ""),
                                getattr(claim, "expression_after", ""),
                            )
                        )
                        if claim is not None
                        else claim_id
                    )
                    target_index = max(
                        range(len(events)),
                        key=lambda index: (
                            claim_event_score(claim_text, events[index]),
                            events[index].end_seconds - events[index].start_seconds,
                            -events[index].start_seconds,
                        ),
                    )
                    target = events[target_index]
                    events[target_index] = target.model_copy(
                        update={"math_claim_ids": [*target.math_claim_ids, claim_id]}
                    )
                normalized_timeline = events
            if (
                not extra_claims
                and not missing_claims
                and not missing_timeline_claims
                and normalized_timeline == state.plan.timeline
            ):
                continue
            state.plan = state.plan.model_copy(
                update={
                    "math_claims": normalized_claims,
                    "timeline": normalized_timeline,
                }
            )
            state.plan_reviewed = False
            state.plan_review_round = 0
            state.plan_review_feedback = ""
            state.plan_review_signature = ""
            state.identical_plan_review_count = 0
            self._reset_technical_spec(state)
            changed = True
            if extra_claims:
                ctx.continuity_warnings.append(
                    f"Scene {state.plan.scene_id} 已删除 Detail 擅自增加的数学断言: "
                    + ", ".join(sorted(extra_claims))
                )
            if missing_claims:
                ctx.continuity_warnings.append(
                    f"Scene {state.plan.scene_id} 已从 LessonSpec 补齐数学断言: "
                    + ", ".join(sorted(missing_claims))
                )
            if missing_timeline_claims:
                ctx.continuity_warnings.append(
                    f"Scene {state.plan.scene_id} 已将断言绑定到时间线证据: "
                    + ", ".join(sorted(missing_timeline_claims))
                )
            self._emit(
                "plan_claim_contract_repaired",
                scene_id=state.plan.scene_id,
                removed_claim_ids=extra_claims,
            )
            self._write_stage_artifact(
                ctx,
                f"scene_{state.plan.scene_id}_plan.json",
                {"schema_version": 1, "plan": state.plan.model_dump(mode="json")},
            )
        if changed:
            ctx.scenes = [
                state.plan
                for state in sorted(ctx.scene_states.values(), key=lambda item: item.plan.scene_id)
            ]
            ctx.plan_review_status = "pending"
            ctx.continuity_review_status = "pending"
        return changed

    def _compile_scene_plans(self, ctx: PipelineContext) -> None:
        """在 LLM 计划审查前执行一次确定性计划编译。"""

        claim_contracts_changed = self._normalize_scene_claim_contracts(ctx)
        transition_claims_changed = self._normalize_transition_claim_contracts(ctx)
        dangling_handoffs_changed = self._normalize_dangling_handoffs(ctx)
        timeline_changed = False
        for state in sorted(ctx.scene_states.values(), key=lambda item: item.plan.scene_id):
            # 已经有代码/视频的场景不能在恢复时静默改写时间线；它们
            # 必须沿用原合同或由显式重规划流程处理。尚未编码的计划则
            # 可以安全吸收模型遗漏的末尾定格时间。
            if (
                state.failed
                or state.give_up
                or not state.plan_ready
                or state.code
                or state.rendered
            ):
                continue
            normalized_plan, repairs = normalize_scene_timeline_contract(state.plan)
            if not repairs:
                continue
            state.plan = normalized_plan
            self._reset_technical_spec(state)
            state.plan_reviewed = False
            state.plan_review_round = 0
            state.plan_review_feedback = ""
            state.plan_review_signature = ""
            state.identical_plan_review_count = 0
            timeline_changed = True
            ctx.continuity_warnings.append(
                f"Scene {state.plan.scene_id} 已自动补齐时间线：" + "；".join(repairs)
            )
            self._emit(
                "continuity_contract_repaired",
                scene_id=state.plan.scene_id,
                repairs=repairs,
            )
        if (
            claim_contracts_changed
            or transition_claims_changed
            or dangling_handoffs_changed
            or timeline_changed
        ):
            ctx.plan_review_status = "pending"
            ctx.scenes = [
                state.plan
                for state in sorted(ctx.scene_states.values(), key=lambda item: item.plan.scene_id)
            ]
            self._checkpoint(ctx, State.PLAN_REVIEWING)

        plans = [
            state.plan
            for state in sorted(ctx.scene_states.values(), key=lambda item: item.plan.scene_id)
            if not state.failed and not state.give_up and state.plan_ready
        ]
        result = PlanCompiler().compile(
            ctx.outlines,
            plans,
            ctx.continuity_bible,
            ctx.lesson_spec,
            ctx.teaching_graph,
        )
        # 视觉/人工诊断产生的计划层问题必须跨越一次重新编译保留下来；否则
        # 下一轮只重算确定性问题，会把“为什么要回到 Planner”丢掉，模型很
        # 容易再次生成同一份方案。重规划成功后由计划审查屏障显式清除。
        by_scene: dict[int, list[PlanReviewIssue]] = {
            scene_id: [issue for issue in issues if issue.field == "visual_evaluation"]
            for scene_id, issues in ctx.plan_compile_issues.items()
            if any(issue.field == "visual_evaluation" for issue in issues)
        }
        for issue in result.issues:
            target_ids = issue.scene_ids or [item.scene_id for item in plans]
            converted = PlanReviewIssue(
                category=issue.category,
                severity=issue.severity,
                field=issue.field,
                message=issue.message,
                fix_instruction=issue.fix_instruction,
            )
            for scene_id in target_ids:
                by_scene.setdefault(scene_id, []).append(converted)
        ctx.plan_compile_issues = by_scene
        if result.issues and ctx.plan_review_status != "skipped":
            # 旧运行可能已经把计划标记为 passed，但新版本的确定性编译器
            # 发现了此前未检查的错误。不能直接进入 TechnicalSpec/Coder，
            # 否则 resume 会重复失败在同一个下游阶段。
            affected_ids = {
                scene_id
                for issue in result.issues
                for scene_id in (issue.scene_ids or [item.scene_id for item in plans])
            }
            for state in ctx.scene_states.values():
                if state.plan.scene_id not in affected_ids or state.failed or state.give_up:
                    continue
                state.plan_reviewed = False
                state.plan_review_round = 0
                state.plan_review_feedback = ""
                state.plan_review_signature = ""
                state.identical_plan_review_count = 0
            ctx.plan_review_status = "pending"
        payload = {
            "schema_version": 1,
            "is_valid": result.is_valid,
            "issues": [issue.model_dump(mode="json") for issue in result.issues],
        }
        self._write_stage_artifact(ctx, "plan_compile.json", payload)

    def _normalize_dangling_handoffs(self, ctx: PipelineContext) -> bool:
        """删除下一场景没有声明接管的临时 handoff 对象。

        handoff 是当前场景到下一场景的边界合同。模型常把章节标题、
        预览公式或过渡箭头标成 keep/create，却没有在下一场景的
        inherited_elements 中声明它们；若继续保留，Coder 会被迫导出
        没有消费者的对象，并触发连续性审查往返。对本场景新增对象，
        这种情况可以确定地降级为 optional 并移出 handoff。
        """

        ordered_states = [
            state
            for state in sorted(ctx.scene_states.values(), key=lambda item: item.plan.scene_id)
            if state.plan_ready
        ]
        changed = False
        for index, state in enumerate(ordered_states[:-1]):
            next_state = ordered_states[index + 1]
            if state.code or state.rendered or next_state.code or next_state.rendered:
                continue
            next_inherited_ids = {item.element_id for item in next_state.plan.inherited_elements}
            persistent_handoff_ids = {
                item.element_id
                for item in state.plan.handoff
                if item.action in {"inherit", "keep", "create"}
            }
            dangling_ids = persistent_handoff_ids - next_inherited_ids
            if not dangling_ids:
                continue

            new_ids = {item.element_id for item in state.plan.new_elements}
            dangling_new_ids = dangling_ids & new_ids
            if not dangling_new_ids:
                # 继承对象的去留涉及真实边界状态，交给 Plan Reviewer，
                # 不在这里擅自改变上一场景的收场语义。
                continue

            new_elements = [
                (
                    item.model_copy(update={"required": False})
                    if item.element_id in dangling_new_ids and item.required
                    else item
                )
                for item in state.plan.new_elements
            ]
            handoff = [
                item for item in state.plan.handoff if item.element_id not in dangling_new_ids
            ]
            state.plan = state.plan.model_copy(
                update={"new_elements": new_elements, "handoff": handoff}
            )
            state.plan_reviewed = False
            state.plan_review_round = 0
            state.plan_review_feedback = ""
            state.plan_review_signature = ""
            state.identical_plan_review_count = 0
            self._reset_technical_spec(state)
            changed = True
            repair = (
                f"Scene {state.plan.scene_id} 的 handoff 对象未被 Scene "
                f"{next_state.plan.scene_id} 接管，已改为场景内临时对象: "
                + ", ".join(sorted(dangling_new_ids))
            )
            ctx.continuity_warnings.append(repair)
            self._emit(
                "continuity_contract_repaired",
                scene_id=state.plan.scene_id,
                repairs=[repair],
            )
        if changed:
            ctx.scenes = [
                state.plan
                for state in sorted(ctx.scene_states.values(), key=lambda item: item.plan.scene_id)
            ]
            ctx.continuity_review_status = "pending"
        return changed

    def _normalize_transition_claim_contracts(self, ctx: PipelineContext) -> bool:
        """在 Detail 后修正过渡场景误占用教学断言的概要合同。

        断言归属在概要阶段锁定，Detail/Plan Review 只能补充该场景的
        证据，不能把过渡场景变成数学推导场景。恢复旧运行时也需要执行
        同一迁移，否则旧 manifest 会继续把相同 claim_ids 传给错误场景。
        """

        original_outlines = list(ctx.outlines)
        normalized_outlines, assignments = normalize_transition_claim_assignments(
            ctx.outlines,
            ctx.teaching_graph.scene_claims,
        )
        outlines_changed = normalized_outlines != ctx.outlines
        graph_changed = assignments != ctx.teaching_graph.scene_claims
        if not outlines_changed and not graph_changed:
            return False

        outline_by_id = {outline.scene_id: outline for outline in normalized_outlines}
        changed = outlines_changed or graph_changed
        for state in sorted(ctx.scene_states.values(), key=lambda item: item.plan.scene_id):
            outline = outline_by_id.get(state.plan.scene_id)
            if outline is None or state.code or state.rendered:
                continue
            next_claim_ids = list(outline.claim_ids)
            if state.plan.claim_ids == next_claim_ids:
                continue
            update: dict[str, object] = {"claim_ids": next_claim_ids}
            if not next_claim_ids:
                update["math_claims"] = []
                update["timeline"] = [
                    event.model_copy(update={"math_claim_ids": []}) for event in state.plan.timeline
                ]
            state.plan = state.plan.model_copy(update=update)
            state.plan_reviewed = False
            state.plan_review_round = 0
            state.plan_review_feedback = ""
            state.plan_review_signature = ""
            state.identical_plan_review_count = 0
            self._reset_technical_spec(state)

        ctx.outlines = normalized_outlines
        ctx.teaching_graph = ctx.teaching_graph.model_copy(update={"scene_claims": assignments})
        ctx.scenes = [
            state.plan
            for state in sorted(ctx.scene_states.values(), key=lambda item: item.plan.scene_id)
        ]
        ctx.plan_review_status = "pending"
        ctx.continuity_review_status = "pending"
        moved = [
            outline.scene_id
            for before, outline in zip(original_outlines, normalized_outlines, strict=True)
            if before.claim_ids != outline.claim_ids
        ]
        if moved:
            ctx.continuity_warnings.append(
                "已将过渡场景的数学断言重新分配到教学场景: " + ", ".join(map(str, moved))
            )
        self._emit(
            "continuity_contract_repaired",
            repairs=["过渡场景不再承担核心数学断言"],
        )
        return changed

    @staticmethod
    def _plan_review_feedback(issues: list[PlanReviewIssue]) -> str:
        return "\n\n".join(
            f"[{issue.category}] 字段 {issue.field or '未指定'}: {issue.message}\n"
            + (f"证据: {issue.evidence}\n" if issue.evidence else "")
            + f"修正要求: {issue.fix_instruction}"
            for issue in issues
        )

    def _repair_plan_handoff_issues(
        self,
        ctx: PipelineContext,
        scene_id: int,
        state: SceneState,
        issues: list[PlanReviewIssue],
    ) -> list[str]:
        """机械修复计划审查中可由 element_id 直接确定的交接问题。

        Plan Reviewer 经常能够准确指出“某个 new_element 应该进入
        handoff”，但 Planner 重规划时又把同一对象改回 optional，形成
        ``review -> replan -> review`` 的无效往返。对于带有明确 element_id
        和 handoff/inherited_elements 字段的问题，可以直接同步前后场景的
        边界合同；数学内容和没有明确对象身份的语义问题仍交给 Planner。
        """

        if (
            state.failed
            or state.give_up
            or state.code
            or state.rendered
            or state.slurm_job is not None
        ):
            return []
        contract_issues = [
            issue
            for issue in issues
            if issue.category == "contract"
            and issue.field in {"handoff", "inherited_elements", "new_elements"}
        ]
        # 交接机械修复只能处理“元素缺失/角色不闭合”这类纯合同问题。
        # 若同一轮还存在几何、数学、时序或 renderer 阻断，优先让
        # Planner 针对完整反馈重写方案；否则这里修改邻接边界后立刻
        # 触发重启，下一轮又会看到同一几何问题并重复修改，形成
        # ``mechanical repair -> restart -> mechanical repair`` 的死循环。
        if not contract_issues or any(issue.category != "contract" for issue in issues):
            return []

        ordered_states = sorted(ctx.scene_states.values(), key=lambda item: item.plan.scene_id)
        state_by_id = {item.plan.scene_id: item for item in ordered_states}
        known_element_ids = {
            item.element_id
            for candidate in ordered_states
            for element_group in (
                candidate.plan.inherited_elements,
                candidate.plan.elements_to_remove,
                candidate.plan.new_elements,
            )
            for item in element_group
        }
        issue_text = "\n".join(
            f"{issue.message}\n{issue.fix_instruction}" for issue in contract_issues
        )
        element_ids = [
            element_id
            for element_id in sorted(known_element_ids, key=lambda item: (-len(item), item))
            if re.search(
                rf"(?<![A-Za-z0-9_.-]){re.escape(element_id)}(?![A-Za-z0-9_.-])",
                issue_text,
            )
        ]
        if not element_ids:
            return []

        mentioned_scene_ids = {
            int(value)
            for value in re.findall(r"(?:场景|scene)\s*([0-9]+)", issue_text, re.IGNORECASE)
        }
        next_state = next(
            (candidate for candidate in ordered_states if candidate.plan.scene_id > scene_id),
            None,
        )
        target_state = next(
            (
                state_by_id[target_id]
                for target_id in sorted(mentioned_scene_ids)
                if target_id > scene_id and target_id in state_by_id
            ),
            None,
        )
        if target_state is None and (
            next_state is not None
            and any(
                marker in issue_text
                for marker in ("下一场景", "后续场景", "传递", "交接", "handoff")
            )
        ):
            target_state = next_state

        previous_state = state_by_id.get(scene_id - 1)
        changed_states: set[int] = set()
        repairs: list[str] = []

        # “上一场景已移除”与“下一场景必须继承”是两个互斥的生命周期
        # 结论。旧的机械修复只看到了 Reviewer 提到的 element_id，就把
        # 已经在上一场景退出的对象重新提升成 keep，随后
        # normalize_scene_plan_contract 又依据 remove 合同把它删掉；这
        # 会触发 ``repair -> re-review -> repair`` 的无效循环，并且还会
        # 把本来已经通过审查的上一场景改回未审查。此类冲突必须交给
        # Planner 重写 opening/transition 文本，不能伪造一份不存在的
        # 上游导出。
        previous_removed_ids = (
            {item.element_id for item in previous_state.plan.elements_to_remove}
            if previous_state is not None
            else set()
        )
        previous_removed_ids.update(
            item.element_id
            for item in (previous_state.plan.handoff if previous_state is not None else [])
            if item.action == "remove"
        )
        if previous_state is not None:
            previous_closing = " ".join(previous_state.plan.closing_state).lower()
            if re.search(r"(?:所有|全部|整体).{0,20}(?:淡出|消失|清空|移除)", previous_closing):
                previous_removed_ids.update(
                    item.element_id
                    for item in (
                        *previous_state.plan.inherited_elements,
                        *previous_state.plan.new_elements,
                    )
                )

        def find_element(plan: ScenePlan, element_id: str) -> tuple[str, VisualElementState] | None:
            for group_name, elements in (
                ("inherited_elements", plan.inherited_elements),
                ("elements_to_remove", plan.elements_to_remove),
                ("new_elements", plan.new_elements),
            ):
                for element in elements:
                    if element.element_id == element_id:
                        return group_name, element
            return None

        def replace_element(
            plan: ScenePlan,
            group_name: str,
            element_id: str,
            replacement: VisualElementState,
        ) -> ScenePlan:
            elements = list(getattr(plan, group_name))
            for index, element in enumerate(elements):
                if element.element_id == element_id:
                    elements[index] = replacement
                    return plan.model_copy(update={group_name: elements})
            return plan

        def upsert_handoff(
            plan: ScenePlan,
            element: VisualElementState,
            action: str,
        ) -> ScenePlan:
            handoff = list(plan.handoff)
            transition = (
                f"Scene {plan.scene_id} {element.element_id} 已建立/接管，"
                "在下一场景保持同一变量名和视觉状态"
            )
            candidate = SceneHandoff(
                element_id=element.element_id,
                variable_name=element.variable_name,
                action=action,
                semantic_state=element.semantic_state,
                transition=transition,
            )
            for index, existing in enumerate(handoff):
                if existing.element_id == element.element_id:
                    if existing != candidate:
                        handoff[index] = existing.model_copy(
                            update={
                                "variable_name": element.variable_name or existing.variable_name,
                                "action": action,
                                "semantic_state": (
                                    element.semantic_state or existing.semantic_state
                                ),
                                "transition": existing.transition or transition,
                            }
                        )
                    return plan.model_copy(update={"handoff": handoff})
            handoff.append(candidate)
            return plan.model_copy(update={"handoff": handoff})

        def add_inherited(plan: ScenePlan, element: VisualElementState) -> ScenePlan:
            existing = find_element(plan, element.element_id)
            inherited = element.model_copy(update={"required": True})
            if existing is not None:
                group_name, current = existing
                if group_name == "inherited_elements":
                    aligned = current.model_copy(
                        update={
                            "required": True,
                            "variable_name": element.variable_name or current.variable_name,
                        }
                    )
                    return replace_element(plan, group_name, element.element_id, aligned)
                return plan
            return plan.model_copy(
                update={"inherited_elements": [*plan.inherited_elements, inherited]}
            )

        def mark_boundary(state_to_update: SceneState, plan: ScenePlan) -> None:
            if plan == state_to_update.plan:
                return
            state_to_update.plan = plan
            changed_states.add(plan.scene_id)

        for element_id in element_ids:
            current_declaration = find_element(state.plan, element_id)
            previous_declaration = (
                find_element(previous_state.plan, element_id)
                if previous_state is not None
                else None
            )

            # 当前计划已经有完整 inherited + handoff 合同的元素不需要再次
            # “修复”。Reviewer 可能只是在报告 semantic_state 与自由文本
            # 的轻微措辞差异；重复执行下面的 upsert 会不断重写相邻场景，
            # 令正常的几何/数学重规划永远到不了预算耗尽分支。
            if current_declaration is not None and current_declaration[0] == "inherited":
                current_element = current_declaration[1]
                current_handoff = next(
                    (item for item in state.plan.handoff if item.element_id == element_id),
                    None,
                )
                current_removed = any(
                    item.element_id == element_id for item in state.plan.elements_to_remove
                )
                expected_actions = {"remove"} if current_removed else {"inherit", "keep"}
                if (
                    current_element.required
                    and current_handoff is not None
                    and current_handoff.action in expected_actions
                ):
                    continue

            # 当前场景新建的元素：提升为边界对象，并同步到下一场景。
            if current_declaration is not None and current_declaration[0] == "new_elements":
                # 最后一个场景没有消费者。若它的 closing/timeline 已经
                # 结束时淡出该对象，所谓“必须加入 handoff”是审查模型把
                # 场景内对象误当成交接对象；提升后又会被终态规范器改回
                # optional，造成机械修复永远返回 changed。没有下一场景时
                # 直接保留场景内生命周期，交给正常的计划审查即可。
                if target_state is None:
                    continue
                _, element = current_declaration
                promoted = element.model_copy(update={"required": True})
                current_plan = replace_element(
                    state.plan,
                    "new_elements",
                    element_id,
                    promoted,
                )
                if target_state is not None:
                    current_plan = upsert_handoff(current_plan, promoted, "create")
                    target_plan = add_inherited(target_state.plan, promoted)
                    target_plan = upsert_handoff(target_plan, promoted, "keep")
                    mark_boundary(target_state, target_plan)
                    repairs.append(
                        f"Scene {scene_id} 的 {element_id} 已提升为 required，并交接给 "
                        f"Scene {target_state.plan.scene_id}"
                    )
                mark_boundary(state, current_plan)
                continue

            # 当前场景缺少但上一场景存在的元素：补齐上一场景导出、当前
            # inherited 以及当前到下一场景的交接。
            if previous_state is not None and previous_declaration is not None:
                previous_group, previous_element = previous_declaration
                if element_id in previous_removed_ids:
                    # 上游已经明确负责淡出/移除；当前场景若仍在
                    # opening_state 或 transition_in 中提及它，应通过
                    # 计划重规划改成“本场景重新创建/独立开始”，而不是
                    # 在交接合同中重新制造一个可继承对象。
                    continue
                promoted_previous = previous_element.model_copy(update={"required": True})
                previous_plan = replace_element(
                    previous_state.plan,
                    previous_group,
                    element_id,
                    promoted_previous,
                )
                previous_plan = upsert_handoff(
                    previous_plan,
                    promoted_previous,
                    "create" if previous_group == "new_elements" else "keep",
                )
                mark_boundary(previous_state, previous_plan)

                current_plan = add_inherited(state.plan, promoted_previous)
                current_element = find_element(current_plan, element_id)
                if current_element is not None:
                    current_plan = upsert_handoff(current_plan, current_element[1], "keep")
                if target_state is not None:
                    target_plan = add_inherited(target_state.plan, promoted_previous)
                    target_element = find_element(target_plan, element_id)
                    if target_element is not None:
                        target_plan = upsert_handoff(target_plan, target_element[1], "keep")
                    mark_boundary(target_state, target_plan)
                mark_boundary(state, current_plan)
                repairs.append(
                    f"已将上一场景的 {element_id} 补入 Scene {scene_id} 的 inherited_elements"
                )

        if not changed_states:
            return []

        # 让刚补齐的上一场景声明成为后继场景的唯一来源，并吸收
        # required/handoff 的机械字段修复；不改写视觉和数学创作内容。
        normalization_scope = {scene_id}
        if target_state is not None:
            normalization_scope.add(target_state.plan.scene_id)
        for candidate in ordered_states:
            if candidate.plan.scene_id not in changed_states | normalization_scope:
                continue
            previous = state_by_id.get(candidate.plan.scene_id - 1)
            normalized, contract_repairs = normalize_scene_plan_contract(
                candidate.plan,
                ctx.continuity_bible or ContinuityBible(),
                previous_plan=previous.plan if previous is not None else None,
                has_next_scene=any(
                    item.plan.scene_id > candidate.plan.scene_id for item in ordered_states
                ),
            )
            if normalized != candidate.plan:
                candidate.plan = normalized
                changed_states.add(candidate.plan.scene_id)
            repairs.extend(
                f"Scene {candidate.plan.scene_id} 合同规范：{repair}" for repair in contract_repairs
            )

        with self._state_lock:
            for changed_scene_id in changed_states:
                changed_state = state_by_id[changed_scene_id]
                changed_state.plan_reviewed = False
                changed_state.plan_review_round = 0
                changed_state.plan_review_feedback = ""
                changed_state.plan_review_signature = ""
                changed_state.identical_plan_review_count = 0
                self._reset_technical_spec(changed_state)
                self._write_stage_artifact(
                    ctx,
                    f"scene_{changed_scene_id}_plan.json",
                    {
                        "schema_version": 1,
                        "plan": changed_state.plan.model_dump(mode="json"),
                    },
                )
            ctx.scenes = [candidate.plan for candidate in ordered_states]
            ctx.plan_review_status = "pending"
            ctx.continuity_review_status = "pending"
            self._checkpoint(ctx, State.PLAN_REVIEWING)
        return repairs

    def _plan_review_failure(
        self,
        ctx: PipelineContext,
        scene_id: int,
        state: SceneState,
        reason: str,
    ) -> None:
        with self._state_lock:
            state.failed = True
            state.give_up = False
            state.failure_category = "planning"
            state.failure_reason = reason[:50_000]
            ctx.plan_review_status = "failed"
            self._checkpoint(ctx, State.PLAN_REVIEWING)
        self._emit("scene_plan_review_fail", scene_id=scene_id, reason=reason)
        self._emit("scene_failed", scene_id=scene_id, reason=reason)

    def _run_plan_review_batch(
        self,
        ctx: PipelineContext,
        active_states: list[SceneState],
    ) -> dict[int, PlanReviewResult]:
        """优先一次审查初始整批计划；不支持/失败时由调用方逐场景回退。"""

        reviewer = PlanReviewerAgent()
        review_batch = getattr(reviewer, "review_batch", None)
        if not callable(review_batch):
            return {}
        deterministic_by_scene = {
            state.plan.scene_id: dedupe_plan_review_issues(
                [
                    *deterministic_plan_issues(
                        state.plan,
                        ctx.continuity_bible,
                        safe_fallback=state.safe_fallback_used,
                        lesson_spec=ctx.lesson_spec,
                    ),
                    *ctx.plan_compile_issues.get(state.plan.scene_id, []),
                ]
            )
            for state in active_states
        }
        try:
            with self._llm_sem:
                return review_batch(
                    [state.plan for state in active_states],
                    user_prompt=ctx.user_prompt,
                    continuity_bible=ctx.continuity_bible,
                    deterministic_by_scene=deterministic_by_scene,
                    renderer=ctx.render_profile.renderer,
                    lesson_spec=ctx.lesson_spec,
                    teaching_graph=ctx.teaching_graph,
                    safe_fallback_scene_ids={
                        state.plan.scene_id for state in active_states if state.safe_fallback_used
                    },
                )
        except TypeError:
            # 兼容外部替换的旧/简化批量接口，不把签名差异误报成规划失败。
            try:
                with self._llm_sem:
                    return review_batch(
                        [state.plan for state in active_states],
                        user_prompt=ctx.user_prompt,
                        continuity_bible=ctx.continuity_bible,
                        deterministic_by_scene=deterministic_by_scene,
                        renderer=ctx.render_profile.renderer,
                    )
            except Exception as exc:
                ctx.continuity_warnings.append(f"批量计划审查不可用，已回退逐场景审查: {exc}")
                return {}
        except Exception as exc:
            ctx.continuity_warnings.append(f"批量计划审查失败，已回退逐场景审查: {exc}")
            return {}

    def _run_plan_review_barrier(
        self,
        ctx: PipelineContext,
        *,
        _mechanical_restart_count: int = 0,
    ) -> None:
        """在 Coder 前逐场景审查计划，阻断不可实现或数学错误的方案。"""

        active_states = [
            state
            for state in ctx.scene_states.values()
            if not state.failed and not state.give_up and state.plan_ready
        ]
        if not active_states:
            ctx.plan_review_status = "passed"
            return

        self._emit("plan_reviewing", scene_count=len(active_states))
        max_rounds = max(1, settings.MAX_PLAN_REVIEW_ROUNDS)
        max_replans = max(1, settings.MAX_PLAN_REPLAN_ATTEMPTS)
        # ``plan_review_round`` 描述当前这份计划的审查轮数；每次重规划
        # 后它会归零。因此必须另行累计 Planner 调用次数，否则模型在
        # 同一冲突上反复返回等价计划时，内层 while 永远不会结束。
        replan_attempts: dict[int, int] = {}
        batch_results = self._run_plan_review_batch(
            ctx,
            [
                state
                for state in sorted(active_states, key=lambda item: item.plan.scene_id)
                if state.plan_review_round == 0
            ],
        )
        restart_required = False
        for scene_id, state in sorted(ctx.scene_states.items()):
            if state.failed or state.give_up or not state.plan_ready or state.plan_reviewed:
                continue
            while not state.plan_reviewed:
                if self._stop_event.is_set() or state.failed or state.give_up:
                    break
                self._emit("scene_plan_reviewing", scene_id=scene_id)
                deterministic = deterministic_plan_issues(
                    state.plan,
                    ctx.continuity_bible,
                    safe_fallback=state.safe_fallback_used,
                    lesson_spec=ctx.lesson_spec,
                )
                deterministic = dedupe_plan_review_issues(
                    [*deterministic, *ctx.plan_compile_issues.get(scene_id, [])]
                )
                result = batch_results.pop(scene_id, None)
                if result is None:
                    try:
                        with self._llm_sem:
                            result = PlanReviewerAgent().review(
                                state.plan,
                                user_prompt=ctx.user_prompt,
                                all_plans=[
                                    item.plan
                                    for item in sorted(
                                        ctx.scene_states.values(),
                                        key=lambda item: item.plan.scene_id,
                                    )
                                    if not item.failed and not item.give_up and item.plan_ready
                                ],
                                continuity_bible=ctx.continuity_bible,
                                deterministic_issues=deterministic,
                                renderer=ctx.render_profile.renderer,
                                safe_fallback=state.safe_fallback_used,
                                lesson_spec=ctx.lesson_spec,
                                teaching_graph=ctx.teaching_graph,
                            )
                    except Exception as exc:
                        self._plan_review_failure(
                            ctx,
                            scene_id,
                            state,
                            f"Scene {scene_id} 计划审查调用失败: {exc}",
                        )
                        break

                all_issues, issues, non_blocking_issues = classify_plan_review_issues(
                    state.plan,
                    deterministic_issues=deterministic,
                    result=result,
                )
                self._write_stage_artifact(
                    ctx,
                    f"plan_review_scene_{scene_id}_{state.plan_review_round + 1}.json",
                    {
                        "schema_version": 1,
                        "scene_id": scene_id,
                        "plan_sha256": sha256_text(state.plan.model_dump_json()),
                        "deterministic_issues": [
                            item.model_dump(mode="json") for item in deterministic
                        ],
                        "result": result.model_dump(mode="json"),
                        "issues": [item.model_dump(mode="json") for item in all_issues],
                        "blocking_issues": [item.model_dump(mode="json") for item in issues],
                        "warning_issues": [
                            item.model_dump(mode="json") for item in non_blocking_issues
                        ],
                    },
                )
                if non_blocking_issues:
                    warning_messages = [
                        f"Scene {scene_id} 计划审查提示：{issue.message}"
                        for issue in non_blocking_issues
                    ]
                    ctx.continuity_warnings.extend(warning_messages)
                    self._emit(
                        "scene_plan_review_warning",
                        scene_id=scene_id,
                        warnings=warning_messages[:20],
                    )
                if not issues:
                    with self._state_lock:
                        state.plan_reviewed = True
                        state.plan_review_round = 0
                        state.plan_review_feedback = ""
                        state.plan_review_signature = ""
                        state.identical_plan_review_count = 0
                        self._checkpoint(ctx, State.PLAN_REVIEWING)
                    self._emit("scene_plan_review_pass", scene_id=scene_id)
                    break

                feedback = self._plan_review_feedback(issues)
                mechanical_repairs = self._repair_plan_handoff_issues(
                    ctx,
                    scene_id,
                    state,
                    issues,
                )
                if mechanical_repairs:
                    ctx.continuity_warnings.append(
                        f"Scene {scene_id} 计划审查后自动修复交接合同："
                        + "；".join(mechanical_repairs)
                    )
                    self._emit(
                        "continuity_contract_repaired",
                        scene_id=scene_id,
                        repairs=mechanical_repairs,
                    )
                    self._compile_scene_plans(ctx)
                    batch_results.clear()
                    restart_required = True
                    break
                signature = sha256_text(
                    state.plan.model_dump_json() + "\n--- plan review ---\n" + feedback
                )[:16]
                with self._state_lock:
                    state.plan_review_round += 1
                    if state.plan_review_signature == signature:
                        state.identical_plan_review_count += 1
                    else:
                        state.plan_review_signature = signature
                        state.identical_plan_review_count = 1
                    state.plan_review_feedback = feedback
                    review_round = state.plan_review_round
                    identical_count = state.identical_plan_review_count
                    self._checkpoint(ctx, State.PLAN_REVIEWING)

                exhausted = (
                    review_round >= max_rounds
                    or identical_count >= settings.MAX_IDENTICAL_REVIEW_ATTEMPTS
                )
                if exhausted:
                    if self._activate_safe_fallback(ctx, scene_id, state, feedback):
                        if self._stop_event.is_set():
                            break
                        continue
                    self._plan_review_failure(
                        ctx,
                        scene_id,
                        state,
                        f"Scene {scene_id} 计划审查未通过（第 {review_round} 轮）：{feedback}",
                    )
                    break

                replan_count = replan_attempts.get(scene_id, 0)
                if replan_count >= max_replans:
                    # 每次重规划都会重置当前计划的审查轮数，因此复杂几何
                    # 方案可能永远到不了上面的 ``review_round`` 上限。重规划
                    # 预算耗尽本身也是明确的收敛信号：若反馈确认是高风险几何，
                    # 应优先切换为保守教学方案，而不是直接把场景判死。
                    if self._activate_safe_fallback(ctx, scene_id, state, feedback):
                        break
                    self._plan_review_failure(
                        ctx,
                        scene_id,
                        state,
                        f"Scene {scene_id} 计划重规划已达到最大次数 {max_replans}，"
                        f"仍未解决以下问题：{feedback}",
                    )
                    break
                replan_attempts[scene_id] = replan_count + 1
                ctx.continuity_warnings.append(
                    f"Scene {scene_id} 计划重规划尝试 {replan_count + 1}/{max_replans}"
                )
                outline = next(item for item in ctx.outlines if item.scene_id == scene_id)
                planner = PlannerAgent()
                replan_kwargs: dict[str, object] = {
                    "stream": False,
                    "renderer": ctx.render_profile.renderer,
                }
                if self._supports_keyword(planner.plan_detail, "continuity_bible"):
                    replan_kwargs["continuity_bible"] = ctx.continuity_bible
                if self._supports_keyword(planner.plan_detail, "continuity_feedback"):
                    replan_kwargs["continuity_feedback"] = (
                        "计划审查反馈（必须逐条修正，不能把不可实现方案原样保留）：\n" + feedback
                    )
                if self._supports_keyword(planner.plan_detail, "continuity_context"):
                    replan_kwargs["continuity_context"] = self._continuity_plan_context(
                        ctx, scene_id
                    )
                if self._supports_keyword(planner.plan_detail, "lesson_spec"):
                    replan_kwargs["lesson_spec"] = ctx.lesson_spec
                if self._supports_keyword(planner.plan_detail, "teaching_graph"):
                    replan_kwargs["teaching_graph"] = ctx.teaching_graph
                try:
                    with self._llm_sem:
                        revised_plan = planner.plan_detail(
                            outline,
                            ctx.outlines,
                            ctx.user_prompt,
                            **replan_kwargs,
                        )
                except Exception as exc:
                    self._plan_review_failure(
                        ctx,
                        scene_id,
                        state,
                        f"Scene {scene_id} 计划重规划失败: {exc}",
                    )
                    break
                previous_plan = None
                previous_state = ctx.scene_states.get(scene_id - 1)
                if previous_state is not None and previous_state.plan_ready:
                    previous_plan = previous_state.plan
                revised_plan, contract_repairs = normalize_scene_plan_contract(
                    revised_plan,
                    ctx.continuity_bible or ContinuityBible(),
                    previous_plan=previous_plan,
                    has_next_scene=any(
                        item.plan.scene_id > scene_id
                        for item in ctx.scene_states.values()
                        if item.plan_ready
                    ),
                )
                code_invalidated = bool(
                    state.code or state.reviewed or state.rendered or state.slurm_job
                )
                if code_invalidated:
                    self._cancel_unfinished_scene_job(state, reason="计划重规划")
                with self._state_lock:
                    state.plan = revised_plan
                    self._reset_technical_spec(state)
                    # 视觉诊断已随本次 plan_detail 传入 Planner；清除旧的
                    # 外部注入问题，避免下一轮把同一条反馈再次当作新发现。
                    ctx.plan_compile_issues[scene_id] = [
                        issue
                        for issue in ctx.plan_compile_issues.get(scene_id, [])
                        if issue.field != "visual_evaluation"
                    ]
                    # 重规划后是全新的计划；不能沿用旧计划已经消耗的审查
                    # 轮数/重复指纹，否则一次合法修正就会被立即判定为耗尽。
                    state.plan_review_feedback = ""
                    state.plan_review_round = 0
                    state.plan_review_signature = ""
                    state.identical_plan_review_count = 0
                    if code_invalidated:
                        state.code = ""
                        state.class_name = ""
                        state.reviewed = False
                        state.rewrite_feedback = ""
                        state.slurm_job = None
                        state.artifact = None
                        state.rendered = False
                        state.exported_elements_code = ""
                        state.exported_elements = []
                        self._remove_element_manifest_scene(ctx, scene_id)
                        state.failure_reason = ""
                        state.failure_category = ""
                        self._reset_visual_receipt(
                            ctx,
                            state,
                            clear_candidate=True,
                            reset_attempts=True,
                        )
                        self._write_private(ctx.paths.scenes / f"scene_{scene_id}.py", "")
                    if ctx.continuity_bible is not None:
                        # 连续性审查预算是整个运行级别的预算。计划重规划
                        # 会使状态重新进入 pending，但不能把已经消耗的
                        # review round 清零，否则规划器可以通过反复改写
                        # 计划无限绕过 MAX_CONTINUITY_FIX_ROUNDS。
                        ctx.continuity_review_status = "pending"
                    ctx.scenes = [
                        item.plan
                        for item in sorted(
                            ctx.scene_states.values(), key=lambda item: item.plan.scene_id
                        )
                    ]
                    if contract_repairs:
                        ctx.continuity_warnings.append(
                            f"Scene {scene_id} 计划重规划后自动修复合同："
                            + "；".join(contract_repairs)
                        )
                    self._checkpoint(ctx, State.PLAN_REVIEWING)
                if code_invalidated:
                    self._request_continuity_rebuild(
                        ctx,
                        scene_id,
                        reason="计划重规划后代码已失效",
                        include_failed=True,
                    )
                self._compile_scene_plans(ctx)
                # 批量结果基于重规划前的整片快照；当前场景改变后，剩余
                # 场景的缓存结果也可能已过期，必须重新逐场景审查。
                batch_results.clear()
                self._emit("scene_plan_replanned", scene_id=scene_id)

            if state.failed:
                break

            if restart_required:
                break

        if self._stop_event.is_set() or ctx.continuity_rebuild_required:
            return
        if restart_required:
            # 机械交接修复可能同时修改上一场景或下一场景；不能让当前
            # for 循环跳过这些已经被标记为 plan_reviewed=False 的计划，
            # 否则代码屏障会在它们未经复审时继续执行。
            if _mechanical_restart_count >= 3:
                reason = "计划交接的机械修复未能在有限轮次内收敛"
                with self._state_lock:
                    ctx.plan_review_status = "failed"
                    ctx.continuity_warnings.append(reason)
                    self._checkpoint(ctx, State.PLAN_REVIEWING)
                return
            return self._run_plan_review_barrier(
                ctx,
                _mechanical_restart_count=_mechanical_restart_count + 1,
            )
        failed = any(state.failed for state in ctx.scene_states.values() if state.plan_ready)
        with self._state_lock:
            ctx.plan_review_status = "failed" if failed else "passed"
            self._checkpoint(ctx, State.REVIEWING)

    @staticmethod
    def _dedupe_continuity_issues(
        issues: list[ContinuityIssue],
    ) -> list[ContinuityIssue]:
        """按场景、类别和消息去重，避免同一冲突重复喂给 Planner。"""

        unique: list[ContinuityIssue] = []
        seen: set[tuple[tuple[int, ...], str, str]] = set()
        for issue in issues:
            key = (tuple(sorted(set(issue.scene_ids))), issue.category, issue.message)
            if key not in seen:
                seen.add(key)
                unique.append(issue)
        return unique

    @staticmethod
    def _continuity_issue_target_fields(issue: ContinuityIssue) -> list[str]:
        """提取连续性冲突真正涉及的分镜字段。

        新版审查器会直接返回 ``target_fields``；旧版响应没有该字段时，
        从错误消息和修正指令中推断，避免重规划时把整份快照原样复制回来。
        """

        allowed = (
            "visual_design",
            "camera_movement",
            "visual_flow",
            "key_moments",
            "computation",
            "persistent_elements",
            "opening_state",
            "closing_state",
            "transition_in",
            "transition_out",
            "continuity_references",
            "global_visual_state",
            "inherited_elements",
            "elements_to_remove",
            "new_elements",
        )
        source = f"{issue.message}\n{issue.fix_instruction}"
        fields = [field for field in issue.target_fields if field in allowed]
        fields.extend(field for field in allowed if field in source and field not in fields)
        return fields

    @staticmethod
    def _continuity_plan_context(ctx: PipelineContext, scene_id: int) -> str:
        """构造有界的当前场景及相邻场景交接快照。

        这里不能直接序列化完整 ScenePlan：连续性重规划通常在计划已经
        包含长 computation、timeline 和多组元素时触发，完整快照会超过
        ``plan_detail`` 的必需区块预算，反而阻止修复请求发出。当前场景
        保留数学/交接字段；相邻场景只保留决定边界的摘要。
        """

        def compact_plan(plan: ScenePlan, *, boundary_only: bool) -> dict:
            data = plan.model_dump(mode="json")
            if boundary_only:
                keys = (
                    "scene_id",
                    "title",
                    "claim_ids",
                    "visual_unit_id",
                    "teaching_role",
                    "opening_state",
                    "closing_state",
                    "transition_in",
                    "transition_out",
                    "inherited_elements",
                    "elements_to_remove",
                    "new_elements",
                    "handoff",
                )
            else:
                keys = (
                    "scene_id",
                    "title",
                    "purpose",
                    "math_concept",
                    "claim_ids",
                    "visual_unit_id",
                    "teaching_role",
                    "visual_design",
                    "camera_movement",
                    "visual_flow",
                    "key_moments",
                    "computation",
                    "opening_state",
                    "closing_state",
                    "transition_in",
                    "transition_out",
                    "continuity_references",
                    "inherited_elements",
                    "elements_to_remove",
                    "new_elements",
                    "timeline",
                    "math_claims",
                    "handoff",
                )
            compact = {key: data.get(key) for key in keys}
            for key in (
                "purpose",
                "math_concept",
                "visual_design",
                "camera_movement",
                "computation",
                "transition_in",
                "transition_out",
            ):
                if isinstance(compact.get(key), str):
                    compact[key] = compact[key][:2_000]
            for key in (
                "visual_flow",
                "key_moments",
                "opening_state",
                "closing_state",
                "continuity_references",
            ):
                if isinstance(compact.get(key), list):
                    compact[key] = [str(item)[:800] for item in compact[key][:12]]
            for key in ("inherited_elements", "elements_to_remove", "new_elements"):
                if isinstance(compact.get(key), list):
                    compact[key] = [
                        {
                            field: item.get(field, "")
                            for field in (
                                "element_id",
                                "variable_name",
                                "role",
                                "kind",
                                "semantic_state",
                                "color_key",
                                "anchor",
                                "required",
                                "reason",
                            )
                        }
                        for item in compact[key][:20]
                    ]
            if not boundary_only:
                for key in ("timeline", "math_claims"):
                    if isinstance(compact.get(key), list):
                        compact[key] = [
                            {
                                field: str(item.get(field, ""))[:1_200]
                                for field in (
                                    "event_id",
                                    "start_seconds",
                                    "end_seconds",
                                    "action",
                                    "element_ids",
                                    "math_claim_ids",
                                    "claim_id",
                                    "statement",
                                    "expression_before",
                                    "expression_after",
                                    "relation",
                                    "justification",
                                )
                                if field in item
                            }
                            for item in compact[key][:30]
                        ]
            return compact

        snapshot = []
        for current_id, state in sorted(ctx.scene_states.items()):
            if abs(current_id - scene_id) > 1 or state.failed or state.give_up:
                continue
            snapshot.append(
                {
                    "relation": (
                        "previous"
                        if current_id < scene_id
                        else "next"
                        if current_id > scene_id
                        else "current"
                    ),
                    "scene_id": current_id,
                    "plan": compact_plan(state.plan, boundary_only=current_id != scene_id),
                }
            )
        return json.dumps(snapshot, ensure_ascii=False, indent=2)[:22_000]

    @staticmethod
    def _supports_continuity_context(planner: object) -> bool:
        """兼容旧版/测试 Planner，同时让新版 Planner 接收交接快照。"""

        try:
            parameters = inspect.signature(planner.plan_detail).parameters
        except (AttributeError, TypeError, ValueError):
            return False
        return "continuity_context" in parameters or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
        )

    def _run_continuity_review(self, ctx: PipelineContext) -> None:
        """审查全片分镜，并只重规划仍未编码的冲突场景。"""

        if ctx.continuity_bible is None:
            return
        active_states = [
            state for state in ctx.scene_states.values() if not state.failed and not state.give_up
        ]
        if any(not state.plan_ready for state in active_states):
            return

        self._emit("continuity_reviewing", scene_count=len(active_states))
        max_rounds = max(0, settings.MAX_CONTINUITY_FIX_ROUNDS)
        while True:
            with self._state_lock:
                ctx.continuity_review_round += 1
                current_round = ctx.continuity_review_round
                ctx.continuity_review_status = "reviewing"
                self._checkpoint(ctx, State.REVIEWING)

            plans = [
                state.plan
                for state in sorted(ctx.scene_states.values(), key=lambda item: item.plan.scene_id)
                if not state.failed and not state.give_up
            ]
            deterministic = deterministic_continuity_issues(
                plans,
                ctx.continuity_bible,
                ctx.lesson_spec,
                ctx.teaching_graph,
            )
            try:
                with self._llm_sem:
                    result = ContinuityReviewerAgent().review(
                        ctx.continuity_bible,
                        ctx.outlines,
                        plans,
                        deterministic_issues=deterministic,
                        renderer=ctx.render_profile.renderer,
                        lesson_spec=ctx.lesson_spec,
                        teaching_graph=ctx.teaching_graph,
                        stream=False,
                    )
                llm_issues = result.issues if not result.is_valid else []
            except Exception as exc:
                warning = f"连续性审查调用失败（第 {current_round} 轮）: {exc}"
                ctx.continuity_warnings.append(warning)
                # 若确定性检查已经找到可定位问题，仍尝试一次局部修正；否则
                # 标记 warning 放行，避免连续性增强把整条流水线判死。
                llm_issues = []
                if not deterministic:
                    with self._state_lock:
                        ctx.continuity_review_status = "warning"
                        self._checkpoint(ctx, State.REVIEWING)
                    self._emit(
                        "continuity_review_accepted_with_warning",
                        reason=warning,
                        reason_type="llm_error",
                        round=current_round,
                        max_rounds=max_rounds,
                    )
                    self._emit("continuity_warning", reason=warning)
                    return
                self._emit("continuity_warning", reason=warning)

            issues = self._dedupe_continuity_issues([*deterministic, *llm_issues])
            if not issues:
                with self._state_lock:
                    ctx.continuity_review_status = "passed"
                    self._checkpoint(ctx, State.REVIEWING)
                self._emit("continuity_pass", round=current_round)
                return

            affected_ids = sorted({scene_id for issue in issues for scene_id in issue.scene_ids})
            uneditable = [
                scene_id
                for scene_id in affected_ids
                if scene_id in ctx.scene_states
                and (
                    ctx.scene_states[scene_id].code
                    or ctx.scene_states[scene_id].slurm_job is not None
                    or ctx.scene_states[scene_id].rendered
                )
            ]
            if current_round > max_rounds or uneditable:
                reasons = [
                    f"Scene {scene_id}: "
                    + "; ".join(issue.message for issue in issues if scene_id in issue.scene_ids)
                    for scene_id in affected_ids
                ]
                warning = (
                    "连续性冲突未自动重规划："
                    + ("已进入编码/渲染阶段" if uneditable else "达到最大连续性修正轮数")
                    + "。"
                    + " ".join(reasons)
                )
                with self._state_lock:
                    ctx.continuity_review_status = "warning"
                    ctx.continuity_warnings.append(warning)
                    self._checkpoint(ctx, State.REVIEWING)
                self._emit("continuity_warning", reason=warning)
                event_data = {
                    "reason": warning,
                    "reason_type": (
                        "max_rounds" if current_round > max_rounds else "already_started"
                    ),
                    "scene_ids": affected_ids,
                    "round": current_round,
                    "max_rounds": max_rounds,
                }
                if current_round > max_rounds:
                    self._emit("continuity_review_exhausted", **event_data)
                self._emit("continuity_review_accepted_with_warning", **event_data)
                return

            feedback_by_scene: dict[int, list[str]] = {scene_id: [] for scene_id in affected_ids}
            target_fields_by_scene: dict[int, list[str]] = {
                scene_id: [] for scene_id in affected_ids
            }
            for issue in issues:
                feedback = issue.fix_instruction or issue.message
                target_fields = self._continuity_issue_target_fields(issue)
                field_hint = (
                    "\n必须重写字段（不得复制这些字段中的冲突原文）：" + ", ".join(target_fields)
                    if target_fields
                    else ""
                )
                for scene_id in issue.scene_ids:
                    feedback_by_scene.setdefault(scene_id, []).append(
                        f"[{issue.category}] {issue.message}\n修正要求: {feedback}{field_hint}"
                    )
                    for field_name in target_fields:
                        if field_name not in target_fields_by_scene.setdefault(scene_id, []):
                            target_fields_by_scene[scene_id].append(field_name)

            self._emit(
                "continuity_fixing",
                scene_ids=affected_ids,
                round=current_round,
                max_rounds=max_rounds,
            )
            for scene_id in affected_ids:
                state = ctx.scene_states.get(scene_id)
                if state is None or state.failed or state.give_up:
                    continue
                outline = next(outline for outline in ctx.outlines if outline.scene_id == scene_id)
                planner = PlannerAgent()
                replan_kwargs = {
                    "stream": False,
                    "renderer": ctx.render_profile.renderer,
                    "continuity_bible": ctx.continuity_bible,
                    "continuity_feedback": "\n\n".join(feedback_by_scene[scene_id]),
                }
                if self._supports_continuity_context(planner):
                    replan_kwargs["continuity_context"] = self._continuity_plan_context(
                        ctx, scene_id
                    )
                if self._supports_keyword(planner.plan_detail, "lesson_spec"):
                    replan_kwargs["lesson_spec"] = ctx.lesson_spec
                if self._supports_keyword(planner.plan_detail, "teaching_graph"):
                    replan_kwargs["teaching_graph"] = ctx.teaching_graph
                try:
                    with self._llm_sem:
                        revised_plan = planner.plan_detail(
                            outline, ctx.outlines, ctx.user_prompt, **replan_kwargs
                        )
                except Exception as exc:
                    warning = (
                        f"连续性重规划调用失败（Scene {scene_id}，第 {current_round} 轮）：{exc}"
                    )
                    with self._state_lock:
                        ctx.continuity_review_status = "warning"
                        ctx.continuity_warnings.append(warning)
                        self._checkpoint(ctx, State.REVIEWING)
                    self._emit(
                        "continuity_review_accepted_with_warning",
                        reason=warning,
                        reason_type="replan_error",
                        scene_ids=[scene_id],
                        round=current_round,
                        max_rounds=max_rounds,
                    )
                    self._emit("continuity_warning", reason=warning)
                    return
                outline_index = next(
                    index for index, item in enumerate(ctx.outlines) if item.scene_id == scene_id
                )
                previous_plan = (
                    ctx.scene_states[ctx.outlines[outline_index - 1].scene_id].plan
                    if outline_index > 0
                    else None
                )
                next_outline = (
                    ctx.outlines[outline_index + 1]
                    if outline_index + 1 < len(ctx.outlines)
                    else None
                )
                revised_plan = apply_deterministic_continuity_repairs(
                    revised_plan,
                    ctx.continuity_bible,
                    target_fields_by_scene.get(scene_id, []),
                    previous_plan=previous_plan,
                    next_outline=next_outline,
                )
                revised_plan, contract_repairs = normalize_scene_plan_contract(
                    revised_plan,
                    ctx.continuity_bible,
                    previous_plan=previous_plan,
                    has_next_scene=outline_index + 1 < len(ctx.outlines),
                )
                with self._state_lock:
                    state.plan = revised_plan
                    self._reset_technical_spec(state)
                    state.plan_ready = True
                    state.plan_reviewed = False
                    state.plan_review_round = 0
                    state.plan_review_feedback = ""
                    state.plan_review_signature = ""
                    state.identical_plan_review_count = 0
                    ctx.plan_review_status = "pending"
                    if contract_repairs:
                        ctx.continuity_warnings.append(
                            f"Scene {scene_id} 重规划后自动修复连续性合同："
                            + "；".join(contract_repairs)
                        )
                    ctx.scenes = [
                        item.plan
                        for item in sorted(
                            ctx.scene_states.values(), key=lambda item: item.plan.scene_id
                        )
                    ]
                    self._checkpoint(ctx, State.DETAILING)
                if contract_repairs:
                    self._emit(
                        "continuity_contract_repaired",
                        scene_id=scene_id,
                        repairs=contract_repairs,
                    )
                self._compile_scene_plans(ctx)
                self._emit("scene_detailed", scene_id=scene_id, title=revised_plan.title)

    def _scene_worker(self, ctx: PipelineContext, scene_id: int, state: SceneState) -> None:
        """执行已通过顺序代码屏障的 Scene 渲染→修复→重新提交流程。"""
        acquired = scene_id in self._reserved_existing_scenes
        try:
            if self._stop_event.is_set():
                return
            # 1) 分镜
            if not state.plan_ready:
                self._phase_emit("detailing")
                self._scene_detail(ctx, scene_id, state)
                if state.failed or state.give_up:
                    return
            # 2) 编码 + 审查循环 (直到审查通过)
            while not state.reviewed:
                if self._stop_event.is_set() or state.failed or state.give_up:
                    return
                if not state.code or state.rewrite_feedback:
                    self._phase_emit("coding")
                    self._scene_code(ctx, scene_id, state)
                    if state.failed or state.give_up:
                        return
                elif (
                    not ctx.dry_run
                    and settings.LOCAL_SMOKE_RENDER_ENABLED
                    and state.local_smoke_status != "passed"
                ):
                    self._local_smoke_render(ctx, state)
                self._phase_emit("reviewing")
                self._scene_review(ctx, scene_id, state)
            # 3) dry-run: 不提交渲染
            if ctx.dry_run:
                return
            # 4) 提交 + 渲染循环 (直到渲染成功/失败/放弃)
            while not state.rendered:
                if self._stop_event.is_set() or state.failed or state.give_up:
                    return
                if state.slurm_job is None:
                    if not acquired:
                        if not self._try_acquire_slot():
                            time.sleep(settings.MONITOR_POLL_INTERVAL)
                            continue
                        acquired = True
                    self._scene_submit(ctx, scene_id, state)
                    if state.slurm_job is None:
                        # 提交失败 → 释放名额
                        self._release_slot()
                        acquired = False
                        return
                    if self._cancel_requested.is_set() or self._stop_event.is_set():
                        self.cancel_all()
                        return
                ok = self._scene_wait_render(ctx, state)
                if ok:
                    return
                if self._cancel_requested.is_set() or self._stop_event.is_set():
                    return
                if state.failed or state.give_up:
                    return
                # 渲染失败 → 修复后重新提交 (名额保留)
                if state.slurm_job is None:
                    # 基础设施故障重排队路径已经清除了旧 Job；不要把
                    # 没有错误日志的节点故障交给 AutoFix，也不要释放名额。
                    continue
                self._scene_fix(ctx, scene_id, state)
                # AutoFix 改变了代码，必须再次经过 Reviewer；major 反馈仍按
                # 正常编码循环重写，绝不能把未经复审的修复代码直接提交。
                while not state.reviewed:
                    if self._stop_event.is_set() or state.failed or state.give_up:
                        return
                    if state.rewrite_feedback:
                        self._phase_emit("coding")
                        self._scene_code(ctx, scene_id, state)
                    elif (
                        not ctx.dry_run
                        and settings.LOCAL_SMOKE_RENDER_ENABLED
                        and state.local_smoke_status != "passed"
                    ):
                        self._local_smoke_render(ctx, state)
                    self._phase_emit("reviewing")
                    self._scene_review(ctx, scene_id, state)
        except Exception as exc:
            with self._state_lock:
                # 视觉门/产物回写可能在渲染成功后抛错；失败状态不能同时
                # 保留 rendered=True，否则调度器会跳过该场景，合并阶段
                # 还可能误把不完整状态当成成功。清除派生凭据，resume
                # 时可重新验证旧 Job 或重新提交。
                if state.rendered:
                    state.rendered = False
                    state.artifact = None
                    self._reset_visual_receipt(ctx, state)
                self._mark_failed(state, f"Scene {scene_id} 流水线异常: {exc}", "system")
            try:
                self._checkpoint(ctx, State.MONITORING)
            except Exception as checkpoint_error:
                self._record_checkpoint_failure(checkpoint_error)
            self._emit("scene_failed", scene_id=scene_id, reason=str(exc))
        finally:
            # 无论成功/失败/异常, 都要释放 in-flight 名额, 避免其他场景死等
            if acquired:
                self._release_slot()
            if not self._stop_event.is_set():
                try:
                    self._checkpoint(ctx, State.MONITORING)
                except Exception as checkpoint_error:
                    self._record_checkpoint_failure(checkpoint_error)

    def _try_acquire_slot(self) -> bool:
        if self._resource_coordinator:
            return self._resource_coordinator.try_acquire_slurm()
        limit = settings.SLURM_MAX_IN_FLIGHT
        with self._slot_lock:
            if limit and self._in_flight >= limit:
                return False
            self._in_flight += 1
            return True

    def _reserve_existing_slot(self, scene_id: int) -> None:
        """让 resume 时已存在的远程作业先占用名额，防止继续超量提交。"""

        if self._resource_coordinator:
            self._resource_coordinator.register_existing_slurm()
        else:
            with self._slot_lock:
                self._in_flight += 1
        self._reserved_existing_scenes.add(scene_id)

    def _release_slot(self) -> None:
        if self._resource_coordinator:
            self._resource_coordinator.release_slurm()
            return
        with self._slot_lock:
            self._in_flight = max(0, self._in_flight - 1)

    def _phase_emit(self, stage: str) -> None:
        """每个阶段只发一次 stage_start (线程安全), 供 TUI/仪表盘显示阶段标题。"""
        with self._phase_lock:
            if stage in self._emitted_phases:
                return
            self._emitted_phases.add(stage)
        self._emit("stage_start", stage=stage)

    def _scene_detail(self, ctx: PipelineContext, scene_id: int, state: SceneState) -> None:
        self._emit(
            "scene_detailing",
            scene_id=scene_id,
            title=state.plan.title,
        )
        rag_context = self._retrieve_rag(
            ctx,
            f"{state.plan.title}\n{state.plan.purpose}\n{state.plan.math_concept}",
            receipt_key=f"scene:{scene_id}:detail",
            stage="detail",
            source_kinds={"manim_doc", "example", "recipe"},
        )
        with self._llm_sem:
            outline = next(o for o in ctx.outlines if o.scene_id == scene_id)
            planner = PlannerAgent()
            detail_kwargs = {
                "stream": False,
                "renderer": ctx.render_profile.renderer,
            }
            if ctx.continuity_bible is not None and self._supports_keyword(
                planner.plan_detail, "continuity_bible"
            ):
                detail_kwargs["continuity_bible"] = ctx.continuity_bible
            if self._supports_keyword(planner.plan_detail, "rag_context"):
                detail_kwargs["rag_context"] = rag_context
            if self._supports_keyword(planner.plan_detail, "lesson_spec"):
                detail_kwargs["lesson_spec"] = ctx.lesson_spec
            if self._supports_keyword(planner.plan_detail, "teaching_graph"):
                detail_kwargs["teaching_graph"] = ctx.teaching_graph
            plan = planner.plan_detail(outline, ctx.outlines, ctx.user_prompt, **detail_kwargs)
        if ctx.continuity_bible is not None:
            previous_plan = None
            previous_state = ctx.scene_states.get(scene_id - 1)
            if previous_state is not None and previous_state.plan_ready:
                previous_plan = previous_state.plan
            plan, repairs = normalize_scene_plan_contract(
                plan,
                ctx.continuity_bible,
                previous_plan=previous_plan,
                has_next_scene=any(item.scene_id > scene_id for item in ctx.outlines),
            )
            if repairs:
                reason = "；".join(repairs)
                with self._state_lock:
                    ctx.continuity_warnings.append(
                        f"Scene {scene_id} 已自动修复连续性合同：{reason}"
                    )
                self._emit(
                    "continuity_contract_repaired",
                    scene_id=scene_id,
                    repairs=repairs,
                )
        with self._state_lock:
            state.plan = plan
            self._reset_technical_spec(state)
            state.plan_ready = True
            state.plan_reviewed = False
            state.plan_review_round = 0
            state.plan_review_feedback = ""
            state.plan_review_signature = ""
            state.identical_plan_review_count = 0
            ctx.scenes = [
                item.plan
                for item in sorted(ctx.scene_states.values(), key=lambda item: item.plan.scene_id)
            ]
            self._checkpoint(ctx, State.DETAILING)
            self._write_stage_artifact(
                ctx,
                f"scene_{scene_id}_plan.json",
                {"schema_version": 1, "plan": plan.model_dump(mode="json")},
            )
        self._emit("scene_detailed", scene_id=scene_id, title=plan.title)

    def _prepare_inherited_context(
        self, ctx: PipelineContext, scene_id: int, state: SceneState
    ) -> None:
        """为当前场景固定上一场景的最小代码级交接。"""

        if scene_id <= 1:
            state.inherited_elements_code = ""
            return
        previous = ctx.scene_states.get(scene_id - 1)
        if previous is None:
            raise ValueError(
                f"Scene {scene_id} 声明了继承元素，但缺少上一场景状态，"
                "禁止在没有真实交接代码的情况下继续编码。"
            )
        current_code_hash = sha256_text(previous.code) if previous.code else ""
        export_hashes_match = bool(previous.exported_elements) and all(
            item.source_code_sha256 in {"", current_code_hash}
            for item in previous.exported_elements
        )
        if previous.code and (not previous.exported_elements_code or not export_hashes_match):
            exported_code, exported_elements = extract_scene_continuity_elements(
                previous.code,
                previous.plan,
            )
            previous.exported_elements_code = exported_code
            previous.exported_elements = [
                item.model_copy(
                    update={
                        "source_scene_id": previous.plan.scene_id,
                        "source_code_sha256": current_code_hash,
                    }
                )
                for item in exported_elements
            ]
        inherited_ids = {item.element_id for item in state.plan.inherited_elements}
        context_mode = settings.CONTINUITY_CONTEXT_MODE
        if inherited_ids and previous.exported_elements:
            selected = [
                item.code for item in previous.exported_elements if item.element_id in inherited_ids
            ]
            selected_ids = {
                item.element_id
                for item in previous.exported_elements
                if item.element_id in inherited_ids
            }
            missing_ids = inherited_ids - selected_ids
            if missing_ids:
                raise ValueError(
                    f"Scene {scene_id} 所需继承元素未由 Scene {scene_id - 1} 导出: "
                    + ", ".join(sorted(missing_ids))
                )
            state.inherited_elements_code = (
                previous.exported_elements_code if context_mode == "full" else "\n\n".join(selected)
            )
        elif inherited_ids:
            selected_entries = ctx.element_manifest.for_elements(inherited_ids)
            selected_ids = {entry.element_id for entry in selected_entries}
            missing_ids = inherited_ids - selected_ids
            if missing_ids:
                raise ValueError(
                    f"Scene {scene_id} 所需继承元素没有可验证的状态账本记录: "
                    + ", ".join(sorted(missing_ids))
                )
            state.inherited_elements_code = (
                "\n\n".join(entry.source_code for entry in ctx.element_manifest.entries)
                if context_mode == "full"
                else "\n\n".join(entry.source_code for entry in selected_entries)
            )
            if not state.inherited_elements_code.strip():
                raise ValueError(f"Scene {scene_id} 的继承元素记录为空，禁止继续编码")
        else:
            # 没有结构化继承合同的旧计划保留历史交接行为；只有显式
            # stateless 才关闭它，避免升级后破坏旧清单的跨场景动画。
            state.inherited_elements_code = (
                "" if context_mode == "stateless" else previous.exported_elements_code
            )

    @staticmethod
    def _refresh_scene_export(state: SceneState) -> None:
        """从审查通过的最终代码提取下一场景可复用的纯定义。"""

        exported_code, exported_elements = extract_scene_continuity_elements(
            state.code,
            state.plan,
        )
        state.exported_elements_code = exported_code
        code_hash = sha256_text(state.code)
        state.exported_elements = [
            item.model_copy(
                update={
                    "source_scene_id": state.plan.scene_id,
                    "source_code_sha256": code_hash,
                }
            )
            for item in exported_elements
        ]

    @staticmethod
    def _remove_element_manifest_scene(ctx: PipelineContext, scene_id: int) -> None:
        """代码失效时删除该场景旧快照，避免后继场景读取陈旧定义。"""

        old_ids = set(ctx.element_manifest.scene_exports.get(scene_id, []))
        if old_ids:
            ctx.element_manifest = ctx.element_manifest.model_copy(
                update={
                    "entries": [
                        entry
                        for entry in ctx.element_manifest.entries
                        if entry.element_id not in old_ids or entry.source_scene_id != scene_id
                    ],
                    "scene_exports": {
                        key: value
                        for key, value in ctx.element_manifest.scene_exports.items()
                        if key != scene_id
                    },
                    "last_scene_id": (
                        max(
                            (key for key in ctx.element_manifest.scene_exports if key != scene_id),
                            default=None,
                        )
                    ),
                }
            )
        retained_boundaries = {
            key: value for key, value in ctx.state_ledger.boundaries.items() if key != scene_id
        }
        retained_references = {
            element_id
            for boundary in retained_boundaries.values()
            for element_id in (*boundary.opening_element_ids, *boundary.closing_element_ids)
        }
        ledger_elements = []
        for item in ctx.state_ledger.elements:
            if item.source_scene_id != scene_id:
                ledger_elements.append(item)
            elif item.element_id in retained_references:
                # 同一 element_id 可能已被后续场景重新导出，而较早的边界
                # 仍然引用它。删除后续场景时不能把这个历史引用变成悬空
                # ID；保留为不可继续继承的 tombstone，直到所有边界都被
                # 删除。其源代码只用于历史追踪，不再作为当前上下文提供。
                ledger_elements.append(
                    item.model_copy(update={"active": False, "required_next": False})
                )
        ctx.state_ledger = StateLedger.model_validate(
            {
                **ctx.state_ledger.model_dump(mode="python"),
                "elements": ledger_elements,
                "boundaries": retained_boundaries,
                "current_scene_id": max(retained_boundaries, default=None),
            }
        )

    def _update_element_manifest(self, ctx: PipelineContext, state: SceneState) -> None:
        ctx.element_manifest = ctx.element_manifest.update_scene(
            state.plan,
            state.exported_elements,
        )
        self._write_stage_artifact(
            ctx,
            "element_manifest.json",
            ctx.element_manifest.model_dump(mode="json"),
        )

    def _update_state_ledger(self, ctx: PipelineContext, state: SceneState) -> None:
        """把计划边界和最终导出区写入语义状态账本。"""

        declaration_by_id = {
            item.element_id: item
            for item in [*state.plan.inherited_elements, *state.plan.new_elements]
        }
        ledger_elements: list[LedgerElement] = []
        for element in state.exported_elements:
            declaration = declaration_by_id.get(element.element_id)
            ledger_elements.append(
                LedgerElement(
                    element_id=element.element_id,
                    variable_name=element.variable_name,
                    semantic_state=(declaration.semantic_state if declaration else ""),
                    mathematical_state="；".join(state.plan.closing_state),
                    color_key=(declaration.color_key if declaration else ""),
                    anchor=(declaration.anchor if declaration else ""),
                    active=True,
                    required_next=bool(declaration and declaration.required),
                    source_scene_id=state.plan.scene_id,
                    source_code_sha256=element.source_code_sha256 or sha256_text(state.code),
                    export_code_sha256=sha256_text(element.code),
                )
            )
        removed_ids = {item.element_id for item in state.plan.elements_to_remove}
        # StateLedger 同时保存历史场景边界。元素在当前场景退出后不能从
        # ``elements`` 物理删除，否则历史 Scene N 的 closing_element_ids
        # 会引用一个不存在的 element，下一次更新账本时触发完整性校验失败。
        # 保留不可变的代码快照并标记 inactive，既能支持历史追踪，也能
        # 让当前场景的 opening/closing 边界继续可验证。
        current_elements = [
            (
                item.model_copy(update={"active": False, "required_next": False})
                if item.element_id in removed_ids
                else item
            )
            for item in ctx.state_ledger.elements
        ]
        known = {item.element_id for item in current_elements}
        current_elements.extend(item for item in ledger_elements if item.element_id not in known)
        # update_scene 会按 element_id 覆盖当前场景重新导出的快照；历史
        # 元素则保留为 inactive tombstone，供旧边界和恢复诊断引用。
        ctx.state_ledger = ctx.state_ledger.model_copy(update={"elements": current_elements})
        ctx.state_ledger = ctx.state_ledger.update_scene(
            scene_id=state.plan.scene_id,
            elements=ledger_elements,
            opening_element_ids=[item.element_id for item in state.plan.inherited_elements],
            closing_element_ids=[item.element_id for item in state.exported_elements],
            opening_math_state="；".join(state.plan.opening_state),
            closing_math_state="；".join(state.plan.closing_state),
            removed_element_ids=[item.element_id for item in state.plan.elements_to_remove],
            transition_in=state.plan.transition_in,
            transition_out=state.plan.transition_out,
            visual_state_digest=sha256_text(state.plan.global_visual_state.model_dump_json()),
            exported_code_sha256=sha256_text(state.exported_elements_code)
            if state.exported_elements_code
            else "",
            artifact_video_sha256=(state.artifact.video_sha256 if state.artifact else ""),
        )
        self._write_stage_artifact(
            ctx,
            "state_ledger.json",
            {
                "schema_version": 1,
                "digest": ctx.state_ledger.digest(),
                "ledger": ctx.state_ledger.model_dump(mode="json"),
            },
        )

    def _scene_code(self, ctx: PipelineContext, scene_id: int, state: SceneState) -> None:
        rewriting = bool(state.rewrite_feedback)
        self._emit(
            "scene_rewriting" if rewriting else "scene_coding",
            scene_id=scene_id,
            title=state.plan.title,
            reason=state.rewrite_feedback if rewriting else "",
        )
        rag_context = self._retrieve_rag(
            ctx,
            "\n".join(
                (
                    state.plan.title,
                    state.plan.math_concept,
                    state.plan.computation,
                    state.plan.camera_movement,
                    ctx.render_profile.renderer,
                )
            ),
            receipt_key=f"scene:{scene_id}:code",
            stage="code",
            source_kinds={"manim_doc", "example", "recipe"},
            preferred_source_kinds={"recipe"},
            code_sha256=sha256_text(state.code) if state.code else "",
            inherited_elements_sha256=sha256_text(state.inherited_elements_code)
            if state.inherited_elements_code
            else "",
        )
        code_fallback_used = False
        code_fallback_reason = ""
        program_used = False

        def compile_ir_candidate() -> tuple[str, str]:
            program = build_scene_program_from_contract(state.plan, state.technical_spec)
            candidate = compile_scene_program(program, state.plan)
            validation = self._validate(candidate, renderer=ctx.render_profile.renderer)
            if not validation.is_valid:
                raise SceneProgramCompileError(validation.feedback)
            try:
                extract_scene_continuity_elements(candidate, state.plan)
            except ValueError as exc:
                raise SceneProgramCompileError(str(exc)) from exc
            if state.technical_spec is not None:
                lifecycle_result = validate_animation_lifecycle(
                    candidate,
                    state.technical_spec,
                    renderer=ctx.render_profile.renderer,
                )
                if not lifecycle_result.is_valid:
                    raise SceneProgramCompileError("；".join(lifecycle_result.errors))
            self._write_stage_artifact(
                ctx,
                f"scene_{scene_id}_program.json",
                {"schema_version": 1, "program": program.model_dump(mode="json")},
            )
            return candidate, validation.scene_classes[0]

        if settings.CODEGEN_MODE == "ir":
            code, class_name = compile_ir_candidate()
            program_used = True
            self._emit("scene_program_compiled", scene_id=scene_id)
        else:
            try:
                with self._llm_sem:
                    code, class_name = self._generate_validated_code(
                        state.plan,
                        feedback=state.rewrite_feedback or "",
                        previous_code=state.code if state.rewrite_feedback else "",
                        stream=False,
                        renderer=ctx.render_profile.renderer,
                        continuity_bible=ctx.continuity_bible,
                        inherited_elements_code=state.inherited_elements_code,
                        inherited_elements=state.plan.inherited_elements,
                        elements_to_remove=state.plan.elements_to_remove,
                        element_manifest=(
                            ctx.element_manifest.model_copy(
                                update={
                                    "entries": ctx.element_manifest.for_elements(
                                        {item.element_id for item in state.plan.inherited_elements}
                                    )
                                }
                            )
                            if state.plan.inherited_elements
                            else None
                        ),
                        technical_spec=state.technical_spec,
                        rag_context=rag_context,
                        lesson_spec=ctx.lesson_spec,
                        teaching_graph=ctx.teaching_graph,
                    )
            except Exception as exc:
                if settings.CODEGEN_MODE == "hybrid":
                    try:
                        code, class_name = compile_ir_candidate()
                    except Exception:
                        code = ""
                    else:
                        program_used = True
                        code_fallback_reason = str(exc)[:5_000]
                        self._emit(
                            "scene_program_compiled",
                            scene_id=scene_id,
                            reason="Coder 生成失败后使用结构化程序编译",
                        )
                if not program_used:
                    # Coder 的网络/截断/结构化输出故障不应直接把一个已经通过
                    # Plan/TechnicalSpec 的场景判死。使用不依赖 LLM 的最小代码
                    # 作为最后保险；它仍必须通过与正常候选完全相同的校验链。
                    fallback_code = build_safe_scene_code(state.plan, state.technical_spec)
                    fallback_validation = self._validate(
                        fallback_code,
                        renderer=ctx.render_profile.renderer,
                    )
                    fallback_continuity_error = ""
                    try:
                        extract_scene_continuity_elements(fallback_code, state.plan)
                    except ValueError as fallback_exc:
                        fallback_continuity_error = str(fallback_exc)
                    fallback_lifecycle_error = ""
                    if (
                        state.technical_spec is not None
                        and fallback_validation.is_valid
                        and not fallback_continuity_error
                    ):
                        lifecycle_result = validate_animation_lifecycle(
                            fallback_code,
                            state.technical_spec,
                            renderer=ctx.render_profile.renderer,
                        )
                        if not lifecycle_result.is_valid:
                            fallback_lifecycle_error = "; ".join(lifecycle_result.errors)
                    if (
                        not fallback_validation.is_valid
                        or fallback_continuity_error
                        or fallback_lifecycle_error
                    ):
                        raise RuntimeError(
                            "Coder 生成失败，且安全代码降级未通过确定性校验："
                            + "；".join(
                                part
                                for part in (
                                    "; ".join(fallback_validation.errors),
                                    fallback_continuity_error,
                                    fallback_lifecycle_error,
                                )
                                if part
                            )
                        ) from exc
                    code = fallback_code
                    class_name = fallback_validation.scene_classes[0]
                    code_fallback_used = True
                    code_fallback_reason = str(exc)[:5_000]
                    self._emit(
                        "scene_code_fallback",
                        scene_id=scene_id,
                        reason=code_fallback_reason,
                    )
        path = ctx.paths.scenes / f"scene_{scene_id}.py"
        self._write_private(path, code)
        with self._state_lock:
            state.code = code
            state.class_name = class_name
            state.rewrite_feedback = ""
            state.failure_reason = ""
            state.failure_category = ""
            state.reviewed = False
            state.infra_retries = 0
            state.artifact = None
            state.rendered = False
            state.exported_elements_code = ""
            state.exported_elements = []
            state.local_smoke_status = "pending"
            if code_fallback_used:
                state.safe_fallback_used = True
                state.safe_fallback_reason = (
                    "Coder 输出不可用，已使用最小安全代码降级：" + code_fallback_reason
                )[:5_000]
            self._remove_element_manifest_scene(ctx, scene_id)
            self._reset_visual_receipt(ctx, state)
            self._checkpoint(ctx, State.CODING)
        self._local_smoke_render(ctx, state)
        api_result = lint_manim_api(
            code,
            renderer=ctx.render_profile.renderer,
            scene_plan=state.plan,
        )
        if api_result.warnings:
            self._emit("scene_api_warning", scene_id=scene_id, warnings=list(api_result.warnings))
        self._emit("scene_coded", scene_id=scene_id, file_path=str(path))

    def _scene_review(self, ctx: PipelineContext, scene_id: int, state: SceneState) -> None:
        if settings.SKIP_REVIEW:
            try:
                self._refresh_scene_export(state)
                self._update_element_manifest(ctx, state)
                self._update_state_ledger(ctx, state)
            except ValueError as exc:
                with self._state_lock:
                    state.rewrite_feedback = f"连续性导出区无效: {exc}"
                    state.reviewed = False
                    self._checkpoint(ctx, State.REVIEWING)
                self._emit("scene_review_fail", scene_id=scene_id, severity="major")
                return
            with self._state_lock:
                state.reviewed = True
                self._apply_incremental_for_scene(ctx, scene_id, state)
                if state.rendered:
                    self._update_state_ledger(ctx, state)
                self._checkpoint(ctx, State.REVIEWING)
            self._emit("scene_review_skipped", scene_id=scene_id)
            if state.rendered and state.artifact and state.artifact.origin == "reused":
                self._emit("scene_reused", scene_id=scene_id)
            return
        self._emit("scene_reviewing", scene_id=scene_id)
        try:
            # 导出区是确定性的交接合同。先校验再调用 Reviewer，避免把重复
            # 导出标记、缺失元素等可直接修复的问题交给 LLM，尤其避免长上下文
            # 让 Reviewer 输出被截断。
            extract_scene_continuity_elements(state.code, state.plan)
            if state.technical_spec is not None:
                lifecycle_result = validate_animation_lifecycle(
                    state.code,
                    state.technical_spec,
                    renderer=ctx.render_profile.renderer,
                )
                if not lifecycle_result.is_valid:
                    raise ValueError("动画生命周期错误：" + "；".join(lifecycle_result.errors))
        except ValueError as exc:
            result = ReviewResult(
                is_valid=False,
                severity="major",
                feedback=f"连续性导出区无效，无法交接给下一场景: {exc}",
            )
        else:
            with self._llm_sem:
                reviewer = ReviewerAgent()
                review_kwargs = {"renderer": ctx.render_profile.renderer}
                if ctx.continuity_bible is not None:
                    review_kwargs["continuity_bible"] = ctx.continuity_bible
                if state.inherited_elements_code:
                    review_kwargs["inherited_elements_code"] = state.inherited_elements_code
                if state.technical_spec is not None and self._supports_keyword(
                    reviewer.review, "technical_spec"
                ):
                    review_kwargs["technical_spec"] = state.technical_spec
                if state.safe_fallback_used and self._supports_keyword(
                    reviewer.review, "safe_fallback"
                ):
                    review_kwargs["safe_fallback"] = True
                if self._supports_keyword(reviewer.review, "lesson_spec"):
                    review_kwargs["lesson_spec"] = ctx.lesson_spec
                result = reviewer.review(state.code, state.plan, **review_kwargs)
        if result.is_valid:
            try:
                self._refresh_scene_export(state)
                self._update_element_manifest(ctx, state)
                self._update_state_ledger(ctx, state)
            except ValueError as exc:
                result = ReviewResult(
                    is_valid=False,
                    severity="major",
                    feedback=f"连续性导出区无效，无法交接给下一场景: {exc}",
                )
        self._apply_review_result(ctx, scene_id, state, result)

    def _scene_submit(self, ctx: PipelineContext, scene_id: int, state: SceneState) -> None:
        self._phase_emit("dispatching")
        source_path = ctx.paths.scenes / f"scene_{scene_id}.py"
        try:
            on_disk_code = source_path.read_text(encoding="utf-8")
        except OSError as exc:
            with self._state_lock:
                self._mark_failed(state, f"提交前无法读取场景代码: {exc}", "coding")
                self._checkpoint(ctx, State.DISPATCHING)
            self._emit("scene_failed", scene_id=scene_id, reason=state.failure_reason)
            return
        if on_disk_code != state.code:
            with self._state_lock:
                self._mark_failed(
                    state,
                    "提交前代码一致性校验失败：磁盘文件已在流水线外被修改",
                    "coding",
                )
                self._checkpoint(ctx, State.DISPATCHING)
            self._emit("scene_failed", scene_id=scene_id, reason=state.failure_reason)
            return
        validation = self._validate(on_disk_code, renderer=ctx.render_profile.renderer)
        if not validation.is_valid:
            with self._state_lock:
                self._mark_failed(state, "提交前校验失败:\n" + validation.feedback, "coding")
                self._checkpoint(ctx, State.DISPATCHING)
            self._emit("scene_failed", scene_id=scene_id, reason=state.failure_reason)
            return
        api_result = lint_manim_api(
            on_disk_code,
            renderer=ctx.render_profile.renderer,
            scene_plan=state.plan,
        )
        if not api_result.is_valid:
            with self._state_lock:
                self._mark_failed(
                    state,
                    "提交前 Manim API 静态检查失败:\n" + "\n".join(api_result.errors),
                    "coding",
                )
                self._checkpoint(ctx, State.DISPATCHING)
            self._emit("scene_failed", scene_id=scene_id, reason=state.failure_reason)
            return
        if api_result.warnings:
            self._emit("scene_api_warning", scene_id=scene_id, warnings=list(api_result.warnings))
        if state.technical_spec is not None:
            lifecycle_result = validate_animation_lifecycle(
                on_disk_code,
                state.technical_spec,
                renderer=ctx.render_profile.renderer,
            )
            if not lifecycle_result.is_valid:
                with self._state_lock:
                    self._mark_failed(
                        state,
                        "提交前动画生命周期校验失败:\n"
                        + "\n".join(f"- {error}" for error in lifecycle_result.errors),
                        "coding",
                    )
                    self._checkpoint(ctx, State.DISPATCHING)
                self._emit("scene_failed", scene_id=scene_id, reason=state.failure_reason)
                return
        state.class_name = validation.scene_classes[0]
        job: SlurmJob | None = None
        try:
            job = self.slurm.submit_scene(
                scene_id,
                source_path,
                state.class_name,
                scenes_dir=ctx.paths.scenes,
                logs_dir=ctx.paths.logs,
                videos_dir=ctx.paths.videos,
                code_sha256=sha256_text(state.code),
                render_profile=ctx.render_profile,
            )
            expected_code_hash = sha256_text(state.code)
            if job.code_sha256 and job.code_sha256 != expected_code_hash:
                raise RuntimeError("Slurm 返回的作业代码哈希与当前场景不一致")
            # 外部/测试 Dispatcher 可能没有回填代码身份；由提交边界补齐，
            # 确保后续 manifest 和 resume 仍然具备不可歧义的代码凭据。
            job.code_sha256 = expected_code_hash
            with self._state_lock:
                state.slurm_job = job
                state.artifact = None
                state.rendered = False
                state.failure_reason = ""
                state.failure_category = ""
                self._reset_visual_receipt(ctx, state)
                try:
                    self._checkpoint(ctx, State.DISPATCHING)
                except Exception as checkpoint_error:
                    # sbatch 已经返回 Job ID；此时不能伪装成普通提交失败并
                    # 自动重提。停止所有 worker，保留内存中的 Job ID，交由
                    # 外层 cancel_all 做一次安全取消和失败收尾。
                    state.failed = True
                    state.give_up = True
                    state.failure_category = "system"
                    state.failure_reason = (
                        f"Slurm Job {job.job_id} 已提交，但本地检查点持久化失败: "
                        f"{checkpoint_error}；保留 Job ID 并禁止自动重提"
                    )
                    self._record_checkpoint_failure(checkpoint_error)
                    self._emit("scene_failed", scene_id=scene_id, reason=state.failure_reason)
                    return
            self._emit("scene_submitted", scene_id=scene_id, job_id=job.job_id)
            if self._cancel_requested.is_set():
                # Ctrl-C 可能恰好发生在 submit_scene 返回之后、批量取消器
                # 扫描之前；此处再扫一次，避免新 Job 漏掉。
                self.cancel_all()
        except Exception as exc:
            # 一旦拿到 Job ID 就绝不能把提交后的异常伪装成提交失败并自动重提。
            if job is not None:
                with self._state_lock:
                    state.slurm_job = job
                    state.give_up = True
                    state.failed = True
                    state.failure_category = "system"
                    state.failure_reason = (
                        f"Slurm Job {job.job_id} 已提交，但本地检查点持久化失败: {exc}；"
                        "保留 Job ID 并禁止自动重提"
                    )
                self._record_checkpoint_failure(exc)
                self._emit("scene_failed", scene_id=scene_id, reason=state.failure_reason)
                return
            with self._state_lock:
                self._mark_failed(state, f"Slurm 提交失败: {exc}", "infrastructure")
                self._checkpoint(ctx, State.DISPATCHING)
            self._emit("scene_failed", scene_id=scene_id, reason=str(exc))

    def _scene_wait_render(self, ctx: PipelineContext, state: SceneState) -> bool:
        """阻塞轮询当前作业直到结束; 返回是否渲染成功。"""
        job = state.slurm_job
        if job is None:
            return False
        self._phase_emit("monitoring")
        if self._slurm_monitor is not None:
            self._slurm_monitor.register(job)
            ok = self._slurm_monitor.wait(job.job_id, stop_event=self._stop_event)
            if ok is None:
                return False
            # 共享 Monitor 已经把最终状态、产物元数据和错误原因写回 Job；
            # 以下逻辑与旧的单 Job 路径共用，保证 AutoFix/基础设施重排队
            # 的行为不因监控实现切换而改变。
            monitor = None
        else:
            monitor = JobMonitor(self.slurm)
            monitor.add_job(job)
            while monitor.pending:
                if (
                    self._cancel_requested.is_set()
                    or self._stop_event.is_set()
                    or ctx.continuity_rebuild_required
                ):
                    return False
                previous_started_at = job.started_at
                if monitor.poll_once():
                    break
                if job.started_at is not None and job.started_at != previous_started_at:
                    # 首次从 squeue 得到实际启动时间后立即持久化，进程中断再
                    # resume 时不会丢失运行计时基准。
                    with self._state_lock:
                        self._checkpoint(ctx, State.MONITORING)
                time.sleep(settings.MONITOR_POLL_INTERVAL)
        if ctx.continuity_rebuild_required:
            return False
        if monitor is not None:
            ok = monitor.results.get(job.job_id)
        if ok is None:
            with self._state_lock:
                state.give_up = True
                state.failure_category = "infrastructure"
                state.failure_reason = "渲染作业状态未知，已放弃"
                self._checkpoint(ctx, State.MONITORING)
            return False
        if ok:
            with self._state_lock:
                state.artifact = self._artifact_from_job(ctx, state, job)
                state.rendered = True
                state.failure_reason = ""
                state.failure_category = ""
                self._reset_visual_receipt(ctx, state)
                self._update_state_ledger(ctx, state)
                self._checkpoint(ctx, State.MONITORING)
            self._emit("scene_rendered", scene_id=job.scene_id)
            # 不再等整个渲染批次结束：最先完成的场景立即接受视觉检查。
            # 视觉修复若改变上游交接，会设置 stop_event 并取消后继任务。
            if ctx.visual_eval_profile.enabled:
                self._visual_gate(ctx, scene_ids={job.scene_id})
            return True
        # 基础设施终态与业务代码无关，即使关闭 AutoFix 也应直接重新排队；
        # 不能因为 direct render 使用 auto_fix=False 就把节点故障交给用户手工重提。
        if job.status in RETRYABLE_INFRA_STATES:
            with self._state_lock:
                if state.infra_retries < settings.MAX_INFRA_RETRIES:
                    state.infra_retries += 1
                    state.slurm_job = None
                    state.artifact = None
                    state.rendered = False
                    state.failure_category = "infrastructure"
                    state.failure_reason = (
                        f"Slurm 基础设施状态 {job.status}，将重新排队 "
                        f"({state.infra_retries}/{settings.MAX_INFRA_RETRIES})"
                    )
                    self._checkpoint(ctx, State.MONITORING)
                    retry = True
                else:
                    state.give_up = True
                    state.failure_category = "infrastructure"
                    state.failure_reason = (
                        f"基础设施故障重试次数已用尽 ({settings.MAX_INFRA_RETRIES}): {job.status}"
                    )
                    self._checkpoint(ctx, State.MONITORING)
                    retry = False
            if retry:
                self._emit(
                    "scene_retrying",
                    scene_id=job.scene_id,
                    reason=state.failure_reason,
                    attempt=state.infra_retries,
                )
                return False
        if not ctx.auto_fix or job.status not in FIXABLE_RENDER_STATES:
            with self._state_lock:
                if not ctx.auto_fix:
                    self._mark_failed(
                        state,
                        job.failure_reason or f"Slurm 状态: {job.status}",
                        "render",
                    )
                else:
                    state.give_up = True
                    state.failure_category = "render"
                    state.failure_reason = (
                        job.failure_reason or f"基础设施失败，不修改代码: {job.status}"
                    )
                self._checkpoint(ctx, State.MONITORING)
        self._emit(
            "scene_failed",
            scene_id=job.scene_id,
            reason=job.failure_reason or f"Slurm 状态: {job.status}",
        )
        return False

    @staticmethod
    def _error_fingerprint(error_log: str) -> str:
        """渲染错误日志的稳定指纹: 数字统一归一化、跳过进度行。

        同一环境/代码错误即使时间戳、帧号、进度百分比不同, 指纹也应相同,
        用于判断"修复后错误是否仍然一样"。
        """
        import hashlib

        normalized: list[str] = []
        for line in error_log.splitlines():
            low = line.lower()
            # 跳过 Manim 进度条行 (含 % 和 |) 与纯时间行
            if "%" in low and "|" in low:
                continue
            # 每次重试都会产生新的 attempt 目录和部分临时哈希文件名；这些
            # 随机 token 不能影响“同一根因”的判定。
            low = re.sub(r"\b\d{8}-\d{6}-[0-9a-f]{8}\b", "<run>", low)
            low = re.sub(r"\battempt_[0-9a-f]{12}\b", "attempt_<token>", low)
            low = re.sub(r"\b[0-9a-f]{12,}\b", "<hex>", low)
            low = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", low)
            low = re.sub(r"\d+", "#", low).strip()
            if low:
                normalized.append(low)
        return hashlib.sha256("\n".join(normalized).encode("utf-8", errors="replace")).hexdigest()[
            :16
        ]

    def _give_up_reason(self, base: str, error_log: str) -> str:
        """放弃原因附带最近错误日志尾部, 方便用户直接定位根因。"""
        if not error_log:
            return base
        excerpt = "\n".join(error_log.splitlines()[-8:]).strip()
        if len(excerpt) > 600:
            excerpt = excerpt[-600:]
        return f"{base}\n最近错误日志(尾部):\n{excerpt}"

    def _request_continuity_rebuild(
        self,
        ctx: PipelineContext,
        scene_id: int,
        *,
        reason: str = "上游场景代码变化",
        preserve_visual_candidates: bool = False,
        include_failed: bool = False,
    ) -> None:
        """上游场景改码后，安全取消并清空所有下游代码/产物。"""

        downstream = [
            (sid, state)
            for sid, state in sorted(ctx.scene_states.items())
            if sid > scene_id and (include_failed or (not state.failed and not state.give_up))
        ]
        if not downstream:
            return
        cancellation_failed = False
        for sid, state in downstream:
            try:
                self._cancel_unfinished_scene_job(state, reason=f"使用旧连续性上下文（{reason}）")
            except RuntimeError as exc:
                with self._state_lock:
                    state.give_up = True
                    state.failed = True
                    state.failure_category = "continuity"
                    state.failure_reason = str(exc)
                    ctx.continuity_warnings.append(state.failure_reason)
                    cancellation_failed = True
                continue
            with self._state_lock:
                state.code = ""
                state.class_name = ""
                state.reviewed = False
                state.rewrite_feedback = ""
                state.review_signature = ""
                state.identical_review_count = 0
                state.slurm_job = None
                state.artifact = None
                state.rendered = False
                state.give_up = False
                state.failed = False
                state.failure_reason = ""
                state.failure_category = ""
                state.inherited_elements_code = ""
                state.exported_elements_code = ""
                state.exported_elements = []
                self._reset_technical_spec(state)
                self._remove_element_manifest_scene(ctx, sid)
                state.safe_fallback_used = False
                state.safe_fallback_reason = ""
                self._reset_visual_receipt(
                    ctx,
                    state,
                    clear_candidate=not preserve_visual_candidates,
                    reset_attempts=not preserve_visual_candidates,
                )
                self._write_private(ctx.paths.scenes / f"scene_{sid}.py", "")
        if cancellation_failed:
            with self._state_lock:
                # 未确认取消前绝不能进入新的连续性上下文；将本轮标记为
                # 阻断并交给外层异常收尾，确保不会悄悄继续或重复提交。
                ctx.continuity_rebuild_required = False
                self._checkpoint(ctx, State.REVIEWING)
            self._stop_event.set()
            return
        with self._state_lock:
            ctx.continuity_rebuild_required = True
            ctx.continuity_warnings.append(
                f"Scene {scene_id} 因{reason}，已请求重建后续场景的连续性上下文。"
            )
        self._stop_event.set()

    def _scene_fix(self, ctx: PipelineContext, scene_id: int, state: SceneState) -> None:
        self._phase_emit("fixing")
        job = state.slurm_job
        if job is None:
            return
        # 与 Coder/Reviewer 一样每个 worker 独立构造 Agent，避免并发修复
        # 共享 OpenAI client/流式状态。
        fixer = AutoFixerAgent()
        error_log = self.slurm.get_error_log(job=job)
        if not error_log:
            with self._state_lock:
                state.give_up = True
                state.failure_category = "render"
                state.failure_reason = job.failure_reason or "渲染失败且没有错误日志"
                self._checkpoint(ctx, State.FIXING)
            self._emit("scene_give_up", scene_id=scene_id, reason=state.failure_reason)
            return
        route = classify_failure(error_log, phase="render", status=job.status)
        self._emit(
            "failure_routed",
            scene_id=scene_id,
            category=route.category,
            handler=route.handler,
            reason=route.reason,
        )
        if route.category == "infrastructure" or fixer.is_infrastructure_error(error_log):
            with self._state_lock:
                state.give_up = True
                state.failure_category = "infrastructure"
                state.failure_reason = self._give_up_reason(
                    "检测到环境或 Slurm 配置错误，未让 LLM 重写业务代码", error_log
                )
                self._checkpoint(ctx, State.FIXING)
            self._emit("scene_give_up", scene_id=scene_id, reason=state.failure_reason)
            return
        patch_builder = getattr(fixer, "deterministic_patches", None)
        deterministic_patches = (
            patch_builder(state.code, error_log) if callable(patch_builder) else []
        )
        if deterministic_patches:
            patch_result = ReviewResult(
                is_valid=False,
                severity="minor",
                feedback="根据渲染日志发现可唯一定位的旧 API 调用",
                fixes=deterministic_patches,
            )
            if self._apply_precise_review_fixes(ctx, scene_id, state, patch_result):
                # 当前失败 Job 已经结束；补丁产生新代码后必须清除旧
                # Job，避免渲染循环再次轮询同一个已结束作业。
                with self._state_lock:
                    state.slurm_job = None
                    self._checkpoint(ctx, State.FIXING)
                self._request_continuity_rebuild(
                    ctx,
                    scene_id,
                    preserve_visual_candidates=state.visual_best_candidate is not None,
                    include_failed=True,
                )
                self._emit("scene_render_patch_applied", scene_id=scene_id)
                return
        # 连续相同错误 → 判定为环境/配置问题, 提前放弃, 不再浪费修复次数
        fp = self._error_fingerprint(error_log)
        with self._state_lock:
            if fp and fp == state.last_error_fp:
                state.identical_error_count += 1
            else:
                state.identical_error_count = 1
                state.last_error_fp = fp
            # 连续相同错误 → 提前放弃, 避免修复器在同一个环境错误上空转。
            # 但必须叠加 fix_attempts>=2 门槛: 修复器至少要修过 2 次才允许据此放弃。
            if (
                state.identical_error_count >= settings.MAX_FIX_IDENTICAL_ERRORS
                and state.fix_attempts >= 2
            ):
                state.give_up = True
                state.failure_category = "infrastructure"
                state.failure_reason = self._give_up_reason(
                    f"连续 {state.identical_error_count} 次渲染错误完全相同且修复未能消除，"
                    "疑似环境/配置问题，已放弃",
                    error_log,
                )
                self._checkpoint(ctx, State.FIXING)
                terminal = True
            elif state.fix_attempts >= settings.MAX_FIX_ATTEMPTS:
                state.give_up = True
                state.failure_category = "render"
                state.failure_reason = self._give_up_reason("达到最大渲染修复次数", error_log)
                self._checkpoint(ctx, State.FIXING)
                terminal = True
            else:
                state.fix_attempts += 1
                # 在 LLM 调用前持久化次数，避免进程中断后无限重试同一轮。
                self._checkpoint(ctx, State.FIXING)
                terminal = False
            attempt = state.fix_attempts
        if terminal:
            self._emit("scene_give_up", scene_id=scene_id, reason=state.failure_reason)
            return
        rag_context = self._retrieve_rag(
            ctx,
            "\n".join(
                (
                    error_log[-20_000:],
                    state.plan.math_concept,
                    state.plan.computation,
                    ctx.render_profile.renderer,
                )
            ),
            receipt_key=f"scene:{scene_id}:fix:{attempt}",
            stage="fix",
            source_kinds={"manim_doc", "example", "recipe"},
            preferred_source_kinds={"recipe"},
            code_sha256=sha256_text(state.code) if state.code else "",
            inherited_elements_sha256=sha256_text(state.inherited_elements_code)
            if state.inherited_elements_code
            else "",
        )
        self._emit(
            "scene_fixing",
            scene_id=scene_id,
            attempt=attempt,
            max_attempts=settings.MAX_FIX_ATTEMPTS,
        )
        with self._llm_sem:
            fix_kwargs: dict[str, object] = {"renderer": ctx.render_profile.renderer}
            if state.technical_spec is not None and self._supports_keyword(
                fixer.fix, "technical_spec"
            ):
                fix_kwargs["technical_spec"] = state.technical_spec
            if rag_context and self._supports_keyword(fixer.fix, "rag_context"):
                fix_kwargs["rag_context"] = rag_context
            if self._supports_keyword(fixer.fix, "lesson_spec"):
                fix_kwargs["lesson_spec"] = ctx.lesson_spec
            if self._supports_keyword(fixer.fix, "teaching_graph"):
                fix_kwargs["teaching_graph"] = ctx.teaching_graph
            candidate = fixer.fix(state.code, error_log, **fix_kwargs)
            validation = self._validate(candidate, renderer=ctx.render_profile.renderer)
            continuity_error = ""
            try:
                extract_scene_continuity_elements(candidate, state.plan)
            except ValueError as exc:
                continuity_error = str(exc)
            lifecycle_error = ""
            if state.technical_spec is not None and validation.is_valid and not continuity_error:
                lifecycle_result = validate_animation_lifecycle(
                    candidate,
                    state.technical_spec,
                    renderer=ctx.render_profile.renderer,
                )
                if not lifecycle_result.is_valid:
                    lifecycle_error = "\n".join(lifecycle_result.errors)
            if not validation.is_valid or continuity_error or lifecycle_error:
                candidate, class_name = self._generate_validated_code(
                    state.plan,
                    feedback=(
                        "AutoFix 结果未通过确定性校验：\n"
                        f"{validation.feedback}\n"
                        + (
                            f"连续性导出合同错误：\n{continuity_error}\n"
                            if continuity_error
                            else ""
                        )
                        + (f"动画生命周期错误：\n{lifecycle_error}\n" if lifecycle_error else "")
                        + f"\n原始渲染错误：\n{error_log}"
                    ),
                    previous_code=candidate,
                    stream=False,
                    renderer=ctx.render_profile.renderer,
                    continuity_bible=ctx.continuity_bible,
                    inherited_elements_code=state.inherited_elements_code,
                    inherited_elements=state.plan.inherited_elements,
                    elements_to_remove=state.plan.elements_to_remove,
                    element_manifest=(
                        ctx.element_manifest.model_copy(
                            update={
                                "entries": ctx.element_manifest.for_elements(
                                    {item.element_id for item in state.plan.inherited_elements}
                                )
                            }
                        )
                        if state.plan.inherited_elements
                        else None
                    ),
                    technical_spec=state.technical_spec,
                    rag_context=rag_context,
                    lesson_spec=ctx.lesson_spec,
                    teaching_graph=ctx.teaching_graph,
                )
            else:
                class_name = validation.scene_classes[0]
        code_changed = candidate != state.code
        self._write_private(ctx.paths.scenes / f"scene_{scene_id}.py", candidate)
        with self._state_lock:
            state.code = candidate
            state.class_name = class_name
            state.review_round = 0
            state.reviewed = False
            state.failure_reason = ""
            state.failure_category = ""
            state.infra_retries = 0
            state.slurm_job = None
            state.artifact = None
            state.rendered = False
            state.exported_elements_code = ""
            state.exported_elements = []
            state.local_smoke_status = "pending"
            self._remove_element_manifest_scene(ctx, scene_id)
            self._reset_visual_receipt(ctx, state)
            self._checkpoint(ctx, State.FIXING)
        if code_changed:
            self._request_continuity_rebuild(
                ctx,
                scene_id,
                preserve_visual_candidates=state.visual_best_candidate is not None,
                include_failed=True,
            )
        self._emit(
            "scene_coded",
            scene_id=scene_id,
            file_path=str(ctx.paths.scenes / f"scene_{scene_id}.py"),
        )
        # 注意: identical_error_count 不在这里重置 —— 只有当"错误指纹变化"时才重置
        # (见上面的 else 分支), 从而让"修复后错误完全相同"能在第 2 次相同错误时提前放弃。

    def _activate_safe_fallback(
        self,
        ctx: PipelineContext,
        scene_id: int,
        state: SceneState,
        feedback: str,
    ) -> bool:
        """把高风险几何场景切换为保守方案，并清空其下游交接。"""

        if (
            not settings.SAFE_FALLBACK_ENABLED
            or state.safe_fallback_used
            or not is_high_confidence_geometry_conflict(state.plan, feedback)
        ):
            return False

        fallback_plan = build_safe_fallback_plan(
            state.plan,
            ctx.continuity_bible or ContinuityBible(),
            reason=fallback_reason_summary(feedback),
        )
        previous_plan = None
        previous_state = ctx.scene_states.get(scene_id - 1)
        if previous_state is not None and previous_state.plan_ready:
            previous_plan = previous_state.plan
        if ctx.continuity_bible is not None:
            fallback_plan, contract_repairs = normalize_scene_plan_contract(
                fallback_plan,
                ctx.continuity_bible,
                previous_plan=previous_plan,
                has_next_scene=any(
                    item.plan.scene_id > scene_id
                    for item in ctx.scene_states.values()
                    if item.plan_ready
                ),
            )
        else:
            contract_repairs = []
        reason = fallback_reason_summary(feedback)
        rewrite_feedback = (
            "系统已将本场景切换为保守教学方案。"
            "只展示已确认的基础图形、面积标签、等式和结论；"
            "禁止重新加入未经验证的切割、旋转、碎片移动或无缝拼接。"
        )
        if contract_repairs:
            rewrite_feedback += "\n同时修复连续性合同：" + "；".join(contract_repairs)
        self._cancel_unfinished_scene_job(state, reason="切换保守方案")
        with self._state_lock:
            state.plan = fallback_plan
            self._reset_technical_spec(state)
            state.safe_fallback_used = True
            state.safe_fallback_reason = reason
            state.review_round = 0
            state.plan_review_round = 0
            state.plan_reviewed = False
            state.plan_review_feedback = ""
            state.plan_review_signature = ""
            state.identical_plan_review_count = 0
            state.code = ""
            state.class_name = ""
            state.reviewed = False
            state.rewrite_feedback = rewrite_feedback
            state.review_signature = ""
            state.identical_review_count = 0
            state.artifact = None
            state.rendered = False
            state.slurm_job = None
            state.exported_elements_code = ""
            state.exported_elements = []
            self._remove_element_manifest_scene(ctx, scene_id)
            state.give_up = False
            state.failed = False
            state.failure_reason = ""
            state.failure_category = ""
            if ctx.continuity_bible is not None:
                ctx.continuity_review_status = "pending"
            ctx.plan_review_status = "pending"
            ctx.continuity_warnings.append(f"Scene {scene_id} 已切换为保守教学方案：{reason}")
            ctx.scenes = [
                item.plan
                for item in sorted(ctx.scene_states.values(), key=lambda item: item.plan.scene_id)
            ]
            self._reset_visual_receipt(ctx, state, clear_candidate=True, reset_attempts=True)
            self._checkpoint(ctx, State.PLAN_REVIEWING)
            self._write_private(ctx.paths.scenes / f"scene_{scene_id}.py", "")

        # 即使当前场景是最后一个场景，降级也改变了 plan_reviewed 状态；
        # 让外层 dry-run/正式流水线重新经过计划审查，而不是直接进入渲染。
        with self._state_lock:
            ctx.continuity_rebuild_required = True
        self._stop_event.set()
        self._request_continuity_rebuild(
            ctx,
            scene_id,
            reason="切换为保守教学方案",
            include_failed=True,
        )
        self._emit(
            "scene_safe_fallback",
            scene_id=scene_id,
            reason=reason,
        )
        return True

    def _apply_precise_review_fixes(
        self,
        ctx: PipelineContext,
        scene_id: int,
        state: SceneState,
        result: ReviewResult,
    ) -> bool:
        """在不改动教学合同的前提下应用 Reviewer 的精确局部修复。

        ``math``/``continuity`` finding 不一定意味着计划错误；例如计划
        正确地要求某个顶点，而 Coder 只把代码里的坐标写错。若 Reviewer
        同时给出了唯一可匹配的局部替换，先在代码层应用并重新跑全部确定性
        校验，避免把纯实现错误错误升级为 Planner 重规划循环。
        """

        candidate = state.code
        applied = 0
        for fix in result.fixes:
            if not fix.find or fix.find == fix.replace:
                continue
            if candidate.count(fix.find) != 1:
                return False
            candidate = candidate.replace(fix.find, fix.replace, 1)
            applied += 1
        if applied == 0 or candidate == state.code:
            return False

        validation = self._validate(candidate, renderer=ctx.render_profile.renderer)
        if not validation.is_valid:
            return False
        try:
            extract_scene_continuity_elements(candidate, state.plan)
        except ValueError:
            return False
        if state.technical_spec is not None:
            lifecycle_result = validate_animation_lifecycle(
                candidate,
                state.technical_spec,
                renderer=ctx.render_profile.renderer,
            )
            if not lifecycle_result.is_valid:
                return False

        self._write_private(ctx.paths.scenes / f"scene_{scene_id}.py", candidate)
        with self._state_lock:
            state.code = candidate
            state.class_name = validation.scene_classes[0]
            state.reviewed = False
            state.review_round = 0
            state.review_signature = ""
            state.identical_review_count = 0
            state.rewrite_feedback = ""
            state.artifact = None
            state.rendered = False
            state.exported_elements_code = ""
            state.exported_elements = []
            state.local_smoke_status = "pending"
            self._remove_element_manifest_scene(ctx, scene_id)
            self._reset_visual_receipt(ctx, state)
            self._checkpoint(ctx, State.REVIEWING)
        self._emit(
            "scene_coded",
            scene_id=scene_id,
            file_path=str(ctx.paths.scenes / f"scene_{scene_id}.py"),
        )
        self._emit("scene_review_fix_applied", scene_id=scene_id, severity=result.severity)
        return True

    def _apply_review_result(
        self, ctx: PipelineContext, scene_id: int, state: SceneState, result: ReviewResult
    ) -> bool:
        """应用单场景审查结果。"""
        self._write_stage_artifact(
            ctx,
            f"code_review_scene_{scene_id}_{state.review_round + 1}.json",
            {
                "schema_version": 1,
                "scene_id": scene_id,
                "code_sha256": sha256_text(state.code),
                "result": result.model_dump(mode="json"),
            },
        )
        if result.is_valid:
            warning_messages = [
                f"Scene {scene_id} 代码审查提示：{warning}" for warning in result.warnings
            ]
            with self._state_lock:
                state.review_round = 0
                state.review_signature = ""
                state.identical_review_count = 0
                state.reviewed = True
                state.failure_reason = ""
                state.failure_category = ""
                self._apply_incremental_for_scene(ctx, scene_id, state)
                self._update_state_ledger(ctx, state)
                if warning_messages:
                    ctx.continuity_warnings.extend(warning_messages)
                self._checkpoint(ctx, State.REVIEWING)
            self._emit("scene_review_pass", scene_id=scene_id)
            if warning_messages:
                self._emit(
                    "scene_review_warning",
                    scene_id=scene_id,
                    warnings=warning_messages[:20],
                )
            if state.rendered and state.artifact and state.artifact.origin == "reused":
                self._emit("scene_reused", scene_id=scene_id)
            return True

        original_feedback = result.feedback or ""
        if result.fixes and self._apply_precise_review_fixes(ctx, scene_id, state, result):
            return True
        if result.severity == "minor":
            # 精确修复已经在上面经过完整确定性校验；如果失败，不再
            # 使用旧的“只做 AST 校验”的分支，避免把未经连续性/生命
            # 周期校验的代码当成可接受修复。将其升级为普通重写反馈。
            result = ReviewResult(
                is_valid=False,
                severity="major",
                feedback=(original_feedback or "局部修复未通过完整的 AST、连续性或生命周期校验"),
                fixes=result.fixes,
                findings=result.findings,
                warnings=result.warnings,
            )
        plan_finding = next(
            (
                finding
                for finding in result.findings
                if finding.severity == "major"
                and finding.category in {"math", "continuity"}
                and self._review_finding_requires_plan_repair(finding)
            ),
            None,
        )
        if plan_finding is not None:
            target = "planner" if plan_finding.category == "math" else "continuity"
            feedback = original_feedback or plan_finding.repair or plan_finding.why
            self._schedule_visual_plan_repair(
                ctx,
                scene_id,
                state,
                feedback,
                target,
                source="代码审查",
            )
            self._emit(
                "scene_plan_repair_requested",
                scene_id=scene_id,
                target=target,
            )
            return True
        fix_details = "\n".join(
            f"- [{fix.reason}] {fix.find!r} → {fix.replace!r}" for fix in result.fixes
        )
        review_signature = sha256_text(
            f"{state.code}\n--- reviewer feedback ---\n{original_feedback}\n{fix_details}"
        )[:16]
        with self._state_lock:
            state.review_round += 1
            review_round = state.review_round
            if state.review_signature == review_signature:
                state.identical_review_count += 1
            else:
                state.review_signature = review_signature
                state.identical_review_count = 1
            identical_review_count = state.identical_review_count
            # Reviewer 已消耗一轮，在后续改写前先持久化计数。
            self._checkpoint(ctx, State.REVIEWING)

        if identical_review_count >= settings.MAX_IDENTICAL_REVIEW_ATTEMPTS:
            if self._activate_safe_fallback(ctx, scene_id, state, original_feedback):
                return True
            with self._state_lock:
                state.give_up = True
                state.failure_category = "review"
                state.failure_reason = (
                    f"相同代码和审查反馈连续重复 {identical_review_count} 次，已停止重复重写"
                )
                self._checkpoint(ctx, State.REVIEWING)
            self._emit("scene_give_up", scene_id=scene_id, reason=state.failure_reason)
            return True

        if review_round >= settings.MAX_REVIEW_ROUNDS:
            if self._activate_safe_fallback(ctx, scene_id, state, original_feedback):
                return True
            with self._state_lock:
                state.give_up = True
                state.failure_category = "review"
                state.failure_reason = "达到最大审查轮次，代码仍未通过"
                self._checkpoint(ctx, State.REVIEWING)
            self._emit("scene_give_up", scene_id=scene_id, reason=state.failure_reason)
            return True

        # major → 排队重写 (下一轮调度进入编码阶段)
        with self._state_lock:
            state.rewrite_feedback = (
                f"## Reviewer 审查意见\n{original_feedback}\n\n"
                f"## 需修复的问题\n{fix_details}\n\n"
                f"请根据以上反馈逐项修正代码，保留正确部分，只修复指出的问题。"
            )
            state.artifact = None
            state.rendered = False
            self._reset_visual_receipt(ctx, state)
            self._checkpoint(ctx, State.REVIEWING)
        self._emit("scene_review_fail", scene_id=scene_id, severity="major")
        return True

    @staticmethod
    def _review_finding_requires_plan_repair(finding: ReviewFinding) -> bool:
        """区分“计划本身错误”和“代码实现了错误数学/连续性动作”。

        Reviewer 的 ``math``/``continuity`` 分类同时覆盖两种问题：例如
        “分镜要求错误公式”应回到 Planner，而“代码用 apply_function 实现
        线性剪切”应留在 Coder 修复。后者具有当前代码的行号、证据和 API
        级 repair；若仅按 category 路由，会把代码错误误送回计划层，造成
        重规划并丢失本来可以局部修复的正确分镜。
        """

        text = " ".join((finding.why, finding.repair)).lower()
        plan_markers = (
            "sceneplan",
            "lessonspec",
            "teachinggraph",
            "教学合同",
            "数学合同",
            "计划层",
            "计划本身",
            "当前计划",
            "分镜本身",
            "重新规划",
            "重规划",
            "无法通过代码修复",
            "claim_id 分配",
        )
        return any(marker.lower() in text for marker in plan_markers)

    def _apply_incremental_for_scene(
        self, ctx: PipelineContext, scene_id: int, state: SceneState
    ) -> None:
        """Review 通过后，按代码与渲染配置身份安全复用旧产物。"""
        if not ctx.incremental or not ctx.base_manifest or not ctx.base_run_id:
            return
        base_scene = ctx.base_manifest.scenes.get(scene_id)
        if base_scene is None:
            ctx.scenes_to_render.append(scene_id)
            return
        artifact = base_scene.artifact
        if (
            base_scene.rendered
            and artifact is not None
            and artifact.verified
            and sha256_text(state.code) == base_scene.code_sha256
            and artifact.code_sha256 == base_scene.code_sha256
            and artifact.render_profile_sha256 == ctx.render_profile.digest()
            and not artifact.environment_warning
        ):
            base_root = RunRepository(settings.WORKSPACE_DIR).run_root(ctx.base_run_id)
            old_video = get_reusable_video_path(ctx.base_manifest, scene_id, base_root)
            if old_video:
                try:
                    if (
                        old_video.stat().st_size == artifact.metadata.size_bytes
                        and sha256_file(old_video) == artifact.video_sha256
                    ):
                        copied_video = self._copy_reused_video(
                            ctx,
                            scene_id,
                            artifact,
                            old_video,
                        )
                    else:
                        copied_video = None
                except (OSError, RuntimeError, ValueError) as exc:
                    logger.warning("Scene %s 增量视频复制失败，将重新渲染: %s", scene_id, exc)
                    copied_video = None
                if copied_video is not None:
                    relative_video = copied_video.relative_to(ctx.paths.root).as_posix()
                    state.rendered = True
                    state.slurm_job = None
                    state.artifact = SceneArtifact(
                        origin="reused",
                        # 复用视频已复制到当前 run；清理 base run 后当前 run
                        # 仍必须能够恢复和重新合并，因此凭据路径也归属于当前 run。
                        source_run_id=ctx.paths.run_id,
                        job_id=artifact.job_id,
                        scene_id=scene_id,
                        scene_class_name=artifact.scene_class_name,
                        code_sha256=artifact.code_sha256,
                        render_profile_sha256=artifact.render_profile_sha256,
                        video_path=relative_video,
                        video_sha256=artifact.video_sha256,
                        metadata=artifact.metadata,
                        verified=True,
                        environment_fingerprint=dict(artifact.environment_fingerprint),
                        environment_warning=artifact.environment_warning,
                    )
                    self._reset_visual_receipt(ctx, state)
                    if scene_id not in ctx.scenes_to_reuse:
                        ctx.scenes_to_reuse.append(scene_id)
                    return
        if scene_id not in ctx.scenes_to_render:
            ctx.scenes_to_render.append(scene_id)

    @staticmethod
    def _copy_reused_video(
        ctx: PipelineContext,
        scene_id: int,
        artifact: SceneArtifact,
        source: Path,
    ) -> Path:
        """把增量复用的视频复制到当前 run，并以原子替换完成落盘。"""

        destination_dir = ctx.paths.videos / f"scene_{scene_id}" / "reused"
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination_dir.chmod(0o700)
        destination = destination_dir / f"{artifact.scene_class_name}.mp4"
        temporary = destination.with_name(f".{destination.name}.{uuid4().hex[:8]}.tmp")
        try:
            shutil.copyfile(source, temporary)
            temporary.chmod(0o600)
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            if destination.stat().st_size <= 0 or sha256_file(destination) != artifact.video_sha256:
                raise RuntimeError("复制后的视频哈希或大小校验失败")
            return destination
        finally:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)

    def _artifact_from_job(
        self,
        ctx: PipelineContext,
        state: SceneState,
        job: SlurmJob,
    ) -> SceneArtifact:
        if (
            job.output_path is None or job.output_metadata is None
        ) and not self.slurm.validate_completed_job(job):
            raise RuntimeError(job.failure_reason or "渲染产物验证失败")
        if job.output_path is None or job.output_metadata is None:
            raise RuntimeError("渲染产物缺少路径或媒体元数据")
        try:
            relative_video = (
                job.output_path.resolve().relative_to(ctx.paths.root.resolve()).as_posix()
            )
        except ValueError as exc:
            raise RuntimeError("渲染产物不在当前 run 目录内") from exc
        code_hash = sha256_text(state.code)
        if job.code_sha256 != code_hash:
            raise RuntimeError("渲染作业记录的代码哈希与当前场景不一致")
        if job.render_profile.digest() != ctx.render_profile.digest():
            raise RuntimeError("渲染作业记录的配置与当前运行不一致")
        return SceneArtifact(
            origin="rendered",
            source_run_id=ctx.paths.run_id,
            job_id=job.job_id,
            scene_id=job.scene_id,
            scene_class_name=job.scene_class_name,
            code_sha256=code_hash,
            render_profile_sha256=ctx.render_profile.digest(),
            video_path=relative_video,
            video_sha256=sha256_file(job.output_path),
            metadata=job.output_metadata,
            verified=True,
            environment_fingerprint=dict(job.environment_fingerprint),
            environment_warning=job.environment_warning,
        )

    @staticmethod
    def _artifact_video_path(ctx: PipelineContext, artifact: SceneArtifact) -> Path:
        if not artifact.verified:
            raise RuntimeError(f"Scene {artifact.scene_id} 的渲染产物未经验证")
        source_root = (
            ctx.paths.root
            if artifact.source_run_id == ctx.paths.run_id
            else RunRepository(settings.WORKSPACE_DIR).run_root(artifact.source_run_id)
        )
        video = restore_run_path(source_root, artifact.video_path)
        if not video.is_file() or video.stat().st_size <= 0:
            raise RuntimeError(f"Scene {artifact.scene_id} 的渲染产物不存在: {video}")
        if sha256_file(video) != artifact.video_sha256:
            raise RuntimeError(f"Scene {artifact.scene_id} 的渲染产物哈希不一致")
        return video

    @staticmethod
    def _scene_visual_context(ctx: PipelineContext, state: SceneState) -> str:
        payload = {
            "scene_plan": state.plan.model_dump(mode="json"),
            "lesson_spec": ctx.lesson_spec.model_dump(mode="json"),
            "teaching_graph": ctx.teaching_graph.model_dump(mode="json"),
            "global_continuity": (
                ctx.continuity_bible.model_dump(mode="json") if ctx.continuity_bible else None
            ),
            "element_manifest": {
                "entries": [
                    entry.model_dump(mode="json")
                    for entry in ctx.element_manifest.for_elements(
                        {item.element_id for item in state.plan.inherited_elements}
                        | {item.element_id for item in state.plan.new_elements}
                    )
                ]
            },
            "state_ledger": {
                "elements": [
                    item.model_dump(mode="json")
                    for item in ctx.state_ledger.for_elements(
                        {item.element_id for item in state.plan.inherited_elements}
                        | {item.element_id for item in state.plan.new_elements}
                    )
                ],
                "boundary": (
                    ctx.state_ledger.boundaries.get(state.plan.scene_id).model_dump(mode="json")
                    if state.plan.scene_id in ctx.state_ledger.boundaries
                    else None
                ),
            },
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)[:30_000]

    @staticmethod
    def _visual_repair_target(result: object) -> str:
        """根据视觉问题类别决定回到计划层还是代码层。"""

        issues = list(getattr(result, "issues", []) or [])
        for issue in issues:
            target = getattr(issue, "repair_target", "unknown")
            if target in {"planner", "continuity"}:
                return target
        categories = {getattr(issue, "category", "other") for issue in issues}
        if "mathematics" in categories or "relevance" in categories:
            return "planner"
        if "consistency" in categories:
            return "continuity"
        if categories & {"readability", "layout", "clipping", "overlap", "contrast"}:
            return "coder"
        # 没有 issue 但整体维度低时，保留旧版“质量低→代码优化”的行为；
        # 显式标记为 other/unknown 的问题则不应盲目修改代码。
        return "coder" if not issues else "unknown"

    def _schedule_visual_plan_repair(
        self,
        ctx: PipelineContext,
        scene_id: int,
        state: SceneState,
        feedback: str,
        target: str,
        source: str = "视觉评估",
    ) -> None:
        """把视觉发现转成有证据的计划层问题，禁止用 Coder 修补错误计划。"""

        category = "math" if target == "planner" else "continuity"
        issue = PlanReviewIssue(
            category=category,
            severity="major",
            field="visual_evaluation",
            message=f"{source}发现需要计划层处理的问题：{feedback[:4_500]}",
            fix_instruction=(
                "重新检查 LessonSpec 中的数学断言、定义域和教学依赖，修正 ScenePlan 后再编码。"
                if target == "planner"
                else "重新检查相邻场景的 opening/closing 状态和元素交接合同，修正 ScenePlan 后再编码。"
            ),
        )
        with self._state_lock:
            existing = ctx.plan_compile_issues.setdefault(scene_id, [])
            if not any(
                item.field == issue.field and item.message == issue.message for item in existing
            ):
                existing.append(issue)
            state.plan_reviewed = False
            state.plan_review_round = 0
            state.plan_review_feedback = issue.message
            state.plan_review_signature = ""
            state.identical_plan_review_count = 0
            state.code = ""
            state.class_name = ""
            state.reviewed = False
            state.rewrite_feedback = ""
            state.slurm_job = None
            state.artifact = None
            state.rendered = False
            state.exported_elements_code = ""
            state.exported_elements = []
            state.local_smoke_status = "pending"
            state.safe_fallback_used = False
            state.safe_fallback_reason = ""
            self._reset_technical_spec(state)
            self._remove_element_manifest_scene(ctx, scene_id)
            self._reset_visual_receipt(ctx, state, clear_candidate=True, reset_attempts=False)
            ctx.plan_review_status = "pending"
            if ctx.continuity_bible is not None:
                ctx.continuity_review_status = "pending"
            ctx.final_video = None
            ctx.final_video_sha256 = ""
            self._write_private(ctx.paths.scenes / f"scene_{scene_id}.py", "")
            self._checkpoint(
                ctx,
                State.PLAN_REVIEWING if source == "代码审查" else State.VISUAL_EVALUATING,
            )
            ctx.continuity_rebuild_required = True
            self._stop_event.set()
        self._request_continuity_rebuild(
            ctx,
            scene_id,
            reason=f"视觉评估要求{target}层重规划",
            preserve_visual_candidates=False,
            include_failed=True,
        )

    @staticmethod
    def _redact_visual_error(error: Exception) -> str:
        detail = redact_text(
            str(error).strip() or type(error).__name__,
            (settings.VISUAL_LLM_API_KEY,),
        )
        return detail[:10_000]

    def _restore_visual_candidate_into_state(
        self,
        ctx: PipelineContext,
        scene_id: int,
        state: SceneState,
        candidate: VisualCandidate,
    ) -> bool:
        """恢复已验证候选；返回代码是否相对当前状态发生变化。"""

        # 所有恢复分支都必须在写回当前状态前验证候选视频；不能只验证
        # 候选代码和报告，否则视频被删除/替换后仍会把 rendered=True 写入
        # manifest，直到合并阶段才暴露问题。
        self._artifact_video_path(ctx, candidate.artifact)
        with self._state_lock:
            inherited_hash = sha256_text(state.inherited_elements_code)
            if candidate.inherited_elements_sha256 != inherited_hash:
                raise RuntimeError(f"Scene {scene_id} 的视觉候选基于不同的继承上下文，拒绝恢复")
            if candidate.artifact.code_sha256 != sha256_text(candidate.code):
                raise RuntimeError(f"Scene {scene_id} 的视觉候选代码哈希与视频产物不一致")
            if candidate.artifact.scene_class_name != candidate.class_name:
                raise RuntimeError(f"Scene {scene_id} 的视觉候选类名与视频产物不一致")
            code_changed = candidate.code != state.code
        self._write_private(ctx.paths.scenes / f"scene_{scene_id}.py", candidate.code)
        with self._state_lock:
            state.code = candidate.code
            state.class_name = candidate.class_name
            state.slurm_job = candidate.slurm_job
            state.artifact = candidate.artifact
            state.rendered = True
            state.reviewed = True
            state.failed = False
            state.give_up = False
            state.failure_reason = ""
            state.failure_category = ""
            state.rewrite_feedback = ""
            state.exported_elements_code = candidate.exported_elements_code
            state.exported_elements = list(candidate.exported_elements)
            state.visual_status = "passed" if candidate.passed else "warning"
            state.visual_score = candidate.score
            state.visual_report_file = candidate.report_file
            state.visual_report_sha256 = candidate.report_sha256
            state.visual_artifact_sha256 = candidate.artifact.video_sha256
            state.visual_feedback = ""
        return code_changed

    def _visual_gate(
        self,
        ctx: PipelineContext,
        *,
        scene_ids: set[int] | None = None,
    ) -> bool:
        """评估精确渲染产物；渲染线程可通过 scene_ids 触发即时评估。"""

        with self._visual_eval_lock:
            return self._visual_gate_locked(ctx, scene_ids=scene_ids)

    def _visual_gate_locked(
        self,
        ctx: PipelineContext,
        *,
        scene_ids: set[int] | None = None,
    ) -> bool:
        """视觉门的串行实现；调用者必须持有 _visual_eval_lock。"""

        profile = ctx.visual_eval_profile
        if not profile.enabled:
            with self._state_lock:
                for state in ctx.scene_states.values():
                    state.visual_status = "skipped"
            return False

        # 若视觉修复链路自身失败，恢复此前得分最高且可验证的候选，避免丢掉
        # 原本可用的视频。恢复旧代码后必须重建所有下游连续性上下文。
        for scene_id, state in sorted(ctx.scene_states.items()):
            if scene_ids is not None and scene_id not in scene_ids:
                continue
            with self._state_lock:
                candidate = state.visual_best_candidate
                rendered = state.rendered
                failed = state.failed
                give_up = state.give_up
                inherited_code = state.inherited_elements_code
                current_code = state.code
            if rendered or candidate is None or not (failed or give_up):
                continue
            if candidate.inherited_elements_sha256 != sha256_text(inherited_code):
                continue
            self._artifact_video_path(ctx, candidate.artifact)
            changed = self._restore_visual_candidate_into_state(ctx, scene_id, state, candidate)
            with self._state_lock:
                state.visual_status = "warning"
                state.visual_feedback = "视觉修复链路失败，已恢复最佳可用版本"
            self._emit(
                "scene_visual_warning",
                scene_id=scene_id,
                score=candidate.score,
                reason="视觉修复链路失败，已恢复最佳可用版本",
            )
            if changed or candidate.code != current_code:
                self._request_continuity_rebuild(
                    ctx,
                    scene_id,
                    reason="视觉修复失败后恢复最佳候选",
                    preserve_visual_candidates=True,
                    include_failed=True,
                )
            self._checkpoint(ctx, State.VISUAL_EVALUATING)
            with self._state_lock:
                return bool(ctx.continuity_rebuild_required)

        targets: list[tuple[int, SceneState, SceneArtifact]] = []
        for scene_id, state in sorted(ctx.scene_states.items()):
            if scene_ids is not None and scene_id not in scene_ids:
                continue
            with self._state_lock:
                artifact = state.artifact
                if not state.rendered or artifact is None:
                    continue
                terminal = state.visual_status in {"passed", "warning", "unknown"}
                if terminal and state.visual_artifact_sha256 == artifact.video_sha256:
                    continue
                state.visual_status = "evaluating"
            targets.append((scene_id, state, artifact))
            self._emit("scene_visual_evaluating", scene_id=scene_id)

        if not targets:
            return False

        self._emit("stage_start", stage="visual_evaluating")
        self._checkpoint(ctx, State.VISUAL_EVALUATING)

        from concurrent.futures import ThreadPoolExecutor, as_completed

        from kd1_anime.eval import Evaluator

        def evaluate_one(scene_id: int, state: SceneState, artifact: SceneArtifact):
            video = self._artifact_video_path(ctx, artifact)
            frame_dir = (
                ctx.paths.root / "eval_frames" / f"scene_{scene_id}" / artifact.video_sha256[:12]
            )
            evaluator = Evaluator(
                enable_visual_eval=True,
                visual_eval_model=profile.model,
                output_dir=ctx.paths.root / "eval_reports",
            )
            with self._state_lock:
                scene_context = self._scene_visual_context(ctx, state)
            with self._visual_llm_slot():
                result, samples = evaluator.evaluate_scene_video(
                    video,
                    description=ctx.original_prompt or ctx.user_prompt,
                    scene_context=scene_context,
                    output_dir=frame_dir,
                    frame_count=profile.frame_count,
                )
            return result, samples

        outcomes: dict[int, tuple[object | None, list, str]] = {}
        with ThreadPoolExecutor(max_workers=settings.VISUAL_LLM_PARALLEL_WORKERS) as pool:
            futures = {
                pool.submit(evaluate_one, scene_id, state, artifact): (scene_id, artifact)
                for scene_id, state, artifact in targets
            }
            for future in as_completed(futures):
                scene_id, _artifact = futures[future]
                try:
                    result, samples = future.result()
                    outcomes[scene_id] = (result, samples, "")
                except Exception as exc:
                    outcomes[scene_id] = (None, [], self._redact_visual_error(exc))

        first_fix_scene: int | None = None
        first_fix_feedback = ""
        first_plan_scene: int | None = None
        first_plan_feedback = ""
        first_plan_target = ""
        for scene_id, state, artifact in targets:
            result, samples, error = outcomes[scene_id]
            with self._state_lock:
                visual_fix_attempt = state.visual_fix_attempts
                inherited_code = state.inherited_elements_code
                code_snapshot = state.code
                class_name_snapshot = state.class_name
                job_snapshot = state.slurm_job
                exported_code_snapshot = state.exported_elements_code
                exported_elements_snapshot = list(state.exported_elements)
            report_dir = ctx.paths.root / "eval_reports" / f"scene_{scene_id}"
            report_dir.mkdir(parents=True, exist_ok=True)
            report_dir.chmod(0o700)
            report_path = report_dir / (
                f"attempt_{visual_fix_attempt:02d}_{artifact.video_sha256[:12]}.json"
            )
            report: dict = {
                "schema_version": 1,
                "scope": "scene",
                "scene_id": scene_id,
                "model": profile.model,
                "visual_profile_sha256": profile.digest(),
                "attempt": visual_fix_attempt,
                "artifact_sha256": artifact.video_sha256,
                "code_sha256": artifact.code_sha256,
                "inherited_elements_sha256": sha256_text(inherited_code),
                "frames": [
                    {
                        "frame_id": sample.frame_id,
                        "scene_id": sample.scene_id,
                        "boundary_id": sample.boundary_id,
                        "timestamp_seconds": sample.timestamp_seconds,
                        "role": sample.role,
                        "path": sample.path.resolve()
                        .relative_to(ctx.paths.root.resolve())
                        .as_posix(),
                        "sha256": sample.image_sha256,
                    }
                    for sample in samples
                ],
                "error": error,
            }
            if result is not None:
                report["result"] = result.to_dict()
            atomic_write_json(report_path, report)
            report_relative = report_path.relative_to(ctx.paths.root).as_posix()
            report_hash = sha256_file(report_path)

            if result is None:
                with self._state_lock:
                    state.visual_report_file = report_relative
                    state.visual_report_sha256 = report_hash
                    state.visual_artifact_sha256 = artifact.video_sha256
                    state.visual_status = "unknown"
                    state.visual_score = None
                    state.visual_feedback = error
                self._emit("scene_visual_unknown", scene_id=scene_id, reason=error)
                continue

            score = result.overall_score
            feedback = result.feedback()
            passed = not result.needs_fix(profile.threshold)
            candidate = VisualCandidate(
                score=score,
                has_major_issue=result.has_major_issue,
                passed=passed,
                inherited_elements_sha256=sha256_text(inherited_code),
                code=code_snapshot,
                class_name=class_name_snapshot,
                slurm_job=job_snapshot,
                artifact=artifact,
                exported_elements_code=exported_code_snapshot,
                exported_elements=exported_elements_snapshot,
                report_file=report_relative,
                report_sha256=report_hash,
            )
            with self._state_lock:
                state.visual_report_file = report_relative
                state.visual_report_sha256 = report_hash
                state.visual_artifact_sha256 = artifact.video_sha256
                state.visual_score = score
                state.visual_feedback = feedback
                previous = state.visual_best_candidate
                candidate_rank = (
                    candidate.passed,
                    not candidate.has_major_issue,
                    candidate.score,
                )
                previous_rank = (
                    (previous.passed, not previous.has_major_issue, previous.score)
                    if previous is not None
                    else (False, False, 0.0)
                )
                if previous is None or candidate_rank > previous_rank:
                    state.visual_best_candidate = candidate

            if passed:
                with self._state_lock:
                    state.visual_status = "passed"
                    state.visual_feedback = ""
                self._emit(
                    "scene_visual_pass",
                    scene_id=scene_id,
                    score=score,
                )
                continue

            with self._state_lock:
                can_fix = ctx.auto_fix and state.visual_fix_attempts < profile.max_fix_attempts
            if can_fix:
                target = self._visual_repair_target(result)
                with self._state_lock:
                    state.visual_status = "needs_fix"
                if target in {"planner", "continuity"}:
                    if first_plan_scene is None or scene_id < first_plan_scene:
                        first_plan_scene = scene_id
                        first_plan_feedback = feedback
                        first_plan_target = target
                elif target == "coder" and first_fix_scene is None:
                    # 一次只修改最早失败场景。它的导出元素会影响所有后继场景；
                    # 先修后重建可避免并行修复基于即将过期的连续性上下文。
                    first_fix_scene = scene_id
                    first_fix_feedback = feedback
                continue

            # 主观质量问题不应使已经成功渲染的视频消失。达到上限时选择
            # 历次得分最高候选，并将未解决问题明确记录为 warning。
            with self._state_lock:
                best = state.visual_best_candidate
            if (
                best is not None
                and best.artifact.video_sha256 != artifact.video_sha256
                and best.inherited_elements_sha256 == sha256_text(inherited_code)
            ):
                self._artifact_video_path(ctx, best.artifact)
                changed = self._restore_visual_candidate_into_state(ctx, scene_id, state, best)
                with self._state_lock:
                    state.visual_status = "warning"
                    state.visual_feedback = "视觉修复耗尽后已恢复最高分候选"
                if changed:
                    self._request_continuity_rebuild(
                        ctx,
                        scene_id,
                        reason="视觉修复耗尽后恢复最高分候选",
                        preserve_visual_candidates=True,
                        include_failed=True,
                    )
            else:
                with self._state_lock:
                    state.visual_status = "warning"
                    if not state.visual_feedback:
                        state.visual_feedback = (
                            "已达到视觉修复上限" if ctx.auto_fix else "已关闭自动修复"
                        )
            self._emit(
                "scene_visual_warning",
                scene_id=scene_id,
                score=score,
                reason=("已达到视觉修复上限" if ctx.auto_fix else "已关闭自动修复"),
            )

        if first_plan_scene is not None:
            state = ctx.scene_states[first_plan_scene]
            with self._state_lock:
                state.visual_fix_attempts += 1
                visual_fix_attempt = state.visual_fix_attempts
                state.visual_feedback = first_plan_feedback
            self._emit(
                "scene_visual_plan_fixing",
                scene_id=first_plan_scene,
                target=first_plan_target,
                attempt=visual_fix_attempt,
                max_attempts=profile.max_fix_attempts,
            )
            self._schedule_visual_plan_repair(
                ctx,
                first_plan_scene,
                state,
                first_plan_feedback,
                first_plan_target,
            )
            return True

        if first_fix_scene is not None:
            state = ctx.scene_states[first_fix_scene]
            with self._state_lock:
                state.visual_fix_attempts += 1
                state.rewrite_feedback = (
                    "## Visual Evaluation Feedback\n"
                    f"{first_fix_feedback}\n\n"
                    "请使用原有全局视觉配置修复这些可见问题，不要改变正确的数学内容或"
                    "跨场景元素合同。"
                )
                state.reviewed = False
                state.rendered = False
                state.artifact = None
                state.slurm_job = None
                ctx.final_video = None
                ctx.final_video_sha256 = ""
                visual_fix_attempt = state.visual_fix_attempts
            self._emit(
                "scene_visual_fixing",
                scene_id=first_fix_scene,
                attempt=visual_fix_attempt,
                max_attempts=profile.max_fix_attempts,
            )
            self._request_continuity_rebuild(
                ctx,
                first_fix_scene,
                reason="视觉评估要求修改场景代码",
                preserve_visual_candidates=True,
                include_failed=True,
            )
            self._checkpoint(ctx, State.VISUAL_EVALUATING)
            return True

        self._checkpoint(ctx, State.VISUAL_EVALUATING)
        with self._state_lock:
            return bool(ctx.continuity_rebuild_required)

    def _boundary_visual_gate(self, ctx: PipelineContext) -> bool:
        """在合并前检查真实相邻场景边界，必要时触发一次定向修复。"""

        profile = ctx.visual_eval_profile
        if not profile.enabled:
            return False
        rendered = [
            (scene_id, self._artifact_video_path(ctx, state.artifact))
            for scene_id, state in sorted(ctx.scene_states.items())
            if state.rendered and state.artifact is not None
        ]
        if len(rendered) < 2:
            return False
        boundary_bindings = [
            {
                "scene_id": scene_id,
                "video_sha256": state.artifact.video_sha256,
                "code_sha256": state.artifact.code_sha256,
                "inherited_elements_sha256": sha256_text(state.inherited_elements_code),
            }
            for scene_id, state in sorted(ctx.scene_states.items())
            if state.rendered and state.artifact is not None
        ]
        from kd1_anime.eval import Evaluator

        max_boundaries = min(3, max(1, profile.frame_count // 2))
        frame_dir = ctx.paths.root / "eval_frames" / "boundaries"
        report_path = ctx.paths.root / "eval_reports" / "boundaries.json"
        try:
            evaluator = Evaluator(
                enable_visual_eval=True,
                visual_eval_model=profile.model,
                output_dir=ctx.paths.root / "eval_reports",
            )
            samples = evaluator.extract_boundary_samples(
                rendered,
                frame_dir,
                max_boundaries=max_boundaries,
            )
            if not samples:
                return False
            with self._state_lock:
                boundaries = dict(ctx.state_ledger.boundaries)
                for sample in samples:
                    if sample.scene_id is None or sample.scene_id not in boundaries:
                        continue
                    boundary = boundaries[sample.scene_id]
                    if sample.role == "boundary_end":
                        boundary = boundary.model_copy(
                            update={"ending_frame_sha256": sample.image_sha256}
                        )
                    elif sample.role == "boundary_start":
                        boundary = boundary.model_copy(
                            update={"opening_frame_sha256": sample.image_sha256}
                        )
                    boundaries[sample.scene_id] = boundary
                ctx.state_ledger = ctx.state_ledger.model_copy(update={"boundaries": boundaries})
                self._write_stage_artifact(
                    ctx,
                    "state_ledger.json",
                    {
                        "schema_version": 1,
                        "digest": ctx.state_ledger.digest(),
                        "ledger": ctx.state_ledger.model_dump(mode="json"),
                    },
                )
            with self._visual_llm_slot():
                result = evaluator.visual_evaluator.evaluate_video_frames(
                    samples,
                    ctx.original_prompt or ctx.user_prompt,
                    scene_context=json.dumps(
                        {
                            "lesson_spec": ctx.lesson_spec.model_dump(mode="json"),
                            "teaching_graph": ctx.teaching_graph.model_dump(mode="json"),
                            "state_ledger": ctx.state_ledger.model_dump(mode="json"),
                            "boundaries": [
                                ctx.state_ledger.boundaries.get(scene_id, {}).model_dump(
                                    mode="json"
                                )
                                if scene_id in ctx.state_ledger.boundaries
                                else {}
                                for scene_id, _ in rendered
                            ],
                        },
                        ensure_ascii=False,
                    )[:30_000],
                    scope="complete video",
                )
            report = {
                "schema_version": 1,
                "scope": "scene_boundaries",
                "model": profile.model,
                "visual_profile_sha256": profile.digest(),
                "state_ledger_digest": ctx.state_ledger.digest(),
                "scene_bindings": boundary_bindings,
                "frames": [
                    {
                        "frame_id": sample.frame_id,
                        "scene_id": sample.scene_id,
                        "boundary_id": sample.boundary_id,
                        "role": sample.role,
                        "path": sample.path.relative_to(ctx.paths.root).as_posix(),
                        "sha256": sample.image_sha256,
                    }
                    for sample in samples
                ],
                "result": result.to_dict(),
                "error": "",
            }
        except Exception as exc:
            report = {
                "schema_version": 1,
                "scope": "scene_boundaries",
                "model": profile.model,
                "visual_profile_sha256": profile.digest(),
                "state_ledger_digest": ctx.state_ledger.digest(),
                "scene_bindings": boundary_bindings,
                "frames": [],
                "result": None,
                "error": self._redact_visual_error(exc),
            }
            atomic_write_json(report_path, report)
            self._emit("boundary_visual_unknown", reason=report["error"])
            return False
        atomic_write_json(report_path, report)
        if not result.needs_fix(profile.threshold):
            self._emit("boundary_visual_pass", score=result.overall_score)
            return False
        target = self._visual_repair_target(result)
        if target == "unknown" or not ctx.auto_fix:
            self._emit("boundary_visual_warning", score=result.overall_score, target=target)
            return False
        issue_boundary_ids = {
            boundary_id for issue in result.issues for boundary_id in issue.boundary_ids
        }
        boundary_start_scene_ids = [
            sample.scene_id
            for sample in samples
            if sample.role == "boundary_start"
            and sample.scene_id is not None
            and (not issue_boundary_ids or sample.boundary_id in issue_boundary_ids)
        ]
        referenced_scene_ids = [
            sample.scene_id for sample in samples if sample.scene_id is not None
        ]
        scene_id = (
            min(boundary_start_scene_ids or referenced_scene_ids)
            if (boundary_start_scene_ids or referenced_scene_ids)
            else rendered[0][0]
        )
        state = ctx.scene_states[scene_id]
        with self._state_lock:
            exhausted = state.visual_fix_attempts >= profile.max_fix_attempts
            if not exhausted:
                state.visual_fix_attempts += 1
                attempt = state.visual_fix_attempts
            else:
                attempt = state.visual_fix_attempts
        if exhausted:
            self._emit("boundary_visual_warning", score=result.overall_score, target=target)
            return False
        feedback = result.feedback()
        if target in {"planner", "continuity"}:
            self._emit(
                "scene_visual_plan_fixing",
                scene_id=scene_id,
                target=target,
                attempt=attempt,
                max_attempts=profile.max_fix_attempts,
            )
            self._schedule_visual_plan_repair(ctx, scene_id, state, feedback, target)
            return True
        with self._state_lock:
            state.rewrite_feedback = (
                "## Boundary Visual Evaluation Feedback\n"
                f"{feedback}\n\n"
                "请只修复相邻场景边界的可见问题，不改变数学合同和继承元素身份。"
            )
            state.reviewed = False
            state.rendered = False
            state.artifact = None
            state.slurm_job = None
            ctx.final_video = None
            ctx.final_video_sha256 = ""
        self._emit(
            "scene_visual_fixing",
            scene_id=scene_id,
            attempt=attempt,
            max_attempts=profile.max_fix_attempts,
        )
        self._request_continuity_rebuild(
            ctx,
            scene_id,
            reason="边界视觉评估要求修改场景代码",
            preserve_visual_candidates=True,
            include_failed=True,
        )
        return True

    def _final_visual_report(self, ctx: PipelineContext) -> None:
        """对合并成片生成报告；不据此执行无法可靠归因的代码重写。"""

        profile = ctx.visual_eval_profile
        if not profile.enabled or ctx.final_video is None:
            return
        self._emit("stage_start", stage="visual_evaluating", scope="final")
        from kd1_anime.eval import Evaluator

        report_path = ctx.paths.root / "eval_reports" / "final_visual.json"
        samples = []
        try:
            evaluator = Evaluator(
                enable_visual_eval=True,
                visual_eval_model=profile.model,
                output_dir=ctx.paths.root / "eval_reports",
            )
            if evaluator.visual_evaluator is None:  # 仅用于静态收窄；正常配置不可能发生
                raise RuntimeError("视觉评估器未初始化")
            frame_dir = ctx.paths.root / "eval_frames" / "final" / ctx.final_video_sha256[:12]
            rendered = [
                (scene_id, self._artifact_video_path(ctx, state.artifact))
                for scene_id, state in sorted(ctx.scene_states.items())
                if state.rendered and state.artifact is not None
            ]
            boundary_count = min(3, profile.frame_count // 2)
            regular_count = profile.frame_count - boundary_count * 2
            regular_samples = (
                evaluator.extract_video_samples(
                    ctx.final_video,
                    frame_dir,
                    frame_count=regular_count,
                )
                if regular_count > 0
                else []
            )
            boundary_samples = (
                evaluator.extract_boundary_samples(
                    rendered,
                    ctx.paths.root / "eval_frames" / "boundaries" / ctx.final_video_sha256[:12],
                    max_boundaries=boundary_count,
                )
                if boundary_count > 0 and len(rendered) >= 2
                else []
            )
            samples = [
                sample.model_copy(update={"frame_id": f"F{index:02d}"})
                for index, sample in enumerate([*regular_samples, *boundary_samples], start=1)
            ]
            if not samples:
                raise RuntimeError("没有可用于成片视觉评估的关键帧")
            with self._visual_llm_slot():
                result = evaluator.visual_evaluator.evaluate_video_frames(
                    samples,
                    ctx.original_prompt or ctx.user_prompt,
                    scene_context=json.dumps(
                        {
                            "lesson_spec": ctx.lesson_spec.model_dump(mode="json"),
                            "teaching_graph": ctx.teaching_graph.model_dump(mode="json"),
                            "scene_plans": [
                                state.plan.model_dump(mode="json")
                                for state in ctx.scene_states.values()
                            ],
                        },
                        ensure_ascii=False,
                    )[:30_000],
                    scope="complete video",
                )
            report = {
                "schema_version": 1,
                "scope": "complete_video",
                "model": profile.model,
                "visual_profile_sha256": profile.digest(),
                "video_sha256": ctx.final_video_sha256,
                "state_ledger_digest": ctx.state_ledger.digest(),
                "frames": [
                    {
                        "frame_id": sample.frame_id,
                        "scene_id": sample.scene_id,
                        "boundary_id": sample.boundary_id,
                        "timestamp_seconds": sample.timestamp_seconds,
                        "role": sample.role,
                        "path": sample.path.resolve()
                        .relative_to(ctx.paths.root.resolve())
                        .as_posix(),
                        "sha256": sample.image_sha256,
                    }
                    for sample in samples
                ],
                "result": result.to_dict(),
                "error": "",
            }
            self._emit("final_visual_complete", score=result.overall_score)
        except Exception as exc:
            report = {
                "schema_version": 1,
                "scope": "complete_video",
                "model": profile.model,
                "visual_profile_sha256": profile.digest(),
                "video_sha256": ctx.final_video_sha256,
                "state_ledger_digest": ctx.state_ledger.digest(),
                "frames": [
                    {
                        "frame_id": sample.frame_id,
                        "scene_id": sample.scene_id,
                        "boundary_id": sample.boundary_id,
                        "timestamp_seconds": sample.timestamp_seconds,
                        "role": sample.role,
                        "path": sample.path.resolve()
                        .relative_to(ctx.paths.root.resolve())
                        .as_posix(),
                        "sha256": sample.image_sha256,
                    }
                    for sample in samples
                ],
                "result": None,
                "error": self._redact_visual_error(exc),
            }
            self._emit("final_visual_unknown", reason=report["error"])
        atomic_write_json(report_path, report)

    def _merge(self, ctx: PipelineContext) -> None:
        """按代码、配置和视频哈希验证每个产物后再合并。"""
        self._emit("stage_start", stage="merging")
        rendered_artifacts: list[SceneArtifact] = []
        for scene_id, state in sorted(ctx.scene_states.items()):
            if not state.rendered:
                continue
            artifact = state.artifact
            if artifact is None:
                raise RuntimeError(f"Scene {scene_id} 标记为已渲染，但缺少产物凭据")
            if artifact.scene_id != scene_id:
                raise RuntimeError(f"Scene {scene_id} 的产物场景 ID 不一致")
            if artifact.scene_class_name != state.class_name:
                raise RuntimeError(f"Scene {scene_id} 的产物类名与当前代码不一致")
            if artifact.code_sha256 != sha256_text(state.code):
                raise RuntimeError(f"Scene {scene_id} 的产物代码哈希与当前代码不一致")
            if artifact.render_profile_sha256 != ctx.render_profile.digest():
                raise RuntimeError(f"Scene {scene_id} 的产物渲染配置与当前运行不一致")
            if ctx.visual_eval_profile.enabled:
                accepted = state.visual_status in {"passed", "warning", "unknown"}
                if not accepted:
                    raise RuntimeError(
                        f"Scene {scene_id} 尚未完成视觉评估（状态: {state.visual_status}）"
                    )
                if state.visual_artifact_sha256 != artifact.video_sha256:
                    raise RuntimeError(f"Scene {scene_id} 的视觉评估记录不属于当前视频")
            rendered_artifacts.append(artifact)
        incomplete = [sid for sid, state in ctx.scene_states.items() if not state.rendered]
        if not rendered_artifacts:
            raise RuntimeError("流水线未能完成。\n没有场景成功渲染")
        if incomplete and not settings.ALLOW_PARTIAL_OUTPUT:
            for sid in incomplete:
                state = ctx.scene_states[sid]
                if not state.failure_reason:
                    state.failure_reason = "场景未成功渲染"
            reasons = "\n".join(
                f"Scene {sid}: {ctx.scene_states[sid].failure_reason or '未完成'}"
                for sid in incomplete
            )
            self._emit("partial_output_blocked", incomplete=incomplete)
            raise RuntimeError("流水线未能完成。\n" + reasons)

        self._checkpoint(ctx, State.MERGING)
        resolved_output = ctx.paths.output.expanduser().resolve()
        output_is_run_local = resolved_output == ctx.paths.root.resolve() or (
            ctx.paths.root.resolve() in resolved_output.parents
        )
        # 评估-改进循环的第二次合并必须强制重新拼接, 不能复用上一轮的旧视频
        force_remerge = getattr(ctx, "eval_round", 0) > 0
        can_reuse_checkpointed_output = (
            ctx.final_video is not None
            and ctx.final_video.expanduser().resolve() == resolved_output
        )
        checkpointed_output_matches = (
            can_reuse_checkpointed_output
            and bool(ctx.final_video_sha256)
            and resolved_output.is_file()
            and sha256_file(resolved_output) == ctx.final_video_sha256
        )
        if (
            output_is_run_local
            and resolved_output.is_file()
            and resolved_output.stat().st_size > 0
            and not settings.OVERWRITE_OUTPUT
            and not force_remerge
            and checkpointed_output_matches
        ):
            ctx.final_video = resolved_output
        else:
            # run 内输出属于当前私有目录，即使旧检查点没有输入收据，也可以让
            # VideoMerger 写临时文件并在验证成功后原子替换；不能提前删除旧文件，
            # 否则新的 FFmpeg 失败会丢失上一次仍可用的结果。
            video_paths = [self._artifact_video_path(ctx, item) for item in rendered_artifacts]
            merge_kwargs = {
                "replace_existing": (
                    output_is_run_local
                    or (force_remerge and not output_is_run_local and checkpointed_output_matches)
                ),
                "render_profile": ctx.render_profile,
            }
            if self._supports_keyword(self.merger.merge, "merge_profile"):
                merge_kwargs["merge_profile"] = ctx.merge_profile
            ctx.final_video = self.merger.merge(video_paths, ctx.paths.output, **merge_kwargs)
        ctx.final_video_sha256 = sha256_file(ctx.final_video)
        minimum = ctx.lesson_spec.requested_duration_min_seconds
        maximum = ctx.lesson_spec.requested_duration_max_seconds
        if minimum is not None or maximum is not None:
            final_metadata = probe_video(ctx.final_video)
            if (
                ctx.expected_final_duration is not None
                and abs(final_metadata.duration_seconds - ctx.expected_final_duration) > 0.35
            ):
                ctx.continuity_warnings.append(
                    "最终视频时长与计划预估存在差异: "
                    f"{final_metadata.duration_seconds:.2f}s != "
                    f"{ctx.expected_final_duration:.2f}s；以 ffprobe 实际时长为准"
                )
            if (
                minimum is not None
                and final_metadata.duration_seconds < minimum - 0.35
                and not incomplete
            ):
                raise RuntimeError("最终视频时长低于 LessonSpec 要求的下限")
            if maximum is not None and final_metadata.duration_seconds > maximum + 0.35:
                raise RuntimeError("最终视频时长超过 LessonSpec 要求的上限")
        self._checkpoint(ctx, State.MERGING)
        self._emit(
            "merge_complete",
            path=str(ctx.final_video),
            size_mb=ctx.final_video.stat().st_size / (1024 * 1024),
            partial=bool(incomplete),
            incomplete=incomplete,
        )

    def _eval(self, ctx: PipelineContext) -> bool:
        """评估最终视频; 返回 True 表示触发改进 (需要重新调度低分场景)。"""
        if not settings.ENABLE_AUTO_EVAL:
            return False
        self._emit("stage_start", stage="evaluating")
        self._checkpoint(ctx, State.EVALUATING)
        eval_round = getattr(ctx, "eval_round", 0)
        if eval_round >= settings.MAX_EVAL_ROUNDS:
            self._emit(
                "eval_max_rounds_reached",
                rounds=eval_round,
                max_rounds=settings.MAX_EVAL_ROUNDS,
            )
            return False
        from kd1_anime.eval import Evaluator

        if not (ctx.final_video and ctx.final_video.exists()):
            self._emit("eval_skipped", reason="no_final_video")
            return False
        try:
            evaluator = Evaluator(
                # 场景级闭环与成片视觉报告由 _visual_gate / _final_visual_report
                # 使用独立多模态端点完成；这里仅保留确定性代码与效率评估，
                # 避免重复计费以及把视觉端点故障误当成改码依据。
                enable_visual_eval=False,
                output_dir=ctx.paths.root / "eval_reports",
            )
            evaluate_code = evaluator.evaluate_code
            if self._supports_keyword(evaluate_code, "renderer"):
                scene_eval_results = {
                    scene_id: evaluate_code(
                        state.code,
                        renderer=ctx.render_profile.renderer,
                    )
                    for scene_id, state in ctx.scene_states.items()
                    if state.code
                }
            else:
                # 兼容旧的评估器替身/外部集成；正式 Evaluator 会按当前
                # run 的 RenderProfile 验证 OpenGL/Cairo 规则。
                scene_eval_results = {
                    scene_id: evaluate_code(state.code)
                    for scene_id, state in ctx.scene_states.items()
                    if state.code
                }
            eval_result = evaluator.evaluate_run(
                ctx.paths.run_id,
                ctx.paths.root,
                description=ctx.original_prompt or ctx.user_prompt,
                enable_visual=False,
            )
            eval_result.save(ctx.paths.root / "eval_result.json")
        except Exception as exc:
            # 可选评估在修改任何流水线状态前失败时，只记录并跳过。
            self._emit("eval_error", error=str(exc))
            return False

        overall_score = eval_result.overall_score
        code_values = [
            score.score for score in eval_result.scores if score.metric.value.startswith("code_")
        ]
        visual_values = [
            state.visual_score
            for state in ctx.scene_states.values()
            if state.visual_score is not None and state.visual_status in {"passed", "warning"}
        ]
        self._emit(
            "eval_complete",
            overall_score=overall_score,
            code_score=(sum(code_values) / len(code_values) if code_values else None),
            visual_score=(sum(visual_values) / len(visual_values) if visual_values else None),
            errors=eval_result.errors,
            threshold=settings.EVAL_THRESHOLD,
        )
        if overall_score is None:
            self._emit("eval_skipped", reason="no_valid_metrics", errors=eval_result.errors)
            return False
        if overall_score >= settings.EVAL_THRESHOLD:
            self._emit(
                "eval_passed",
                score=overall_score,
                threshold=settings.EVAL_THRESHOLD,
            )
            return False

        low_score_scenes = [
            scene_id
            for scene_id, scene_eval in scene_eval_results.items()
            if scene_eval.overall_score is not None
            and scene_eval.overall_score < settings.EVAL_THRESHOLD
        ]
        if not low_score_scenes:
            self._emit(
                "eval_improvement_skipped",
                reason="low_score_not_attributable_to_scene_code",
            )
            return False

        self._emit(
            "eval_below_threshold",
            score=overall_score,
            threshold=settings.EVAL_THRESHOLD,
            action="triggering_improvement",
        )
        with self._state_lock:
            ctx.eval_round = eval_round + 1
            ctx.scenes_to_improve = low_score_scenes
            # 当前代码交接是按场景顺序建立的：Scene N 的导出定义会成为
            # Scene N+1 的输入。因此重新生成任一场景时，不能只清空该场景，
            # 否则后续场景会继续携带旧上下文和旧视频。保守地从最早低分
            # 场景开始失效所有后继场景，保留它们的导演分镜以减少 LLM 成本。
            first_changed_scene = min(low_score_scenes)
            invalidated_ids = [
                scene_id for scene_id in sorted(ctx.scene_states) if scene_id >= first_changed_scene
            ]
            for scene_id in invalidated_ids:
                state = ctx.scene_states[scene_id]
                state.rendered = False
                state.reviewed = False
                state.failed = False
                state.give_up = False
                state.code = ""
                state.class_name = ""
                state.slurm_job = None
                state.artifact = None
                state.fix_attempts = 0
                state.infra_retries = 0
                state.rewrite_feedback = ""
                state.review_signature = ""
                state.identical_review_count = 0
                state.failure_reason = ""
                state.failure_category = ""
                state.last_error_fp = ""
                state.identical_error_count = 0
                state.safe_fallback_used = False
                state.safe_fallback_reason = ""
                state.inherited_elements_code = ""
                state.exported_elements_code = ""
                state.exported_elements = []
                state.local_smoke_status = "pending"
                # 代码和交接上下文同时失效；否则后续场景会从旧的
                # ElementManifest 读取已经被评估淘汰的定义，TechnicalSpec
                # 也可能在下一轮被误认为仍然与输入匹配。
                self._reset_technical_spec(state)
                self._remove_element_manifest_scene(ctx, scene_id)
                state.visual_status = "pending" if ctx.visual_eval_profile.enabled else "skipped"
                state.visual_fix_attempts = 0
                state.visual_score = None
                state.visual_report_file = ""
                state.visual_report_sha256 = ""
                state.visual_artifact_sha256 = ""
                state.visual_feedback = ""
                state.visual_best_candidate = None
                code_path = ctx.paths.scenes / f"scene_{scene_id}.py"
                self._write_private(code_path, "")
            ctx.scenes_to_render.clear()
            ctx.scenes_to_reuse.clear()
            ctx.final_video = None
            ctx.final_video_sha256 = ""
            # 轮次和场景重置必须先持久化；检查点失败应终止流水线，不能被
            # 当作可忽略的视觉评估错误，否则 resume 会复用上一轮旧视频。
            self._checkpoint(ctx, State.EVALUATING)
        return True
