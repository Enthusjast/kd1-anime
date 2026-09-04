"""有限状态机：串联规划、代码生成、审查、Slurm 渲染、修复与拼接。"""

from __future__ import annotations

import inspect
import json
import os
import re
import shutil
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

from kd1_anime.agents.auto_fixer import AutoFixerAgent
from kd1_anime.agents.coder import CoderAgent
from kd1_anime.agents.continuity import (
    ContinuityIssue,
    ContinuityReviewerAgent,
    apply_deterministic_continuity_repairs,
    deterministic_continuity_issues,
    extract_continuity_elements,
    validate_export_contract,
)
from kd1_anime.agents.planner import (
    ContinuityBible,
    ExtractedElement,
    PlannerAgent,
    SceneOutline,
    ScenePlan,
    VisualElementState,
)
from kd1_anime.agents.reviewer import ReviewerAgent, ReviewResult
from kd1_anime.agents.validator import CodeValidationResult, validate_manim_code
from kd1_anime.cluster.slurm import (
    FAILURE_STATES,
    JobMonitor,
    SlurmDispatcher,
    SlurmJob,
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
from kd1_anime.rendering import RenderProfile, SceneArtifact, sha256_file
from kd1_anime.resources import ResourceCoordinator
from kd1_anime.run_store import (
    MANIFEST_NAME,
    RunManifest,
    RunRepository,
    StoredSceneState,
    StoredVisualCandidate,
    VisualEvalProfile,
    atomic_write_json,
    get_reusable_video_path,
    lock_run,
    restore_run_path,
    restore_slurm_job,
    sha256_text,
    store_slurm_job,
    write_manifest,
)

logger = get_logger(__name__)
console = Console()
Callback = Callable[[str, dict], None]
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
    # 导演分镜是否已完成(区别于占位 plan)。场景独立推进时据此判断下一步。
    plan_ready: bool = False
    # Reviewer major 反馈待重写时暂存; 调度器据此把场景重新排入编码阶段。
    rewrite_feedback: str = ""
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
    outlines: list[SceneOutline] = field(default_factory=list)
    scenes: list[ScenePlan] = field(default_factory=list)
    scene_states: dict[int, SceneState] = field(default_factory=dict)
    continuity_bible: ContinuityBible | None = None
    continuity_review_status: str = "passed"
    continuity_review_round: int = 0
    continuity_warnings: list[str] = field(default_factory=list)
    final_video: Path | None = None
    final_video_sha256: str = ""
    render_profile: RenderProfile = field(default_factory=RenderProfile.current)
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
        self._emitted_phases: set[str] = set()
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

    def _emit(self, event: str, **data) -> None:
        if self._callback:
            self._callback(event, data)

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
        return RagRuntimeProfile(
            enabled=bool(runtime["enabled"]),
            status=runtime["status"],
            index_sha256=(runtime.get("index") or {}).get("index_sha256", ""),
            embedding_model=str(runtime.get("embedding_model", "")),
            reranker_model=str(runtime.get("reranker_model", "")),
            top_k=self.rag.config.RAG_TOP_K,
            rerank_top_n=self.rag.config.RAG_RERANK_TOP_N,
            max_context_chars=self.rag.config.RAG_MAX_CONTEXT_CHARS,
        )

    def _retrieve_rag(
        self,
        ctx: PipelineContext,
        query: str,
        *,
        receipt_key: str,
        stage: str,
        source_kinds: set[str] | None = None,
        code_sha256: str = "",
        inherited_elements_sha256: str = "",
    ) -> str:
        """获取 RAG 上下文并保存不含密钥的检索收据。"""

        try:
            result = self.rag.search(
                query[:50_000],
                stage=stage,
                source_kinds=source_kinds,
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
                warning=f"RAG 检索异常，已跳过: {exc}",
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
    def _mark_failed(state: SceneState, reason: str) -> None:
        state.failed = True
        state.failure_reason = reason

    @staticmethod
    def _validate(code: str, *, renderer: str | None = None) -> CodeValidationResult:
        return validate_manim_code(code, renderer=renderer)

    @staticmethod
    def _write_private(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.parent.chmod(0o700)
        path.write_text(content, encoding="utf-8")
        path.chmod(0o600)

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
                    plan_ready=scene.plan_ready,
                    rewrite_feedback=scene.rewrite_feedback,
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
                output_path=str(ctx.paths.output),
                render_profile=ctx.render_profile,
                outlines=ctx.outlines,
                scenes=scenes,
                continuity_bible=ctx.continuity_bible,
                continuity_review_status=ctx.continuity_review_status,
                continuity_review_round=ctx.continuity_review_round,
                continuity_warnings=ctx.continuity_warnings[-100:],
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

    def _record_checkpoint_failure(self, error: Exception) -> None:
        """记录持久化故障并停止其它 worker，避免继续推进未保存的状态。"""

        with self._state_lock:
            if self._checkpoint_error is None:
                self._checkpoint_error = error
        self._stop_event.set()
        self._emit("checkpoint_failed", error=str(error))

    @staticmethod
    def _scene_phase(scene: SceneState) -> str:
        if scene.visual_status == "evaluating":
            return "visual_evaluating"
        if scene.rendered and scene.visual_status in {"passed", "warning", "unknown"}:
            return "visual_accepted"
        if scene.rendered and scene.artifact:
            return "rendered"
        if scene.failed or scene.give_up:
            return "failed"
        if scene.slurm_job:
            return "monitoring"
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
                plan_ready=getattr(stored, "plan_ready", False),
                rewrite_feedback=getattr(stored, "rewrite_feedback", ""),
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
        context = PipelineContext(
            user_prompt=manifest.user_prompt,
            original_prompt=manifest.user_prompt,
            paths=paths,
            dry_run=manifest.dry_run,
            interactive=manifest.interactive,
            auto_fix=manifest.auto_fix,
            outlines=manifest.outlines,
            scenes=[scene_states[key].plan for key in sorted(scene_states)],
            scene_states=scene_states,
            continuity_bible=manifest.continuity_bible,
            continuity_review_status=manifest.continuity_review_status,
            continuity_review_round=manifest.continuity_review_round,
            continuity_warnings=list(manifest.continuity_warnings),
            final_video=final_video,
            final_video_sha256=manifest.final_video_sha256,
            incremental=manifest.incremental,
            base_run_id=manifest.base_run_id,
            render_profile=manifest.render_profile,
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
        rag_context: str = "",
    ) -> tuple[str, str]:
        agent = CoderAgent()
        current_feedback = feedback
        current_previous = previous_code
        last_validation: CodeValidationResult | None = None
        for _ in range(settings.CODE_VALIDATION_ATTEMPTS):
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
            if rag_context and self._supports_keyword(agent.generate_code, "rag_context"):
                code_kwargs["rag_context"] = rag_context
            code = agent.generate_code(
                plan,
                **code_kwargs,
            )

            validation = self._validate(code, renderer=renderer)
            if validation.is_valid:
                return code, validation.scene_classes[0]
            last_validation = validation
            # 提供详细的修复指导
            feedback_parts = [f"确定性校验未通过，必须修复以下问题：\n{validation.feedback}"]

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
            + (last_validation.feedback if last_validation else "未知错误"),
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
    ) -> Path | None:
        """增量渲染：只重新渲染受 prompt 变化影响的场景。"""
        # CLI 会在进入流水线前做真实网络探测；库调用方至少也必须通过
        # 同一份配置完整性检查，避免运行几分钟后才因空 Key 失败。
        settings.require_llm_key()
        if settings.ENABLE_VISUAL_EVAL and not dry_run:
            settings.require_visual_llm()
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
    ) -> Path | None:
        # 保持 programmatic API 与 CLI 的配置门槛一致。网络可用性由 CLI
        # 的启动探测负责，底层 Agent 仍会在真正调用时给出详细错误。
        settings.require_llm_key()
        if settings.ENABLE_VISUAL_EVAL and not dry_run:
            settings.require_visual_llm()
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
            self._callback = callback
            self._cancel_requested.clear()
            self._stop_event.clear()
            self._manifest = manifest
            ctx = self._context_from_manifest(manifest, root)
            ctx.interactive = interactive
            self._ctx = ctx
            self._reconcile_rag_context(ctx)

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

            if incomplete_dry_run:
                # 旧版本会把含失败场景的 dry-run 误标为完成，导致 resume
                # 看到 DONE 后直接返回。显式恢复这类清单时，从代码审查屏障
                # 重新开始；已完成的场景仍会被顺序屏障安全跳过。
                state = State.CODING if ctx.scene_states else State.PLANNING

            # resume 明确表示用户愿意再次尝试；先重置放弃标记，再处理 ERROR。
            # 旧逻辑先判断 ERROR，导致“所有场景都已放弃”时无法进入这里的
            # 重试分支，仪表盘提示可恢复但实际直接报“无可用场景”。
            reset_give_up = False
            reset_failed = False
            for scene in ctx.scene_states.values():
                if scene.give_up:
                    scene.give_up = False
                    scene.review_round = 0
                    scene.fix_attempts = 0
                    scene.failure_reason = ""
                    reset_give_up = True
                # 失败清单也允许显式 resume。旧实现只在 ERROR 快照中清除
                # failed，若最后一次检查点是 MONITORING，调度器会永久跳过
                # 这些场景，表现为“恢复成功但没有重新开始”。
                if scene.failed and not scene.rendered:
                    scene.failed = False
                    scene.failure_reason = ""
                    reset_failed = True

            if state is State.ERROR:
                # 允许从 ERROR 状态恢复：无场景时重跑概要规划；已有场景
                # 则让场景级 worker 根据 plan_ready/code/reviewed 自己选择
                # 分镜、编码或审查，不因某个失败快照而永久卡死。
                if not ctx.scene_states:
                    state = State.PLANNING
                else:
                    for scene in ctx.scene_states.values():
                        if scene.failed:
                            scene.failed = False
                            scene.failure_reason = ""
                    state = State.CODING
                self._emit("run_resuming_from_error", run_id=run_id, state=state.name)

            if reset_give_up:
                # resume 时仪表盘可能已激活, 避免直接打印破坏 Live 渲染
                with suppress(Exception):
                    from kd1_anime.dashboard import quiet

                    if not quiet():
                        console.print("[yellow]发现已放弃的场景，将重置并重试[/]")
                if state not in {State.CODING, State.REVIEWING, State.DISPATCHING}:
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

            # 核对恢复的 Slurm 作业: 上次会话提交的作业可能早已结束/已不存在,
            # 直接监控会得到连续 UNKNOWN → CANCEL_FAILED → 场景永久判死。
            self._reconcile_restored_jobs(ctx)

            # 上一次运行可能在连续性重规划达到上限后被中断。warning 不是
            # 已确认通过；显式 resume 应开启一轮新的有限修正，让新版的
            # 字段级反馈和确定性合同修复有机会处理旧分镜，而不是直接带着
            # 已知冲突进入编码阶段。
            if (
                ctx.continuity_bible is not None
                and ctx.continuity_review_status == "warning"
                and any(
                    not scene.rendered and not scene.failed and not scene.give_up
                    for scene in ctx.scene_states.values()
                )
            ):
                ctx.continuity_review_status = "pending"
                ctx.continuity_review_round = 0
                ctx.continuity_warnings.append("恢复运行：重新开启连续性审查与有限修正")

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
                # 在任何并行 Detail worker 启动前保存概要和 continuity bible；
                # 进程此时中断时 resume 不会丢失全片规范或重新生成一份不同的规范。
                self._checkpoint(ctx, State.DETAILING)
                self._emit(
                    "plan_complete",
                    scenes=ctx.scenes,
                    visual_enabled=ctx.visual_eval_profile.enabled,
                )
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
                if ctx.dry_run:
                    break
                if ctx.continuity_rebuild_required:
                    # 上游 AutoFix 已经停止本轮渲染并清空下游；下一轮从
                    # 顺序编码屏障重新建立继承代码，再恢复并行渲染。
                    ctx.continuity_rebuild_required = False
                    self._stop_event.clear()
                    self._checkpoint(ctx, State.CODING)
                    continue
                if self._visual_gate(ctx):
                    if ctx.continuity_rebuild_required:
                        ctx.continuity_rebuild_required = False
                        self._stop_event.clear()
                    self._checkpoint(ctx, State.CODING)
                    continue
                self._merge(ctx)
                self._final_visual_report(ctx)
                improve = self._eval(ctx)

            # ---- 收尾 ----
            if ctx.dry_run:
                unfinished = [
                    (scene_id, scene)
                    for scene_id, scene in sorted(ctx.scene_states.items())
                    if not scene.reviewed or scene.failed or scene.give_up
                ]
                if unfinished:
                    details = "; ".join(
                        f"Scene {scene_id}: {scene.failure_reason or '未通过编码/审查'}"
                        for scene_id, scene in unfinished
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
            return State[self._manifest.state]
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
                    source_kinds={"manim_doc", "example"},
                )
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
        if len(outlines) > settings.MAX_SCENES:
            raise RuntimeError(
                f"Planner 生成了 {len(outlines)} 个场景，超过 MAX_SCENES={settings.MAX_SCENES}"
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
            return
        try:
            rag_context = self._retrieve_rag(
                ctx,
                ctx.user_prompt
                + "\n"
                + "\n".join(f"{item.title}: {item.math_concept}" for item in ctx.outlines),
                receipt_key="continuity",
                stage="continuity",
                source_kinds={"manim_doc", "example"},
            )
            bible_kwargs: dict[str, object] = {
                "stream": False,
                "renderer": ctx.render_profile.renderer,
            }
            if self._supports_keyword(planner_method, "rag_context"):
                bible_kwargs["rag_context"] = rag_context
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
            return
        ctx.continuity_review_status = "pending"
        ctx.continuity_review_round = 0
        self._emit("continuity_bible_ready")

    def _llm_slot(self):
        """在调度器已初始化信号量时复用它，否则提供无锁上下文。"""

        from contextlib import nullcontext

        return self._llm_sem if hasattr(self, "_llm_sem") else nullcontext()

    def _visual_llm_slot(self):
        """批处理时复用进程级视觉模型配额。"""

        from contextlib import nullcontext

        if self._resource_coordinator is not None:
            return self._resource_coordinator.visual_llm
        return nullcontext()

    # ------------------------------------------------------------------
    # 场景级并行调度 (per-scene pipeline)
    # 每个 Scene 一个工作线程, 独立推进 分镜→编码→审查→提交→渲染→修复。
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

    def _run_scheduler(self, ctx: PipelineContext) -> None:
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

        # 正式运行先完成所有场景的 Detail，再做一次全片连续性审查；通过后才
        # 进入编码/审查/渲染。编码/审查必须顺序执行，因为 Scene N 的真实
        # 最终 Mobject 定义要作为 Scene N+1 的输入；渲染和监控仍然并行。
        self._run_detail_barrier(ctx)
        if self._checkpoint_error is not None:
            raise RuntimeError(
                f"运行状态持久化失败，流水线已停止: {self._checkpoint_error}"
            ) from self._checkpoint_error
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

        self._run_code_review_barrier(ctx)
        if self._checkpoint_error is not None:
            raise RuntimeError(
                f"运行状态持久化失败，流水线已停止: {self._checkpoint_error}"
            ) from self._checkpoint_error

        threads: list[threading.Thread] = []
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
        if self._checkpoint_error is not None:
            raise RuntimeError(
                f"运行状态持久化失败，流水线已停止: {self._checkpoint_error}"
            ) from self._checkpoint_error

    def _run_code_review_barrier(self, ctx: PipelineContext) -> None:
        """按 Scene ID 顺序完成编码、审查，并固定代码级连续性上下文。"""

        for scene_id, state in sorted(ctx.scene_states.items()):
            if self._stop_event.is_set() or state.failed or state.give_up:
                continue
            try:
                previous_context = state.inherited_elements_code
                self._prepare_inherited_context(ctx, scene_id, state)
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
                    state.artifact = None
                    state.rendered = False
                    state.slurm_job = None
                    state.exported_elements_code = ""
                    state.exported_elements = []
                    self._reset_visual_receipt(
                        ctx,
                        state,
                        clear_candidate=True,
                        reset_attempts=True,
                    )
                    self._write_private(ctx.paths.scenes / f"scene_{scene_id}.py", "")
                reusable_visual = state.visual_best_candidate
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
                    self._checkpoint(ctx, State.REVIEWING)
                    continue
                while not state.reviewed:
                    if self._stop_event.is_set() or state.failed or state.give_up:
                        break
                    if not state.code or state.rewrite_feedback:
                        self._phase_emit("coding")
                        self._scene_code(ctx, scene_id, state)
                    self._phase_emit("reviewing")
                    self._scene_review(ctx, scene_id, state)
                if state.reviewed and state.code and not state.exported_elements_code:
                    self._refresh_scene_export(state)
                self._checkpoint(ctx, State.REVIEWING)
            except Exception as exc:
                with self._state_lock:
                    self._mark_failed(state, f"Scene {scene_id} 编码/审查失败: {exc}")
                    try:
                        self._checkpoint(ctx, State.REVIEWING)
                    except Exception as checkpoint_error:
                        self._record_checkpoint_failure(checkpoint_error)
                self._emit("scene_failed", scene_id=scene_id, reason=state.failure_reason)

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
                self._mark_failed(state, f"Scene {scene_id} 分镜生成失败: {exc}")
            try:
                self._checkpoint(ctx, State.DETAILING)
            except Exception as checkpoint_error:
                self._record_checkpoint_failure(checkpoint_error)
            self._emit("scene_failed", scene_id=scene_id, reason=str(exc))

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
        """构造当前场景及相邻场景的最新交接快照。"""

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
                    "plan": state.plan.model_dump(mode="json"),
                }
            )
        return json.dumps(snapshot, ensure_ascii=False, indent=2)

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
            deterministic = deterministic_continuity_issues(plans, ctx.continuity_bible)
            try:
                with self._llm_sem:
                    result = ContinuityReviewerAgent().review(
                        ctx.continuity_bible,
                        ctx.outlines,
                        plans,
                        deterministic_issues=deterministic,
                        renderer=ctx.render_profile.renderer,
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
                with self._llm_sem:
                    revised_plan = planner.plan_detail(
                        outline, ctx.outlines, ctx.user_prompt, **replan_kwargs
                    )
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
                with self._state_lock:
                    state.plan = revised_plan
                    state.plan_ready = True
                    ctx.scenes = [
                        item.plan
                        for item in sorted(
                            ctx.scene_states.values(), key=lambda item: item.plan.scene_id
                        )
                    ]
                    self._checkpoint(ctx, State.DETAILING)
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
                    self._phase_emit("reviewing")
                    self._scene_review(ctx, scene_id, state)
        except Exception as exc:
            with self._state_lock:
                self._mark_failed(state, f"Scene {scene_id} 流水线异常: {exc}")
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
            source_kinds={"manim_doc", "example"},
        )
        with self._llm_sem:
            outline = next(o for o in ctx.outlines if o.scene_id == scene_id)
            planner = PlannerAgent()
            detail_kwargs = {
                "stream": False,
                "renderer": ctx.render_profile.renderer,
            }
            if ctx.continuity_bible is not None and callable(
                getattr(planner, "plan_continuity_bible", None)
            ):
                detail_kwargs["continuity_bible"] = ctx.continuity_bible
            if self._supports_keyword(planner.plan_detail, "rag_context"):
                detail_kwargs["rag_context"] = rag_context
            plan = planner.plan_detail(outline, ctx.outlines, ctx.user_prompt, **detail_kwargs)
        with self._state_lock:
            state.plan = plan
            state.plan_ready = True
            self._checkpoint(ctx, State.DETAILING)
        self._emit("scene_detailed", scene_id=scene_id, title=plan.title)

    def _prepare_inherited_context(
        self, ctx: PipelineContext, scene_id: int, state: SceneState
    ) -> None:
        """为当前场景固定上一场景的代码级交接，恢复旧运行时按需提取。"""

        if scene_id <= 1:
            state.inherited_elements_code = ""
            return
        previous = ctx.scene_states.get(scene_id - 1)
        if previous is None:
            state.inherited_elements_code = ""
            ctx.continuity_warnings.append(
                f"Scene {scene_id} 缺少上一场景状态，编码将不带代码级继承上下文。"
            )
            return
        current_code_hash = sha256_text(previous.code) if previous.code else ""
        export_hashes_match = bool(previous.exported_elements) and all(
            item.source_code_sha256 in {"", current_code_hash}
            for item in previous.exported_elements
        )
        if previous.code and (not previous.exported_elements_code or not export_hashes_match):
            exported_code, exported_elements = extract_continuity_elements(previous.code)
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
        state.inherited_elements_code = previous.exported_elements_code

    @staticmethod
    def _refresh_scene_export(state: SceneState) -> None:
        """从审查通过的最终代码提取下一场景可复用的纯定义。"""

        exported_code, exported_elements = extract_continuity_elements(state.code)
        validate_export_contract(state.plan, exported_elements)
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
            source_kinds={"manim_doc", "example"},
            code_sha256=sha256_text(state.code) if state.code else "",
            inherited_elements_sha256=sha256_text(state.inherited_elements_code)
            if state.inherited_elements_code
            else "",
        )
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
                rag_context=rag_context,
            )
        path = ctx.paths.scenes / f"scene_{scene_id}.py"
        self._write_private(path, code)
        with self._state_lock:
            state.code = code
            state.class_name = class_name
            state.rewrite_feedback = ""
            state.reviewed = False
            state.infra_retries = 0
            state.artifact = None
            state.rendered = False
            state.exported_elements_code = ""
            state.exported_elements = []
            self._reset_visual_receipt(ctx, state)
            self._checkpoint(ctx, State.CODING)
        self._emit("scene_coded", scene_id=scene_id, file_path=str(path))

    def _scene_review(self, ctx: PipelineContext, scene_id: int, state: SceneState) -> None:
        if settings.SKIP_REVIEW:
            try:
                self._refresh_scene_export(state)
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
            _, exported_elements = extract_continuity_elements(state.code)
            validate_export_contract(state.plan, exported_elements)
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
                result = reviewer.review(state.code, state.plan, **review_kwargs)
        if result.is_valid:
            try:
                self._refresh_scene_export(state)
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
                self._mark_failed(state, f"提交前无法读取场景代码: {exc}")
                self._checkpoint(ctx, State.DISPATCHING)
            self._emit("scene_failed", scene_id=scene_id, reason=state.failure_reason)
            return
        if on_disk_code != state.code:
            with self._state_lock:
                self._mark_failed(state, "提交前代码一致性校验失败：磁盘文件已在流水线外被修改")
                self._checkpoint(ctx, State.DISPATCHING)
            self._emit("scene_failed", scene_id=scene_id, reason=state.failure_reason)
            return
        validation = self._validate(on_disk_code, renderer=ctx.render_profile.renderer)
        if not validation.is_valid:
            with self._state_lock:
                self._mark_failed(state, "提交前校验失败:\n" + validation.feedback)
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
                self._reset_visual_receipt(ctx, state)
                try:
                    self._checkpoint(ctx, State.DISPATCHING)
                except Exception as checkpoint_error:
                    # sbatch 已经返回 Job ID；此时不能伪装成普通提交失败并
                    # 自动重提。停止所有 worker，保留内存中的 Job ID，交由
                    # 外层 cancel_all 做一次安全取消和失败收尾。
                    state.failed = True
                    state.give_up = True
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
                    state.failure_reason = (
                        f"Slurm Job {job.job_id} 已提交，但本地检查点持久化失败: {exc}；"
                        "保留 Job ID 并禁止自动重提"
                    )
                self._record_checkpoint_failure(exc)
                self._emit("scene_failed", scene_id=scene_id, reason=state.failure_reason)
                return
            with self._state_lock:
                self._mark_failed(state, f"Slurm 提交失败: {exc}")
                self._checkpoint(ctx, State.DISPATCHING)
            self._emit("scene_failed", scene_id=scene_id, reason=str(exc))

    def _scene_wait_render(self, ctx: PipelineContext, state: SceneState) -> bool:
        """阻塞轮询当前作业直到结束; 返回是否渲染成功。"""
        job = state.slurm_job
        if job is None:
            return False
        self._phase_emit("monitoring")
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
        ok = monitor.results.get(job.job_id)
        if ok is None:
            with self._state_lock:
                state.give_up = True
                state.failure_reason = "渲染作业状态未知，已放弃"
                self._checkpoint(ctx, State.MONITORING)
            return False
        if ok:
            with self._state_lock:
                state.artifact = self._artifact_from_job(ctx, state, job)
                state.rendered = True
                self._reset_visual_receipt(ctx, state)
                self._checkpoint(ctx, State.MONITORING)
            self._emit("scene_rendered", scene_id=job.scene_id)
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
                    state.failure_reason = (
                        f"Slurm 基础设施状态 {job.status}，将重新排队 "
                        f"({state.infra_retries}/{settings.MAX_INFRA_RETRIES})"
                    )
                    self._checkpoint(ctx, State.MONITORING)
                    retry = True
                else:
                    state.give_up = True
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
                    self._mark_failed(state, job.failure_reason or f"Slurm 状态: {job.status}")
                else:
                    state.give_up = True
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
        for sid, state in downstream:
            job = state.slurm_job
            if job is not None and not state.rendered:
                # 监控线程可能刚把作业标成终态，但还没来得及把
                # ``state.rendered`` 写回。对已经结束的作业不再调用 scancel，
                # 否则集群会返回 Invalid job id，进而把一个本可安全重建的
                # 下游场景误判为“取消失败”。
                terminal = job.cancelled or job.status in {
                    "COMPLETED",
                    "CANCELLED",
                    *FAILURE_STATES,
                }
                if not terminal and not self.slurm.cancel_job(job.job_id):
                    with self._state_lock:
                        state.give_up = True
                        state.failed = True
                        state.failure_reason = (
                            f"Scene {sid} 的旧 Job {job.job_id} 使用旧连续性上下文，"
                            "取消失败，禁止自动重提"
                        )
                    continue
                if not terminal:
                    job.cancelled = True
                    job.status = "CANCELLED"
                    job.failure_reason = f"{reason}，连续性重建时取消"
            with self._state_lock:
                state.code = ""
                state.class_name = ""
                state.reviewed = False
                state.rewrite_feedback = ""
                state.slurm_job = None
                state.artifact = None
                state.rendered = False
                state.give_up = False
                state.failed = False
                state.failure_reason = ""
                state.inherited_elements_code = ""
                state.exported_elements_code = ""
                state.exported_elements = []
                self._reset_visual_receipt(
                    ctx,
                    state,
                    clear_candidate=not preserve_visual_candidates,
                    reset_attempts=not preserve_visual_candidates,
                )
                self._write_private(ctx.paths.scenes / f"scene_{sid}.py", "")
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
                state.failure_reason = job.failure_reason or "渲染失败且没有错误日志"
                self._checkpoint(ctx, State.FIXING)
            self._emit("scene_give_up", scene_id=scene_id, reason=state.failure_reason)
            return
        if fixer.is_infrastructure_error(error_log):
            with self._state_lock:
                state.give_up = True
                state.failure_reason = self._give_up_reason(
                    "检测到环境或 Slurm 配置错误，未让 LLM 重写业务代码", error_log
                )
                self._checkpoint(ctx, State.FIXING)
            self._emit("scene_give_up", scene_id=scene_id, reason=state.failure_reason)
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
                state.failure_reason = self._give_up_reason(
                    f"连续 {state.identical_error_count} 次渲染错误完全相同且修复未能消除，"
                    "疑似环境/配置问题，已放弃",
                    error_log,
                )
                self._checkpoint(ctx, State.FIXING)
                terminal = True
            elif state.fix_attempts >= settings.MAX_FIX_ATTEMPTS:
                state.give_up = True
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
            source_kinds={"manim_doc", "example"},
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
            if rag_context and self._supports_keyword(fixer.fix, "rag_context"):
                fix_kwargs["rag_context"] = rag_context
            candidate = fixer.fix(state.code, error_log, **fix_kwargs)
            validation = self._validate(candidate, renderer=ctx.render_profile.renderer)
            if not validation.is_valid:
                candidate, class_name = self._generate_validated_code(
                    state.plan,
                    feedback=(
                        "AutoFix 结果未通过确定性校验：\n"
                        f"{validation.feedback}\n\n原始渲染错误：\n{error_log}"
                    ),
                    previous_code=candidate,
                    stream=False,
                    renderer=ctx.render_profile.renderer,
                    continuity_bible=ctx.continuity_bible,
                    inherited_elements_code=state.inherited_elements_code,
                    inherited_elements=state.plan.inherited_elements,
                    elements_to_remove=state.plan.elements_to_remove,
                    rag_context=rag_context,
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
            state.infra_retries = 0
            state.slurm_job = None
            state.artifact = None
            state.rendered = False
            state.exported_elements_code = ""
            state.exported_elements = []
            self._reset_visual_receipt(ctx, state)
            self._checkpoint(ctx, State.FIXING)
        if code_changed:
            self._request_continuity_rebuild(
                ctx,
                scene_id,
                preserve_visual_candidates=state.visual_best_candidate is not None,
            )
        self._emit(
            "scene_coded",
            scene_id=scene_id,
            file_path=str(ctx.paths.scenes / f"scene_{scene_id}.py"),
        )
        # 注意: identical_error_count 不在这里重置 —— 只有当"错误指纹变化"时才重置
        # (见上面的 else 分支), 从而让"修复后错误完全相同"能在第 2 次相同错误时提前放弃。

    def _apply_review_result(
        self, ctx: PipelineContext, scene_id: int, state: SceneState, result: ReviewResult
    ) -> bool:
        """应用单场景审查结果。"""
        if result.is_valid:
            with self._state_lock:
                state.review_round = 0
                state.reviewed = True
                self._apply_incremental_for_scene(ctx, scene_id, state)
                self._checkpoint(ctx, State.REVIEWING)
            self._emit("scene_review_pass", scene_id=scene_id)
            if state.rendered and state.artifact and state.artifact.origin == "reused":
                self._emit("scene_reused", scene_id=scene_id)
            return True

        with self._state_lock:
            state.review_round += 1
            review_round = state.review_round
            # Reviewer 已消耗一轮，在后续改写前先持久化计数。
            self._checkpoint(ctx, State.REVIEWING)
        original_feedback = result.feedback or ""
        fix_details = "\n".join(
            f"- [{fix.reason}] {fix.find!r} → {fix.replace!r}" for fix in result.fixes
        )

        if review_round >= settings.MAX_REVIEW_ROUNDS:
            with self._state_lock:
                state.give_up = True
                state.failure_reason = "达到最大审查轮次，代码仍未通过"
                self._checkpoint(ctx, State.REVIEWING)
            self._emit("scene_give_up", scene_id=scene_id, reason=state.failure_reason)
            return True

        if result.severity == "minor":
            candidate = state.code
            applied_count = 0
            for fix in result.fixes:
                if candidate.count(fix.find) == 1:
                    candidate = candidate.replace(fix.find, fix.replace, 1)
                    applied_count += 1
            validation = (
                self._validate(candidate, renderer=ctx.render_profile.renderer)
                if applied_count > 0
                else None
            )
            if validation and validation.is_valid:
                self._write_private(ctx.paths.scenes / f"scene_{scene_id}.py", candidate)
                with self._state_lock:
                    state.code = candidate
                    state.class_name = validation.scene_classes[0]
                    state.artifact = None
                    state.rendered = False
                    self._reset_visual_receipt(ctx, state)
                    self._checkpoint(ctx, State.REVIEWING)
                self._emit(
                    "scene_coded",
                    scene_id=scene_id,
                    file_path=str(ctx.paths.scenes / f"scene_{scene_id}.py"),
                )
                self._emit("scene_review_fail", scene_id=scene_id, severity="minor")
                return True
            # minor 修复失败 → 升级为 major, 保留原始反馈
            result = ReviewResult(
                is_valid=False,
                severity="major",
                feedback=(
                    f"## Reviewer 审查意见（minor 修复未能全部应用）\n{original_feedback}\n\n"
                    f"## 修复建议详情\n{fix_details}\n\n"
                    f"## 确定性校验\n{validation.feedback if validation else '未生成有效代码'}"
                ),
            )

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
            "global_continuity": (
                ctx.continuity_bible.model_dump(mode="json") if ctx.continuity_bible else None
            ),
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)[:30_000]

    @staticmethod
    def _redact_visual_error(error: Exception) -> str:
        detail = str(error).strip() or type(error).__name__
        if settings.VISUAL_LLM_API_KEY:
            detail = detail.replace(settings.VISUAL_LLM_API_KEY, "<redacted>")
        return detail[:10_000]

    def _restore_visual_candidate_into_state(
        self,
        ctx: PipelineContext,
        scene_id: int,
        state: SceneState,
        candidate: VisualCandidate,
    ) -> bool:
        """恢复已验证候选；返回代码是否相对当前状态发生变化。"""

        inherited_hash = sha256_text(state.inherited_elements_code)
        if candidate.inherited_elements_sha256 != inherited_hash:
            raise RuntimeError(f"Scene {scene_id} 的视觉候选基于不同的继承上下文，拒绝恢复")
        code_changed = candidate.code != state.code
        self._write_private(ctx.paths.scenes / f"scene_{scene_id}.py", candidate.code)
        state.code = candidate.code
        state.class_name = candidate.class_name
        state.slurm_job = candidate.slurm_job
        state.artifact = candidate.artifact
        state.rendered = True
        state.reviewed = True
        state.failed = False
        state.give_up = False
        state.failure_reason = ""
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

    def _visual_gate(self, ctx: PipelineContext) -> bool:
        """逐场景评估精确渲染产物；返回是否安排了代码改进。"""

        profile = ctx.visual_eval_profile
        if not profile.enabled:
            for state in ctx.scene_states.values():
                state.visual_status = "skipped"
            return False

        # 若视觉修复链路自身失败，恢复此前得分最高且可验证的候选，避免丢掉
        # 原本可用的视频。恢复旧代码后必须重建所有下游连续性上下文。
        for scene_id, state in sorted(ctx.scene_states.items()):
            candidate = state.visual_best_candidate
            if state.rendered or candidate is None or not (state.failed or state.give_up):
                continue
            if candidate.inherited_elements_sha256 != sha256_text(state.inherited_elements_code):
                continue
            current_code = state.code
            self._artifact_video_path(ctx, candidate.artifact)
            changed = self._restore_visual_candidate_into_state(ctx, scene_id, state, candidate)
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
            return bool(ctx.continuity_rebuild_required)

        targets: list[tuple[int, SceneState, SceneArtifact]] = []
        for scene_id, state in sorted(ctx.scene_states.items()):
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
            with self._visual_llm_slot():
                result, samples = evaluator.evaluate_scene_video(
                    video,
                    description=ctx.original_prompt or ctx.user_prompt,
                    scene_context=self._scene_visual_context(ctx, state),
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
        for scene_id, state, artifact in targets:
            result, samples, error = outcomes[scene_id]
            report_dir = ctx.paths.root / "eval_reports" / f"scene_{scene_id}"
            report_dir.mkdir(parents=True, exist_ok=True)
            report_dir.chmod(0o700)
            report_path = report_dir / (
                f"attempt_{state.visual_fix_attempts:02d}_{artifact.video_sha256[:12]}.json"
            )
            report: dict = {
                "schema_version": 1,
                "scope": "scene",
                "scene_id": scene_id,
                "model": profile.model,
                "visual_profile_sha256": profile.digest(),
                "attempt": state.visual_fix_attempts,
                "artifact_sha256": artifact.video_sha256,
                "code_sha256": artifact.code_sha256,
                "inherited_elements_sha256": sha256_text(state.inherited_elements_code),
                "frames": [
                    {
                        "frame_id": sample.frame_id,
                        "timestamp_seconds": sample.timestamp_seconds,
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

            state.visual_report_file = report_relative
            state.visual_report_sha256 = report_hash
            state.visual_artifact_sha256 = artifact.video_sha256
            if result is None:
                state.visual_status = "unknown"
                state.visual_score = None
                state.visual_feedback = error
                self._emit("scene_visual_unknown", scene_id=scene_id, reason=error)
                continue

            state.visual_score = result.overall_score
            state.visual_feedback = result.feedback()
            candidate = VisualCandidate(
                score=result.overall_score,
                has_major_issue=result.has_major_issue,
                passed=not result.needs_fix(profile.threshold),
                inherited_elements_sha256=sha256_text(state.inherited_elements_code),
                code=state.code,
                class_name=state.class_name,
                slurm_job=state.slurm_job,
                artifact=artifact,
                exported_elements_code=state.exported_elements_code,
                exported_elements=list(state.exported_elements),
                report_file=report_relative,
                report_sha256=report_hash,
            )
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

            if not result.needs_fix(profile.threshold):
                state.visual_status = "passed"
                state.visual_feedback = ""
                self._emit(
                    "scene_visual_pass",
                    scene_id=scene_id,
                    score=result.overall_score,
                )
                continue

            can_fix = ctx.auto_fix and state.visual_fix_attempts < profile.max_fix_attempts
            if can_fix:
                state.visual_status = "needs_fix"
                if first_fix_scene is None:
                    # 一次只修改最早失败场景。它的导出元素会影响所有后继场景；
                    # 先修后重建可避免并行修复基于即将过期的连续性上下文。
                    first_fix_scene = scene_id
                    first_fix_feedback = result.feedback()
                continue

            # 主观质量问题不应使已经成功渲染的视频消失。达到上限时选择
            # 历次得分最高候选，并将未解决问题明确记录为 warning。
            best = state.visual_best_candidate
            if (
                best is not None
                and best.artifact.video_sha256 != artifact.video_sha256
                and best.inherited_elements_sha256 == sha256_text(state.inherited_elements_code)
            ):
                changed = self._restore_visual_candidate_into_state(ctx, scene_id, state, best)
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
                state.visual_status = "warning"
                if not state.visual_feedback:
                    state.visual_feedback = (
                        "已达到视觉修复上限" if ctx.auto_fix else "已关闭自动修复"
                    )
            self._emit(
                "scene_visual_warning",
                scene_id=scene_id,
                score=state.visual_score,
                reason=("已达到视觉修复上限" if ctx.auto_fix else "已关闭自动修复"),
            )

        if first_fix_scene is not None:
            state = ctx.scene_states[first_fix_scene]
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
            self._emit(
                "scene_visual_fixing",
                scene_id=first_fix_scene,
                attempt=state.visual_fix_attempts,
                max_attempts=profile.max_fix_attempts,
            )
            self._request_continuity_rebuild(
                ctx,
                first_fix_scene,
                reason="视觉评估要求修改场景代码",
                preserve_visual_candidates=True,
            )
            self._checkpoint(ctx, State.VISUAL_EVALUATING)
            return True

        self._checkpoint(ctx, State.VISUAL_EVALUATING)
        return bool(ctx.continuity_rebuild_required)

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
            samples = evaluator.extract_video_samples(
                ctx.final_video,
                frame_dir,
                frame_count=profile.frame_count,
            )
            with self._visual_llm_slot():
                result = evaluator.visual_evaluator.evaluate_video_frames(
                    samples,
                    ctx.original_prompt or ctx.user_prompt,
                    scene_context=json.dumps(
                        [state.plan.model_dump(mode="json") for state in ctx.scene_states.values()],
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
                "frames": [
                    {
                        "frame_id": sample.frame_id,
                        "timestamp_seconds": sample.timestamp_seconds,
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
                "frames": [
                    {
                        "frame_id": sample.frame_id,
                        "timestamp_seconds": sample.timestamp_seconds,
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
            ctx.final_video = self.merger.merge(
                video_paths,
                ctx.paths.output,
                replace_existing=(
                    output_is_run_local
                    or (force_remerge and not output_is_run_local and checkpointed_output_matches)
                ),
                render_profile=ctx.render_profile,
            )
        ctx.final_video_sha256 = sha256_file(ctx.final_video)
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
            scene_eval_results = {
                scene_id: evaluator.evaluate_code(state.code)
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
                state.failure_reason = ""
                state.last_error_fp = ""
                state.identical_error_count = 0
                state.inherited_elements_code = ""
                state.exported_elements_code = ""
                state.exported_elements = []
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
