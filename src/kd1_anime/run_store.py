"""持久化运行状态，支持查询和安全恢复中断的流水线。"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from kd1_anime.agents.planner import (
    ContinuityBible,
    ElementManifest,
    ExtractedElement,
    LessonSpec,
    SceneOutline,
    ScenePlan,
    TeachingGraph,
)
from kd1_anime.agents.state_ledger import StateLedger
from kd1_anime.agents.technical_planner import TechnicalSpec
from kd1_anime.cluster.slurm import SlurmJob
from kd1_anime.config import resolve_runtime_path
from kd1_anime.rag.models import RagReceipt, RagRuntimeProfile
from kd1_anime.rendering import (
    RenderProfile,
    SceneArtifact,
    VideoMetadata,
)

MANIFEST_NAME = "manifest.json"
MANIFEST_SCHEMA_VERSION = 5
READABLE_MANIFEST_SCHEMA_VERSIONS = frozenset({4, 5})
RUN_ID_PATTERN = re.compile(r"\d{8}-\d{6}-[0-9a-f]{8}")
RESUME_LLM_STATES = frozenset(
    {
        "INIT",
        "PLANNING",
        "DETAILING",
        "PLAN_REVIEWING",
        "CODING",
        "REVIEWING",
        "FIXING",
        "VISUAL_EVALUATING",
        "EVALUATING",
        "ERROR",
    }
)
RunStatus = Literal["running", "interrupted", "failed", "completed", "dry_run_complete"]
FSMState = Literal[
    "INIT",
    "PLANNING",
    "DETAILING",
    "PLAN_REVIEWING",
    "CODING",
    "REVIEWING",
    "DISPATCHING",
    "MONITORING",
    "FIXING",
    "VISUAL_EVALUATING",
    "MERGING",
    "EVALUATING",
    "DONE",
    "ERROR",
]
ScenePhase = Literal[
    "pending",
    "detailed",
    "plan_reviewing",
    "plan_reviewed",
    "technical_planning",
    "technical_validating",
    "coded",
    "reviewed",
    "monitoring",
    "rendered",
    "visual_evaluating",
    "visual_accepted",
    "failed",
]
VisualStatus = Literal[
    "pending",
    "evaluating",
    "passed",
    "needs_fix",
    "warning",
    "unknown",
    "skipped",
]
TechnicalStatus = Literal["pending", "generating", "passed", "failed"]
LocalSmokeStatus = Literal["pending", "running", "passed", "failed", "skipped"]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def sha256_text(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class StoredSlurmJob(BaseModel):
    """只保存恢复 Slurm 监控所需的数据；路径均相对当前 run 根目录。"""

    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(pattern=r"^\d+$")
    scene_id: int = Field(ge=1)
    script_path: str
    log_out: str
    log_err: str
    media_dir: str
    scene_class_name: str = Field(min_length=1, max_length=200)
    submitted_at: float = Field(gt=0)
    started_at: float | None = Field(default=None, gt=0)
    code_sha256: str = Field(default="", pattern=r"^(?:[0-9a-f]{64})?$")
    render_profile: RenderProfile = Field(default_factory=RenderProfile.current)
    output_path: str | None = None
    output_metadata: VideoMetadata | None = None
    output_sha256: str = Field(default="", pattern=r"^(?:[0-9a-f]{64})?$")
    elapsed_seconds: float | None = Field(default=None, ge=0)
    status: str = Field(min_length=1, max_length=100)
    failure_reason: str = Field(default="", max_length=50_000)
    cancelled: bool = False


class VisualEvalProfile(BaseModel):
    """持久化的非敏感视觉评估策略；API Key 和端点不写入运行清单。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    model: str = Field(default="", max_length=300)
    frame_count: int = Field(default=6, ge=1, le=8)
    threshold: float = Field(default=3.5, ge=1.0, le=5.0)
    max_fix_attempts: int = Field(default=2, ge=0, le=5)
    evaluator_version: Literal["1"] = "1"

    def digest(self) -> str:
        payload = self.model_dump_json(exclude_none=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class StoredVisualCandidate(BaseModel):
    """可在视觉修复失败时恢复的、完整且经过验证的场景候选。"""

    model_config = ConfigDict(extra="forbid")

    score: float = Field(ge=1.0, le=5.0)
    has_major_issue: bool = False
    passed: bool = False
    # 全 0 表示旧收据未记录继承上下文；它不会与任何真实 SHA-256 匹配，
    # 因而可读取但不能被不安全地自动恢复。
    inherited_elements_sha256: str = Field(default="0" * 64, pattern=r"^[0-9a-f]{64}$")
    code_file: str
    code_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    class_name: str = Field(min_length=1, max_length=200)
    slurm_job: StoredSlurmJob | None = None
    artifact: SceneArtifact
    exported_elements_code: str = Field(default="", max_length=30_000)
    exported_elements: list[ExtractedElement] = Field(default_factory=list, max_length=100)
    report_file: str
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class StoredSceneState(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    plan: ScenePlan
    code_file: str = ""
    code_sha256: str = Field(default="", pattern=r"^(?:[0-9a-f]{64})?$")
    class_name: str = Field(default="", max_length=200)
    review_round: int = Field(default=0, ge=0)
    fix_attempts: int = Field(default=0, ge=0)
    infra_retries: int = Field(default=0, ge=0)
    reviewed: bool = False
    plan_ready: bool = False
    plan_review_round: int = Field(default=0, ge=0)
    plan_reviewed: bool = False
    plan_review_feedback: str = Field(default="", max_length=50_000)
    plan_review_signature: str = Field(default="", pattern=r"^(?:[0-9a-f]{16})?$")
    identical_plan_review_count: int = Field(default=0, ge=0)
    technical_spec: TechnicalSpec | None = None
    technical_spec_sha256: str = Field(default="", pattern=r"^(?:[0-9a-f]{64})?$")
    technical_input_sha256: str = Field(default="", pattern=r"^(?:[0-9a-f]{64})?$")
    technical_status: TechnicalStatus = "pending"
    technical_error: str = Field(default="", max_length=50_000)
    local_smoke_status: LocalSmokeStatus = "pending"
    rewrite_feedback: str = Field(default="", max_length=50_000)
    review_signature: str = Field(default="", pattern=r"^(?:[0-9a-f]{16})?$")
    identical_review_count: int = Field(default=0, ge=0)
    last_error_fp: str = Field(default="", max_length=64)
    identical_error_count: int = Field(default=0, ge=0)
    slurm_job: StoredSlurmJob | None = None
    artifact: SceneArtifact | None = None
    phase: ScenePhase = "pending"
    rendered: bool = False
    give_up: bool = False
    failed: bool = False
    failure_reason: str = Field(default="", max_length=50_000)
    failure_category: Literal[
        "",
        "planning",
        "continuity",
        "coding",
        "review",
        "render",
        "infrastructure",
        "llm",
        "system",
    ] = ""
    # 复杂几何方案审查耗尽后是否已切换到保守教学方案；恢复时不能重复
    # 触发同一降级，否则会无限消耗 LLM 重试预算。
    safe_fallback_used: bool = False
    safe_fallback_reason: str = Field(default="", max_length=5_000)
    inherited_elements_code: str = Field(default="", max_length=30_000)
    exported_elements_code: str = Field(default="", max_length=30_000)
    exported_elements: list[ExtractedElement] = Field(default_factory=list, max_length=100)
    visual_status: VisualStatus = "skipped"
    visual_fix_attempts: int = Field(default=0, ge=0)
    visual_score: float | None = Field(default=None, ge=1.0, le=5.0)
    visual_report_file: str = ""
    visual_report_sha256: str = Field(default="", pattern=r"^(?:[0-9a-f]{64})?$")
    visual_artifact_sha256: str = Field(default="", pattern=r"^(?:[0-9a-f]{64})?$")
    visual_feedback: str = Field(default="", max_length=20_000)
    visual_best_candidate: StoredVisualCandidate | None = None


class RunManifest(BaseModel):
    """版本化运行清单。未知字段会被拒绝，防止静默误读未来格式。"""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    schema_version: Literal[4, 5] = MANIFEST_SCHEMA_VERSION
    revision: int = Field(default=0, ge=0)
    run_id: str
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    status: RunStatus = "running"
    state: FSMState = "INIT"
    user_prompt: str = Field(max_length=1_000_000)
    dry_run: bool = False
    interactive: bool = False
    auto_fix: bool = True
    # Direct ``render`` runs contain user-supplied code and intentionally skip
    # every generation/review LLM stage, including on resume.
    direct_render: bool = False
    approve_plan: bool = False
    plan_approved: bool = False
    output_path: str
    render_profile: RenderProfile = Field(default_factory=RenderProfile.current)
    outlines: list[SceneOutline] = Field(default_factory=list)
    scenes: dict[int, StoredSceneState] = Field(default_factory=dict)
    # v5 在概要阶段固定全片教学合同和断言依赖；v4 读取时使用空默认值，
    # 但 validate_for_resume 会拒绝继续修改旧清单。
    lesson_spec: LessonSpec = Field(default_factory=LessonSpec)
    teaching_graph: TeachingGraph = Field(default_factory=TeachingGraph)
    state_ledger: StateLedger = Field(default_factory=StateLedger)
    expected_final_duration: float | None = Field(default=None, ge=0, le=3_600)
    # v4/v5 运行都会固定全片连续性规范；v5 另外持久化教学合同与状态账本。
    continuity_bible: ContinuityBible | None = None
    element_manifest: ElementManifest = Field(default_factory=ElementManifest)
    plan_review_status: Literal["pending", "reviewing", "passed", "failed", "skipped"] = "skipped"
    plan_review_cycle_signature: str = Field(default="", pattern=r"^(?:[0-9a-f]{64})?$")
    plan_review_cycle_count: int = Field(default=0, ge=0)
    continuity_review_status: Literal["pending", "reviewing", "passed", "warning"] = "passed"
    continuity_review_round: int = Field(default=0, ge=0)
    # 连续性审查耗尽后，resume 最多自动重查一次；默认值兼容已有清单。
    continuity_resume_recheck_used: bool = False
    continuity_warnings: list[str] = Field(default_factory=list, max_length=100)
    final_video: str | None = None
    final_video_sha256: str = Field(default="", pattern=r"^(?:[0-9a-f]{64})?$")
    error: str = Field(default="", max_length=50_000)

    # 增量渲染支持
    incremental: bool = False
    base_run_id: str | None = Field(default=None, pattern=r"^(?:\d{8}-\d{6}-[0-9a-f]{8})?$")
    eval_round: int = Field(default=0, ge=0)
    continuity_rebuild_required: bool = False
    visual_eval_profile: VisualEvalProfile = Field(default_factory=VisualEvalProfile)
    rag_profile: RagRuntimeProfile = Field(default_factory=RagRuntimeProfile)
    rag_receipts: dict[str, RagReceipt] = Field(default_factory=dict, max_length=256)
    rag_warnings: list[str] = Field(default_factory=list, max_length=100)

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, value: str) -> str:
        if not RUN_ID_PATTERN.fullmatch(value):
            raise ValueError("run_id 格式无效")
        return value

    @field_validator("created_at", "updated_at")
    @classmethod
    def validate_aware_datetime(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("时间戳必须包含时区")
        return value

    def integrity_errors(self) -> list[str]:
        """检查跨字段一致性；供读取清单时拒绝语义损坏的数据。

        这些约束不能只靠 Pydantic 的字段类型表达，例如 rendered=True
        必须有经过验证的 artifact。
        """

        errors: list[str] = []
        if self.schema_version not in READABLE_MANIFEST_SCHEMA_VERSIONS:
            errors.append(f"不支持的 manifest schema_version: {self.schema_version}")
        for scene_id, scene in self.scenes.items():
            if scene.plan.scene_id != scene_id:
                errors.append(f"Scene key {scene_id} 与 plan.scene_id {scene.plan.scene_id} 不一致")
            if scene.plan_reviewed and not scene.plan_ready:
                errors.append(f"Scene {scene_id} 标记为 plan_reviewed 但 plan_ready=false")
            if scene.technical_status == "passed":
                if scene.technical_spec is None:
                    errors.append(f"Scene {scene_id} 标记为 technical passed 但缺少 TechnicalSpec")
                elif not scene.technical_input_sha256:
                    errors.append(f"Scene {scene_id} 的 TechnicalSpec 缺少输入哈希")
                elif (
                    sha256_text(scene.technical_spec.model_dump_json())
                    != scene.technical_spec_sha256
                ):
                    errors.append(f"Scene {scene_id} 的 TechnicalSpec 哈希不一致")
            elif (
                scene.technical_spec is not None
                and scene.technical_spec_sha256
                and sha256_text(scene.technical_spec.model_dump_json())
                != scene.technical_spec_sha256
            ):
                errors.append(f"Scene {scene_id} 的 TechnicalSpec 哈希不一致")
            if scene.technical_spec is None and (
                scene.technical_spec_sha256 or scene.technical_input_sha256
            ):
                errors.append(f"Scene {scene_id} 存在 TechnicalSpec 哈希但缺少 TechnicalSpec")
            if scene.rendered and scene.artifact is None:
                errors.append(f"Scene {scene_id} 标记为 rendered 但缺少 artifact")
            if scene.artifact is not None:
                artifact = scene.artifact
                if not scene.rendered:
                    errors.append(f"Scene {scene_id} 存在 artifact 但 rendered=false")
                if scene.rendered and not artifact.verified:
                    errors.append(f"Scene {scene_id} 标记为 rendered 但 artifact 未验证")
                if artifact.scene_id != scene_id:
                    errors.append(f"Scene {scene_id} 的 artifact.scene_id 不一致")
                if artifact.scene_class_name != scene.class_name:
                    errors.append(f"Scene {scene_id} 的 artifact 类名不一致")
                if scene.code_sha256 and artifact.code_sha256 != scene.code_sha256:
                    errors.append(f"Scene {scene_id} 的 artifact 代码哈希不一致")
            if scene.slurm_job is not None:
                job = scene.slurm_job
                if job.scene_id != scene_id:
                    errors.append(f"Scene {scene_id} 的 Slurm Job 场景 ID 不一致")
                if scene.code_sha256 and job.code_sha256 != scene.code_sha256:
                    errors.append(f"Scene {scene_id} 的 Slurm Job 代码哈希不一致")
            if scene.visual_status in {"passed", "warning", "unknown"} and not scene.rendered:
                errors.append(f"Scene {scene_id} 视觉状态为 {scene.visual_status} 但没有渲染产物")
            if bool(scene.visual_report_file) != bool(scene.visual_report_sha256):
                errors.append(f"Scene {scene_id} 的视觉报告路径与哈希不完整")
            if scene.visual_status in {"passed", "warning", "unknown"}:
                if not scene.visual_report_file:
                    errors.append(f"Scene {scene_id} 的视觉终态缺少评估报告")
                if (
                    scene.artifact is not None
                    and scene.visual_artifact_sha256 != scene.artifact.video_sha256
                ):
                    errors.append(f"Scene {scene_id} 的视觉评估记录与当前视频哈希不一致")
            candidate = scene.visual_best_candidate
            if candidate is not None:
                if not candidate.artifact.verified:
                    errors.append(f"Scene {scene_id} 的最佳视觉候选产物未经验证")
                if candidate.artifact.scene_id != scene_id:
                    errors.append(f"Scene {scene_id} 的最佳视觉候选场景 ID 不一致")
                if candidate.artifact.code_sha256 != candidate.code_sha256:
                    errors.append(f"Scene {scene_id} 的最佳视觉候选代码哈希不一致")
                if candidate.slurm_job and candidate.slurm_job.code_sha256 != candidate.code_sha256:
                    errors.append(f"Scene {scene_id} 的最佳视觉候选 Job 代码哈希不一致")
        if self.rag_profile.index_sha256:
            for receipt_key, receipt in self.rag_receipts.items():
                if receipt.index_sha256 and receipt.index_sha256 != self.rag_profile.index_sha256:
                    errors.append(f"RAG 收据 {receipt_key} 的索引哈希与运行配置不一致")
        if self.plan_review_status == "passed":
            for scene_id, scene in self.scenes.items():
                if (
                    scene.plan_ready
                    and not scene.plan_reviewed
                    and not scene.failed
                    and not scene.give_up
                ):
                    errors.append(f"Scene {scene_id} 未通过计划审查但运行标记为 passed")
        entry_ids = [entry.element_id for entry in self.element_manifest.entries]
        if len(entry_ids) != len(set(entry_ids)):
            errors.append("element_manifest 包含重复 element_id")
        for entry in self.element_manifest.entries:
            if sha256_text(entry.source_code) != entry.source_code_sha256:
                errors.append(f"element_manifest 元素 {entry.element_id} 的源代码哈希不一致")
            if entry.source_scene_id not in self.scenes:
                errors.append(
                    f"element_manifest 元素 {entry.element_id} 引用了不存在的 Scene "
                    f"{entry.source_scene_id}"
                )
        for scene_id, exported_ids in self.element_manifest.scene_exports.items():
            if scene_id not in self.scenes:
                errors.append(f"element_manifest.scene_exports 引用了不存在的 Scene {scene_id}")
            if len(exported_ids) != len(set(exported_ids)):
                errors.append(f"Scene {scene_id} 的 element_manifest 导出 ID 重复")
        for scene_id, boundary in self.state_ledger.boundaries.items():
            if scene_id not in self.scenes:
                errors.append(f"StateLedger 引用了不存在的 Scene {scene_id}")
            if boundary.scene_id != scene_id:
                errors.append(f"StateLedger 边界 key {scene_id} 与 scene_id 不一致")
            scene = self.scenes.get(scene_id)
            if scene is not None:
                if (
                    boundary.exported_code_sha256
                    and scene.exported_elements_code
                    and boundary.exported_code_sha256 != sha256_text(scene.exported_elements_code)
                ):
                    errors.append(f"StateLedger Scene {scene_id} 的导出代码哈希不一致")
                if (
                    boundary.artifact_video_sha256
                    and scene.artifact is not None
                    and boundary.artifact_video_sha256 != scene.artifact.video_sha256
                ):
                    errors.append(f"StateLedger Scene {scene_id} 的视频哈希不一致")
        ledger_ids = [item.element_id for item in self.state_ledger.elements]
        if len(ledger_ids) != len(set(ledger_ids)):
            errors.append("StateLedger 包含重复 element_id")
        for element in self.state_ledger.elements:
            if element.source_scene_id not in self.scenes:
                errors.append(
                    f"StateLedger 元素 {element.element_id} 引用了不存在的 Scene "
                    f"{element.source_scene_id}"
                )
        if self.status == "completed" and not self.final_video:
            errors.append("运行标记为 completed 但缺少 final_video")
        return errors

    def validate_for_resume(self) -> None:
        """在恢复/增量复用前拒绝语义损坏的清单。

        ``status`` 命令仍允许读取并展示损坏清单，便于诊断；真正复用其中
        的代码、Job 或元素交接时必须 fail-closed，不能只依赖字段类型校验。
        """

        if self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"manifest schema_version={self.schema_version} 仅支持只读查看；"
                f"恢复需要 v{MANIFEST_SCHEMA_VERSION}，请重新生成运行。"
            )
        errors = self.integrity_errors()
        if errors:
            raise ValueError("运行清单完整性校验失败: " + "; ".join(errors))


def _run_relative(root: Path, path: Path) -> str:
    root = root.resolve()
    resolved = path.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"运行产物路径不在 run 目录内: {resolved}") from exc


def restore_run_path(root: Path, relative: str) -> Path:
    """解析清单中的相对路径，同时拒绝绝对路径、遍历和符号链接逃逸。"""

    candidate = Path(relative)
    if not relative or candidate.is_absolute():
        raise ValueError(f"清单包含无效的运行相对路径: {relative!r}")
    root = root.resolve()
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"清单路径越出 run 目录: {relative!r}") from exc
    return resolved


def store_slurm_job(job: SlurmJob, root: Path) -> StoredSlurmJob:
    return StoredSlurmJob(
        job_id=job.job_id,
        scene_id=job.scene_id,
        script_path=_run_relative(root, job.script_path),
        log_out=_run_relative(root, job.log_out),
        log_err=_run_relative(root, job.log_err),
        media_dir=_run_relative(root, job.media_dir),
        scene_class_name=job.scene_class_name,
        submitted_at=job.submitted_at,
        started_at=job.started_at,
        code_sha256=job.code_sha256,
        render_profile=job.render_profile,
        output_path=_run_relative(root, job.output_path) if job.output_path else None,
        output_metadata=job.output_metadata,
        output_sha256=job.output_sha256,
        elapsed_seconds=job.elapsed_seconds,
        status=job.status,
        failure_reason=job.failure_reason,
        cancelled=job.cancelled,
    )


def restore_slurm_job(stored: StoredSlurmJob, root: Path) -> SlurmJob:
    return SlurmJob(
        job_id=stored.job_id,
        scene_id=stored.scene_id,
        script_path=restore_run_path(root, stored.script_path),
        log_out=restore_run_path(root, stored.log_out),
        log_err=restore_run_path(root, stored.log_err),
        media_dir=restore_run_path(root, stored.media_dir),
        scene_class_name=stored.scene_class_name,
        submitted_at=stored.submitted_at,
        started_at=stored.started_at,
        code_sha256=stored.code_sha256,
        render_profile=stored.render_profile,
        output_path=(restore_run_path(root, stored.output_path) if stored.output_path else None),
        output_metadata=stored.output_metadata,
        output_sha256=stored.output_sha256,
        elapsed_seconds=stored.elapsed_seconds,
        status=stored.status,
        failure_reason=stored.failure_reason,
        cancelled=stored.cancelled,
    )


def atomic_write_text(path: Path, payload: str, *, mode: int = 0o600) -> None:
    """以同目录临时文件 + fsync + os.replace 原子写入私有文本文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=".atomic-", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    fd_open = True
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd_open = False  # fdopen 接管描述符，离开 with 后负责关闭
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        path.chmod(mode)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        if fd_open:
            with suppress(OSError):
                os.close(fd)
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, payload: Any, *, mode: int = 0o600) -> None:
    """原子写入 JSON，供评估报告等非 manifest 持久化使用。"""

    atomic_write_text(
        path,
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        mode=mode,
    )


def write_manifest(path: Path, manifest: RunManifest) -> None:
    """以同目录临时文件 + os.replace 原子更新清单。"""

    if manifest.schema_version != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"只允许写入 v{MANIFEST_SCHEMA_VERSION} manifest，"
            f"不能修改 v{manifest.schema_version} 旧清单"
        )
    manifest.updated_at = utc_now()
    atomic_write_text(path, manifest.model_dump_json(indent=2) + "\n")


class RunRepository:
    """在配置的 workspace 下定位运行，且不接受任意路径。"""

    def __init__(self, workspace_dir: Path) -> None:
        self.runs_root = resolve_runtime_path(workspace_dir) / "runs"
        self.list_errors: list[tuple[str, str]] = []

    def run_root(self, run_id: str) -> Path:
        if not RUN_ID_PATTERN.fullmatch(run_id):
            raise ValueError("run_id 格式无效")
        candidate = self.runs_root / run_id
        if candidate.exists() and (
            candidate.is_symlink() or candidate.resolve().parent != self.runs_root
        ):
            raise ValueError("运行目录不是配置的 workspace/runs 下的真实目录")
        return candidate

    def manifest_path(self, run_id: str) -> Path:
        return self.run_root(run_id) / MANIFEST_NAME

    def load(self, run_id: str) -> RunManifest:
        path = self.manifest_path(run_id)
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"找不到运行清单: {run_id}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("运行清单顶层必须是 JSON 对象")
        migrated = migrate_manifest_data(raw, self.run_root(run_id))
        manifest = RunManifest.model_validate(migrated)
        if manifest.run_id != run_id:
            raise ValueError("运行目录与 manifest.run_id 不一致")
        return manifest

    def load_for_resume(self, run_id: str) -> RunManifest:
        """读取并验证可修改恢复的 v5 清单。"""

        manifest = self.load(run_id)
        manifest.validate_for_resume()
        return manifest

    def list(self) -> list[RunManifest]:
        self.list_errors = []
        if not self.runs_root.is_dir():
            return []
        manifests: list[RunManifest] = []
        for child in self.runs_root.iterdir():
            if not child.is_dir() or child.is_symlink() or not RUN_ID_PATTERN.fullmatch(child.name):
                continue
            try:
                manifests.append(self.load(child.name))
            except (OSError, ValueError) as exc:
                self.list_errors.append((child.name, str(exc)))
                continue
        return sorted(manifests, key=lambda item: item.updated_at, reverse=True)


@contextmanager
def lock_run(root: Path) -> Iterator[None]:
    """持有进程级排他锁，防止两个 resume 实例操作同一批 Slurm 作业。"""

    if root.is_symlink():
        raise RuntimeError(f"运行目录不能是符号链接: {root}")
    root.mkdir(parents=True, exist_ok=True)
    # 旧版本创建的 run 目录可能仍然是组/其他用户可读的；恢复时重新收紧
    # 权限，避免 manifest、提示词和生成代码继续暴露给同机用户。
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"运行目录不是可信的真实目录: {root}")
    root.chmod(0o700)
    lock_path = root / ".run.lock"
    if lock_path.is_symlink():
        raise RuntimeError(f"运行锁不能是符号链接: {lock_path}")
    open_flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, open_flags, 0o600)
    except OSError as exc:
        raise RuntimeError(f"无法安全打开运行锁: {lock_path}") from exc
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"运行 {root.name} 正被另一个进程使用") from exc
        os.fchmod(descriptor, 0o600)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def get_reusable_video_path(
    old_manifest: RunManifest,
    scene_id: int,
    old_root: Path,
) -> Path | None:
    """获取可复用的旧视频路径。"""
    old_scene = old_manifest.scenes.get(scene_id)
    if old_scene is None or not old_scene.rendered:
        return None

    if old_scene.artifact and old_scene.artifact.verified:
        artifact = old_scene.artifact
        source_root = old_root
        if artifact.source_run_id != old_manifest.run_id:
            source_root = old_root.parent / artifact.source_run_id
            # 跨 run 复用只能访问配置的 runs 目录下的真实兄弟目录，不能让
            # 清单中的合法 run-id 通过符号链接逃逸到 workspace 之外。
            if (
                source_root.is_symlink()
                or not source_root.is_dir()
                or source_root.resolve().parent != old_root.resolve().parent
            ):
                return None
        try:
            video = restore_run_path(source_root, artifact.video_path)
        except (OSError, ValueError):
            return None
        try:
            return video if video.is_file() and video.stat().st_size > 0 else None
        except OSError:
            return None

    old_job = old_scene.slurm_job
    if old_job is None:
        return None

    try:
        old_media_dir = restore_run_path(old_root, old_job.media_dir)
        video = _latest_video_candidate(old_media_dir, old_scene.class_name)
    except (OSError, ValueError):
        return None
    return video


def _latest_video_candidate(media_dir: Path, class_name: str) -> Path | None:
    """在媒体目录中竞态安全地选择最新的完整 MP4。"""

    candidates: list[tuple[Path, float]] = []
    try:
        for path in media_dir.rglob(f"{class_name}.mp4"):
            if "partial_movie_files" in path.parts or "__smoke__" in path.parts:
                continue
            try:
                stat = path.stat()
            except OSError:
                # 渲染器可能正在替换/删除文件；跳过瞬时消失的候选。
                continue
            if stat.st_size > 0:
                candidates.append((path, stat.st_mtime))
    except OSError:
        return None
    return max(candidates, key=lambda item: item[1])[0] if candidates else None


def migrate_manifest_data(raw: dict, root: Path) -> dict:
    """读取 v4/v5 清单；v4 只允许查看，不进行猜测迁移。"""

    version = raw.get("schema_version", 1)
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValueError(f"manifest schema_version 必须是整数: {version!r}")
    if version in READABLE_MANIFEST_SCHEMA_VERSIONS:
        if version == MANIFEST_SCHEMA_VERSION:
            required_fields = {"lesson_spec", "teaching_graph", "state_ledger"}
            missing_fields = sorted(required_fields - raw.keys())
            if missing_fields:
                raise ValueError("v5 manifest 缺少必需字段: " + ", ".join(missing_fields))
        return raw
    if version < MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"不支持旧版 manifest schema_version={version}；当前版本为 "
            f"{MANIFEST_SCHEMA_VERSION}。旧运行不能安全恢复，请重新生成。"
        )
    raise ValueError(
        f"不支持的 manifest schema_version: {version}（当前版本为 {MANIFEST_SCHEMA_VERSION}）"
    )
