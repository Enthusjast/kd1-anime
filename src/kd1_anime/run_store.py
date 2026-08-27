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
    ExtractedElement,
    SceneOutline,
    ScenePlan,
)
from kd1_anime.cluster.slurm import SlurmJob
from kd1_anime.config import resolve_runtime_path
from kd1_anime.rag.models import RagReceipt, RagRuntimeProfile
from kd1_anime.rendering import (
    RenderProfile,
    SceneArtifact,
    VideoMetadata,
    sha256_file,
    verify_video,
)

MANIFEST_NAME = "manifest.json"
MANIFEST_SCHEMA_VERSION = 3
RUN_ID_PATTERN = re.compile(r"\d{8}-\d{6}-[0-9a-f]{8}")
RESUME_LLM_STATES = frozenset(
    {
        "INIT",
        "PLANNING",
        "DETAILING",
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

    schema_version: Literal[3] = MANIFEST_SCHEMA_VERSION
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
    output_path: str
    render_profile: RenderProfile = Field(default_factory=RenderProfile.current)
    outlines: list[SceneOutline] = Field(default_factory=list)
    scenes: dict[int, StoredSceneState] = Field(default_factory=dict)
    # 新运行在分镜生成前固定全片规范；旧 manifest 缺少这些字段时按已完成兼容，
    # 不会在恢复已有代码/作业时擅自重规划场景。
    continuity_bible: ContinuityBible | None = None
    continuity_review_status: Literal["pending", "reviewing", "passed", "warning"] = "passed"
    continuity_review_round: int = Field(default=0, ge=0)
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
        必须有经过验证的 artifact。单独提供方法是为了让 v1 迁移先完成，
        再检查当前 schema，避免把可安全迁移的旧清单提前判死。
        """

        errors: list[str] = []
        for scene_id, scene in self.scenes.items():
            if scene.plan.scene_id != scene_id:
                errors.append(f"Scene key {scene_id} 与 plan.scene_id {scene.plan.scene_id} 不一致")
            if scene.rendered and scene.artifact is None:
                errors.append(f"Scene {scene_id} 标记为 rendered 但缺少 artifact")
            if scene.artifact is not None:
                artifact = scene.artifact
                if not scene.rendered:
                    errors.append(f"Scene {scene_id} 存在 artifact 但 rendered=false")
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
        if self.status == "completed" and not self.final_video:
            errors.append("运行标记为 completed 但缺少 final_video")
        return errors


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
            if "partial_movie_files" in path.parts:
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


def _legacy_phase(scene: dict) -> str:
    if scene.get("rendered"):
        return "rendered"
    if scene.get("give_up") or scene.get("failed"):
        return "failed"
    if scene.get("slurm_job"):
        return "monitoring"
    if scene.get("reviewed"):
        return "reviewed"
    if scene.get("code_file"):
        return "coded"
    if scene.get("plan_ready"):
        return "detailed"
    return "pending"


def _legacy_plan_ready(scene: dict) -> bool:
    """从 v1 清单推导分镜是否已完成，避免恢复时重复调用 Planner。"""

    if scene.get("plan_ready") is True:
        return True
    if any(
        scene.get(key) for key in ("rendered", "reviewed", "code_file", "code_sha256", "class_name")
    ):
        return True
    plan = scene.get("plan")
    if not isinstance(plan, dict):
        return False
    # 早期清单可能只保存概要占位计划；只有关键详细字段都存在且不是
    # 占位省略号时才认为 plan_detail 已完成。
    detail_values = [
        plan.get("visual_design"),
        plan.get("camera_movement"),
        plan.get("visual_flow"),
        plan.get("key_moments"),
        plan.get("computation"),
    ]
    return all(value and value != "…" and value != ["…"] for value in detail_values)


def _migrate_legacy_artifact(
    *,
    root: Path,
    run_id: str,
    scene_id: int,
    scene: dict,
    profile: RenderProfile,
) -> SceneArtifact | None:
    job = scene.get("slurm_job")
    code_hash = scene.get("code_sha256", "")
    class_name = scene.get("class_name", "")
    if not scene.get("rendered") or not job or not code_hash or not class_name:
        return None
    if not str(job.get("job_id", "")).isdigit():
        return None
    try:
        media_dir = restore_run_path(root, job["media_dir"])
        video = _latest_video_candidate(media_dir, class_name)
        if video is None:
            return None
        metadata = verify_video(video, profile)
        return SceneArtifact(
            origin="rendered",
            source_run_id=run_id,
            job_id=str(job["job_id"]),
            scene_id=scene_id,
            scene_class_name=class_name,
            code_sha256=code_hash,
            render_profile_sha256=profile.digest(),
            video_path=_run_relative(root, video),
            video_sha256=sha256_file(video),
            metadata=metadata,
            verified=True,
        )
    except (KeyError, OSError, RuntimeError, ValueError):
        return None


def migrate_manifest_data(raw: dict, root: Path) -> dict:
    """把旧清单升级为当前 schema；未来版本明确拒绝降级读取。"""

    version = raw.get("schema_version", 1)
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValueError(f"manifest schema_version 必须是整数: {version!r}")
    if version == MANIFEST_SCHEMA_VERSION:
        return raw
    if version == 2:
        data = dict(raw)
        data["schema_version"] = MANIFEST_SCHEMA_VERSION
        data.setdefault("visual_eval_profile", VisualEvalProfile().model_dump(mode="json"))
        scenes = data.get("scenes", {})
        if not isinstance(scenes, dict):
            raise ValueError("旧版运行清单的 scenes 必须是对象")
        migrated_scenes: dict[str, dict] = {}
        for raw_scene_id, raw_scene in scenes.items():
            if not isinstance(raw_scene, dict):
                raise ValueError(f"旧版运行清单 Scene {raw_scene_id} 必须是对象")
            scene = dict(raw_scene)
            scene.setdefault("visual_status", "skipped")
            migrated_scenes[str(raw_scene_id)] = scene
        data["scenes"] = migrated_scenes
        return data
    if version != 1:
        raise ValueError(f"不支持的 manifest schema_version: {version}")

    data = dict(raw)
    if not isinstance(data.get("run_id"), str):
        raise ValueError("旧版运行清单缺少有效 run_id")
    raw_scenes = data.get("scenes", {})
    if not isinstance(raw_scenes, dict):
        raise ValueError("旧版运行清单的 scenes 必须是对象")
    profile = RenderProfile.current()
    data["schema_version"] = MANIFEST_SCHEMA_VERSION
    data["revision"] = 0
    data["render_profile"] = profile.model_dump(mode="json")
    migrated_scenes: dict[str, dict] = {}
    for raw_scene_id, raw_scene in raw_scenes.items():
        try:
            scene_id = int(raw_scene_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"旧版运行清单包含无效场景 ID: {raw_scene_id!r}") from exc
        if not isinstance(raw_scene, dict):
            raise ValueError(f"旧版运行清单 Scene {scene_id} 必须是对象")
        scene = dict(raw_scene)
        scene["plan_ready"] = _legacy_plan_ready(scene)
        scene["phase"] = _legacy_phase(scene)
        if scene["plan_ready"] and scene["phase"] == "pending":
            scene["phase"] = "detailed"
        scene["artifact"] = None
        job = scene.get("slurm_job")
        if job is not None and not isinstance(job, dict):
            raise ValueError(f"旧版运行清单 Scene {scene_id} 的 slurm_job 必须是对象")
        if job and str(job.get("job_id", "")).startswith("reused-"):
            scene["slurm_job"] = None
            scene["rendered"] = False
            scene["phase"] = "reviewed" if scene.get("reviewed") else "coded"
            scene["failure_reason"] = "旧版复用记录无法安全验证，将重新渲染"
        elif job:
            migrated_job = dict(job)
            migrated_job["code_sha256"] = scene.get("code_sha256", "")
            migrated_job["render_profile"] = profile.model_dump(mode="json")
            migrated_job["output_path"] = None
            migrated_job["output_metadata"] = None
            migrated_job["elapsed_seconds"] = None
            scene["slurm_job"] = migrated_job
            artifact = _migrate_legacy_artifact(
                root=root,
                run_id=data["run_id"],
                scene_id=scene_id,
                scene=scene,
                profile=profile,
            )
            if artifact:
                scene["artifact"] = artifact.model_dump(mode="json")
                migrated_job["output_path"] = artifact.video_path
                migrated_job["output_metadata"] = artifact.metadata.model_dump(mode="json")
                migrated_job["output_sha256"] = artifact.video_sha256
            elif scene.get("rendered"):
                scene["rendered"] = False
                scene["phase"] = "reviewed"
                scene["failure_reason"] = "旧版渲染产物无法按当前配置验证，将重新渲染"
        elif scene.get("rendered"):
            # v1 中 rendered=True 并不保证存在真实 Slurm 作业或最终视频。
            # 没有可核验作业记录时不能制造 artifact，保守地重新渲染。
            scene["rendered"] = False
            scene["phase"] = "reviewed" if scene.get("reviewed") else "coded"
            scene["failure_reason"] = "旧版场景缺少可验证的渲染作业，将重新渲染"
        migrated_scenes[str(scene_id)] = scene
    data["scenes"] = migrated_scenes
    return data
