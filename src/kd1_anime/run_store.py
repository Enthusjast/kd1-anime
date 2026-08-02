"""持久化运行状态，支持查询和安全恢复中断的流水线。"""

from __future__ import annotations

import fcntl
import hashlib
import os
import re
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from kd1_anime.agents.planner import SceneOutline, ScenePlan
from kd1_anime.cluster.slurm import SlurmJob
from kd1_anime.config import resolve_runtime_path

MANIFEST_NAME = "manifest.json"
MANIFEST_SCHEMA_VERSION = 1
RUN_ID_PATTERN = re.compile(r"\d{8}-\d{6}-[0-9a-f]{8}")
RunStatus = Literal["running", "interrupted", "failed", "completed", "dry_run_complete"]


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
    status: str = Field(min_length=1, max_length=100)
    failure_reason: str = Field(default="", max_length=50_000)
    cancelled: bool = False


class StoredSceneState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan: ScenePlan
    code_file: str = ""
    code_sha256: str = Field(default="", pattern=r"^(?:[0-9a-f]{64})?$")
    class_name: str = Field(default="", max_length=200)
    review_round: int = Field(default=0, ge=0)
    fix_attempts: int = Field(default=0, ge=0)
    slurm_job: StoredSlurmJob | None = None
    rendered: bool = False
    give_up: bool = False
    failed: bool = False
    failure_reason: str = Field(default="", max_length=50_000)


class RunManifest(BaseModel):
    """版本化运行清单。未知字段会被拒绝，防止静默误读未来格式。"""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    schema_version: Literal[1] = MANIFEST_SCHEMA_VERSION
    run_id: str
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    status: RunStatus = "running"
    state: str = "INIT"
    user_prompt: str = Field(max_length=1_000_000)
    dry_run: bool = False
    interactive: bool = False
    auto_fix: bool = True
    output_path: str
    outlines: list[SceneOutline] = Field(default_factory=list)
    scenes: dict[int, StoredSceneState] = Field(default_factory=dict)
    final_video: str | None = None
    error: str = Field(default="", max_length=50_000)

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
        status=stored.status,
        failure_reason=stored.failure_reason,
        cancelled=stored.cancelled,
    )


def write_manifest(path: Path, manifest: RunManifest) -> None:
    """以同目录临时文件 + os.replace 原子更新清单。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.updated_at = utc_now()
    payload = manifest.model_dump_json(indent=2) + "\n"
    fd, temporary_name = tempfile.mkstemp(prefix=".manifest-", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        path.chmod(0o600)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


class RunRepository:
    """在配置的 workspace 下定位运行，且不接受任意路径。"""

    def __init__(self, workspace_dir: Path) -> None:
        self.runs_root = resolve_runtime_path(workspace_dir) / "runs"

    def run_root(self, run_id: str) -> Path:
        if not RUN_ID_PATTERN.fullmatch(run_id):
            raise ValueError("run_id 格式无效")
        candidate = self.runs_root / run_id
        if candidate.exists() and (
            candidate.is_symlink() or candidate.resolve().parent != self.runs_root
        ):
            raise ValueError("运行目录不是 workspace/runs 下的真实目录")
        return candidate

    def manifest_path(self, run_id: str) -> Path:
        return self.run_root(run_id) / MANIFEST_NAME

    def load(self, run_id: str) -> RunManifest:
        path = self.manifest_path(run_id)
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"找不到运行清单: {run_id}")
        manifest = RunManifest.model_validate_json(path.read_text(encoding="utf-8"))
        if manifest.run_id != run_id:
            raise ValueError("运行目录与 manifest.run_id 不一致")
        return manifest

    def list(self) -> list[RunManifest]:
        if not self.runs_root.is_dir():
            return []
        manifests: list[RunManifest] = []
        for child in self.runs_root.iterdir():
            if not child.is_dir() or child.is_symlink() or not RUN_ID_PATTERN.fullmatch(child.name):
                continue
            try:
                manifests.append(self.load(child.name))
            except (OSError, ValueError):
                continue
        return sorted(manifests, key=lambda item: item.updated_at, reverse=True)


@contextmanager
def lock_run(root: Path) -> Iterator[None]:
    """持有进程级排他锁，防止两个 resume 实例操作同一批 Slurm 作业。"""

    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".run.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
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
