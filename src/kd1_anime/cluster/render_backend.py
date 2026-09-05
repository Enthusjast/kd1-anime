"""统一的场景渲染后端。

渲染生命周期只依赖这里的最小协议：提交、轮询、验证、取消和读取日志。
Slurm 与本地前台进程因此可以复用同一套 Orchestrator/JobMonitor 逻辑，
而不会让本地模式绕过产物哈希、ffprobe 或恢复检查。
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from contextlib import suppress
from pathlib import Path
from typing import Literal, Protocol
from uuid import uuid4

from kd1_anime.cluster.resource_estimator import RenderResourceProfile
from kd1_anime.cluster.slurm import (
    SlurmDispatcher,
    SlurmJob,
    SlurmPollSnapshot,
    _dashboard_quiet,
)
from kd1_anime.config import settings
from kd1_anime.rendering import RenderProfile

RenderBackendName = Literal["slurm", "local"]
# 兼容旧代码的名字；作业字段已经被抽象为可由任一后端承载的 RenderJob。
RenderJob = SlurmJob


class RenderBackend(Protocol):
    """Orchestrator 使用的渲染后端最小接口。"""

    name: RenderBackendName

    def submit_scene(
        self,
        scene_id: int,
        python_file: Path,
        scene_class_name: str = "Scene",
        *,
        scenes_dir: Path | None = None,
        logs_dir: Path | None = None,
        videos_dir: Path | None = None,
        code_sha256: str = "",
        render_profile: RenderProfile | None = None,
        resource_profile: RenderResourceProfile | None = None,
    ) -> RenderJob: ...

    def poll_all_statuses(self, job_ids: list[str]) -> dict[str, str]: ...

    def validate_completed_job(self, job: RenderJob) -> bool: ...

    def cancel_job(self, job_id: str) -> bool: ...

    def get_error_log(
        self,
        scene_id: int | None = None,
        job_id: str | None = None,
        *,
        job: RenderJob | None = None,
    ) -> str | None: ...


class SlurmRenderBackend(SlurmDispatcher):
    """现有 Slurm 调度器的统一后端适配名。"""

    name: RenderBackendName = "slurm"


class LocalRenderBackend(SlurmDispatcher):
    """在当前主机前台启动 Manim 的正式渲染后端。

    每次尝试都有独立 media/log 目录；进程使用新的 session，因此取消时
    可以清理 Manim 及其 ffmpeg 子进程。进程句柄只保存在当前 Python 进程
    内，manifest 不保存 PID，恢复时不会错误认领旧本地进程。
    """

    name: RenderBackendName = "local"

    def __init__(self) -> None:
        super().__init__()
        self._processes: dict[str, subprocess.Popen[str]] = {}
        self._log_handles: dict[str, tuple[object, object]] = {}
        self._jobs: dict[str, SlurmJob] = {}
        self._process_lock = threading.RLock()

    @staticmethod
    def _local_job_id() -> str:
        return f"local-{uuid4().hex[:12]}"

    @staticmethod
    def _build_command(
        python_file: Path,
        scene_class_name: str,
        media_dir: Path,
        profile: RenderProfile,
    ) -> list[str]:
        args = [
            "-m",
            "manim",
            "render",
            f"--renderer={profile.renderer}",
            f"-q{profile.quality}",
            "--resolution",
            f"{profile.pixel_width},{profile.pixel_height}",
            "--fps",
            str(profile.frame_rate),
            "--media_dir",
            str(media_dir),
            str(python_file),
            scene_class_name,
        ]
        if profile.renderer == "opengl":
            # Manim 0.20 的 OpenGL 路径只有显式写入电影时才可靠地产出 MP4。
            args.insert(-2, "--write_to_movie")
        return [sys.executable, *args]

    @staticmethod
    def _validate_dirs(
        scenes_dir: Path | None,
        logs_dir: Path | None,
        videos_dir: Path | None,
    ) -> tuple[Path, Path, Path]:
        resolved = tuple(
            Path(item or default).expanduser().resolve()
            for item, default in (
                (scenes_dir, settings.SCENES_DIR),
                (logs_dir, settings.LOGS_DIR),
                (videos_dir, settings.VIDEOS_DIR),
            )
        )
        for directory in resolved:
            if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
                raise RuntimeError(f"本地渲染目录不是可信的真实目录: {directory}")
            directory.mkdir(parents=True, exist_ok=True)
            directory.chmod(0o700)
        return resolved  # type: ignore[return-value]

    @staticmethod
    def _open_log(path: Path):
        if path.is_symlink():
            raise RuntimeError(f"本地渲染日志不能是符号链接: {path}")
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        os.fchmod(descriptor, 0o600)
        return os.fdopen(descriptor, "w", encoding="utf-8")

    def submit_scene(
        self,
        scene_id: int,
        python_file: Path,
        scene_class_name: str = "Scene",
        *,
        scenes_dir: Path | None = None,
        logs_dir: Path | None = None,
        videos_dir: Path | None = None,
        code_sha256: str = "",
        render_profile: RenderProfile | None = None,
        resource_profile: RenderResourceProfile | None = None,
    ) -> SlurmJob:
        scene_id = self._validate_scene_id(scene_id)
        scene_class_name = self._validate_scene_class_name(scene_class_name)
        source = Path(python_file).expanduser().resolve()
        raw_source = Path(python_file).expanduser()
        if raw_source.is_symlink() or not source.is_file():
            raise RuntimeError(f"本地渲染场景代码不存在或不安全: {source}")
        _scenes_root, logs_root, videos_root = self._validate_dirs(scenes_dir, logs_dir, videos_dir)
        profile = render_profile or RenderProfile.current()
        attempt = uuid4().hex[:12]
        job_id = self._local_job_id()
        media_dir = videos_root / f"scene_{scene_id}" / f"attempt_{attempt}"
        if media_dir.is_symlink():
            raise RuntimeError(f"本地渲染媒体目录不能是符号链接: {media_dir}")
        media_dir.mkdir(parents=True, exist_ok=True)
        media_dir.chmod(0o700)
        log_out = logs_root / f"scene_{scene_id}_{job_id}.out"
        log_err = logs_root / f"scene_{scene_id}_{job_id}.err"
        command = self._build_command(source, scene_class_name, media_dir, profile)
        prlimit = shutil.which("prlimit")
        if prlimit:
            command = [
                prlimit,
                f"--as={settings.LOCAL_RENDER_MEMORY_MB * 1024 * 1024}",
                f"--cpu={settings.LOCAL_RENDER_TIMEOUT + 5}",
                "--",
                *command,
            ]
        env = os.environ.copy()
        env["MANIM_RENDERER"] = profile.renderer
        if profile.renderer == "opengl":
            env["PYOPENGL_PLATFORM"] = profile.opengl_platform
        # 不让本地调用继承一个指向不同项目/用户的媒体配置。
        env.pop("MANIM_MEDIA_DIR", None)
        out_handle = None
        err_handle = None
        try:
            out_handle = self._open_log(log_out)
            err_handle = self._open_log(log_err)
            process = subprocess.Popen(
                command,
                cwd=source.parent,
                env=env,
                stdout=out_handle,
                stderr=err_handle,
                text=True,
                start_new_session=True,
            )
        except (OSError, RuntimeError):
            with suppress(OSError):
                if out_handle is not None:
                    out_handle.close()
            with suppress(OSError):
                if err_handle is not None:
                    err_handle.close()
            raise
        submitted_at = time.time()
        job = SlurmJob(
            job_id=job_id,
            scene_id=scene_id,
            script_path=source,
            log_out=log_out,
            log_err=log_err,
            media_dir=media_dir,
            scene_class_name=scene_class_name,
            submitted_at=submitted_at,
            code_sha256=code_sha256,
            render_profile=profile,
            resource_profile=resource_profile,
            backend="local",
        )
        with self._process_lock:
            self._processes[job_id] = process
            self._log_handles[job_id] = (out_handle, err_handle)
            self._jobs[job_id] = job
        if not _dashboard_quiet():
            print(f"[Local] 已启动渲染进程, Job ID: {job_id}")
        return job

    def _close_handles(self, job_id: str) -> None:
        handles = self._log_handles.pop(job_id, ())
        for handle in handles:
            with suppress(OSError, AttributeError):
                handle.close()  # type: ignore[union-attr]

    def poll_all_statuses_snapshot(self, job_ids: list[str]) -> SlurmPollSnapshot:
        statuses: dict[str, str] = {}
        start_times: dict[str, float] = {}
        diagnostics: list[str] = []
        with self._process_lock:
            for job_id in job_ids:
                process = self._processes.get(job_id)
                job = self._jobs.get(job_id)
                if process is None or job is None:
                    # 只有当前进程创建的句柄可被认领；恢复时由 Orchestrator
                    # 在进入这里前清除旧 local Job 并重新提交。
                    statuses[job_id] = "GONE"
                    diagnostics.append(f"本地 Job {job_id} 不属于当前进程")
                    continue
                return_code = process.poll()
                if return_code is None:
                    statuses[job_id] = "RUNNING"
                    start_times[job_id] = job.submitted_at
                elif return_code == 0:
                    statuses[job_id] = "COMPLETED"
                    self._close_handles(job_id)
                else:
                    statuses[job_id] = "CANCELLED" if return_code < 0 else "FAILED"
                    self._close_handles(job_id)
        snapshot = SlurmPollSnapshot(statuses, start_times, "；".join(diagnostics))
        self.last_start_times = dict(start_times)
        self.last_status_diagnostic = snapshot.diagnostic
        return snapshot

    def poll_all_statuses(self, job_ids: list[str]) -> dict[str, str]:
        return dict(self.poll_all_statuses_snapshot(job_ids).statuses)

    def cancel_job(self, job_id: str) -> bool:
        with self._process_lock:
            process = self._processes.get(job_id)
        if process is None:
            # 当前进程没有这个句柄时不假装重新获取它；它可能是恢复出来的
            # 旧作业，调用方会把它作为不可认领任务重新排队。
            return True
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except (OSError, ProcessLookupError):
                return False
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                with suppress(OSError, ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                with suppress(OSError):
                    process.wait(timeout=5)
        self._close_handles(job_id)
        return True

    def close(self) -> None:
        """结束当前进程创建的全部本地任务。"""

        with self._process_lock:
            job_ids = list(self._processes)
        for job_id in job_ids:
            self.cancel_job(job_id)

    def get_error_log(self, *args, **kwargs) -> str | None:
        return super().get_error_log(*args, **kwargs)


def create_render_backend(name: str | None = None) -> RenderBackend:
    """创建后端并严格拒绝未知名称。"""

    selected = name or settings.RENDER_BACKEND
    if selected == "slurm":
        return SlurmRenderBackend()
    if selected == "local":
        return LocalRenderBackend()
    raise ValueError(f"不支持的渲染后端: {selected!r}")


__all__ = [
    "LocalRenderBackend",
    "RenderBackend",
    "RenderBackendName",
    "RenderJob",
    "SlurmRenderBackend",
    "create_render_backend",
]
