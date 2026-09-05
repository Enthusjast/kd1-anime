"""渲染后端组合服务。"""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

from kd1_anime.cluster.render_backend import RenderBackend, RenderBackendName
from kd1_anime.cluster.slurm import SlurmJob, SlurmMonitorCoordinator
from kd1_anime.config import settings
from kd1_anime.rendering import RenderProfile


class RenderingService:
    """为 FSM 提供统一的后端操作和环境预检。"""

    def __init__(self, backend: RenderBackend, name: RenderBackendName) -> None:
        self.backend = backend
        self.name = name

    def set_backend(self, backend: RenderBackend, name: RenderBackendName) -> None:
        self.backend = backend
        self.name = name

    def preflight(self, profile: RenderProfile) -> None:
        if self.name == "local":
            missing = [name for name in ("ffmpeg", "ffprobe") if not shutil.which(name)]
            try:
                manim_available = importlib.util.find_spec("manim") is not None
            except (ImportError, ModuleNotFoundError, ValueError):
                manim_available = False
            if not manim_available:
                missing.append("manim")
            if missing:
                raise RuntimeError(
                    "本地渲染环境缺少命令/模块: " + ", ".join(dict.fromkeys(missing))
                )
            return

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

    def submit_scene(self, *args, **kwargs) -> SlurmJob:
        return self.backend.submit_scene(*args, **kwargs)

    def cancel_job(self, job_id: str) -> bool:
        return self.backend.cancel_job(job_id)

    def get_error_log(self, *args, **kwargs) -> str | None:
        return self.backend.get_error_log(*args, **kwargs)

    def classify_gone(self, job: SlurmJob) -> str | None:
        classifier = getattr(self.backend, "_classify_gone", None)
        if not callable(classifier):
            return None
        return classifier(job)

    def monitor(
        self,
        *,
        run_timeout: int | None = None,
        on_job_update=None,
    ) -> SlurmMonitorCoordinator:
        return SlurmMonitorCoordinator(
            self.backend,
            run_timeout=run_timeout,
            on_job_update=on_job_update,
        )


__all__ = ["RenderingService"]
