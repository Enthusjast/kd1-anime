"""根据场景复杂度生成保守的 Slurm 资源建议。"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator

from kd1_anime.agents.planner import ScenePlan
from kd1_anime.agents.risk import assess_scene_risk
from kd1_anime.agents.technical_planner import TechnicalSpec
from kd1_anime.rendering import RenderProfile


def _parse_time_limit(value: str) -> int:
    match = re.fullmatch(r"(?:(\d+)-)?(\d{1,3}):(\d{2}):(\d{2})", str(value).strip())
    if match is None or int(match.group(3)) >= 60 or int(match.group(4)) >= 60:
        raise ValueError("time_limit 必须使用 [days-]HH:MM:SS 格式")
    return (
        (int(match.group(1) or 0) * 24 + int(match.group(2))) * 60 + int(match.group(3))
    ) * 60 + int(match.group(4))


def _format_time_limit(seconds: int) -> str:
    seconds = max(60, int(seconds))
    days, remainder = divmod(seconds, 86_400)
    hours, remainder = divmod(remainder, 3_600)
    minutes, seconds = divmod(remainder, 60)
    prefix = f"{days}-" if days else ""
    return f"{prefix}{hours:02d}:{minutes:02d}:{seconds:02d}"


class RenderResourceProfile(BaseModel):
    """实际写入渲染脚本并随 Job 持久化的资源配置。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cpus_per_task: int = Field(ge=1, le=1_024)
    mem_gb: int | None = Field(default=None, ge=1, le=1_024)
    time_limit: str = Field(pattern=r"^(?:\d+-)?\d{1,3}:\d{2}:\d{2}$")
    gpu_type: str = Field(default="", max_length=100)
    gpu_count: int = Field(default=1, ge=1, le=64)
    estimated: bool = False
    reasons: tuple[str, ...] = ()

    @field_validator("time_limit")
    @classmethod
    def validate_time_limit(cls, value: str) -> str:
        _parse_time_limit(value)
        return value

    @field_validator("gpu_type")
    @classmethod
    def validate_gpu_type(cls, value: str) -> str:
        if value and not re.fullmatch(r"[A-Za-z0-9_.:@,+/-]+", value):
            raise ValueError("gpu_type 不是安全的 Slurm 标识")
        return value

    @classmethod
    def from_settings(cls) -> RenderResourceProfile:
        from kd1_anime.config import settings

        memory = _memory_gb(settings.SLURM_MEM_GB)
        return cls(
            cpus_per_task=settings.SLURM_CPUS_PER_TASK,
            mem_gb=memory,
            time_limit=settings.SLURM_TIME_LIMIT,
            gpu_type=settings.SLURM_GPU_TYPE,
            gpu_count=settings.SLURM_GPU_COUNT,
        )


def _memory_gb(value: str) -> int | None:
    if not value:
        return None
    match = re.fullmatch(r"(\d+)([KMGTP])?", value.strip(), flags=re.IGNORECASE)
    if match is None:
        return None
    amount = int(match.group(1))
    unit = (match.group(2) or "G").upper()
    factors = {"K": 1 / (1024**2), "M": 1 / 1024, "G": 1, "T": 1024, "P": 1024**2}
    return max(1, int(amount * factors[unit]))


def estimate_render_resources(
    scene_plan: ScenePlan,
    technical_spec: TechnicalSpec | None,
    render_profile: RenderProfile,
    *,
    cpus_per_task: int,
    mem_gb: str = "",
    time_limit: str = "01:00:00",
    gpu_type: str = "",
    gpu_count: int = 1,
    apply_estimate: bool = False,
) -> RenderResourceProfile:
    """返回可直接用于脚本的资源配置。

    默认保留用户配置；``apply_estimate=True`` 时只向上增加资源，不会
    降低用户给出的 CPU、内存、时间或 GPU 数量。Cairo 永远不申请 GPU。
    """

    risk = assess_scene_risk(scene_plan, technical_spec)
    reasons = list(risk.reasons)
    base_cpus = max(1, int(cpus_per_task))
    base_memory = _memory_gb(mem_gb)
    base_seconds = _parse_time_limit(time_limit)
    cpus = base_cpus
    memory = base_memory
    seconds = base_seconds
    if apply_estimate:
        if risk.level == "medium":
            cpus = max(cpus, base_cpus + 1)
            seconds = max(seconds, base_seconds + 15 * 60)
        elif risk.level == "high":
            cpus = max(cpus, base_cpus + 2)
            seconds = max(seconds, base_seconds + 30 * 60)
        if technical_spec is not None and len(technical_spec.objects) >= 8:
            memory = max(memory or 0, (memory or 4) + 2)
        if risk.level == "high" and memory is not None:
            memory = max(memory, memory + 4)
    use_gpu = render_profile.renderer == "opengl"
    return RenderResourceProfile(
        cpus_per_task=cpus,
        mem_gb=memory,
        time_limit=_format_time_limit(seconds),
        gpu_type=gpu_type if use_gpu else "",
        gpu_count=max(1, int(gpu_count)) if use_gpu else 1,
        estimated=bool(apply_estimate),
        reasons=tuple(reasons),
    )


__all__ = ["RenderResourceProfile", "estimate_render_resources"]
