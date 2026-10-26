"""Backend-neutral render job identity and polling observations."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from kd1_anime.cluster.resource_estimator import RenderResourceProfile
from kd1_anime.rendering import RenderProfile, VideoMetadata


@dataclass
class RenderJob:
    job_id: str
    scene_id: int
    script_path: Path
    log_out: Path
    log_err: Path
    media_dir: Path
    scene_class_name: str
    submitted_at: float
    # Slurm 实际开始运行时间；可由 squeue 恢复，旧清单没有该字段时为空。
    started_at: float | None = None
    execution_sha256: str = ""
    code_sha256: str = ""
    render_profile: RenderProfile = field(default_factory=RenderProfile.current)
    resource_profile: RenderResourceProfile | None = None
    output_path: Path | None = None
    output_metadata: VideoMetadata | None = None
    output_sha256: str = ""
    elapsed_seconds: float | None = None
    status: str = "PENDING"
    failure_reason: str = ""
    cancelled: bool = False
    environment_fingerprint: dict[str, str] = field(default_factory=dict)
    environment_warning: str = ""
    backend: Literal["slurm", "local"] = "slurm"


@dataclass(frozen=True, slots=True)
class PollObservation:
    """一次调度器查询的不可变快照。

    不把启动时间和诊断信息挂在 Dispatcher 的共享可变属性上，避免并行
    Scene 轮询时互相覆盖结果。
    """

    statuses: dict[str, str]
    start_times: dict[str, float]
    diagnostic: str = ""
    observed_at: float = field(default_factory=time.time)
