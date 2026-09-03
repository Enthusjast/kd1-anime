"""Slurm 调度、批量监控和渲染脚本生成。"""

from __future__ import annotations

import getpass
import json
import re
import shlex
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from rich.console import Console

from kd1_anime.config import settings
from kd1_anime.rendering import RenderProfile, VideoMetadata, sha256_file, verify_video

console = Console()


def _dashboard_quiet() -> bool:
    """Live 仪表盘激活时抑制普通输出, 避免破坏 Rich Live 渲染。"""
    try:
        from kd1_anime.dashboard import quiet

        return quiet()
    except Exception:
        return False


TERMINAL_STATES = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "BOOT_FAIL",
    "PREEMPTED",
    "DEADLINE",
    "REVOKED",
    "SPECIAL_EXIT",
}
FAILURE_STATES = TERMINAL_STATES - {"COMPLETED"}
RUNNING_STATES = {"RUNNING", "COMPLETING", "STAGE_OUT"}
QUEUED_STATES = {"PENDING", "CONFIGURING", "REQUEUED", "SUSPENDED"}
MONITOR_ABORT_STATES = {
    "QUEUE_TIMEOUT",
    "RUN_TIMEOUT",
    "UNKNOWN_TIMEOUT",
    "MONITOR_QUERY_FAILED",
    "CANCEL_FAILED",
}


@dataclass
class SlurmJob:
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
    code_sha256: str = ""
    render_profile: RenderProfile = field(default_factory=RenderProfile.current)
    output_path: Path | None = None
    output_metadata: VideoMetadata | None = None
    output_sha256: str = ""
    elapsed_seconds: float | None = None
    status: str = "PENDING"
    failure_reason: str = ""
    cancelled: bool = False
    environment_fingerprint: dict[str, str] = field(default_factory=dict)
    environment_warning: str = ""


@dataclass(frozen=True, slots=True)
class SlurmPollSnapshot:
    """一次调度器查询的不可变快照。

    不把启动时间和诊断信息挂在 Dispatcher 的共享可变属性上，避免并行
    Scene 轮询时互相覆盖结果。
    """

    statuses: dict[str, str]
    start_times: dict[str, float]
    diagnostic: str = ""
    observed_at: float = field(default_factory=time.time)


class JobMonitor:
    """可增量推进的 Slurm 作业监控器（非阻塞轮询）。

    与 wait_for_all_jobs 不同, 它把"轮询一次"与"等待"解耦:
    - add_job(): 加入新提交的作业 (调度器可以边提交边监控)
    - poll_once(): 推进一次轮询, 返回本轮是否有作业结束
    - pending/results: 查询未结束/已结束作业

    供 orchestrator 的场景级调度器使用, 让渲染与其他场景的 LLM 阶段并行推进。
    """

    def __init__(
        self,
        dispatcher: SlurmDispatcher,
        *,
        queue_timeout: int | None = None,
        run_timeout: int | None = None,
        unknown_timeout: int | None = None,
        artifact_grace: int | None = None,
        max_unknown: int | None = None,
        poll_interval: int | None = None,
        use_legacy_timeout: bool = True,
    ) -> None:
        self.dispatcher = dispatcher
        self.queue_timeout = (
            settings.MONITOR_QUEUE_TIMEOUT if queue_timeout is None else queue_timeout
        )
        self.run_timeout = settings.MONITOR_RUN_TIMEOUT if run_timeout is None else run_timeout
        self.unknown_timeout = (
            settings.MONITOR_UNKNOWN_TIMEOUT if unknown_timeout is None else unknown_timeout
        )
        self.artifact_grace = (
            settings.MONITOR_ARTIFACT_GRACE if artifact_grace is None else artifact_grace
        )
        self.max_unknown = settings.MONITOR_MAX_UNKNOWN if max_unknown is None else max_unknown
        self.poll_interval = (
            settings.MONITOR_POLL_INTERVAL if poll_interval is None else poll_interval
        )
        # 兼容只设置旧 MONITOR_TIMEOUT 的配置；显式修改新的拆分项时以新项为准。
        legacy_timeout = settings.MONITOR_TIMEOUT
        if use_legacy_timeout and legacy_timeout is not None:
            if self.queue_timeout == 3600:
                self.queue_timeout = legacy_timeout
            if self.run_timeout == 3600:
                self.run_timeout = legacy_timeout
            if self.unknown_timeout == 300:
                self.unknown_timeout = legacy_timeout
        self.pending: dict[str, SlurmJob] = {}
        self.results: dict[str, bool] = {}
        self.jobs: dict[str, SlurmJob] = {}
        self.indeterminate_streaks: dict[str, int] = {}
        self.indeterminate_since: dict[str, float] = {}
        self.gone_since: dict[str, float] = {}
        self.completed_since: dict[str, float] = {}
        self.running_since: dict[str, float] = {}
        self.queued_since: dict[str, float] = {}
        self.log_positions: dict[str, int] = {}

    def add_job(self, job: SlurmJob) -> None:
        if job.job_id in self.results:
            return
        self.pending.setdefault(job.job_id, job)
        self.jobs[job.job_id] = job
        self.indeterminate_streaks.setdefault(job.job_id, 0)
        if job.status == "COMPLETED":
            self.completed_since.setdefault(job.job_id, time.time())
        # resume 时清单可能已经记录作业处于 RUNNING；保守地从已知启动/提交
        # 时间计时，避免恢复后把一项运行很久的作业重新从零开始计时。
        if job.status in RUNNING_STATES:
            self.running_since.setdefault(job.job_id, job.started_at or job.submitted_at)
        elif job.status not in TERMINAL_STATES:
            self.queued_since.setdefault(job.job_id, job.submitted_at)

    def _quiet(self) -> bool:
        """Live 仪表盘激活时抑制 Monitor 文本输出, 避免破坏 Rich Live。"""
        return _dashboard_quiet()

    def poll_once(self) -> bool:
        """单次轮询所有 pending 作业, 更新状态; 返回本轮是否有作业结束。"""
        if not self.pending:
            return False
        now = time.time()
        try:
            snapshot_method = getattr(self.dispatcher, "poll_all_statuses_snapshot", None)
            legacy_method = getattr(self.dispatcher, "poll_all_statuses", None)
            legacy_is_default = (
                getattr(legacy_method, "__func__", None) is SlurmDispatcher.poll_all_statuses
            )
            if callable(snapshot_method) and legacy_is_default:
                snapshot = snapshot_method(list(self.pending))
                statuses = snapshot.statuses
                start_times = snapshot.start_times
                diagnostic = snapshot.diagnostic
            else:
                statuses = self.dispatcher.poll_all_statuses(list(self.pending))
                start_times = getattr(self.dispatcher, "last_start_times", {})
                diagnostic = getattr(self.dispatcher, "last_status_diagnostic", "")
        except Exception as exc:
            # 查询命令的瞬时异常不能等同于远端作业失败，更不能立即取消
            # 所有健康作业。把本轮当作 UNKNOWN，沿用统一的次数/时间宽限；
            # 若下一轮恢复为 RUNNING/PENDING，连续未知计数会自动清零。
            diagnostic = f"Slurm 状态查询异常: {exc}"
            quiet = self._quiet()
            finished: list[str] = []
            for job_id, job in self.pending.items():
                job.status = "UNKNOWN"
                self.indeterminate_streaks[job_id] += 1
                unknown_since = self.indeterminate_since.setdefault(job_id, now)
                if (
                    self.indeterminate_streaks[job_id] >= self.max_unknown
                    and now - unknown_since >= self.unknown_timeout
                ):
                    reason = f"状态连续未知，已停止监控并尝试取消远端任务；查询诊断: {diagnostic}"
                    self.dispatcher._cancel_for_monitor_failure(
                        job,
                        status="UNKNOWN_TIMEOUT",
                        reason=reason,
                    )
                    self.results[job_id] = False
                    finished.append(job_id)
                elif not quiet:
                    console.print(
                        f"[dim][Monitor][/] Scene {job.scene_id}: UNKNOWN "
                        f"({self.indeterminate_streaks[job_id]}/{self.max_unknown})"
                    )
            for job_id in finished:
                self.pending.pop(job_id, None)
            return bool(finished)
        finished: list[str] = []
        quiet = self._quiet()
        for job_id, job in self.pending.items():
            status = statuses.get(job_id, "UNKNOWN")
            job.status = status
            if job_id in start_times:
                job.started_at = start_times[job_id]

            if status == "COMPLETED":
                self.indeterminate_streaks[job_id] = 0
                self.indeterminate_since.pop(job_id, None)
                self.gone_since.pop(job_id, None)
                completed_at = self.completed_since.setdefault(job_id, now)
                job.elapsed_seconds = max(
                    0.0, now - self.running_since.get(job_id, job.submitted_at)
                )
                self.dispatcher._forward_log(job, self.log_positions)
                valid = self.dispatcher.validate_completed_job(job)
                if not valid and now - completed_at < self.artifact_grace:
                    # Slurm 状态已经完成，但共享文件系统上的最终文件可能还未
                    # 传播，或文件仍在由 ffmpeg 关闭。保留 pending，下一轮重验。
                    if not quiet:
                        console.print(
                            f"[dim][Monitor][/] Scene {job.scene_id} 已完成，等待最终 MP4 同步"
                        )
                    continue
                self.results[job_id] = valid
                finished.append(job_id)
                if not valid:
                    job.status = "FAILED"
                if not quiet and valid:
                    console.print(f"[bold green][Monitor][/] Scene {job.scene_id} 渲染成功")
                elif not quiet:
                    console.print(
                        f"[bold red][Monitor][/] Scene {job.scene_id} 作业结束但产物无效: "
                        f"{job.failure_reason}",
                        markup=False,
                    )
            elif status in FAILURE_STATES:
                self.indeterminate_streaks[job_id] = 0
                self.gone_since.pop(job_id, None)
                self.dispatcher._forward_log(job, self.log_positions)
                job.failure_reason = f"Slurm 状态: {status}"
                self.results[job_id] = False
                finished.append(job_id)
                if not quiet:
                    console.print(f"[bold red][Monitor][/] Scene {job.scene_id} 渲染失败: {status}")
            elif status in ("UNKNOWN", "GONE"):
                if status == "GONE":
                    # 作业已确认不在调度器: 依据产物判定结果, 避免秒退/刚结束的作业被误杀。
                    gone_at = self.gone_since.setdefault(job_id, now)
                    outcome = self.dispatcher._classify_gone(job)
                    if outcome == "COMPLETED":
                        self.indeterminate_streaks[job_id] = 0
                        self.gone_since.pop(job_id, None)
                        job.status = "COMPLETED"
                        job.elapsed_seconds = max(0.0, now - job.submitted_at)
                        self.dispatcher._forward_log(job, self.log_positions)
                        self.results[job_id] = True
                        finished.append(job_id)
                        if not quiet:
                            console.print(f"[bold green][Monitor][/] Scene {job.scene_id} 渲染成功")
                        continue
                    if outcome == "FAILED":
                        self.indeterminate_streaks[job_id] = 0
                        self.gone_since.pop(job_id, None)
                        job.status = "FAILED"
                        self.dispatcher._forward_log(job, self.log_positions)
                        self.results[job_id] = False
                        finished.append(job_id)
                        if not quiet:
                            console.print(
                                f"[bold red][Monitor][/] Scene {job.scene_id} 渲染失败（作业已消失，依据日志判定）"
                            )
                        continue
                    # 作业消失与 COMPLETED 一样可能只是共享文件系统尚未同步。
                    # 不能依赖 completed_since：sacct 尚未返回终态时该字段为空，
                    # 旧逻辑会在 max_unknown 次轮询后过早判死。
                    if now - gone_at < self.artifact_grace:
                        if not quiet:
                            console.print(
                                f"[dim][Monitor][/] Scene {job.scene_id} 已结束，等待最终 MP4 同步"
                            )
                        continue
                else:
                    self.gone_since.pop(job_id, None)
                self.indeterminate_streaks[job_id] += 1
                self.indeterminate_since.setdefault(job_id, now)
                if self.indeterminate_streaks[job_id] >= self.max_unknown:
                    if status == "GONE":
                        # 作业消失且无日志 → 按失败交给修复流程, 而非永久判死
                        job.status = "FAILED"
                        job.failure_reason = "作业已从集群消失且无输出日志，无法确认渲染结果"
                        self.results[job_id] = False
                        finished.append(job_id)
                        if not quiet:
                            console.print(
                                f"[bold red][Monitor][/] Scene {job.scene_id} 渲染失败：作业消失且无输出"
                            )
                    else:
                        unknown_since = self.indeterminate_since[job_id]
                        if now - unknown_since < self.unknown_timeout:
                            if not quiet:
                                console.print(
                                    f"[dim][Monitor][/] Scene {job.scene_id}: UNKNOWN "
                                    f"({int(now - unknown_since)}s/{self.unknown_timeout}s)"
                                )
                            continue
                        reason = "状态连续未知，已停止监控并尝试取消远端任务"
                        if diagnostic:
                            reason += f"；查询诊断: {diagnostic}"
                        self.dispatcher._cancel_for_monitor_failure(
                            job,
                            status="UNKNOWN_TIMEOUT",
                            reason=reason,
                        )
                        self.results[job_id] = False
                        finished.append(job_id)
            else:
                self.indeterminate_streaks[job_id] = 0
                self.indeterminate_since.pop(job_id, None)
                self.gone_since.pop(job_id, None)
                self.completed_since.pop(job_id, None)
                if status in RUNNING_STATES:
                    self.queued_since.pop(job_id, None)
                    self.running_since.setdefault(job_id, job.started_at or now)
                elif job_id in self.running_since:
                    # 作业被抢占退回排队等非运行状态: 停止累计运行时长,
                    # 否则之前累计的 run_timeout 会错误地继续计时
                    self.running_since.pop(job_id, None)
                    self.queued_since[job_id] = now
                elif status in QUEUED_STATES:
                    self.queued_since.setdefault(job_id, now)
                if job_id in self.running_since:
                    self.dispatcher._forward_log(job, self.log_positions)
                    if now - self.running_since[job_id] > self.run_timeout:
                        self.dispatcher._cancel_for_monitor_failure(
                            job,
                            status="RUN_TIMEOUT",
                            reason=f"运行超过 {self.run_timeout} 秒",
                        )
                        self.results[job_id] = False
                        finished.append(job_id)
                        continue
                elif now - self.queued_since.get(job_id, job.submitted_at) > self.queue_timeout:
                    self.dispatcher._cancel_for_monitor_failure(
                        job,
                        status="QUEUE_TIMEOUT",
                        reason=f"排队超过 {self.queue_timeout} 秒",
                    )
                    self.results[job_id] = False
                    finished.append(job_id)
                    continue
                if not quiet:
                    console.print(f"[dim][Monitor][/] Scene {job.scene_id}: {status}")

        for job_id in finished:
            self.pending.pop(job_id, None)
        return bool(finished)


class SlurmDispatcher:
    """Slurm 任务调度器。"""

    def __init__(self) -> None:
        # 保留旧属性供外部集成读取；新的 JobMonitor 使用不可变快照，
        # 不依赖这些共享临时值。
        self.last_start_times: dict[str, float] = {}
        self.last_status_diagnostic: str = ""
        self._poll_lock = threading.RLock()

    @staticmethod
    def _validate_scene_id(scene_id: int) -> int:
        if isinstance(scene_id, bool) or not isinstance(scene_id, int) or scene_id < 1:
            raise ValueError("scene_id 必须是大于 0 的整数")
        return scene_id

    @staticmethod
    def _validate_scene_class_name(scene_class_name: str) -> str:
        if not isinstance(scene_class_name, str) or not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]{0,199}", scene_class_name
        ):
            raise ValueError("Scene 类名必须是合法的 Python 标识符")
        return scene_class_name

    def generate_script(
        self,
        scene_id: int,
        python_file: Path,
        scene_class_name: str = "Scene",
        *,
        scenes_dir: Path | None = None,
        logs_dir: Path | None = None,
        videos_dir: Path | None = None,
        attempt_token: str | None = None,
        render_profile: RenderProfile | None = None,
    ) -> tuple[Path, Path, Path, Path]:
        scene_id = self._validate_scene_id(scene_id)
        scene_class_name = self._validate_scene_class_name(scene_class_name)
        scenes_dir = Path(scenes_dir or settings.SCENES_DIR).resolve()
        logs_dir = Path(logs_dir or settings.LOGS_DIR).resolve()
        videos_dir = Path(videos_dir or settings.VIDEOS_DIR).resolve()
        for directory in (scenes_dir, logs_dir, videos_dir):
            directory.mkdir(parents=True, exist_ok=True)

        script_path = scenes_dir / f"render_{scene_id}.sh"
        log_out = logs_dir / f"scene_{scene_id}_%j.out"
        log_err = logs_dir / f"scene_{scene_id}_%j.err"
        media_dir = videos_dir / f"scene_{scene_id}"
        if attempt_token:
            if not re.fullmatch(r"[0-9a-f]{12}", attempt_token):
                raise ValueError("渲染尝试标识格式无效")
            media_dir /= f"attempt_{attempt_token}"
        media_dir.mkdir(parents=True, exist_ok=True)

        content = self._build_script(
            scene_id=scene_id,
            python_file=python_file.resolve(),
            scene_class_name=scene_class_name,
            media_dir=media_dir,
            log_out=log_out,
            log_err=log_err,
            run_root=scenes_dir.parent,
            render_profile=render_profile,
        )
        script_path.write_text(content, encoding="utf-8")
        script_path.chmod(0o700)
        if not _dashboard_quiet():
            console.print(f"[bold blue][Slurm][/] 已生成渲染脚本: {script_path}")
        return script_path, log_out, log_err, media_dir

    def _build_script(
        self,
        scene_id: int,
        python_file: Path,
        scene_class_name: str,
        media_dir: Path,
        log_out: Path,
        log_err: Path,
        run_root: Path | None = None,
        render_profile: RenderProfile | None = None,
    ) -> str:
        self._validate_scene_id(scene_id)
        self._validate_scene_class_name(scene_class_name)
        for label, value in (
            ("场景代码路径", python_file),
            ("媒体输出路径", media_dir),
            ("标准输出日志路径", log_out),
            ("错误日志路径", log_err),
            ("Scene 类名", scene_class_name),
        ):
            if any(character in str(value) for character in ("\x00", "\r", "\n")):
                raise ValueError(f"{label}必须是单行值")
        profile = render_profile or RenderProfile.current()
        renderer = profile.renderer
        use_gpu = renderer == "opengl"
        if use_gpu and not settings.SLURM_GPU_TYPE:
            raise RuntimeError(
                "MANIM_RENDERER=opengl 时必须配置 SLURM_GPU_TYPE；否则无法保证作业分配到 GPU 节点。"
            )
        if settings.SLURM_REQUIRE_CONTAINER and not settings.SLURM_CONTAINER_IMAGE:
            raise RuntimeError("SLURM_REQUIRE_CONTAINER=true，但未配置 SLURM_CONTAINER_IMAGE。")

        lines = ["#!/bin/bash"]
        if settings.SLURM_QOS:
            lines.append(f"#SBATCH --qos={settings.SLURM_QOS}")
        if settings.SLURM_PARTITION:
            lines.append(f"#SBATCH -p {settings.SLURM_PARTITION}")
        if settings.SLURM_ACCOUNT:
            lines.append(f"#SBATCH --account={settings.SLURM_ACCOUNT}")
        # media_dir 可能包含本次提交专用的 attempt_<token> 子目录，不能再用
        # 固定层级推导 run 根目录；代码文件始终位于 <run>/scenes 下。
        run_root = (run_root or python_file.parent.parent).resolve()
        run_id = run_root.name
        safe_run_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", run_id)[:48]
        job_name = f"kd1-{safe_run_id}-s{scene_id}"[:80]
        lines.extend(
            [
                f"#SBATCH -J {job_name}",
                "#SBATCH -N 1",
                f"#SBATCH --cpus-per-task={settings.SLURM_CPUS_PER_TASK}",
            ]
        )
        # Cairo 不使用 GPU；只为 OpenGL 作业申请 GPU。
        if use_gpu:
            lines.append(f"#SBATCH --gres=gpu:{settings.SLURM_GPU_TYPE}:{settings.SLURM_GPU_COUNT}")
        lines.extend(
            [
                f"#SBATCH -t {settings.SLURM_TIME_LIMIT}",
                f"#SBATCH -o {shlex.quote(str(log_out))}",
                f"#SBATCH -e {shlex.quote(str(log_err))}",
            ]
        )
        if settings.SLURM_MEM_GB:
            lines.append(f"#SBATCH --mem={settings.SLURM_MEM_GB}")

        lines.extend(["", "set -euo pipefail", "umask 077", ""])

        container = settings.SLURM_CONTAINER_IMAGE
        if not container:
            configured_base = (
                shlex.quote(str(settings.SLURM_CONDA_BASE.expanduser()))
                if settings.SLURM_CONDA_BASE
                else ""
            )
            lines.extend(
                [
                    "# 动态定位并激活 conda，不依赖特定集群安装路径",
                    f"CONDA_BASE={configured_base}" if configured_base else 'CONDA_BASE=""',
                    'if [ -z "$CONDA_BASE" ] || [ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then',
                    "    module load miniconda/py312 2>/dev/null || true",
                    "    if command -v conda >/dev/null 2>&1; then",
                    '        CONDA_BASE="$(conda info --base)"',
                    "    fi",
                    "fi",
                    'if [ -z "$CONDA_BASE" ] || [ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then',
                    '    echo "无法定位 conda 基础目录" >&2',
                    "    exit 1",
                    "fi",
                    "unset PYTHONHOME",
                    'source "$CONDA_BASE/etc/profile.d/conda.sh"',
                    f"conda activate {shlex.quote(settings.SLURM_CONDA_ENV)}",
                    "",
                ]
            )

        lines.extend(
            [
                'echo "=========================================="',
                f'echo "Scene {scene_id} 渲染任务"',
                'echo "Job ID: $SLURM_JOB_ID"',
                'echo "Node: $SLURM_NODELIST"',
                'echo "Start: $(date)"',
                'echo "=========================================="',
                f"mkdir -p {shlex.quote(str(media_dir))}",
                f"cd {shlex.quote(str(python_file.parent))}",
            ]
        )

        quality = f"-q{profile.quality}"
        formal_args = [
            "manim",
            "render",
            f"--renderer={renderer}",
            quality,
            "--resolution",
            f"{profile.pixel_width},{profile.pixel_height}",
            "--fps",
            str(profile.frame_rate),
            "--media_dir",
            str(media_dir),
            str(python_file),
            scene_class_name,
        ]

        def add_renderer_specific_args(args: list[str]) -> list[str]:
            result = list(args)
            if use_gpu:
                # Manim 0.20 的 OpenGL 渲染器必须显式传 --write_to_movie；
                # 否则动画可能正常执行但不产出最终 MP4。
                result.insert(-2, "--write_to_movie")
            return result

        formal_args = add_renderer_specific_args(formal_args)
        # Manim 0.20 在导入 mobject 类时就会根据 renderer 选择 Cairo 或
        # OpenGL 继承树。即使是 Cairo，也把变量显式固定，方便计算节点
        # 环境指纹与正式命令保持一致。
        lines.append(f"export MANIM_RENDERER={renderer}")
        if use_gpu:
            # 某些集群包装器会在 CLI 解析前导入场景模块；提前设置平台
            # 可避免 OpenGL 回退到 GLX 并在无显示节点失败。
            lines.append(f"export PYOPENGL_PLATFORM={profile.opengl_platform}")

        if container:
            image = str(Path(container).expanduser().resolve())
            if not Path(image).is_file():
                raise RuntimeError(
                    f"Apptainer 镜像不存在: {image}\n"
                    f"请检查 .env 中的 SLURM_CONTAINER_IMAGE 配置。\n"
                    f"如果不需要容器，请设置 SLURM_CONTAINER_IMAGE 为空或注释掉该行。"
                )
            container_cmd = [
                "apptainer",
                "exec",
                "--containall",
                "--cleanenv",
                "--no-home",
            ]
            container_cmd.extend(
                [
                    "--env",
                    f"MANIM_RENDERER={renderer}",
                ]
            )
            if use_gpu:
                container_cmd.append("--nv")
                # --cleanenv 会清除作业脚本中的普通环境变量；显式传入
                # PyOpenGL 平台，否则容器内可能退回 GLX 并在无显示节点失败。
                container_cmd.extend(
                    [
                        "--env",
                        f"PYOPENGL_PLATFORM={profile.opengl_platform}",
                    ]
                )
            if settings.SLURM_CONTAINER_DISABLE_NETWORK:
                container_cmd.extend(["--net", "--network", "none"])

            def command_for(args: list[str]) -> str:
                return shlex.join(
                    [
                        *container_cmd,
                        "--bind",
                        f"{run_root}:{run_root}",
                        image,
                        *args,
                    ]
                )
        else:
            lines.append('echo "Python: $(which python)"')
            lines.append("echo \"Manim: $(python -c 'import manim; print(manim.__version__)')\"")

            def command_for(args: list[str]) -> str:
                return shlex.join(args)

        smoke_marker = run_root / "artifacts" / f"smoke_scene_{scene_id}.json"

        def marker_line(status: str) -> str:
            payload = json.dumps(
                {
                    "schema_version": 1,
                    "scene_id": scene_id,
                    "status": status,
                    **(
                        {"renderer": renderer, "quality": settings.SMOKE_RENDER_QUALITY}
                        if status == "passed"
                        else {}
                    ),
                },
                ensure_ascii=False,
            )
            return "printf '%s\\n' " + shlex.quote(payload) + f" > {shlex.quote(str(smoke_marker))}"

        if settings.SMOKE_RENDER_ENABLED:
            smoke_dir = media_dir / "__smoke__"
            smoke_width = max(16, (profile.pixel_width // 8) // 2 * 2)
            smoke_height = max(16, (profile.pixel_height // 8) // 2 * 2)
            smoke_args = [
                "manim",
                "render",
                f"--renderer={renderer}",
                f"-q{settings.SMOKE_RENDER_QUALITY}",
                "--resolution",
                f"{smoke_width},{smoke_height}",
                "--fps",
                str(min(profile.frame_rate, 15)),
                "--disable_caching",
                "--media_dir",
                str(smoke_dir),
                str(python_file),
                scene_class_name,
            ]
            smoke_args = add_renderer_specific_args(smoke_args)
            smoke_command = command_for(smoke_args)
            smoke_dir_q = shlex.quote(str(smoke_dir))
            smoke_class_q = shlex.quote(f"{scene_class_name}.mp4")
            smoke_video_q = shlex.quote(str(smoke_dir / "__smoke_video_path.txt"))
            # 通过位置参数把宿主 shell 中的路径传入 probe 子进程；不能
            # 只依赖未 export 的 smoke_video，尤其是 Apptainer --cleanenv
            # 会清除普通环境变量。
            smoke_probe_command = command_for(
                [
                    "sh",
                    "-c",
                    "ffprobe -v error -show_entries format=duration "
                    "-of default=noprint_wrappers=1:nokey=1 \"$1\" >/dev/null",
                    "kd1-smoke-probe",
                    "__KD1_SMOKE_VIDEO__",
                ]
            ).replace("__KD1_SMOKE_VIDEO__", '"$smoke_video"')
            lines.extend(
                [
                    'echo "[Smoke] 开始轻量运行时检查"',
                    f"rm -f {smoke_video_q}",
                    f"mkdir -p {smoke_dir_q}",
                    "if command -v timeout >/dev/null 2>&1; then "
                    f"timeout {settings.SMOKE_RENDER_TIMEOUT}s {smoke_command}; "
                    f"else {smoke_command}; fi",
                    # Manim 的退出码为 0 并不保证 OpenGL 已经写出最终
                    # MP4（例如缺少 --write_to_movie 时）。必须把产物
                    # 存在性作为 canary 的第二个独立成功条件。
                    f"smoke_video=$(find {smoke_dir_q} -type f -name {smoke_class_q} "
                    "! -path '*/partial_movie_files/*' -print -quit)",
                    'if [ -z "$smoke_video" ] || [ ! -s "$smoke_video" ]; then',
                    '    echo "[Smoke] 未生成有效最终 MP4" >&2',
                    f"    mkdir -p {shlex.quote(str(smoke_marker.parent))}",
                    marker_line("failed"),
                    "    exit 1",
                    "fi",
                    # 读取一项元数据，尽早捕获空文件/损坏容器，而不是
                    # 让正式高清渲染完成后才在合并阶段失败。
                    f"if ! {smoke_probe_command}; then",
                    '    echo "[Smoke] MP4 容器无法通过 ffprobe 校验" >&2',
                    f"    mkdir -p {shlex.quote(str(smoke_marker.parent))}",
                    marker_line("failed"),
                    "    exit 1",
                    "fi",
                    f"printf '%s\\n' \"$smoke_video\" > {smoke_video_q}",
                    'echo "[Smoke] 运行时检查通过"',
                    f"mkdir -p {shlex.quote(str(smoke_marker.parent))}",
                    marker_line("passed"),
                ]
            )
        else:
            lines.extend(
                [
                    f"mkdir -p {shlex.quote(str(smoke_marker.parent))}",
                    marker_line("disabled"),
                ]
            )
        lines.append(command_for(formal_args))

        # 记录真正计算节点/容器内的运行时身份；RenderProfile 来自提交端，
        # 不能假设登录节点与执行节点安装了完全相同的版本。
        environment_marker = run_root / "artifacts" / f"environment_scene_{scene_id}.json"
        environment_probe = command_for(
            [
                "python",
                "-c",
                "import importlib.metadata as md,json,os,shutil,subprocess,sys\n"
                "def version(command,flag):\n"
                "    path=shutil.which(command)\n"
                "    if not path: return ''\n"
                "    try:\n"
                "        result=subprocess.run([path,flag],capture_output=True,text=True,timeout=5,check=False)\n"
                "        return ((result.stdout or result.stderr).splitlines() or [''])[0][:300] if result.returncode == 0 else ''\n"
                "    except (OSError,subprocess.TimeoutExpired): return ''\n"
                "try: manim=md.version('manim')\n"
                "except md.PackageNotFoundError: manim=''\n"
                "payload={'python':sys.version.split()[0],'manim':manim,'ffmpeg':version('ffmpeg','-version'),'xelatex':version('xelatex','--version'),'renderer':os.environ.get('MANIM_RENDERER',''),'pyopengl_platform':os.environ.get('PYOPENGL_PLATFORM','')}\n"
                "print(json.dumps(payload,ensure_ascii=False,separators=(',',':')))\n",
            ]
        )
        lines.extend(
            [
                f"mkdir -p {shlex.quote(str(environment_marker.parent))}",
                f"if ENVIRONMENT_JSON=$({environment_probe}); then printf '%s\\n' \"$ENVIRONMENT_JSON\" > {shlex.quote(str(environment_marker))}; else printf '%s\\n' '{{\"status\":\"unavailable\"}}' > {shlex.quote(str(environment_marker))}; fi",
            ]
        )

        lines.extend(
            [
                "",
                'echo "=========================================="',
                'echo "渲染完成: $(date)"',
                shlex.join(["echo", f"输出目录: {media_dir}"]),
                'echo "=========================================="',
                "",
            ]
        )
        return "\n".join(lines)

    def submit(self, script_path: Path) -> str:
        if not script_path.is_file():
            raise RuntimeError(f"渲染脚本不存在: {script_path}")
        sbatch_path = shutil.which("sbatch")
        if not sbatch_path:
            raise RuntimeError("未找到 sbatch，请确认 Slurm 客户端在 PATH 中。")

        last_err = ""
        for attempt in range(1, settings.SLURM_SUBMIT_RETRIES + 1):
            try:
                result = subprocess.run(
                    [sbatch_path, "--parsable", str(script_path)],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                raise RuntimeError(
                    "sbatch 命令超时，提交状态未知。为避免重复提交，程序不会自动重试；"
                    "请先使用 squeue 检查同名任务。"
                ) from None
            else:
                if result.returncode == 0:
                    match = re.match(r"\s*(\d+)", result.stdout)
                    if not match:
                        raise RuntimeError(f"无法解析 sbatch Job ID: {result.stdout}")
                    job_id = match.group(1)
                    if not _dashboard_quiet():
                        console.print(f"[bold green][Slurm][/] 任务已提交, Job ID: {job_id}")
                    return job_id
                last_err = (result.stderr or result.stdout).strip()
                fatal = (
                    "invalid account",
                    "invalid partition",
                    "invalid qos",
                    "permission denied",
                    "access denied",
                )
                if any(word in last_err.lower() for word in fatal):
                    raise RuntimeError(f"Slurm 配置错误（不可重试）:\n{last_err}")
            if attempt < settings.SLURM_SUBMIT_RETRIES:
                delay = settings.SLURM_SUBMIT_RETRY_DELAY * attempt
                if not _dashboard_quiet():
                    console.print(f"[yellow][Slurm][/] 提交失败，{delay:.1f}s 后重试: {last_err}")
                time.sleep(delay)
        raise RuntimeError(f"sbatch 提交失败:\n{last_err}")

    def cancel_job(self, job_id: str) -> bool:
        scancel = shutil.which("scancel")
        if not scancel:
            if not _dashboard_quiet():
                console.print(f"[yellow][Slurm][/] 未找到 scancel，无法取消 Job {job_id}")
            return False
        try:
            result = subprocess.run(
                [scancel, str(job_id)],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            if not _dashboard_quiet():
                console.print(f"[yellow][Slurm][/] 取消 Job {job_id} 失败: {exc}", markup=False)
            return False
        if result.returncode != 0:
            err = (result.stderr or "").strip()
            err_lower = err.lower()
            # 作业已不在调度器 (已结束/被清理) → 无需取消, 视为成功, 无重复作业风险
            if any(k in err_lower for k in ("invalid job id", "unknown job", "does not exist")):
                return True
            if not _dashboard_quiet():
                console.print(
                    f"[yellow][Slurm][/] 取消 Job {job_id} 失败: {err}",
                    markup=False,
                )
            return False
        if not _dashboard_quiet():
            console.print(f"[yellow][Slurm][/] 已取消 Job {job_id}")
        return True

    @staticmethod
    def _normalize_state(raw: str) -> str:
        return raw.strip().split()[0].upper().rstrip("+") if raw.strip() else "UNKNOWN"

    @staticmethod
    def _check_final_status(job_id: str) -> tuple[bool, str]:
        """查询 sacct 获取已结束作业的终态。

        返回 (ok, state): ok=True 表示 sacct 查询成功 (即使无该作业的记录);
        ok=True 且 state=="UNKNOWN" 表示查询成功但没有账务记录。
        """
        ok, statuses = SlurmDispatcher._check_final_statuses([job_id])
        return ok, statuses.get(job_id, "UNKNOWN")

    @staticmethod
    def _check_final_statuses(job_ids: list[str]) -> tuple[bool, dict[str, str]]:
        """一次 sacct 查询多个 Job，并优先使用父 Job 记录。

        逐个调用 sacct 在多场景结束时会把每个 Job 的 10 秒超时串起来；
        同时，sacct 可能只返回 ``<job>.batch`` 等 step 记录。批量解析
        可以同时降低控制面压力并避免把某个 step 的状态误当成父 Job。
        """

        if not job_ids:
            return True, {}
        sacct = shutil.which("sacct")
        if not sacct:
            return False, {}
        try:
            result = subprocess.run(
                [sacct, "-j", ",".join(job_ids), "-n", "-o", "JobIDRaw,State", "--parsable2"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False, {}
        if result.returncode != 0:
            return False, {}

        requested = set(job_ids)
        exact: dict[str, str] = {}
        fallback: dict[str, str] = {}

        def state_priority(state: str) -> int:
            if state in FAILURE_STATES:
                return 3
            if state in RUNNING_STATES or state in QUEUED_STATES:
                return 2
            if state == "COMPLETED":
                return 1
            return 0

        for line in result.stdout.splitlines():
            parts = line.strip().split("|")
            if len(parts) < 2 or not parts[1]:
                continue
            raw_id, raw_state = parts[0], parts[1]
            normalized = SlurmDispatcher._normalize_state(raw_state)
            if raw_id in requested:
                exact[raw_id] = normalized
                continue
            # 兼容 job.batch/job.extern 和数组任务的 step 记录。
            base_id = raw_id.split(".", 1)[0]
            if base_id not in requested:
                continue
            previous = fallback.get(base_id)
            if previous is None or state_priority(normalized) > state_priority(previous):
                fallback[base_id] = normalized

        return True, {
            job_id: exact.get(job_id, fallback.get(job_id, "UNKNOWN")) for job_id in job_ids
        }

    def poll_status(self, job_id: str) -> str:
        return self.poll_all_statuses([job_id]).get(job_id, "UNKNOWN")

    def poll_all_statuses_snapshot(self, job_ids: list[str]) -> SlurmPollSnapshot:
        """线程安全地返回一次完整调度器查询快照。"""

        with self._poll_lock:
            return self._poll_all_statuses_snapshot_unlocked(job_ids)

    def _poll_all_statuses_snapshot_unlocked(self, job_ids: list[str]) -> SlurmPollSnapshot:
        if not job_ids:
            return SlurmPollSnapshot({}, {}, "")
        start_times: dict[str, float] = {}
        diagnostics: list[str] = []
        squeue = shutil.which("squeue")
        seen: dict[str, str] = {}
        squeue_ok = False
        if not squeue:
            diagnostics.append("未找到 squeue")
        else:
            result = self._run_squeue([squeue, "-j", ",".join(job_ids), "-h", "-o", "%i|%T|%S"])
            if result is not None and result.returncode == 0:
                squeue_ok = True
                self._merge_squeue_output(result.stdout, seen, start_times)
            else:
                if result is None:
                    diagnostics.append("squeue -j 超时")
                elif result.stderr:
                    diagnostics.append(result.stderr.strip()[-300:])
                # 部分集群对已消失的 job id 执行 squeue -j 会非零退出；
                # 改用按用户名查询，避免把一个无效 ID 误当成控制面故障。
                uresult = self._run_squeue(
                    [squeue, "-u", getpass.getuser(), "-h", "-o", "%i|%T|%S"]
                )
                if uresult is not None and uresult.returncode == 0:
                    squeue_ok = True
                    self._merge_squeue_output(uresult.stdout, seen, start_times)
                elif uresult is None:
                    diagnostics.append("squeue -u 超时")
                elif uresult.stderr:
                    diagnostics.append(uresult.stderr.strip()[-300:])

        missing_ids = [job_id for job_id in job_ids if job_id not in seen]
        sacct_ok, final_statuses = self._check_final_statuses(missing_ids)
        statuses: dict[str, str] = {}
        for job_id in job_ids:
            if job_id in seen:
                statuses[job_id] = seen[job_id]
            elif sacct_ok and final_statuses.get(job_id, "UNKNOWN") != "UNKNOWN":
                statuses[job_id] = final_statuses[job_id]
            elif sacct_ok and squeue_ok:
                # squeue 可访问但 Job 不在队列，且 sacct 无记录 → GONE。
                statuses[job_id] = "GONE"
            else:
                if not sacct_ok:
                    diagnostics.append(f"sacct 查询 Job {job_id} 失败或超时")
                statuses[job_id] = "UNKNOWN"
        diagnostic = "；".join(dict.fromkeys(item for item in diagnostics if item))
        return SlurmPollSnapshot(statuses, start_times, diagnostic)

    def poll_all_statuses(self, job_ids: list[str]) -> dict[str, str]:
        """批量查询作业状态。

        返回值区分三种情况:
        - 调度器状态 (PENDING/RUNNING/COMPLETED/FAILED/...)
        - "GONE"  : squeue 可查但作业不在队列、且 sacct 无该作业记录
                    → 作业已确定从调度器消失 (已结束且被清理), 无重复作业风险
        - "UNKNOWN": 集群查询失败 (squeue/sacct 缺失、超时或非零退出)
                    → 无法确认作业是否存在, 必须保守处理
        """
        with self._poll_lock:
            snapshot = self._poll_all_statuses_snapshot_unlocked(job_ids)
            # 旧调用方仍可读取兼容属性；新的 JobMonitor 直接消费 snapshot，
            # 不会在并发场景下读取到其它线程的临时值。
            self.last_start_times = dict(snapshot.start_times)
            self.last_status_diagnostic = snapshot.diagnostic
        return dict(snapshot.statuses)

    @staticmethod
    def _run_squeue(args: list[str]) -> subprocess.CompletedProcess | None:
        """执行一条 squeue 查询, 超时返回 None。"""
        try:
            return subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None

    @staticmethod
    def _merge_squeue_output(
        stdout: str,
        seen: dict[str, str],
        start_times: dict[str, float] | None = None,
    ) -> None:
        """解析 `squeue -o "%i|%T|%S"` 输出到状态和启动时间字典。"""
        for line in stdout.splitlines():
            parts = line.strip().split("|")
            if len(parts) >= 2:
                job_id = parts[0]
                seen[job_id] = SlurmDispatcher._normalize_state(parts[1])
                if start_times is not None and len(parts) >= 3:
                    raw_start = parts[2].strip()
                    if raw_start and raw_start not in {"N/A", "Unknown", "UNKNOWN", "-"}:
                        with suppress(ValueError):
                            start_times[job_id] = datetime.fromisoformat(
                                raw_start.replace("Z", "+00:00")
                            ).timestamp()

    @staticmethod
    def _find_final_video(job: SlurmJob) -> Path | None:
        """递归查找作业产出的最终渲染视频。

        manim 0.20 会把成品写到嵌套路径
        <media_dir>/videos/<源文件名>/<quality>/<SceneClass>.mp4
        (不同版本/平台层级略有差异), 因此必须递归查找, 与 VideoMerger 的
        定位逻辑保持一致; 同时排除 partial_movie_files, 并只接受 mtime
        不早于作业提交时间的产物, 避免误复用上一次修复尝试的旧视频。
        """
        if not job.media_dir.is_dir():
            return None
        candidates: list[tuple[Path, float]] = []
        try:
            paths = job.media_dir.rglob(f"{job.scene_class_name}.mp4")
            for path in paths:
                if (
                    path.is_symlink()
                    or "partial_movie_files" in path.parts
                    or "__smoke__" in path.parts
                ):
                    continue
                try:
                    stat = path.stat()
                except OSError:
                    # ffmpeg/manim 可能在扫描期间替换或删除文件；跳过该候选，
                    # 不应让一次竞态把成功作业判成监控器异常。
                    continue
                # submit_scene 为每次尝试创建独立的 attempt_<token> 目录，目录
                # 隔离已经排除了旧尝试污染，因此不再依赖登录节点与计算节点的
                # wall-clock 一致性。旧版/手工构造的共享目录仍保留宽松的 mtime
                # 保护，兼容历史清单并避免误拾取明显陈旧产物。
                isolated_attempt = bool(re.fullmatch(r"attempt_[0-9a-f]{12}", job.media_dir.name))
                if isolated_attempt:
                    fresh_enough = True
                else:
                    fresh_enough = stat.st_mtime >= job.submitted_at - 300.0
                if stat.st_size > 0 and fresh_enough:
                    candidates.append((path, stat.st_mtime))
        except OSError:
            return None
        if not candidates:
            return None
        return max(candidates, key=lambda item: item[1])[0]

    @staticmethod
    def _read_environment_fingerprint(job: SlurmJob) -> dict[str, str]:
        """读取计算节点写入的环境身份；旧 Job 缺少该文件时返回空。"""

        marker = (
            job.script_path.parent.parent / "artifacts" / f"environment_scene_{job.scene_id}.json"
        )
        try:
            if not marker.is_file() or marker.is_symlink():
                return {}
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, dict):
            return {}
        return {
            str(key): str(value)[:300]
            for key, value in payload.items()
            if isinstance(key, str) and isinstance(value, (str, int, float))
        }

    def validate_completed_job(self, job: SlurmJob) -> bool:
        """确认当前作业确实生成了可解析且匹配配置的最终视频。"""

        video = self._find_final_video(job)
        if video is None:
            job.failure_reason = "Slurm 状态为 COMPLETED，但未找到本次作业的最终 MP4"
            return False
        try:
            metadata = verify_video(video, job.render_profile)
            digest = sha256_file(video)
        except (OSError, RuntimeError, ValueError) as exc:
            job.failure_reason = f"Slurm 状态为 COMPLETED，但视频验证失败: {exc}"
            return False
        job.output_path = video
        job.output_metadata = metadata
        # 记录经过 ffprobe 验证的精确文件身份，供恢复和公共 merge_jobs
        # 继续拒绝目录中被替换的同名文件。
        job.output_sha256 = digest
        job.environment_fingerprint = self._read_environment_fingerprint(job)
        expected = {
            "manim": job.render_profile.manim_version,
            "ffmpeg": job.render_profile.ffmpeg_version,
            "xelatex": job.render_profile.xelatex_version,
            "renderer": job.render_profile.renderer,
            "pyopengl_platform": job.render_profile.opengl_platform,
        }
        mismatches = [
            f"{key}: {actual!r} != {value!r}"
            for key, value in expected.items()
            if value and (actual := job.environment_fingerprint.get(key)) and actual != value
        ]
        if mismatches:
            job.environment_warning = "计算节点环境与提交端配置不一致: " + "; ".join(mismatches)
        return True

    @staticmethod
    def _job_log_tail(job: SlurmJob, limit: int = 4000) -> str:
        """合并作业 .out/.err 尾部文本, 用于判定已消失作业的真实结果。"""
        chunks: list[str] = []
        for path in (job.log_err, job.log_out):
            try:
                if path.is_file() and path.stat().st_size > 0:
                    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
                    chunks.append("\n".join(lines[-limit:]))
            except OSError:
                continue
        return "\n".join(chunks)[-4000:]

    def _classify_gone(self, job: SlurmJob) -> str | None:
        """作业已确认不在调度器 (squeue/sacct 均无记录) 时, 依据产物判定结果。

        - 有最终视频 → "COMPLETED" (作业在轮询间隙已结束并成功产出)
        - 有明确失败证据的日志 → "FAILED" (作业实际运行过, 可走自动修复)
        - 无任何产物 → None (可能是刚提交尚未注册, 交给计数延后决断)
        """
        if self.validate_completed_job(job):
            return "COMPLETED"
        log_tail = self._job_log_tail(job)
        # stdout 常包含正常的启动/进度/完成前日志，不能仅凭“有日志”就
        # 把作业判为失败；只有出现 traceback/异常/非零退出等明确证据时
        # 才进入 AutoFix，否则继续等待产物传播或 sacct 记录。
        # 不能使用裸的 ``error ``/``failed`` 子串：正常输出、课程文字或
        # 警告中出现这些词时，作业也可能只是短暂地从队列中消失。只匹配
        # traceback、异常类型、退出码和明确的作业终态证据。
        failure_markers = (
            re.compile(r"traceback\s*\(most recent call last\)", re.IGNORECASE),
            re.compile(r"\b(?:[a-z_][a-z0-9_]*(?:error|exception))\s*:", re.IGNORECASE),
            re.compile(r"\bfatal(?:\s+error)?\b", re.IGNORECASE),
            re.compile(r"\bnon[- ]zero(?:\s+exit)?\b", re.IGNORECASE),
            re.compile(r"\bexit(?:ed)?\s+(?:with\s+)?(?:code|status)\b", re.IGNORECASE),
            re.compile(r"\b(?:killed|cancelled|time limit)\b", re.IGNORECASE),
            re.compile(r"\b(?:render|job|process)\s+(?:failed|failure)\b", re.IGNORECASE),
        )
        if log_tail and any(marker.search(log_tail) for marker in failure_markers):
            job.failure_reason = f"作业已从集群消失，依据日志判定为失败:\n{log_tail}"
            return "FAILED"
        return None

    def wait_for_job(
        self,
        job_id: str,
        scene_id: int,
        poll_interval: int | None = None,
        timeout: int | None = None,
        job: SlurmJob | None = None,
    ) -> bool:
        if job is None:
            job = SlurmJob(
                job_id=job_id,
                scene_id=scene_id,
                script_path=Path(),
                log_out=settings.LOGS_DIR / f"scene_{scene_id}_{job_id}.out",
                log_err=settings.LOGS_DIR / f"scene_{scene_id}_{job_id}.err",
                media_dir=settings.VIDEOS_DIR / f"scene_{scene_id}",
                scene_class_name="Scene",
                submitted_at=time.time(),
            )
        return self.wait_for_all_jobs({job_id: job}, poll_interval=poll_interval, timeout=timeout)[
            job_id
        ]

    def wait_for_all_jobs(
        self,
        jobs: dict[str, SlurmJob],
        poll_interval: int | None = None,
        timeout: int | None = None,
    ) -> dict[str, bool]:
        legacy_timeout = settings.MONITOR_TIMEOUT
        if timeout is not None:
            queue_timeout = run_timeout = timeout
        else:
            queue_timeout = settings.MONITOR_QUEUE_TIMEOUT
            run_timeout = settings.MONITOR_RUN_TIMEOUT
            # 兼容只设置旧 MONITOR_TIMEOUT 的配置；显式修改新的拆分项时以新项为准。
            if legacy_timeout is not None:
                if queue_timeout == 3600:
                    queue_timeout = legacy_timeout
                if run_timeout == 3600:
                    run_timeout = legacy_timeout

        monitor = JobMonitor(
            self,
            queue_timeout=queue_timeout,
            run_timeout=run_timeout,
            poll_interval=poll_interval,
            # 上面已把 timeout 参数解析成最终值；不能再次被旧的
            # MONITOR_TIMEOUT 覆盖，尤其是显式传入 3600 时。
            use_legacy_timeout=False,
        )
        for job in jobs.values():
            monitor.add_job(job)
        while monitor.pending:
            monitor.poll_once()
            if monitor.pending:
                time.sleep(monitor.poll_interval)
        return monitor.results

    def _cancel_for_monitor_failure(self, job: SlurmJob, *, status: str, reason: str) -> None:
        if not _dashboard_quiet():
            console.print(
                f"[bold red][Monitor][/] Scene {job.scene_id} {reason}，取消 Job {job.job_id}"
            )
        try:
            cancelled = self.cancel_job(job.job_id)
        except Exception as exc:
            cancelled = False
            reason = f"{reason}；调用 scancel 时发生异常: {exc}"
        if cancelled:
            job.cancelled = True
            job.status = status
            job.failure_reason = reason
            return
        job.status = "CANCEL_FAILED"
        job.failure_reason = f"{reason}；scancel 失败，禁止自动重提以避免重复作业"

    @staticmethod
    def _forward_log(job: SlurmJob, positions: dict[str, int]) -> None:
        last_pos = positions.get(job.job_id, 0)
        try:
            if not job.log_out.exists():
                return
            size = job.log_out.stat().st_size
            if size < last_pos:
                last_pos = 0
            with job.log_out.open("r", encoding="utf-8", errors="replace") as handle:
                handle.seek(last_pos)
                new = handle.read()
                positions[job.job_id] = handle.tell()
            if not _dashboard_quiet():
                for line in new.rstrip().splitlines():
                    if line.strip():
                        console.print(f"  {line}", markup=False, style="dim")
        except OSError:
            return

    def get_error_log(
        self,
        scene_id: int | None = None,
        job_id: str | None = None,
        *,
        job: SlurmJob | None = None,
    ) -> str | None:
        if job is not None:
            log_paths = (job.log_err, job.log_out)
        elif scene_id is not None and job_id is not None:
            log_paths = (settings.LOGS_DIR / f"scene_{scene_id}_{job_id}.err",)
        else:
            raise ValueError("必须提供 job，或同时提供 scene_id 和 job_id")
        try:
            logs: list[tuple[str, str]] = []
            for path in log_paths:
                if not path.is_file():
                    continue
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
                if lines:
                    label = "stderr" if path == log_paths[0] else "stdout"
                    logs.append((label, "\n".join(lines[-settings.LOG_TAIL_LINES :])))
            if not logs:
                return None
            # stderr 是 AutoFix 的首要诊断来源；截断时优先保留 stderr，避免
            # 大量 stdout 进度日志把真正的 traceback 顶出发送窗口。
            stderr = next((text for label, text in logs if label == "stderr"), "")
            stdout = next((text for label, text in logs if label == "stdout"), "")
            limit = settings.MAX_LOG_CHARS
            if stderr:
                stderr_block = f"[stderr]\n{stderr}"[-limit:]
                remaining = max(0, limit - len(stderr_block) - 2)
                stdout_block = f"[stdout]\n{stdout}"[-remaining:] if remaining else ""
                return f"{stderr_block}\n\n{stdout_block}".strip()
            return f"[stdout]\n{stdout}"[-limit:]
        except OSError as exc:
            if not _dashboard_quiet():
                console.print(f"[yellow][Monitor][/] 读取日志失败: {exc}", markup=False)
            return None

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
    ) -> SlurmJob:
        scene_id = self._validate_scene_id(scene_id)
        scene_class_name = self._validate_scene_class_name(scene_class_name)
        profile = render_profile or RenderProfile.current()
        script_path, log_out_pattern, log_err_pattern, media_dir = self.generate_script(
            scene_id,
            python_file,
            scene_class_name,
            scenes_dir=scenes_dir,
            logs_dir=logs_dir,
            videos_dir=videos_dir,
            # 每次提交使用独立媒体目录，从结构上杜绝自动修复后
            # 把上一次作业的 MP4 误认为当前作业产物。
            attempt_token=uuid4().hex[:12],
            render_profile=profile,
        )
        submitted_at = time.time()
        job_id = self.submit(script_path)
        return SlurmJob(
            job_id=job_id,
            scene_id=scene_id,
            script_path=script_path,
            log_out=Path(str(log_out_pattern).replace("%j", job_id)),
            log_err=Path(str(log_err_pattern).replace("%j", job_id)),
            media_dir=media_dir,
            scene_class_name=scene_class_name,
            submitted_at=submitted_at,
            code_sha256=code_sha256,
            render_profile=profile,
        )


class SlurmMonitorCoordinator:
    """在一个后台线程中批量监控同一运行的所有 Slurm Job。

    场景 worker 只负责注册 Job 和等待结果，不再各自创建 JobMonitor。
    这样既避免重复查询 Slurm 控制面，也让每个 Job 的未知/宽限/超时
    计数只存在一份。
    """

    def __init__(
        self,
        dispatcher: SlurmDispatcher,
        *,
        queue_timeout: int | None = None,
        run_timeout: int | None = None,
        unknown_timeout: int | None = None,
        artifact_grace: int | None = None,
        max_unknown: int | None = None,
        poll_interval: int | None = None,
        on_job_update: Callable[[SlurmJob], None] | None = None,
    ) -> None:
        self.monitor = JobMonitor(
            dispatcher,
            queue_timeout=queue_timeout,
            run_timeout=run_timeout,
            unknown_timeout=unknown_timeout,
            artifact_grace=artifact_grace,
            max_unknown=max_unknown,
            poll_interval=poll_interval,
            use_legacy_timeout=False,
        )
        self._condition = threading.Condition()
        self._monitor_lock = threading.RLock()
        self._queued: list[SlurmJob] = []
        self._on_job_update = on_job_update
        self._last_started_at: dict[str, float | None] = {}
        self._registered: set[str] = set()
        self._events: dict[str, threading.Event] = {}
        self._outcomes: dict[str, bool] = {}
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            name="slurm-monitor",
            daemon=True,
        )
        self._thread.start()

    def register(self, job: SlurmJob) -> None:
        """注册一个新 Job；重复注册同一 Job ID 是幂等操作。"""

        with self._condition:
            if self._closed:
                raise RuntimeError("Slurm 监控协调器已关闭")
            if job.job_id in self._registered:
                return
            self._registered.add(job.job_id)
            self._events[job.job_id] = threading.Event()
            self._queued.append(job)
            self._condition.notify_all()

    def wait(
        self,
        job_id: str,
        *,
        stop_event: threading.Event | None = None,
    ) -> bool | None:
        """等待 Job 结果；外部流水线停止时返回 None，不篡改 Job 结果。"""

        with self._condition:
            event = self._events.get(job_id)
            if event is None:
                raise KeyError(f"未注册 Slurm Job: {job_id}")
        while not event.wait(0.25):
            if stop_event is not None and stop_event.is_set():
                return None
        with self._condition:
            return self._outcomes.get(job_id)

    def close(self, *, timeout: float = 30.0) -> None:
        """停止接收新 Job，并等待当前轮询尽量完成。"""

        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._thread.join(timeout=max(0.0, timeout))

    def cancel_pending(self, *, reason: str = "流水线停止") -> None:
        """取消仍由协调器持有的作业，避免停止后遗留后台轮询线程。"""

        with self._monitor_lock:
            with self._condition:
                queued = self._queued
                self._queued = []
            pending = [*self.monitor.pending.values(), *queued]
            seen: set[str] = set()
            for job in pending:
                if job.job_id in seen:
                    continue
                seen.add(job.job_id)
                if not job.cancelled and job.status not in TERMINAL_STATES:
                    try:
                        cancelled = self.monitor.dispatcher.cancel_job(job.job_id)
                    except Exception as exc:
                        cancelled = False
                        job.failure_reason = f"{reason}；取消作业时发生异常: {exc}"
                    if cancelled:
                        job.cancelled = True
                        job.status = "CANCELLED"
                        job.failure_reason = f"{reason}，已取消远端作业"
                    else:
                        job.status = "CANCEL_FAILED"
                        if not job.failure_reason:
                            job.failure_reason = f"{reason}；取消远端作业失败"
                self.monitor.jobs[job.job_id] = job
                self.monitor.results[job.job_id] = False
                self.monitor.pending.pop(job.job_id, None)
            self._publish_results()

    def _drain_queue(self) -> None:
        with self._condition:
            queued = self._queued
            self._queued = []
        for job in queued:
            self.monitor.add_job(job)

    def _publish_results(self) -> None:
        if self._on_job_update is not None:
            for job_id, job in self.monitor.jobs.items():
                started_at = job.started_at
                if started_at is None or started_at == self._last_started_at.get(job_id):
                    continue
                self._last_started_at[job_id] = started_at
                try:
                    self._on_job_update(job)
                except Exception as exc:
                    # 检查点回调失败不应杀死监控线程并让 worker 永久等待；
                    # 交给外层通过 checkpoint_error/stop_event 收尾。
                    if not _dashboard_quiet():
                        console.print(
                            f"[yellow][Monitor][/] 更新作业检查点失败: {exc}",
                            markup=False,
                        )
        with self._condition:
            for job_id, outcome in self.monitor.results.items():
                if job_id in self._outcomes:
                    continue
                self._outcomes[job_id] = outcome
                event = self._events.get(job_id)
                if event is not None:
                    event.set()

    def _run(self) -> None:
        while True:
            with self._monitor_lock:
                self._drain_queue()
                has_pending = bool(self.monitor.pending)
            with self._condition:
                if self._closed and not self._queued and not has_pending:
                    return
                if not has_pending:
                    self._condition.wait()
                    continue
            with self._monitor_lock:
                self.monitor.poll_once()
                self._publish_results()
                has_pending = bool(self.monitor.pending)
            with self._condition:
                if has_pending:
                    self._condition.wait(timeout=self.monitor.poll_interval)
