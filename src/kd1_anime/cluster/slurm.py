"""Slurm 调度、批量监控和渲染脚本生成。"""

from __future__ import annotations

import re
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

from kd1_anime.config import settings

console = Console()

TERMINAL_STATES = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "BOOT_FAIL",
    "PREEMPTED",
}
FAILURE_STATES = TERMINAL_STATES - {"COMPLETED"}
RUNNING_STATES = {"RUNNING", "COMPLETING", "STAGE_OUT"}
MONITOR_ABORT_STATES = {"QUEUE_TIMEOUT", "RUN_TIMEOUT", "UNKNOWN_TIMEOUT", "CANCEL_FAILED"}


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
    status: str = "PENDING"
    failure_reason: str = ""
    cancelled: bool = False


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
        dispatcher: "SlurmDispatcher",
        *,
        queue_timeout: int | None = None,
        run_timeout: int | None = None,
        poll_interval: int | None = None,
    ) -> None:
        self.dispatcher = dispatcher
        self.queue_timeout = queue_timeout or settings.MONITOR_QUEUE_TIMEOUT
        self.run_timeout = run_timeout or settings.MONITOR_RUN_TIMEOUT
        self.poll_interval = poll_interval or settings.MONITOR_POLL_INTERVAL
        # 兼容只设置旧 MONITOR_TIMEOUT 的配置；显式修改新的拆分项时以新项为准。
        legacy_timeout = settings.MONITOR_TIMEOUT
        if legacy_timeout is not None:
            if self.queue_timeout == 3600:
                self.queue_timeout = legacy_timeout
            if self.run_timeout == 3600:
                self.run_timeout = legacy_timeout
        self.pending: dict[str, SlurmJob] = {}
        self.results: dict[str, bool] = {}
        self.jobs: dict[str, SlurmJob] = {}
        self.unknown_streaks: dict[str, int] = {}
        self.running_since: dict[str, float] = {}
        self.log_positions: dict[str, int] = {}

    def add_job(self, job: SlurmJob) -> None:
        if job.job_id in self.results:
            return
        self.pending.setdefault(job.job_id, job)
        self.jobs[job.job_id] = job
        self.unknown_streaks.setdefault(job.job_id, 0)

    def _quiet(self) -> bool:
        """Live 仪表盘激活时抑制 Monitor 文本输出, 避免破坏 Rich Live。"""
        try:
            from kd1_anime.dashboard import is_active

            return is_active()
        except Exception:
            return False

    def poll_once(self) -> bool:
        """单次轮询所有 pending 作业, 更新状态; 返回本轮是否有作业结束。"""
        if not self.pending:
            return False
        now = time.time()
        statuses = self.dispatcher.poll_all_statuses(list(self.pending))
        finished: list[str] = []
        quiet = self._quiet()
        for job_id, job in self.pending.items():
            status = statuses.get(job_id, "UNKNOWN")
            job.status = status

            if status == "COMPLETED":
                self.unknown_streaks[job_id] = 0
                self.dispatcher._forward_log(job, self.log_positions)
                self.results[job_id] = True
                finished.append(job_id)
                if not quiet:
                    console.print(f"[bold green][Monitor][/] Scene {job.scene_id} 渲染成功")
            elif status in FAILURE_STATES:
                self.unknown_streaks[job_id] = 0
                self.dispatcher._forward_log(job, self.log_positions)
                job.failure_reason = f"Slurm 状态: {status}"
                self.results[job_id] = False
                finished.append(job_id)
                if not quiet:
                    console.print(f"[bold red][Monitor][/] Scene {job.scene_id} 渲染失败: {status}")
            elif status == "UNKNOWN":
                self.unknown_streaks[job_id] += 1
                if self.unknown_streaks[job_id] >= settings.MONITOR_MAX_UNKNOWN:
                    self.dispatcher._cancel_for_monitor_failure(
                        job,
                        status="UNKNOWN_TIMEOUT",
                        reason="状态连续未知，已停止监控并尝试取消远端任务",
                    )
                    self.results[job_id] = False
                    finished.append(job_id)
            else:
                self.unknown_streaks[job_id] = 0
                if status in RUNNING_STATES:
                    self.running_since.setdefault(job_id, now)
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
                elif now - job.submitted_at > self.queue_timeout:
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

    def generate_script(
        self,
        scene_id: int,
        python_file: Path,
        scene_class_name: str = "Scene",
        *,
        scenes_dir: Path | None = None,
        logs_dir: Path | None = None,
        videos_dir: Path | None = None,
    ) -> tuple[Path, Path, Path, Path]:
        scenes_dir = Path(scenes_dir or settings.SCENES_DIR).resolve()
        logs_dir = Path(logs_dir or settings.LOGS_DIR).resolve()
        videos_dir = Path(videos_dir or settings.VIDEOS_DIR).resolve()
        for directory in (scenes_dir, logs_dir, videos_dir):
            directory.mkdir(parents=True, exist_ok=True)

        script_path = scenes_dir / f"render_{scene_id}.sh"
        log_out = logs_dir / f"scene_{scene_id}_%j.out"
        log_err = logs_dir / f"scene_{scene_id}_%j.err"
        media_dir = videos_dir / f"scene_{scene_id}"
        media_dir.mkdir(parents=True, exist_ok=True)

        content = self._build_script(
            scene_id=scene_id,
            python_file=python_file.resolve(),
            scene_class_name=scene_class_name,
            media_dir=media_dir,
            log_out=log_out,
            log_err=log_err,
        )
        script_path.write_text(content, encoding="utf-8")
        script_path.chmod(0o700)
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
    ) -> str:
        renderer = settings.MANIM_RENDERER
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
        run_id = media_dir.parent.parent.name
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

        quality = f"-q{settings.MANIM_QUALITY}"
        manim_args = [
            "manim",
            "render",
            f"--renderer={renderer}",
            quality,
            "--media_dir",
            str(media_dir),
            str(python_file),
            scene_class_name,
        ]
        if use_gpu:
            lines.append(f"export PYOPENGL_PLATFORM={settings.MANIM_OPENGL_PLATFORM}")

        if container:
            image = str(Path(container).expanduser().resolve())
            if not Path(image).is_file():
                raise RuntimeError(
                    f"Apptainer 镜像不存在: {image}\n"
                    f"请检查 .env 中的 SLURM_CONTAINER_IMAGE 配置。\n"
                    f"如果不需要容器，请设置 SLURM_CONTAINER_IMAGE 为空或注释掉该行。"
                )
            run_root = media_dir.parent.parent
            container_cmd = [
                "apptainer",
                "exec",
                "--containall",
                "--cleanenv",
                "--no-home",
            ]
            if use_gpu:
                container_cmd.append("--nv")
            container_cmd.extend(["--bind", f"{run_root}:{run_root}", image, *manim_args])
            lines.append(shlex.join(container_cmd))
        else:
            lines.append('echo "Python: $(which python)"')
            lines.append("echo \"Manim: $(python -c 'import manim; print(manim.__version__)')\"")
            lines.append(shlex.join(manim_args))

        lines.extend(
            [
                "",
                'echo "=========================================="',
                'echo "渲染完成: $(date)"',
                f'echo "输出目录: {media_dir}"',
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
                console.print(f"[yellow][Slurm][/] 提交失败，{delay:.1f}s 后重试: {last_err}")
                time.sleep(delay)
        raise RuntimeError(f"sbatch 提交失败:\n{last_err}")

    def cancel_job(self, job_id: str) -> bool:
        scancel = shutil.which("scancel")
        if not scancel:
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
        except subprocess.TimeoutExpired:
            console.print(f"[yellow][Slurm][/] 取消 Job {job_id} 超时")
            return False
        if result.returncode != 0:
            console.print(
                f"[yellow][Slurm][/] 取消 Job {job_id} 失败: {result.stderr.strip()}",
                markup=False,
            )
            return False
        console.print(f"[yellow][Slurm][/] 已取消 Job {job_id}")
        return True

    @staticmethod
    @staticmethod
    def _normalize_state(raw: str) -> str:
        return raw.strip().split()[0].upper() if raw.strip() else "UNKNOWN"

    @staticmethod
    def _check_final_status(job_id: str) -> str:
        sacct = shutil.which("sacct")
        if not sacct:
            return "UNKNOWN"
        try:
            result = subprocess.run(
                [sacct, "-j", job_id, "-n", "-o", "JobIDRaw,State", "--parsable2"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return "UNKNOWN"
        if result.returncode != 0:
            return "UNKNOWN"
        fallback = "UNKNOWN"
        for line in result.stdout.splitlines():
            parts = line.strip().split("|")
            if len(parts) < 2:
                continue
            raw_id, raw_state = parts[0], parts[1]
            if raw_id == job_id:
                return self._normalize_state(raw_state)
            if fallback == "UNKNOWN" and raw_state:
                fallback = self._normalize_state(raw_state)
        return fallback

    def poll_status(self, job_id: str) -> str:
        return self.poll_all_statuses([job_id]).get(job_id, "UNKNOWN")

    def poll_all_statuses(self, job_ids: list[str]) -> dict[str, str]:
        if not job_ids:
            return {}
        squeue = shutil.which("squeue")
        seen: dict[str, str] = {}
        if squeue:
            try:
                result = subprocess.run(
                    [squeue, "-j", ",".join(job_ids), "-h", "-o", "%i|%T"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                result = None
            if result and result.returncode == 0:
                for line in result.stdout.splitlines():
                    parts = line.strip().split("|", 1)
                    if len(parts) == 2:
                        seen[parts[0]] = self._normalize_state(parts[1])
        return {job_id: seen.get(job_id) or self._check_final_status(job_id) for job_id in job_ids}

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
        interval = poll_interval or settings.MONITOR_POLL_INTERVAL
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
        pending = dict(jobs)
        results: dict[str, bool] = {}
        unknown_streaks = {job_id: 0 for job_id in jobs}
        running_since: dict[str, float] = {}
        log_positions: dict[str, int] = {}

        while pending:
            now = time.time()
            statuses = self.poll_all_statuses(list(pending))
            finished: list[str] = []
            for job_id, job in pending.items():
                status = statuses.get(job_id, "UNKNOWN")
                job.status = status

                if status == "COMPLETED":
                    unknown_streaks[job_id] = 0
                    self._forward_log(job, log_positions)
                    results[job_id] = True
                    finished.append(job_id)
                    console.print(f"[bold green][Monitor][/] Scene {job.scene_id} 渲染成功")
                elif status in FAILURE_STATES:
                    unknown_streaks[job_id] = 0
                    self._forward_log(job, log_positions)
                    job.failure_reason = f"Slurm 状态: {status}"
                    results[job_id] = False
                    finished.append(job_id)
                    console.print(f"[bold red][Monitor][/] Scene {job.scene_id} 渲染失败: {status}")
                elif status == "UNKNOWN":
                    unknown_streaks[job_id] += 1
                    if unknown_streaks[job_id] >= settings.MONITOR_MAX_UNKNOWN:
                        self._cancel_for_monitor_failure(
                            job,
                            status="UNKNOWN_TIMEOUT",
                            reason="状态连续未知，已停止监控并尝试取消远端任务",
                        )
                        results[job_id] = False
                        finished.append(job_id)
                else:
                    unknown_streaks[job_id] = 0
                    if status in RUNNING_STATES:
                        running_since.setdefault(job_id, now)
                    if job_id in running_since:
                        self._forward_log(job, log_positions)
                        if now - running_since[job_id] > run_timeout:
                            self._cancel_for_monitor_failure(
                                job,
                                status="RUN_TIMEOUT",
                                reason=f"运行超过 {run_timeout} 秒",
                            )
                            results[job_id] = False
                            finished.append(job_id)
                            continue
                    elif now - job.submitted_at > queue_timeout:
                        self._cancel_for_monitor_failure(
                            job,
                            status="QUEUE_TIMEOUT",
                            reason=f"排队超过 {queue_timeout} 秒",
                        )
                        results[job_id] = False
                        finished.append(job_id)
                        continue
                    console.print(f"[dim][Monitor][/] Scene {job.scene_id}: {status}")

            for job_id in finished:
                pending.pop(job_id, None)
            if pending:
                time.sleep(interval)
        return results

    def _cancel_for_monitor_failure(self, job: SlurmJob, *, status: str, reason: str) -> None:
        console.print(
            f"[bold red][Monitor][/] Scene {job.scene_id} {reason}，取消 Job {job.job_id}"
        )
        if self.cancel_job(job.job_id):
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
            log_path = job.log_err
        elif scene_id is not None and job_id is not None:
            log_path = settings.LOGS_DIR / f"scene_{scene_id}_{job_id}.err"
        else:
            raise ValueError("必须提供 job，或同时提供 scene_id 和 job_id")
        if not log_path.exists():
            return None
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            tail = "\n".join(lines[-settings.LOG_TAIL_LINES :])
            return tail[-settings.MAX_LOG_CHARS :]
        except OSError as exc:
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
    ) -> SlurmJob:
        script_path, log_out_pattern, log_err_pattern, media_dir = self.generate_script(
            scene_id,
            python_file,
            scene_class_name,
            scenes_dir=scenes_dir,
            logs_dir=logs_dir,
            videos_dir=videos_dir,
        )
        job_id = self.submit(script_path)
        submitted_at = time.time()
        return SlurmJob(
            job_id=job_id,
            scene_id=scene_id,
            script_path=script_path,
            log_out=Path(str(log_out_pattern).replace("%j", job_id)),
            log_err=Path(str(log_err_pattern).replace("%j", job_id)),
            media_dir=media_dir,
            scene_class_name=scene_class_name,
            submitted_at=submitted_at,
        )
