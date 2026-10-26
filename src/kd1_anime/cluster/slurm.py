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
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from rich.console import Console

from kd1_anime.cluster.artifacts import RenderArtifactValidator
from kd1_anime.cluster.execution import ENVIRONMENT_KEYS, ExecutionPolicy
from kd1_anime.cluster.jobs import PollObservation as SlurmPollSnapshot
from kd1_anime.cluster.jobs import RenderJob as SlurmJob
from kd1_anime.cluster.launch import ENVIRONMENT_PROBE
from kd1_anime.cluster.monitoring import JobMonitor as JobMonitor
from kd1_anime.cluster.monitoring import RenderMonitorCoordinator as SlurmMonitorCoordinator
from kd1_anime.cluster.resource_estimator import RenderResourceProfile
from kd1_anime.cluster.status import FAILURE_STATES as FAILURE_STATES
from kd1_anime.cluster.status import MONITOR_ABORT_STATES as MONITOR_ABORT_STATES
from kd1_anime.cluster.status import QUEUED_STATES, RUNNING_STATES
from kd1_anime.cluster.status import TERMINAL_STATES as TERMINAL_STATES
from kd1_anime.config import settings
from kd1_anime.rendering import RenderProfile, verify_video

console = Console()


def _dashboard_quiet() -> bool:
    """Live 仪表盘激活时抑制普通输出, 避免破坏 Rich Live 渲染。"""
    try:
        from kd1_anime.dashboard import quiet

        return quiet()
    except Exception:
        return False


class SlurmDispatcher(RenderArtifactValidator):
    """Slurm 任务调度器。"""

    def _verify_video(self, video, profile):
        return verify_video(video, profile)

    def __init__(self) -> None:
        # 保留旧属性供外部集成读取；新的 JobMonitor 使用不可变快照，
        # 不依赖这些共享临时值。
        self.last_start_times: dict[str, float] = {}
        self.last_status_diagnostic: str = ""
        self._poll_lock = threading.RLock()

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
        resource_profile: RenderResourceProfile | None = None,
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
            resource_profile=resource_profile,
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
        resource_profile: RenderResourceProfile | None = None,
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
        resources = resource_profile or RenderResourceProfile.from_settings()
        smoke_mode = (
            resources.smoke_mode
            if resource_profile is not None and settings.ADAPTIVE_SMOKE_RENDER
            else settings.SMOKE_RENDER_MODE
        )
        renderer = profile.renderer
        use_gpu = renderer == "opengl"
        if use_gpu and not resources.gpu_type:
            raise RuntimeError(
                "MANIM_RENDERER=opengl 时必须配置 SLURM_GPU_TYPE；否则无法保证作业分配到 GPU 节点。"
            )
        if settings.SLURM_REQUIRE_CONTAINER and not settings.SLURM_CONTAINER_IMAGE:
            raise RuntimeError("SLURM_REQUIRE_CONTAINER=true，但未配置 SLURM_CONTAINER_IMAGE。")

        ExecutionPolicy.from_spec(getattr(self, "execution_spec", None)).validate()
        lines = ["#!/bin/bash", "#SBATCH --export=" + ",".join(ENVIRONMENT_KEYS)]
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
        if re.fullmatch(r"attempt_[0-9a-f]{12}", media_dir.name):
            job_name += "-" + media_dir.name.removeprefix("attempt_")
        lines.extend(
            [
                f"#SBATCH -J {job_name}",
                "#SBATCH -N 1",
                f"#SBATCH --cpus-per-task={resources.cpus_per_task}",
            ]
        )
        # Cairo 不使用 GPU；只为 OpenGL 作业申请 GPU。
        if use_gpu:
            lines.append(f"#SBATCH --gres=gpu:{resources.gpu_type}:{resources.gpu_count}")
        lines.extend(
            [
                f"#SBATCH -t {resources.time_limit}",
                f"#SBATCH -o {shlex.quote(str(log_out))}",
                f"#SBATCH -e {shlex.quote(str(log_err))}",
            ]
        )
        if resources.mem_gb:
            lines.append(f"#SBATCH --mem={resources.mem_gb}G")

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
        # Smoke 阶段会受到地址空间限制；限制 BLAS 线程，避免
        # OpenBLAS 在导入 numpy 时创建过多线程映射而误报内存不足。
        lines.extend(
            [
                "export OPENBLAS_NUM_THREADS=1",
                "export OMP_NUM_THREADS=1",
                "export MKL_NUM_THREADS=1",
                "export NUMEXPR_NUM_THREADS=1",
                "export PYTHONNOUSERSITE=1",
            ]
        )
        if use_gpu:
            # 某些集群包装器会在 CLI 解析前导入场景模块；提前设置平台
            # 可避免 OpenGL 回退到 GLX 并在无显示节点失败。
            lines.append(f"export PYOPENGL_PLATFORM={profile.opengl_platform}")

        policy = ExecutionPolicy.from_spec(getattr(self, "execution_spec", None))
        if policy.startup_check():
            lines.append(policy.startup_check())

        def command_for(args: list[str]) -> str:
            wrapped = policy.wrap(
                args, run_root, renderer=renderer, platform=profile.opengl_platform
            )
            return policy.shell_environment_prefix() + shlex.join(wrapped)

        smoke_marker = run_root / "artifacts" / f"smoke_scene_{scene_id}.json"

        def marker_line(status: str) -> str:
            payload = json.dumps(
                {
                    "schema_version": 1,
                    "scene_id": scene_id,
                    "status": status,
                    "mode": smoke_mode,
                    "short_animations": settings.SMOKE_RENDER_SHORT_ANIMATIONS,
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
            smoke_width = max(16, (profile.pixel_width // 8) // 2 * 2)
            smoke_height = max(16, (profile.pixel_height // 8) // 2 * 2)
            smoke_fps = min(profile.frame_rate, 15)

            def smoke_failure(message: str) -> list[str]:
                return [
                    f"echo {shlex.quote('[Smoke] ' + message)} >&2",
                    f"mkdir -p {shlex.quote(str(smoke_marker.parent))}",
                    marker_line("failed"),
                    "exit 1",
                ]

            def smoke_run(command: str) -> str:
                return (
                    "if command -v timeout >/dev/null 2>&1; then "
                    f"timeout {settings.SMOKE_RENDER_TIMEOUT}s {command}; "
                    f"else {command}; fi"
                )

            lines.append('echo "[Smoke] 开始轻量运行时检查"')
            import_check = (
                "import importlib.util, pathlib, sys\n"
                "path=pathlib.Path(sys.argv[1])\n"
                "name=sys.argv[2]\n"
                "spec=importlib.util.spec_from_file_location('kd1_smoke_scene', path)\n"
                "module=importlib.util.module_from_spec(spec)\n"
                "spec.loader.exec_module(module)\n"
                "candidate=getattr(module, name, None)\n"
                "if not isinstance(candidate, type):\n"
                "    raise SystemExit(1)"
            )
            lines.append(
                smoke_run(
                    command_for(["python", "-c", import_check, str(python_file), scene_class_name])
                )
            )
            lines.append('echo "[Smoke] import-only 检查通过"')
            if smoke_mode in {"frame", "both"}:
                frame_dir = media_dir / "__frame_smoke__"
                frame_args = [
                    "manim",
                    "render",
                    f"--renderer={renderer}",
                    f"-q{settings.SMOKE_RENDER_QUALITY}",
                    "--resolution",
                    f"{smoke_width},{smoke_height}",
                    "--fps",
                    str(smoke_fps),
                    "--disable_caching",
                    "--format",
                    "png",
                    "--save_last_frame",
                    "--media_dir",
                    str(frame_dir),
                    str(python_file),
                    scene_class_name,
                ]
                frame_dir_q = shlex.quote(str(frame_dir))
                frame_class_q = shlex.quote(f"{scene_class_name}.png")
                frame_versioned_class_q = shlex.quote(f"{scene_class_name}_*.png")
                frame_command = command_for(frame_args)
                lines.extend(
                    [
                        f"mkdir -p {frame_dir_q}",
                        smoke_run(frame_command),
                        f"smoke_frame=$(find {frame_dir_q} -type f \\( -name {frame_class_q} "
                        f"-o -name {frame_versioned_class_q} \\) "
                        "! -path '*/partial_movie_files/*' -print -quit)",
                        'if [ -z "$smoke_frame" ] || [ ! -s "$smoke_frame" ]; then',
                        *smoke_failure("未生成有效最后一帧 PNG"),
                        "fi",
                        'echo "[Smoke] 最后一帧检查通过"',
                    ]
                )
            if smoke_mode in {"video", "both"}:
                smoke_dir = media_dir / "__smoke__"
                smoke_args = [
                    "manim",
                    "render",
                    f"--renderer={renderer}",
                    f"-q{settings.SMOKE_RENDER_QUALITY}",
                    "--resolution",
                    f"{smoke_width},{smoke_height}",
                    "--fps",
                    str(smoke_fps),
                    "--disable_caching",
                    "--from_animation_number",
                    f"0,{settings.SMOKE_RENDER_SHORT_ANIMATIONS}",
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
                        '-of default=noprint_wrappers=1:nokey=1 "$1" >/dev/null',
                        "kd1-smoke-probe",
                        "__KD1_SMOKE_VIDEO__",
                    ]
                ).replace("__KD1_SMOKE_VIDEO__", '"$smoke_video"')
                lines.extend(
                    [
                        f"rm -f {smoke_video_q}",
                        f"mkdir -p {smoke_dir_q}",
                        smoke_run(smoke_command),
                        # Manim 的退出码为 0 并不保证 OpenGL 已经写出最终
                        # MP4（例如缺少 --write_to_movie 时）。必须把产物
                        # 存在性作为 canary 的第二个独立成功条件。
                        f"smoke_video=$(find {smoke_dir_q} -type f -name {smoke_class_q} "
                        "! -path '*/partial_movie_files/*' -print -quit)",
                        'if [ -z "$smoke_video" ] || [ ! -s "$smoke_video" ]; then',
                        *smoke_failure("未生成有效最终 MP4"),
                        "fi",
                        # 读取一项元数据，尽早捕获空文件/损坏容器，而不是
                        # 让正式高清渲染完成后才在合并阶段失败。
                        f"if ! {smoke_probe_command}; then",
                        *smoke_failure("MP4 容器无法通过 ffprobe 校验"),
                        "fi",
                        f"printf '%s\\n' \"$smoke_video\" > {smoke_video_q}",
                        'echo "[Smoke] MP4 检查通过"',
                    ]
                )
            lines.extend(
                [
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
        spec = getattr(self, "execution_spec", None)
        environment_marker = media_dir / "environment.json"
        identity = json.dumps(
            {
                "execution_sha256": spec.digest() if spec else "",
                "container_sha256": spec.image_sha256 if spec else "",
            }
        )
        environment_probe = command_for(
            [
                "python",
                "-c",
                ENVIRONMENT_PROBE
                + "print(json.dumps(payload,ensure_ascii=False,separators=(',',':')))\n",
                identity,
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
                    env=ExecutionPolicy.environment(),
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

    def poll(self, job_ids):
        # Instance overrides belong to the legacy injection boundary.
        if "poll_all_statuses" in self.__dict__:
            return SlurmPollSnapshot(
                self.poll_all_statuses(job_ids),
                dict(self.last_start_times),
                self.last_status_diagnostic,
            )
        return self.poll_all_statuses_snapshot(job_ids)

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
        attempt_token: str | None = None,
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
            attempt_token=attempt_token or uuid4().hex[:12],
            render_profile=profile,
            resource_profile=resource_profile,
        )
        submitted_at = time.time()
        job_id = self.submit(script_path)
        return SlurmJob(
            execution_sha256=self.execution_spec.digest()
            if getattr(self, "execution_spec", None)
            else "",
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
            resource_profile=resource_profile,
        )


__all__ = [
    "FAILURE_STATES",
    "MONITOR_ABORT_STATES",
    "TERMINAL_STATES",
    "JobMonitor",
    "SlurmDispatcher",
    "SlurmJob",
    "SlurmMonitorCoordinator",
    "SlurmPollSnapshot",
]
