"""
Slurm 调度模块
负责生成 Slurm 脚本、提交任务、查询状态

参考 ustc107-cli 的脚本风格:
- 使用 --qos=qos_stu_default
- 使用短标志 (-p, -J, -N, -t, -o, -e)
- 支持 GPU 类型 (--gres=gpu:{type}:{count})
- set -eo pipefail
"""

import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

from config import settings

console = Console()


@dataclass
class SlurmJob:
    """Slurm 任务信息"""
    job_id: str
    scene_id: int
    script_path: Path
    log_out: Path
    log_err: Path
    status: str = "PENDING"


class SlurmDispatcher:
    """Slurm 任务调度器"""

    def generate_script(
        self,
        scene_id: int,
        python_file: Path,
        scene_class_name: str = "Scene",
    ) -> Path:
        """
        生成 Slurm 渲染脚本 (字符串拼接,无外部模板依赖)

        Args:
            scene_id: 场景 ID
            python_file: Python 代码文件路径
            scene_class_name: Manim 场景类名

        Returns:
            生成的 .sh 脚本路径
        """
        script_path = settings.SCENES_DIR / f"render_{scene_id}.sh"
        log_out = settings.LOGS_DIR / f"scene_{scene_id}_%j.out"
        log_err = settings.LOGS_DIR / f"scene_{scene_id}_%j.err"
        media_dir = settings.VIDEOS_DIR / f"scene_{scene_id}"

        content = self._build_script(
            scene_id=scene_id,
            python_file=python_file.resolve(),
            scene_class_name=scene_class_name,
            media_dir=media_dir.resolve(),
            log_out=log_out,
            log_err=log_err,
        )

        script_path.write_text(content, encoding="utf-8")
        script_path.chmod(0o755)

        console.print(f"[bold blue][Slurm][/] 已生成渲染脚本: {script_path}")
        return script_path

    def _build_script(
        self,
        scene_id: int,
        python_file: Path,
        scene_class_name: str,
        media_dir: Path,
        log_out: Path,
        log_err: Path,
    ) -> str:
        """通过字符串拼接构建 sbatch 脚本内容"""
        lines: list[str] = []

        lines.append("#!/bin/bash")
        lines.append(f"#SBATCH --qos={settings.SLURM_QOS}")
        lines.append(f"#SBATCH -p {settings.SLURM_PARTITION}")
        # 部分集群强制要求 --account (如 stu)
        if settings.SLURM_ACCOUNT:
            lines.append(f"#SBATCH --account={settings.SLURM_ACCOUNT}")
        lines.append(f"#SBATCH -J manim-scene-{scene_id}")
        lines.append("#SBATCH -N 1")
        lines.append(f"#SBATCH --cpus-per-task={settings.SLURM_CPUS_PER_TASK}")

        # GPU 配置 (仅在指定 GPU 类型时添加)
        if settings.SLURM_GPU_TYPE:
            lines.append(
                f"#SBATCH --gres=gpu:{settings.SLURM_GPU_TYPE}:{settings.SLURM_GPU_COUNT}"
            )

        lines.append(f"#SBATCH -t {settings.SLURM_TIME_LIMIT}")
        lines.append(f"#SBATCH -o \"{log_out}\"")
        lines.append(f"#SBATCH -e \"{log_err}\"")

        # 内存配置 (仅在指定时添加)
        if settings.SLURM_MEM_GB:
            lines.append(f"#SBATCH --mem={settings.SLURM_MEM_GB}")

        lines.append("")
        lines.append("set -eo pipefail")
        lines.append("")

        # Conda 环境激活 — 兼容 module 系统损坏 / PYTHONHOME 污染
        lines.append("# 激活 conda 环境")
        lines.append("CONDA_BASE=/public/app/miniconda3/py312_24.4.0-0")
        lines.append("if [ -x \"$CONDA_BASE/bin/conda\" ]; then")
        lines.append("    export PATH=\"$CONDA_BASE/bin:$PATH\"")
        lines.append("else")
        lines.append("    module load miniconda/py312 2>/dev/null || true")
        lines.append("fi")
        lines.append("unset PYTHONHOME")
        lines.append(f"source \"$CONDA_BASE/bin/activate\" {settings.SLURM_CONDA_ENV}")
        lines.append("")

        # 环境信息
        lines.append("echo \"==========================================\"")
        lines.append(f"echo \"Scene {scene_id} 渲染任务\"")
        lines.append("echo \"Job ID: $SLURM_JOB_ID\"")
        lines.append("echo \"Node: $SLURM_NODELIST\"")
        lines.append("echo \"Start: $(date)\"")
        lines.append("echo \"==========================================\"")
        lines.append("")

        # Python 和 Manim 版本
        lines.append("echo \"Python: $(which python)\"")
        lines.append("echo \"Manim: $(python -c 'import manim; print(manim.__version__)')\"")
        lines.append("")

        # 创建输出目录
        lines.append(f"mkdir -p \"{media_dir}\"")
        lines.append("")

        # 执行 Manim 渲染
        # -qh: 高质量 (1080p60)
        # --media_dir: 指定输出目录
        # 所有路径加引号, 防止含空格/特殊字符时参数错位
        lines.append(f"cd \"{python_file.parent}\"")
        lines.append(f"manim -qh --media_dir \"{media_dir}\" \"{python_file}\" \"{scene_class_name}\"")
        lines.append("")

        # 完成信息
        lines.append("echo \"==========================================\"")
        lines.append("echo \"渲染完成: $(date)\"")
        lines.append(f"echo \"输出目录: {media_dir}\"")
        lines.append("echo \"==========================================\"")
        lines.append("")

        return "\n".join(lines)

    def submit(self, script_path: Path) -> str:
        """
        提交 Slurm 任务

        对调度器瞬态错误 (sbatch 非零退出) 进行有界重试.

        Args:
            script_path: .sh 脚本路径

        Returns:
            Job ID

        Raises:
            RuntimeError: 提交失败时抛出
        """
        console.print(f"[bold blue][Slurm][/] 正在提交任务: {script_path}")

        # 脚本文件不存在则直接报错 (带路径)
        if not script_path.is_file():
            raise RuntimeError(
                f"渲染脚本不存在: {script_path}\n"
                f"请检查 SCENES_DIR ({settings.SCENES_DIR}) 是否正确, "
                f"或前一步代码生成是否成功."
            )

        # 用完整路径, 避免 conda env 下 PATH 不包含 /usr/bin
        sbatch_path = shutil.which("sbatch")
        if sbatch_path is None:
            for guess in ["/usr/bin/sbatch", "/opt/slurm/bin/sbatch"]:
                if Path(guess).is_file():
                    sbatch_path = guess
                    break
        if sbatch_path is None:
            raise RuntimeError(
                "未找到 sbatch 命令,请确认 Slurm 已安装并配置.\n"
                f"查找路径: {shutil.which('sbatch') or '无'}, "
                "/usr/bin/sbatch, /opt/slurm/bin/sbatch"
            )

        last_err = ""
        for attempt in range(1, settings.LLM_MAX_RETRIES + 1):
            try:
                result = subprocess.run(
                    [sbatch_path, str(script_path)],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
            except subprocess.TimeoutExpired:
                last_err = "sbatch 命令超时"
                console.print(f"[yellow][Slurm][/] {last_err}, 重试 {attempt}/{settings.LLM_MAX_RETRIES}")
                time.sleep(settings.LLM_RETRY_BASE_DELAY * attempt)
                continue
            except FileNotFoundError:
                raise RuntimeError(
                    f"sbatch 可执行文件不存在: {sbatch_path}\n"
                    f"脚本路径: {script_path}\n"
                    "请确认 Slurm 客户端已正确安装."
                )

            if result.returncode == 0:
                match = re.search(r"Submitted batch job (\d+)", result.stdout)
                if not match:
                    raise RuntimeError(f"无法从 sbatch 输出中解析 Job ID: {result.stdout}")
                job_id = match.group(1)
                console.print(f"[bold green][Slurm][/] 任务已提交, Job ID: {job_id}")
                return job_id

            # 非零退出: 辨别配置错误 (不可重试) 和瞬态错误 (可重试)
            stderr_text = result.stderr.strip()
            stderr_lower = stderr_text.lower()
            fatal_keywords = [
                "invalid account", "invalid partition", "invalid qos",
                "account/partition", "permission denied", "access denied",
            ]
            is_fatal = any(kw in stderr_lower for kw in fatal_keywords)

            # 立即显示真实的 sbatch 错误 (不等重试结束)
            console.print("[bold red][Slurm] sbatch 错误:[/]")
            console.print(stderr_text, markup=False)

            if is_fatal:
                raise RuntimeError(
                    f"Slurm 配置错误 (不可重试):\n{stderr_text}\n"
                    "请检查 .env 中的 SLURM_PARTITION 和 SLURM_QOS."
                )

            last_err = stderr_text
            console.print(
                f"[yellow][Slurm][/] 瞬态错误, 重试 {attempt}/{settings.LLM_MAX_RETRIES}"
            )
            time.sleep(settings.LLM_RETRY_BASE_DELAY * attempt)

        raise RuntimeError(f"sbatch 提交失败 (重试 {settings.LLM_MAX_RETRIES} 次):\n{last_err}")

    def cancel_job(self, job_id: str) -> None:
        """取消指定 Slurm 任务 (用于 Ctrl-C 清理)"""
        try:
            subprocess.run(
                ["scancel", str(job_id)],
                capture_output=True,
                text=True,
                timeout=15,
            )
            console.print(f"[yellow][Slurm][/] 已取消 Job {job_id}[/]")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    @staticmethod
    def _normalize_state(raw: str) -> str:
        """
        归一化 Slurm 状态字符串.

        sacct/squeue 可能返回带后缀的状态, 如 "CANCELLED by 12345".
        只取第一个空白分隔的 token 并大写.
        """
        return raw.strip().split()[0].upper() if raw.strip() else "UNKNOWN"

    def poll_status(self, job_id: str) -> str:
        """
        查询单个任务的状态

        Args:
            job_id: Slurm Job ID

        Returns:
            任务状态字符串 (归一化后, 如 COMPLETED/FAILED/CANCELLED/RUNNING/PENDING/UNKNOWN)
        """
        try:
            result = subprocess.run(
                ["squeue", "-j", job_id, "-h", "-o", "%T"],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return "UNKNOWN"

        status = result.stdout.strip()

        # squeue 返回空表示任务已不在队列中,用 sacct 查询最终状态
        if not status:
            return self._check_final_status(job_id)

        return self._normalize_state(status)

    def _check_final_status(self, job_id: str) -> str:
        """使用 sacct 查询已完成任务的最终状态 (归一化, 去除 ' by N' 后缀)"""
        try:
            result = subprocess.run(
                ["sacct", "-j", job_id, "-n", "-o", "State", "--parsable2"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            # --parsable2 以 | 分隔; 取第一个非空状态 (主作业行)
            for line in result.stdout.strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                state = line.split("|")[0]
                if state:
                    return self._normalize_state(state)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        return "UNKNOWN"

    def wait_for_job(
        self,
        job_id: str,
        scene_id: int,
        poll_interval: int | None = None,
        timeout: int | None = None,
    ) -> bool:
        """
        等待任务完成，运行中自动转发 Slurm stdout 到当前终端

        Args:
            job_id: Slurm Job ID
            scene_id: 场景 ID (用于日志)
            poll_interval: 轮询间隔 (秒)
            timeout: 超时时间 (秒)

        Returns:
            True 表示成功完成, False 表示失败
        """
        interval = poll_interval or settings.MONITOR_POLL_INTERVAL
        max_time = timeout or settings.MONITOR_TIMEOUT
        start_time = time.time()
        unknown_streak = 0
        max_unknown = getattr(settings, "MONITOR_MAX_UNKNOWN", 5)

        # 日志转发
        log_path = settings.LOGS_DIR / f"scene_{scene_id}_{job_id}.out"
        last_pos = 0
        was_running = False

        console.print(f"[bold blue][Monitor][/] 开始监控 Scene {scene_id} (Job {job_id})...")

        while True:
            elapsed = time.time() - start_time
            if elapsed > max_time:
                console.print(f"[bold red][Monitor][/] Scene {scene_id} 超时 ({max_time}s)")
                return False

            status = self.poll_status(job_id)

            # 运行中: 转发 stdout 新增内容到当前终端
            if status == "RUNNING":
                if not was_running:
                    console.print(f"[bold blue][Monitor][/] Scene {scene_id} 开始运行, 转发 Manim 输出:")
                    was_running = True
                try:
                    if log_path.exists():
                        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                            f.seek(last_pos)
                            new = f.read()
                            if new:
                                # 每行加缩进, 与 Rich 输出区分
                                for line in new.rstrip().split("\n"):
                                    if line.strip():
                                        console.print(f"  [dim]{line}[/]", markup=False)
                                last_pos = f.tell()
                except Exception:
                    pass  # 文件读写竞争, 下次再读
            else:
                unknown_streak = 0
                # 结束时再读一次, 捞残留输出
                if was_running and log_path.exists():
                    try:
                        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                            f.seek(last_pos)
                            new = f.read()
                            if new:
                                for line in new.rstrip().split("\n"):
                                    if line.strip():
                                        console.print(f"  [dim]{line}[/]", markup=False)
                    except Exception:
                        pass
                    was_running = False

            if status == "COMPLETED":
                console.print(f"[bold green][Monitor][/] Scene {scene_id} 渲染成功!")
                return True
            elif status in ("FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY", "BOOT_FAIL"):
                console.print(f"[bold red][Monitor][/] Scene {scene_id} 渲染失败: {status}")
                return False
            elif status == "UNKNOWN":
                unknown_streak += 1
                console.print(
                    f"[yellow][Monitor][/] Scene {scene_id}: 状态查询失败 (连续 {unknown_streak}/{max_unknown})[/]"
                )
                if unknown_streak >= max_unknown:
                    console.print(
                        f"[bold red][Monitor][/] Scene {scene_id}: 连续监控失败, 判定为失败[/]"
                    )
                    return False
            else:
                unknown_streak = 0
                console.print(
                    f"[dim][Monitor][/] Scene {scene_id}: {status} "
                    f"({elapsed:.0f}s / {max_time}s)"
                )

            time.sleep(interval)

    def get_error_log(self, scene_id: int, job_id: str) -> str | None:
        """
        读取任务的错误日志

        Args:
            scene_id: 场景 ID
            job_id: Job ID

        Returns:
            错误日志的最后 N 行,如果文件不存在返回 None
        """
        # 日志文件名格式: scene_{id}_{job_id}.err
        log_path = settings.LOGS_DIR / f"scene_{scene_id}_{job_id}.err"

        if not log_path.exists():
            # 精确文件缺失时, 按 job_id 数值最接近的匹配 (而非 mtime, 避免取到重试前旧 job 的日志)
            pattern = f"scene_{scene_id}_*.err"
            matches = list(settings.LOGS_DIR.glob(pattern))
            if matches:
                try:
                    target = int(job_id)
                    # 从文件名解析 job_id, 取数值最接近的
                    def parse_jid(p: Path) -> int:
                        stem = p.stem  # scene_{id}_{job_id}
                        return int(stem.rsplit("_", 1)[-1])
                    log_path = min(matches, key=lambda p: abs(parse_jid(p) - target))
                except (ValueError, IndexError):
                    log_path = max(matches, key=lambda p: p.stat().st_mtime)
            else:
                return None

        try:
            content = log_path.read_text(encoding="utf-8", errors="replace")
            lines = content.split("\n")
            # 只返回最后 N 行
            tail = lines[-settings.LOG_TAIL_LINES:]
            return "\n".join(tail)
        except Exception as e:
            console.print(f"[bold yellow][Monitor][/] 读取日志失败: {e}", markup=False)
            return None

    def submit_scene(
        self,
        scene_id: int,
        python_file: Path,
        scene_class_name: str = "Scene",
    ) -> SlurmJob:
        """
        一站式提交: 生成脚本 + 提交任务

        Args:
            scene_id: 场景 ID
            python_file: Python 代码文件
            scene_class_name: Manim 场景类名

        Returns:
            SlurmJob 信息
        """
        console.print(
            f"[dim][Slurm][/] DEBUG submit_scene: py={python_file} exists={python_file.exists()}, "
            f"SCENES_DIR={settings.SCENES_DIR.resolve()}, sbatch={shutil.which('sbatch')}",
            markup=False,
        )
        script_path = self.generate_script(scene_id, python_file, scene_class_name)
        console.print(f"[dim][Slurm][/] DEBUG: script_path={script_path} exists={script_path.exists()}")
        job_id = self.submit(script_path)

        return SlurmJob(
            job_id=job_id,
            scene_id=scene_id,
            script_path=script_path,
            log_out=settings.LOGS_DIR / f"scene_{scene_id}_{job_id}.out",
            log_err=settings.LOGS_DIR / f"scene_{scene_id}_{job_id}.err",
            status="PENDING",
        )
