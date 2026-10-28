"""批量并行处理模块。

支持从文件读取多个 prompt，并行处理多个动画项目。
"""

from __future__ import annotations

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

from rich.console import Console
from rich.table import Table
from rich.text import Text

from kd1_anime.config import resolve_runtime_path, settings
from kd1_anime.logging import get_logger
from kd1_anime.resources import ResourceCoordinator
from kd1_anime.security import redact_text

logger = get_logger(__name__)
console = Console()


@dataclass
class BatchTask:
    """批量任务定义。"""

    task_id: int
    prompt: str
    output: Path | None = None
    status: Literal[
        "pending", "running", "completed", "dry_run_complete", "waiting", "failed", "interrupted"
    ] = "pending"
    run_id: str | None = None
    error: str | None = None
    state: str | None = None
    run_phase: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None


@dataclass
class BatchConfig:
    """批量处理配置。"""

    max_parallel: int = 3
    dry_run: bool = False
    approve_plan: bool = False
    output_dir: Path | None = None
    incremental: bool = False
    backend: Literal["slurm", "local"] | None = None
    base_run_ids: dict[int, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.approve_plan, bool):
            raise ValueError("approve_plan 必须是 bool")
        if isinstance(self.max_parallel, bool) or not 1 <= self.max_parallel <= 32:
            raise ValueError("max_parallel 必须在 1..32 之间")
        normalized: dict[int, str] = {}
        for raw_task_id, raw_run_id in self.base_run_ids.items():
            task_id = int(raw_task_id)
            run_id = str(raw_run_id)
            if task_id < 1:
                raise ValueError("base_run_ids 的任务 ID 必须大于 0")
            if not re.fullmatch(r"\d{8}-\d{6}-[0-9a-f]{8}", run_id):
                raise ValueError(f"base_run_ids 包含无效 run-id: {run_id!r}")
            normalized[task_id] = run_id
        self.base_run_ids = normalized
        if self.backend not in {None, "slurm", "local"}:
            raise ValueError("backend 必须是 slurm、local 或 null")


def load_prompts_from_file(file_path: Path) -> list[str]:
    """从文件加载 prompts。

    支持的格式：
    - 纯文本：每行一个 prompt
    - JSON：包含 prompts 数组的 JSON 文件
    """
    content = file_path.read_text(encoding="utf-8").strip()

    # 尝试解析为 JSON
    if file_path.suffix == ".json":
        try:
            data = json.loads(content)
            if isinstance(data, list):
                values = data
            elif isinstance(data, dict):
                values = data.get("prompts")
            else:
                values = None
            if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
                raise ValueError("JSON prompt 文件必须是字符串数组或包含 prompts 字符串数组的对象")
            prompts = [item.strip() for item in values if item.strip()]
            if not prompts:
                raise ValueError("prompt 文件未包含有效任务")
            return prompts
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON prompt 文件格式无效: {exc}") from exc

    # 作为纯文本处理，每行一个 prompt
    lines = [line.strip() for line in content.split("\n") if line.strip()]
    # 过滤掉注释行
    prompts = [line for line in lines if not line.startswith("#")]
    if not prompts:
        raise ValueError("prompt 文件未包含有效任务")
    return prompts


def load_batch_config(file_path: Path) -> BatchConfig:
    """从 JSON 文件加载批量配置。"""
    content = file_path.read_text(encoding="utf-8")
    data = json.loads(content)

    return BatchConfig(
        max_parallel=data.get("max_parallel", 3),
        dry_run=data.get("dry_run", False),
        approve_plan=data.get("approve_plan", False),
        output_dir=Path(data["output_dir"]) if "output_dir" in data else None,
        incremental=data.get("incremental", False),
        backend=data.get("backend"),
        base_run_ids=data.get("base_run_ids", {}),
    )


class BatchProcessor:
    """批量并行处理器。"""

    def __init__(self, config: BatchConfig | None = None) -> None:
        self.config = config or BatchConfig()
        self.tasks: list[BatchTask] = []
        self.resources = ResourceCoordinator(
            llm_limit=settings.LLM_PARALLEL_WORKERS,
            visual_llm_limit=settings.VISUAL_LLM_PARALLEL_WORKERS,
            rag_limit=settings.RAG_PARALLEL_WORKERS,
            slurm_limit=settings.SLURM_MAX_IN_FLIGHT,
            local_limit=settings.LOCAL_RENDER_MAX_IN_FLIGHT,
        )
        self._active_lock = threading.RLock()
        self._active_applications: dict[int, object] = {}
        self._cancel_errors: dict[int, str] = {}
        self._secrets = tuple(
            value
            for key, value in settings.model_dump().items()
            if key.endswith("API_KEY") and value
        )
        self._interrupted = threading.Event()

    def add_task(self, prompt: str, output: Path | None = None) -> BatchTask:
        """添加单个任务。"""
        task = BatchTask(
            task_id=len(self.tasks) + 1,
            prompt=prompt,
            output=output,
        )
        self.tasks.append(task)
        return task

    def load_tasks_from_file(self, file_path: Path) -> list[BatchTask]:
        """从文件加载任务。"""
        prompts = load_prompts_from_file(file_path)
        tasks = []
        for prompt in prompts:
            task = self.add_task(prompt)
            tasks.append(task)
        return tasks

    def _safe_text(self, value: object) -> str:
        return redact_text(value, self._secrets)

    def _observe_view(self, task: BatchTask, application):
        view = application.run_view(task.run_id)
        task.state = view.state
        task.run_phase = view.run_phase
        return view

    def _execute_single_task(self, task: BatchTask) -> BatchTask:
        """Execute one independent application without accessing runtime internals."""
        task.status = "running"
        task.start_time = datetime.now()
        application = None
        final_video = None
        execution_error = None

        def observe(name: str, data: dict) -> None:
            if name == "run_started":
                task.run_id = data["run_id"]

        try:
            from kd1_anime.application import Application

            if self._interrupted.is_set():
                return task
            application = Application(resource_coordinator=self.resources)
            with self._active_lock:
                self._active_applications[task.task_id] = application
            if self._interrupted.is_set():
                application.cancel_all()
                return task
            options = {
                "callback": observe,
                "dry_run": self.config.dry_run,
                "interactive": False,
                "approve_plan": self.config.approve_plan,
                "output_path": self._task_output_path(task),
            }
            if self.config.backend is not None:
                options["backend"] = self.config.backend
            if self.config.incremental and task.task_id in self.config.base_run_ids:
                final_video = application.run_incremental(
                    task.prompt, self.config.base_run_ids[task.task_id], **options
                )
            else:
                final_video = application.generate(task.prompt, **options)
        except Exception as exc:
            execution_error = self._safe_text(exc)
            task.status = "failed"
            task.error = execution_error
            logger.error("任务 %s 失败: %s", task.task_id, execution_error)
        finally:
            if application is not None and task.run_id is not None:
                try:
                    view = self._observe_view(task, application)
                    if execution_error is None and not self._interrupted.is_set():
                        if view.status in {"completed", "dry_run_complete"}:
                            task.status = view.status
                            task.error = None
                            if view.status == "completed":
                                task.output = final_video
                        elif view.status in {"failed", "interrupted"}:
                            task.status = view.status
                            task.error = self._safe_text(view.stop_reason or view.status)
                        elif view.status == "running":
                            task.status = "waiting"
                            task.error = self._safe_text(
                                view.stop_reason or "等待用户操作；请检查并恢复此运行"
                            )
                        else:
                            task.status = "failed"
                            task.error = self._safe_text(f"无法识别运行状态: {view.status}")
                except Exception as exc:
                    if execution_error is None:
                        task.status = "failed"
                        task.error = self._safe_text(exc)
            elif execution_error is None and not self._interrupted.is_set():
                task.status = "failed"
                task.error = "Application 未提供 run_started 运行标识"
            with self._active_lock:
                self._active_applications.pop(task.task_id, None)
                cancel_error = self._cancel_errors.get(task.task_id)
            if self._interrupted.is_set():
                task.status = "interrupted"
                task.error = (
                    cancel_error or execution_error or task.error or "用户中断，已请求取消任务"
                )
            if task.status != "completed":
                task.output = None
            task.end_time = datetime.now()

        return task

    def _task_output_path(self, task: BatchTask) -> Path | None:
        if task.output is not None:
            return resolve_runtime_path(task.output)
        if self.config.output_dir is None:
            return None
        output_dir = resolve_runtime_path(self.config.output_dir)
        return output_dir / f"task_{task.task_id:03d}.mp4"

    def _validate_output_targets(self) -> None:
        targets = [self._task_output_path(task) for task in self.tasks]
        concrete = [target for target in targets if target is not None]
        if len({str(target) for target in concrete}) != len(concrete):
            raise ValueError("批量任务包含重复输出路径")
        for target in concrete:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and not settings.OVERWRITE_OUTPUT:
                raise FileExistsError(
                    self._safe_text(f"输出文件已存在且 OVERWRITE_OUTPUT=false: {target}")
                )

    def execute_all(self) -> list[BatchTask]:
        """并行执行所有任务。

        避免在并发 worker 打印期间全屏 clear/重绘: 逐条追加结果行,
        全部结束后再一次性输出汇总表 (此时无并发写, 可安全使用 Table)。
        """
        if not self.tasks:
            console.print("[yellow]没有任务可执行[/]")
            return []
        self._validate_output_targets()

        console.print(
            f"[cyan]开始批量处理[/] 共 {len(self.tasks)} 个任务，最大并行数: {self.config.max_parallel}"
        )

        completed_tasks: list[BatchTask] = []
        executor = ThreadPoolExecutor(max_workers=self.config.max_parallel)
        future_to_task = {
            executor.submit(self._execute_single_task, task): task for task in self.tasks
        }
        try:
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    task = future.result()
                except Exception as exc:
                    task.status = "failed"
                    task.error = self._safe_text(exc)
                    task.end_time = datetime.now()
                    logger.error("任务 %s worker 异常: %s", task.task_id, task.error)
                completed_tasks.append(task)
                elapsed = ""
                if task.start_time and task.end_time:
                    elapsed = f"{(task.end_time - task.start_time).total_seconds():.1f}s"
                if task.status in {"completed", "dry_run_complete"}:
                    color = "green"
                elif task.status in {"interrupted", "waiting"}:
                    color = "yellow"
                else:
                    color = "red"
                detail = task.output if task.status == "completed" else task.error or ""
                run_label = f" run_id={self._safe_text(task.run_id)}" if task.run_id else ""
                console.print(
                    Text(
                        f"任务 {task.task_id} {task.status} ({elapsed}){run_label} "
                        f"{self._safe_text(detail)}",
                        style=color,
                    )
                )
        except KeyboardInterrupt:
            self.cancel_all()
            for future in future_to_task:
                future.cancel()
            for task in self.tasks:
                if task.status in {"pending", "running"}:
                    task.status = "interrupted"
                    task.error = self._cancel_errors.get(task.task_id) or "用户中断，已请求取消任务"
                    task.output = None
                    task.end_time = datetime.now()
            # 等待 worker 完成自己的清理，避免退出时遗留远端作业。
            executor.shutdown(wait=True, cancel_futures=True)
            completed_tasks = list(self.tasks)
            console.print("[yellow]批量处理已中断；已请求取消活动任务，请检查运行状态[/]")
        else:
            executor.shutdown(wait=True)

        # 汇总表 (全部结束后再打印, 避免与并发输出交错)
        table = Table(title="批量处理结果")
        table.add_column("任务 ID", justify="right")
        table.add_column("Prompt", max_width=50)
        table.add_column("状态")
        table.add_column("Run ID", overflow="fold")
        table.add_column("耗时")
        table.add_column("输出/错误")
        for task in sorted(completed_tasks, key=lambda t: t.task_id):
            elapsed = ""
            if task.start_time and task.end_time:
                elapsed = f"{(task.end_time - task.start_time).total_seconds():.1f}s"
            status_text = {
                "completed": Text("完成", style="green"),
                "dry_run_complete": Text("dry_run_complete", style="green"),
                "waiting": Text("WAITING", style="yellow"),
                "failed": Text("失败", style="red"),
                "interrupted": Text("中断", style="yellow"),
                "running": Text("运行中", style="yellow"),
                "pending": Text("待执行", style="dim"),
            }.get(task.status, Text(task.status))
            prompt = self._safe_text(task.prompt)
            output_or_error = self._safe_text(task.output or task.error or "")
            table.add_row(
                str(task.task_id),
                Text(prompt[:50] + ("..." if len(prompt) > 50 else "")),
                status_text,
                Text(self._safe_text(task.run_id or "")),
                elapsed,
                Text(output_or_error),
            )
        console.print(table)

        completed = sum(1 for t in completed_tasks if t.status == "completed")
        failed = sum(1 for t in completed_tasks if t.status == "failed")
        interrupted = sum(1 for t in completed_tasks if t.status == "interrupted")
        waiting = sum(1 for t in completed_tasks if t.status == "waiting")
        dry_run = sum(1 for t in completed_tasks if t.status == "dry_run_complete")
        console.print(
            f"[bold]批量处理结束[/] 成功: {completed}, dry-run: {dry_run}, "
            f"等待: {waiting}, 失败: {failed}, 中断: {interrupted}"
        )
        return completed_tasks

    def cancel_all(self) -> None:
        """Request cancellation through each active application's public policy boundary."""
        self._interrupted.set()
        with self._active_lock:
            active = list(self._active_applications.items())
        for task_id, application in active:
            try:
                application.cancel_all()
            except Exception as exc:
                error = self._safe_text(f"取消失败，外部作业停止状态未确认: {exc}")
                with self._active_lock:
                    self._cancel_errors[task_id] = error
                logger.warning("取消批量任务 %s 失败: %s", task_id, error)

    def generate_summary(self, tasks: list[BatchTask]) -> str:
        """生成批量处理摘要。"""
        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        interrupted = sum(1 for t in tasks if t.status == "interrupted")
        waiting = sum(1 for t in tasks if t.status == "waiting")
        dry_run = sum(1 for t in tasks if t.status == "dry_run_complete")

        total_time = 0
        for task in tasks:
            if task.start_time and task.end_time:
                total_time += (task.end_time - task.start_time).total_seconds()

        summary = f"""
批量处理摘要
============
总任务数: {len(tasks)}
成功: {completed}
失败: {failed}
中断: {interrupted}
等待: {waiting}
dry-run: {dry_run}
总耗时: {total_time:.1f}秒

详细结果:
"""
        for task in tasks:
            output = self._safe_text(task.output or task.error or "N/A")
            run_label = f" run_id={self._safe_text(task.run_id)}" if task.run_id else ""
            summary += f"  任务 {task.task_id}: {task.status}{run_label} {output}\n"

        return summary
