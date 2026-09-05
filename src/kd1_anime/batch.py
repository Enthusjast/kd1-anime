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

from kd1_anime.config import resolve_runtime_path, settings
from kd1_anime.logging import get_logger
from kd1_anime.resources import ResourceCoordinator

logger = get_logger(__name__)
console = Console()


@dataclass
class BatchTask:
    """批量任务定义。"""

    task_id: int
    prompt: str
    output: Path | None = None
    status: Literal["pending", "running", "completed", "failed", "interrupted"] = "pending"
    run_id: str | None = None
    error: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None


@dataclass
class BatchConfig:
    """批量处理配置。"""

    max_parallel: int = 3
    dry_run: bool = False
    output_dir: Path | None = None
    incremental: bool = False
    backend: Literal["slurm", "local"] | None = None
    base_run_ids: dict[int, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
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
        self._active_orchestrators: dict[int, object] = {}
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

    def _execute_single_task(self, task: BatchTask) -> BatchTask:
        """执行单个任务。"""
        task.status = "running"
        task.start_time = datetime.now()
        orchestrator: Orchestrator | None = None

        try:
            # 导入和初始化也属于任务范围；如果安装环境损坏，不能让
            # future.result() 把整个批次提前抛出并遗漏其它任务的收尾。
            from kd1_anime.orchestrator import Orchestrator

            if self._interrupted.is_set():
                task.status = "interrupted"
                task.error = "用户中断，任务已停止"
                return task
            orchestrator = Orchestrator(resource_coordinator=self.resources)
            with self._active_lock:
                self._active_orchestrators[task.task_id] = orchestrator
            if self._interrupted.is_set():
                orchestrator.cancel_all()
                task.status = "interrupted"
                task.error = "用户中断，任务已停止"
                return task
            output_path = self._task_output_path(task)

            if self.config.incremental and task.task_id in self.config.base_run_ids:
                # 增量渲染模式
                base_run_id = self.config.base_run_ids[task.task_id]
                run_kwargs = {
                    "dry_run": self.config.dry_run,
                    "output_path": output_path,
                }
                if self.config.backend is not None:
                    run_kwargs["backend"] = self.config.backend
                final_video = orchestrator.run_incremental(task.prompt, base_run_id, **run_kwargs)
            else:
                # 普通渲染模式
                run_kwargs = {
                    "dry_run": self.config.dry_run,
                    "output_path": output_path,
                }
                if self.config.backend is not None:
                    run_kwargs["backend"] = self.config.backend
                final_video = orchestrator.run(task.prompt, **run_kwargs)

            if self._interrupted.is_set():
                task.status = "interrupted"
                task.error = "用户中断，任务已停止"
            else:
                task.status = "completed"
                task.output = final_video
                if final_video:
                    logger.info(f"任务 {task.task_id} 完成: {final_video}")
            task.run_id = orchestrator._ctx.paths.run_id if orchestrator._ctx else None

        except Exception as exc:
            if self._interrupted.is_set():
                task.status = "interrupted"
                task.error = "用户中断，任务已停止"
            else:
                task.status = "failed"
                task.error = str(exc)
                logger.error(f"任务 {task.task_id} 失败: {exc}")

        finally:
            if orchestrator is not None:
                with self._active_lock:
                    self._active_orchestrators.pop(task.task_id, None)
            if task.run_id is None and orchestrator is not None and orchestrator._ctx is not None:
                task.run_id = orchestrator._ctx.paths.run_id
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
                raise FileExistsError(f"输出文件已存在且 OVERWRITE_OUTPUT=false: {target}")

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
                    task.error = str(exc)
                    task.end_time = datetime.now()
                    logger.error("任务 %s worker 异常: %s", task.task_id, exc)
                completed_tasks.append(task)
                elapsed = ""
                if task.start_time and task.end_time:
                    elapsed = f"{(task.end_time - task.start_time).total_seconds():.1f}s"
                if task.status == "completed":
                    icon, color = "✓", "green"
                elif task.status == "interrupted":
                    icon, color = "⏹", "yellow"
                else:
                    icon, color = "✗", "red"
                detail = task.output if task.status == "completed" else task.error or ""
                console.print(
                    f"[{color}]{icon} 任务 {task.task_id}[/] "
                    f"{task.status} ({elapsed}) {str(detail)[-60:]}",
                    markup=False,
                )
        except KeyboardInterrupt:
            self.cancel_all()
            for future in future_to_task:
                future.cancel()
            for task in self.tasks:
                if task.status in {"pending", "running"}:
                    task.status = "interrupted"
                    task.error = "用户中断，任务已停止"
                    task.end_time = datetime.now()
            # 等待 worker 完成自己的清理，避免退出时遗留远端作业。
            executor.shutdown(wait=True, cancel_futures=True)
            completed_tasks = list(self.tasks)
            console.print("[yellow]批量处理已中断，活动任务已取消[/]")
        else:
            executor.shutdown(wait=True)

        # 汇总表 (全部结束后再打印, 避免与并发输出交错)
        table = Table(title="批量处理结果")
        table.add_column("任务 ID", justify="right")
        table.add_column("Prompt", max_width=50)
        table.add_column("状态")
        table.add_column("耗时")
        table.add_column("输出/错误")
        for task in sorted(completed_tasks, key=lambda t: t.task_id):
            elapsed = ""
            if task.start_time and task.end_time:
                elapsed = f"{(task.end_time - task.start_time).total_seconds():.1f}s"
            status_text = {
                "completed": "[green]✓ 完成[/]",
                "failed": "[red]✗ 失败[/]",
                "interrupted": "[yellow]⏹ 中断[/]",
                "running": "[yellow]⟳ 运行中[/]",
                "pending": "[dim]○ 等待中[/]",
            }.get(task.status, task.status)
            output_or_error = ""
            if task.output:
                output_or_error = str(task.output)[-40:]
            elif task.error:
                output_or_error = task.error[-40:]
            table.add_row(
                str(task.task_id),
                task.prompt[:50] + ("..." if len(task.prompt) > 50 else ""),
                status_text,
                elapsed,
                output_or_error,
            )
        console.print(table)

        completed = sum(1 for t in completed_tasks if t.status == "completed")
        failed = sum(1 for t in completed_tasks if t.status == "failed")
        interrupted = sum(1 for t in completed_tasks if t.status == "interrupted")
        console.print(
            f"[bold]批量处理完成[/] 成功: {completed}, 失败: {failed}, 中断: {interrupted}"
        )
        return completed_tasks

    def cancel_all(self) -> None:
        """取消所有正在执行的 Orchestrator 及其远端 Slurm 作业。"""

        self._interrupted.set()
        with self._active_lock:
            active = list(self._active_orchestrators.values())
        for orchestrator in active:
            try:
                orchestrator.cancel_all()
            except Exception as exc:
                logger.warning("取消批量任务失败: %s", exc)

    def generate_summary(self, tasks: list[BatchTask]) -> str:
        """生成批量处理摘要。"""
        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        interrupted = sum(1 for t in tasks if t.status == "interrupted")

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
总耗时: {total_time:.1f}秒

详细结果:
"""
        for task in tasks:
            status = (
                "✓" if task.status == "completed" else "⏹" if task.status == "interrupted" else "✗"
            )
            output = str(task.output) if task.output else task.error or "N/A"
            summary += f"  {status} 任务 {task.task_id}: {output}\n"

        return summary
