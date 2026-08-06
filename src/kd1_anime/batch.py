"""批量并行处理模块。

支持从文件读取多个 prompt，并行处理多个动画项目。
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

from rich.console import Console
from rich.table import Table

from kd1_anime.logging import get_logger

logger = get_logger(__name__)
console = Console()


@dataclass
class BatchTask:
    """批量任务定义。"""
    task_id: int
    prompt: str
    output: Path | None = None
    status: Literal["pending", "running", "completed", "failed"] = "pending"
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
    base_run_ids: dict[int, str] = field(default_factory=dict)


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
                return [str(p) for p in data if p]
            elif isinstance(data, dict) and "prompts" in data:
                return [str(p) for p in data["prompts"] if p]
        except json.JSONDecodeError:
            pass
    
    # 作为纯文本处理，每行一个 prompt
    lines = [line.strip() for line in content.split("\n") if line.strip()]
    # 过滤掉注释行
    return [line for line in lines if not line.startswith("#")]


def load_batch_config(file_path: Path) -> BatchConfig:
    """从 JSON 文件加载批量配置。"""
    content = file_path.read_text(encoding="utf-8")
    data = json.loads(content)
    
    return BatchConfig(
        max_parallel=data.get("max_parallel", 3),
        dry_run=data.get("dry_run", False),
        output_dir=Path(data["output_dir"]) if "output_dir" in data else None,
        incremental=data.get("incremental", False),
        base_run_ids=data.get("base_run_ids", {}),
    )


class BatchProcessor:
    """批量并行处理器。"""
    
    def __init__(self, config: BatchConfig | None = None) -> None:
        self.config = config or BatchConfig()
        self.tasks: list[BatchTask] = []
    
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
        from kd1_anime.orchestrator import Orchestrator
        
        task.status = "running"
        task.start_time = datetime.now()
        
        try:
            orchestrator = Orchestrator()
            
            if self.config.incremental and task.task_id in self.config.base_run_ids:
                # 增量渲染模式
                base_run_id = self.config.base_run_ids[task.task_id]
                final_video = orchestrator.run_incremental(
                    task.prompt,
                    base_run_id,
                    dry_run=self.config.dry_run,
                )
            else:
                # 普通渲染模式
                final_video = orchestrator.run(
                    task.prompt,
                    dry_run=self.config.dry_run,
                )
            
            task.status = "completed"
            if final_video:
                task.output = final_video
                logger.info(f"任务 {task.task_id} 完成: {final_video}")
        
        except Exception as exc:
            task.status = "failed"
            task.error = str(exc)
            logger.error(f"任务 {task.task_id} 失败: {exc}")
        
        finally:
            task.end_time = datetime.now()
        
        return task
    
    def execute_all(self) -> list[BatchTask]:
        """并行执行所有任务。

        避免在并发 worker 打印期间全屏 clear/重绘: 逐条追加结果行,
        全部结束后再一次性输出汇总表 (此时无并发写, 可安全使用 Table)。
        """
        if not self.tasks:
            console.print("[yellow]没有任务可执行[/]")
            return []

        console.print(f"[cyan]开始批量处理[/] 共 {len(self.tasks)} 个任务，最大并行数: {self.config.max_parallel}")

        completed_tasks: list[BatchTask] = []
        with ThreadPoolExecutor(max_workers=self.config.max_parallel) as executor:
            future_to_task = {
                executor.submit(self._execute_single_task, task): task
                for task in self.tasks
            }
            for future in as_completed(future_to_task):
                task = future.result()
                completed_tasks.append(task)
                elapsed = ""
                if task.start_time and task.end_time:
                    elapsed = f"{(task.end_time - task.start_time).total_seconds():.1f}s"
                icon = "✓" if task.status == "completed" else "✗"
                color = "green" if task.status == "completed" else "red"
                detail = task.output if task.status == "completed" else task.error or ""
                console.print(
                    f"[{color}]{icon} 任务 {task.task_id}[/] "
                    f"{task.status} ({elapsed}) {str(detail)[-60:]}",
                    markup=False,
                )

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
        console.print(f"[bold]批量处理完成[/] 成功: {completed}, 失败: {failed}")
        return completed_tasks
    
    def generate_summary(self, tasks: list[BatchTask]) -> str:
        """生成批量处理摘要。"""
        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        
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
总耗时: {total_time:.1f}秒

详细结果:
"""
        for task in tasks:
            status = "✓" if task.status == "completed" else "✗"
            output = str(task.output) if task.output else task.error or "N/A"
            summary += f"  {status} 任务 {task.task_id}: {output}\n"
        
        return summary
