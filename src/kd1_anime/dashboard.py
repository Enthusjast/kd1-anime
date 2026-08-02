"""并行场景状态仪表盘 — Rich Live 渲染多场景进度。

在 Planner/Coder/Reviewer/Slurm/AutoFixer 并行阶段展示每个 Scene 的
状态图标与最近事件，不展示详细日志文本。非 TTY 环境自动降级为无操作。
"""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

# 全局标志：dashboard 激活时抑制 Agent 直接打印，避免破坏 Live 渲染
_active = False
_lock = threading.Lock()
_current: "SceneDashboard | None" = None


def suspend_all():
    """上下文管理器：暂停当前 Live 仪表盘，允许其他输出后再恢复。

    用于交互式询问（Confirm.ask 等）或需要直接打印的场景。
    """
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        global _current, _active
        dash = _current
        if dash is None or dash.live is None:
            yield
            return
        with _lock:
            _active = False
        try:
            dash.live.stop()
        except Exception:
            pass
        try:
            yield
        finally:
            with _lock:
                _active = True
            try:
                dash.live.start()
            except Exception:
                pass

    return _ctx()


def is_active() -> bool:
    return _active


def suppress_agent_logs() -> bool:
    """返回是否应抑制 Agent 日志（dashboard 激活时）。"""
    return _active


@dataclass
class SceneStatus:
    """单个场景的仪表盘状态。"""

    scene_id: int
    title: str = ""
    stage: str = ""          # 当前阶段: planning/coding/reviewing/rendering/fixing
    state: str = "pending"   # pending/running/completed/failed/skipped
    message: str = ""        # 最近事件摘要

    @property
    def icon(self) -> str:
        """状态图标。"""
        return {
            "pending": "⏳",
            "running": "⟳",
            "completed": "✓",
            "failed": "✗",
            "skipped": "–",
        }.get(self.state, "?")

    @property
    def color(self) -> str:
        return {
            "pending": "dim",
            "running": "cyan",
            "completed": "green",
            "failed": "red",
            "skipped": "dim",
        }.get(self.state, "white")

    def render_row(self) -> list[Text]:
        """生成表格行。"""
        icon = Text(f"  {self.icon}", style=self.color)
        title = Text(self.title, style=self.color)
        stage_txt = Text(self.stage, style="dim")
        msg = Text(self.message[:40], style="dim")
        return [icon, title, stage_txt, msg]


class SceneDashboard:
    """Rich Live 场景状态仪表盘。"""

    def __init__(self) -> None:
        self.live: Live | None = None
        self.stage: str = ""
        self.stage_label: str = ""
        self.scenes: dict[int, SceneStatus] = {}
        self.total: int = 0
        self._event_lock = threading.Lock()

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """启动 Live 仪表盘；非 TTY 环境返回 False。"""
        global _active, _current
        if not sys.stdout.isatty():
            return False
        try:
            self.live = Live(
                self._render(),
                console=console,
                refresh_per_second=10,
                transient=False,
            )
            self.live.start()
            with _lock:
                _active = True
                _current = self
            return True
        except Exception:
            self.live = None
            return False

    def stop(self) -> None:
        """停止 Live 仪表盘。"""
        global _active, _current
        with _lock:
            _active = False
            if _current is self:
                _current = None
        if self.live:
            try:
                self.live.stop()
            except Exception:
                pass
            self.live = None

    # ------------------------------------------------------------------
    # 事件处理
    # ------------------------------------------------------------------

    def on_event(self, event: str, data: dict) -> None:
        """根据 orchestrator 事件更新仪表盘状态。"""
        if not self.live:
            return
        with self._event_lock:
            self._apply_event(event, data)
        try:
            if self.live:
                self.live.update(self._render(), refresh=True)
        except Exception:
            pass

    def _apply_event(self, event: str, data: dict) -> None:
        scene_id = data.get("scene_id")
        if scene_id is not None:
            status = self.scenes.setdefault(
                int(scene_id), SceneStatus(scene_id=int(scene_id))
            )
        else:
            status = None

        if event == "stage_start":
            self.stage = data.get("stage", "")
            self.stage_label = {
                "planning": "场景概要",
                "detailing": "导演分镜",
                "coding": "代码生成",
                "reviewing": "代码审查",
                "dispatching": "提交渲染",
                "monitoring": "监控渲染",
                "fixing": "自动修复",
                "merging": "视频拼接",
                "evaluating": "质量评估",
            }.get(self.stage, self.stage)

        elif event == "plan_complete":
            scenes = data.get("scenes", [])
            self.total = len(scenes)
            for scene in scenes:
                sid = scene.scene_id
                self.scenes.setdefault(sid, SceneStatus(scene_id=sid))
                self.scenes[sid].title = scene.title

        elif event in ("scene_detailing", "scene_coding", "scene_rewriting", "scene_fixing"):
            if status:
                status.state = "running"
                if event == "scene_detailing":
                    status.stage = "分镜"
                    status.title = status.title or data.get("title", "")
                    status.message = "生成分镜中"
                elif event == "scene_coding":
                    status.stage = "编码"
                    status.message = "生成代码中"
                elif event == "scene_rewriting":
                    status.stage = "编码"
                    status.message = "修正代码中"
                elif event == "scene_fixing":
                    status.stage = "修复"
                    status.message = f"自动修复 #{data.get('attempt', 0)}"

        elif event in ("scene_detailed", "scene_coded"):
            if status:
                status.state = "completed"
                if event == "scene_detailed":
                    status.stage = "分镜"
                    status.message = "分镜完成"
                else:
                    status.stage = "编码"
                    status.message = "代码就绪"

        elif event == "scene_review_pass":
            if status:
                status.state = "completed"
                status.stage = "审查"
                status.message = "审查通过"

        elif event == "scene_review_fail":
            if status:
                status.state = "running"
                status.stage = "审查"
                status.message = "需修正"

        elif event == "scene_submitted":
            if status:
                status.state = "running"
                status.stage = "渲染"
                status.message = f"Job {data.get('job_id', '?')}"

        elif event == "scene_rendered":
            if status:
                status.state = "completed"
                status.stage = "渲染"
                status.message = "渲染完成"

        elif event == "scene_failed":
            if status:
                status.state = "failed"
                status.message = (data.get("reason", "") or "")[:40]

        elif event == "scene_give_up":
            if status:
                status.state = "failed"
                status.message = "已放弃"

    # ------------------------------------------------------------------
    # 渲染
    # ------------------------------------------------------------------

    def _render(self) -> Panel:
        total = self.total or max(len(self.scenes), 1)
        completed = sum(1 for s in self.scenes.values() if s.state == "completed")
        failed = sum(1 for s in self.scenes.values() if s.state == "failed")

        header = Text()
        header.append(f"  {self.stage_label or '流水线'}  ", style="bold white")
        header.append(
            f"完成 {completed}/{total}",
            style="green" if completed == total else "yellow",
        )
        if failed:
            header.append(f"  失败 {failed}", style="red")

        table = Table(
            show_header=True,
            header_style="bold cyan",
            box=None,
            padding=(0, 1),
            expand=True,
        )
        table.add_column("状态", justify="center", width=6)
        table.add_column("场景", style="white")
        table.add_column("阶段", justify="center", width=6)
        table.add_column("最近事件", style="dim", overflow="ellipsis")

        if not self.scenes:
            table.add_row(Text("  …", style="dim"), Text("等待场景规划…", style="dim"), "", "")
        else:
            for sid in sorted(self.scenes):
                status = self.scenes[sid]
                icon, title, stage_txt, msg = status.render_row()
                table.add_row(icon, title, stage_txt, msg)

        return Panel(
            table,
            title="[bold cyan]kd1-anime[/]",
            border_style="cyan",
            subtitle=f"[dim]{self.stage}[/]" if self.stage else None,
        )
