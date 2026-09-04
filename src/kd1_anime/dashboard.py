"""并行场景状态仪表盘 — Rich Live 渲染多场景进度。

在 Planner/Coder/Reviewer/Slurm/AutoFixer 并行阶段展示每个 Scene 的
状态图标与最近事件，不展示详细日志文本。非 TTY 环境自动降级为无操作。
"""

from __future__ import annotations

import sys
import threading
import time
from contextlib import suppress
from dataclasses import dataclass, field

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()


class _DashboardState:
    """模块级共享状态 (避免用 global 语句)."""

    def __init__(self) -> None:
        self.active = False
        self.current: SceneDashboard | None = None
        self.lock = threading.Lock()


_state = _DashboardState()


def suspend_all():
    """上下文管理器：暂停当前 Live 仪表盘，允许其他输出后再恢复。

    用于交互式询问（Confirm.ask 等）或需要直接打印的场景。
    """
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        dash = _state.current
        if dash is None or dash.live is None:
            yield
            return
        with _state.lock:
            _state.active = False
        with suppress(Exception):
            dash.live.stop()
        try:
            yield
        finally:
            with _state.lock:
                _state.active = True
            with suppress(Exception):
                dash.live.start()

    return _ctx()


def is_active() -> bool:
    return _state.active


def quiet() -> bool:
    """仪表盘激活时，外部模块应静默普通 console 输出，避免破坏 Live。"""
    return _state.active


def suppress_agent_logs() -> bool:
    """返回是否应抑制 Agent 日志（dashboard 激活时）。"""
    return _state.active


STAGES = ("分镜", "编码", "审查", "渲染")


@dataclass
class SceneStatus:
    """单个场景的仪表盘状态。"""

    scene_id: int
    title: str = ""
    stage: str = ""  # 当前活动阶段 ("" = 阶段间隙, 等待下一个事件)
    state: str = "pending"  # pending/running/completed/failed/skipped
    message: str = ""  # 最近事件摘要
    started_at: float = 0.0  # 当前阶段开始时间 (用于显示耗时)
    done: list[str] = field(default_factory=list)  # 已完成的流水线阶段

    def mark_done(self, stage_name: str) -> None:
        """记录一个阶段完成 (去重)。"""
        if stage_name not in self.done:
            self.done.append(stage_name)

    def invalidate_from(self, stage_name: str) -> None:
        """重做某阶段时清除该阶段及所有下游的完成标记。"""

        start = STAGES.index(stage_name)
        self.done = [name for name in self.done if STAGES.index(name) < start]

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
        """生成表格行: 状态图标 + 标题 + 阶段流水线 + 最近事件。"""
        icon = Text(f"  {self.icon}", style=self.color)
        title = Text(self.title, style=self.color)
        pipeline = Text()
        for idx, name in enumerate(STAGES):
            if name in self.done:
                pipeline.append(f"{name}✓", style="green")
            elif name == self.stage:
                pipeline.append(f"{name}⟳", style="cyan")
            else:
                pipeline.append(f"{name}·", style="dim")
            if idx < len(STAGES) - 1:
                pipeline.append("  ")
        if self.stage == "修复":
            pipeline.append("  ", style="dim")
            pipeline.append("修复⟳", style="cyan")
        message = self.message
        if self.state == "running" and self.started_at and self.stage:
            message = f"{message} · {int(time.time() - self.started_at)}s"
        msg = Text(message[:40], style="dim")
        return [icon, title, pipeline, msg]


class SceneDashboard:
    """Rich Live 场景状态仪表盘。"""

    def __init__(self) -> None:
        self.live: Live | None = None
        self.stage: str = ""
        self.stage_label: str = ""
        self.run_id: str = ""
        self.started_at: float = 0.0
        self.scenes: dict[int, SceneStatus] = {}
        self.total: int = 0
        self._event_lock = threading.Lock()

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """启动 Live 仪表盘；非 TTY 环境返回 False。"""
        if not sys.stdout.isatty():
            return False
        try:
            self.live = Live(
                self._render,  # 传 callable: Rich 每次刷新都会重新渲染, 耗时/进度实时更新
                console=console,
                refresh_per_second=10,
                transient=False,
            )
            self.live.start()
            with _state.lock:
                _state.active = True
                _state.current = self
            return True
        except Exception:
            self.live = None
            return False

    def stop(self) -> None:
        """停止 Live 仪表盘。"""
        with _state.lock:
            _state.active = False
            if _state.current is self:
                _state.current = None
        if self.live:
            with suppress(Exception):
                self.live.stop()
            self.live = None

    # ------------------------------------------------------------------
    # 事件处理
    # ------------------------------------------------------------------

    def on_event(self, event: str, data: dict) -> None:
        """根据 orchestrator 事件更新仪表盘状态 (线程安全)。"""
        if not self.live:
            return
        # 状态更新与 Live 重绘放在同一把锁内: Rich Live 非线程安全,
        # 多个场景线程并发调用时避免画面错乱。
        with self._event_lock:
            self._apply_event(event, data)
            with suppress(Exception):
                self.live.update(self._render(), refresh=True)

    def _mark_running(self, status: SceneStatus, stage: str, message: str) -> None:
        status.state = "running"
        status.stage = stage
        status.message = message
        status.started_at = time.time()

    def _apply_event(self, event: str, data: dict) -> None:
        scene_id = data.get("scene_id")
        if scene_id is not None:
            status = self.scenes.setdefault(int(scene_id), SceneStatus(scene_id=int(scene_id)))
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

        elif event in ("run_started", "run_resumed"):
            self.run_id = data.get("run_id", "") or self.run_id
            if not self.started_at:
                self.started_at = time.time()

        elif event == "plan_complete":
            scenes = data.get("scenes", [])
            self.total = len(scenes)
            if not self.started_at:
                self.started_at = time.time()
            for scene in scenes:
                sid = scene.scene_id
                self.scenes.setdefault(sid, SceneStatus(scene_id=sid))
                self.scenes[sid].title = scene.title

        elif event in (
            "scene_detailing",
            "scene_coding",
            "scene_rewriting",
            "scene_reviewing",
            "scene_fixing",
            "scene_retrying",
        ):
            if status:
                if event == "scene_detailing":
                    status.invalidate_from("分镜")
                    status.title = status.title or data.get("title", "")
                    self._mark_running(status, "分镜", "生成分镜中")
                elif event == "scene_coding":
                    status.invalidate_from("编码")
                    self._mark_running(status, "编码", "生成代码中")
                elif event == "scene_rewriting":
                    status.invalidate_from("编码")
                    self._mark_running(status, "编码", "修正代码中")
                elif event == "scene_reviewing":
                    status.invalidate_from("审查")
                    self._mark_running(status, "审查", "代码审查中")
                elif event == "scene_fixing":
                    status.invalidate_from("审查")
                    self._mark_running(status, "修复", f"自动修复 #{data.get('attempt', 0)}")
                elif event == "scene_retrying":
                    status.invalidate_from("渲染")
                    self._mark_running(status, "渲染", "基础设施故障，重新排队")

        elif event in ("scene_detailed", "scene_coded"):
            if status:
                # 中间阶段完成: 场景仍处于进行中 (青色), 只累积阶段 ✓
                status.state = "running"
                status.stage = ""
                status.started_at = 0.0
                if event == "scene_detailed":
                    status.mark_done("分镜")
                    status.message = "分镜完成"
                else:
                    status.mark_done("编码")
                    status.message = "代码就绪"

        elif event in ("scene_review_pass", "scene_review_skipped"):
            if status:
                status.state = "running"
                status.stage = ""
                status.started_at = 0.0
                status.mark_done("审查")
                status.message = "审查通过" if event == "scene_review_pass" else "已跳过审查"

        elif event == "scene_review_fail":
            if status:
                self._mark_running(status, "审查", "需修正")

        elif event == "scene_submitted":
            if status:
                self._mark_running(status, "渲染", f"Job {data.get('job_id', '?')}")

        elif event in ("scene_rendered", "scene_reused"):
            # 只有渲染完成 (或复用旧视频) 才算整个场景完成, 显示为绿色
            if status:
                status.state = "completed"
                status.stage = "渲染"
                status.started_at = 0.0
                status.mark_done("渲染")
                status.message = "渲染完成" if event == "scene_rendered" else "复用旧视频"

        elif event == "scene_failed":
            if status:
                status.state = "failed"
                status.started_at = 0.0
                status.message = (data.get("reason", "") or "")[:40]

        elif event == "scene_give_up":
            if status:
                status.state = "failed"
                status.started_at = 0.0
                status.message = "已放弃"

        elif event == "merge_complete":
            self.stage = "merging"
            self.stage_label = "视频拼接"

        elif event == "dry_run_complete":
            self.stage = "dry-run"
            self.stage_label = "Dry-run 完成"

    # ------------------------------------------------------------------
    # 渲染
    # ------------------------------------------------------------------

    def _running_phases(self) -> dict[str, int]:
        """聚合当前运行中各阶段 (并行时头部显示 分镜×2 编码×1 ...)。"""
        counts: dict[str, int] = {}
        for scene in self.scenes.values():
            if scene.state == "running" and scene.stage:
                counts[scene.stage] = counts.get(scene.stage, 0) + 1
        return counts

    def _render(self) -> Panel:
        total = self.total
        completed = sum(1 for s in self.scenes.values() if s.state == "completed")
        failed = sum(1 for s in self.scenes.values() if s.state == "failed")

        header = Text()
        header.append(f"  {self.stage_label or '流水线'}  ", style="bold white")
        running = self._running_phases()
        if running:
            header.append(
                "运行中 " + " ".join(f"{k}×{v}" for k, v in sorted(running.items())) + "  ",
                style="cyan",
            )
        if total:
            header.append(
                f"完成 {completed}/{total}",
                style="green" if completed == total else "yellow",
            )
        elif completed:
            header.append(f"完成 {completed}", style="yellow")
        if failed:
            header.append(f"  失败 {failed}", style="red")
        if self.started_at:
            header.append(f"  用时 {int(time.time() - self.started_at)}s", style="dim")

        table = Table(
            show_header=True,
            header_style="bold cyan",
            box=None,
            padding=(0, 1),
            expand=True,
        )
        table.add_column("状态", justify="center", width=6)
        table.add_column("场景", style="white")
        table.add_column("进度", justify="left", min_width=24)
        table.add_column("最近事件", style="dim", overflow="ellipsis")

        if not self.scenes:
            table.add_row(Text("  …", style="dim"), Text("等待场景规划…", style="dim"), "", "")
        else:
            for sid in sorted(self.scenes):
                icon, title, stage_txt, msg = self.scenes[sid].render_row()
                table.add_row(icon, title, stage_txt, msg)

        title = "[bold cyan]kd1-anime[/]"
        if self.run_id:
            title += f"  [dim]{self.run_id}[/]"
        # header 在表格上方显示阶段/进度/耗时聚合
        content: object = Group(header, table)
        return Panel(
            content,
            title=title,
            border_style="cyan",
            subtitle=f"[dim]{self.stage}[/]" if self.stage else None,
        )
