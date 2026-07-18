"""
核心状态机 (Orchestrator)
串联所有 Agent 和模块,驱动整个 Manim 动画生成流水线

状态流转:
  INIT -> PLANNING -> CODING -> REVIEWING -> DISPATCHING -> MONITORING -> (FIXING | MERGING) -> DONE

支持可选的 callback 回调,TUI 通过它获取状态更新.
callback 签名: callback(event: str, data: dict) -> None
"""

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Callable

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config import settings
from agents.planner import PlannerAgent, ScenePlan
from agents.coder import CoderAgent
from agents.reviewer import ReviewerAgent
from agents.auto_fixer import AutoFixerAgent
from cluster.slurm import SlurmDispatcher, SlurmJob
from media.merger import VideoMerger

console = Console()

Callback = Callable[[str, dict], None]


class State(Enum):
    """状态机状态"""
    INIT = auto()
    PLANNING = auto()
    CODING = auto()
    REVIEWING = auto()
    DISPATCHING = auto()
    MONITORING = auto()
    FIXING = auto()
    MERGING = auto()
    DONE = auto()
    ERROR = auto()


@dataclass
class SceneState:
    """单个场景的运行时状态"""
    plan: ScenePlan
    code: str = ""
    review_round: int = 0
    fix_attempts: int = 0
    slurm_job: SlurmJob | None = None
    rendered: bool = False
    give_up: bool = False        # 超过修复上限, 放弃该场景
    failed: bool = False         # 该场景处理出错 (与渲染失败区分)


@dataclass
class PipelineContext:
    """流水线上下文"""
    user_prompt: str
    scenes: list[ScenePlan] = field(default_factory=list)
    scene_states: dict[int, SceneState] = field(default_factory=dict)
    final_video: Path | None = None


class Orchestrator:
    """核心状态机,驱动整个流水线"""

    def __init__(self):
        self.planner = PlannerAgent()
        self.coder = CoderAgent()
        self.reviewer = ReviewerAgent()
        self.auto_fixer = AutoFixerAgent()
        self.slurm = SlurmDispatcher()
        self.merger = VideoMerger()
        self._callback: Callback | None = None
        self._ctx: PipelineContext | None = None

    def _emit(self, event: str, **kwargs) -> None:
        """发送回调事件"""
        if self._callback:
            self._callback(event, kwargs)

    def cancel_all(self) -> None:
        """取消所有已提交但未完成的 Slurm 任务 (用于 Ctrl-C 清理)"""
        if not self._ctx:
            return
        for ss in self._ctx.scene_states.values():
            if ss.slurm_job and not ss.rendered:
                self.slurm.cancel_job(ss.slurm_job.job_id)

    def run(self, user_prompt: str, callback: Callback | None = None) -> Path:
        """
        执行完整的动画生成流水线

        Args:
            user_prompt: 用户的自然语言需求
            callback: 可选的状态回调函数

        Returns:
            最终输出视频路径
        """
        self._callback = callback
        ctx = PipelineContext(user_prompt=user_prompt)
        self._ctx = ctx
        state = State.INIT

        while state != State.DONE and state != State.ERROR:
            try:
                match state:
                    case State.INIT:
                        state = self._handle_init(ctx)
                    case State.PLANNING:
                        state = self._handle_planning(ctx)
                    case State.CODING:
                        state = self._handle_coding(ctx)
                    case State.REVIEWING:
                        state = self._handle_reviewing(ctx)
                    case State.DISPATCHING:
                        state = self._handle_dispatching(ctx)
                    case State.MONITORING:
                        state = self._handle_monitoring(ctx)
                    case State.FIXING:
                        state = self._handle_fixing(ctx)
                    case State.MERGING:
                        state = self._handle_merging(ctx)
            except KeyboardInterrupt:
                console.print("[bold yellow][Orchestrator][/] 中断, 正在清理 Slurm 任务...[/]")
                self.cancel_all()
                raise
            except Exception as e:
                console.print(f"[bold red][Orchestrator][/] 错误: {e}")
                # 清理已提交的作业
                self.cancel_all()
                state = State.ERROR
                raise

        return ctx.final_video

    def _handle_init(self, ctx: PipelineContext) -> State:
        """初始化: 创建必要的目录"""
        settings.SCENES_DIR.mkdir(parents=True, exist_ok=True)
        settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        settings.VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
        settings.OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        return State.PLANNING

    def _handle_planning(self, ctx: PipelineContext) -> State:
        """阶段 1: 场景规划"""
        self._emit("stage_start", stage="planning")

        ctx.scenes = self.planner.plan(ctx.user_prompt)

        for scene in ctx.scenes:
            ctx.scene_states[scene.scene_id] = SceneState(plan=scene)

        self._emit("plan_complete", scenes=ctx.scenes)
        return State.CODING

    def _handle_coding(self, ctx: PipelineContext) -> State:
        """阶段 2: 代码生成"""
        self._emit("stage_start", stage="coding")

        for scene_id, scene_state in ctx.scene_states.items():
            # 已有代码 (从 fixing 回来) 或已放弃的场景, 跳过
            if scene_state.code or scene_state.give_up:
                continue

            self._emit("scene_coding", scene_id=scene_id, title=scene_state.plan.title)

            try:
                scene_state.code = self.coder.generate_code(scene_state.plan)
                code_file = settings.SCENES_DIR / f"scene_{scene_id}.py"
                code_file.write_text(scene_state.code, encoding="utf-8")
                self._emit("scene_coded", scene_id=scene_id, file_path=str(code_file))
            except Exception as e:
                # 单场景失败不毁掉整批, 标记后继续
                console.print(
                    f"[bold red][Orchestrator][/] Scene {scene_id} 代码生成失败: {e}"
                )
                scene_state.failed = True
                self._emit("scene_failed", scene_id=scene_id)

        return State.REVIEWING

    def _handle_reviewing(self, ctx: PipelineContext) -> State:
        """阶段 3: 代码审查"""
        self._emit("stage_start", stage="reviewing")

        all_valid = True

        for scene_id, scene_state in ctx.scene_states.items():
            if scene_state.rendered or scene_state.give_up or scene_state.failed:
                continue

            try:
                result = self.reviewer.review(scene_state.code, scene_state.plan.title)
            except Exception as e:
                console.print(
                    f"[bold red][Orchestrator][/] Scene {scene_id} 审查失败: {e}"
                )
                scene_state.failed = True
                continue

            if result.is_valid:
                self._emit("scene_review_pass", scene_id=scene_id)
            else:
                all_valid = False
                scene_state.review_round += 1

                if scene_state.review_round >= settings.MAX_REVIEW_ROUNDS:
                    self._emit(
                        "scene_review_skip",
                        scene_id=scene_id,
                        reason=f"已达最大审查轮次 ({settings.MAX_REVIEW_ROUNDS})",
                    )
                else:
                    self._emit("scene_review_fail", scene_id=scene_id)
                    try:
                        scene_state.code = self.coder.generate_code(
                            scene_state.plan, result.feedback
                        )
                        code_file = settings.SCENES_DIR / f"scene_{scene_id}.py"
                        code_file.write_text(scene_state.code, encoding="utf-8")
                    except Exception as e:
                        console.print(
                            f"[bold red][Orchestrator][/] Scene {scene_id} 重新生成失败: {e}"
                        )
                        scene_state.failed = True

        if not all_valid:
            # 仍有未达上限且未放弃/失败的未渲染场景 -> 重新审查
            # (审查未通过且未到上限时, 代码已在上方就地重写, 这里回到 REVIEWING 重审)
            needs_rereview = any(
                not ss.rendered and not ss.failed and not ss.give_up
                and ss.review_round < settings.MAX_REVIEW_ROUNDS
                for ss in ctx.scene_states.values()
            )
            if needs_rereview:
                return State.REVIEWING

        return State.DISPATCHING

    def _handle_dispatching(self, ctx: PipelineContext) -> State:
        """阶段 4: 提交 Slurm 任务"""
        self._emit("stage_start", stage="dispatching")

        for scene_id, scene_state in ctx.scene_states.items():
            # 已渲染、已放弃或已失败的场景不提交
            if scene_state.rendered or scene_state.give_up or scene_state.failed:
                continue
            # 已有活跃 job 不重复提交
            if scene_state.slurm_job:
                continue

            code_file = settings.SCENES_DIR / f"scene_{scene_id}.py"
            class_name = self._extract_class_name(scene_state.code)

            try:
                job = self.slurm.submit_scene(
                    scene_id=scene_id,
                    python_file=code_file,
                    scene_class_name=class_name,
                )
                scene_state.slurm_job = job
                self._emit("scene_submitted", scene_id=scene_id, job_id=job.job_id)
            except Exception as e:
                console.print(
                    f"[bold red][Orchestrator][/] Scene {scene_id} 提交失败: {e}"
                )
                scene_state.failed = True
                self._emit("scene_failed", scene_id=scene_id)

        return State.MONITORING

    def _handle_monitoring(self, ctx: PipelineContext) -> State:
        """阶段 5: 监控渲染进度"""
        self._emit("stage_start", stage="monitoring")

        failed_scenes: list[int] = []

        for scene_id, scene_state in ctx.scene_states.items():
            if scene_state.rendered or not scene_state.slurm_job:
                continue

            job = scene_state.slurm_job
            try:
                success = self.slurm.wait_for_job(job.job_id, scene_id)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                console.print(
                    f"[bold red][Orchestrator][/] Scene {scene_id} 监控异常: {e}"
                )
                success = False

            if success:
                scene_state.rendered = True
                scene_state.slurm_job.status = "COMPLETED"
                self._emit("scene_rendered", scene_id=scene_id)
            else:
                scene_state.slurm_job.status = "FAILED"
                failed_scenes.append(scene_id)
                self._emit("scene_failed", scene_id=scene_id)

        if failed_scenes:
            return State.FIXING

        return State.MERGING

    def _handle_fixing(self, ctx: PipelineContext) -> State:
        """阶段 6: 自动修复失败的场景"""
        self._emit("stage_start", stage="fixing")

        fixed_any = False

        for scene_id, scene_state in ctx.scene_states.items():
            if scene_state.rendered or scene_state.give_up or scene_state.failed:
                continue
            if not scene_state.slurm_job or scene_state.slurm_job.status != "FAILED":
                continue

            scene_state.fix_attempts += 1

            if scene_state.fix_attempts > settings.MAX_FIX_ATTEMPTS:
                # 超过修复上限: 放弃该场景, 不再重新提交 (防止死循环)
                scene_state.give_up = True
                scene_state.slurm_job = None
                console.print(
                    f"[bold red][Orchestrator][/] Scene {scene_id} 达到最大修复次数 "
                    f"({settings.MAX_FIX_ATTEMPTS}), 放弃[/]"
                )
                self._emit("scene_give_up", scene_id=scene_id)
                continue

            error_log = self.slurm.get_error_log(scene_id, scene_state.slurm_job.job_id)

            if not error_log:
                scene_state.give_up = True
                scene_state.slurm_job = None
                console.print(
                    f"[bold yellow][Orchestrator][/] Scene {scene_id} 无法获取错误日志, 放弃[/]"
                )
                self._emit("scene_give_up", scene_id=scene_id)
                continue

            self._emit(
                "scene_fixing",
                scene_id=scene_id,
                attempt=scene_state.fix_attempts,
                max_attempts=settings.MAX_FIX_ATTEMPTS,
            )

            try:
                scene_state.code = self.auto_fixer.fix(scene_state.code, error_log)
                code_file = settings.SCENES_DIR / f"scene_{scene_id}.py"
                code_file.write_text(scene_state.code, encoding="utf-8")
                # 修复后重置审查轮次, 让新代码走完整 review 流程
                scene_state.review_round = 0
                scene_state.slurm_job = None
                fixed_any = True
            except Exception as e:
                console.print(
                    f"[bold red][Orchestrator][/] Scene {scene_id} 修复失败: {e}"
                )
                scene_state.failed = True
                scene_state.slurm_job = None

        # 有被修复的场景需要重新走 coding(已就位) -> reviewing -> dispatch
        if fixed_any:
            return State.REVIEWING

        return State.MERGING

    def _handle_merging(self, ctx: PipelineContext) -> State:
        """阶段 7: 视频拼接"""
        self._emit("stage_start", stage="merging")

        # 收集实际渲染成功的场景 ID (而非数量)
        rendered_ids = [
            sid for sid, ss in ctx.scene_states.items() if ss.rendered
        ]

        if not rendered_ids:
            self._emit("no_scenes_rendered")
            return State.ERROR

        self._emit("merging", rendered=len(rendered_ids), total=len(ctx.scene_states))

        ctx.final_video = self.merger.merge_scenes(rendered_ids)

        size_mb = ctx.final_video.stat().st_size / (1024 * 1024)
        self._emit("merge_complete", path=str(ctx.final_video), size_mb=size_mb)

        return State.DONE

    @staticmethod
    def _extract_class_name(code: str) -> str:
        """从 Python 代码中提取 Scene 类名"""
        match = re.search(
            r"class\s+(\w+)\s*\(\s*(?:Scene|ThreeDScene|MovingCameraScene)\s*\)", code
        )
        if match:
            return match.group(1)
        return "Scene"
