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
from rich.prompt import Confirm
from rich.table import Table

from config import settings
from agents.planner import PlannerAgent, ScenePlan, SceneOutline
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
    PLANNING = auto()     # 阶段 1: 拆解场景概要 (轻量)
    DETAILING = auto()    # 阶段 2: 逐场景生成导演分镜 (可单场景重试)
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
    minor_fix_round: int = 0     # 小修小补次数 (超过上限升为 major)
    fix_attempts: int = 0
    slurm_job: SlurmJob | None = None
    rendered: bool = False
    give_up: bool = False
    failed: bool = False


@dataclass
class PipelineContext:
    """流水线上下文"""
    user_prompt: str
    dry_run: bool = False
    interactive: bool = False
    outlines: list[SceneOutline] = field(default_factory=list)  # 阶段 1 结果
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

    def _ask_retry_or_skip(self, scene_id: int, error: str) -> bool:
        """LLM 重试耗尽后询问用户: 跳过 (False) 还是再重试 (True).

        非交互模式直接返回 False (跳过).
        """
        if not self._ctx or not self._ctx.interactive:
            return False

        console.print()
        console.print(
            f"[bold yellow]Scene {scene_id}[/] LLM 调用在 "
            f"{settings.LLM_MAX_RETRIES} 次重试后仍然失败:\n"
            f"  [dim]{error}[/]",
            markup=False,
        )
        try:
            answer = Confirm.ask(
                "[bold]再重试 3 次?[/] (y = 重试, n = 跳过该场景)",
                default=False,
                console=console,
            )
        except (EOFError, KeyboardInterrupt):
            answer = False
        return answer

    def run(
        self,
        user_prompt: str,
        callback: Callback | None = None,
        dry_run: bool = False,
        interactive: bool = False,
    ) -> Path:
        """
        执行完整的动画生成流水线

        Args:
            user_prompt: 用户的自然语言需求
            callback: 可选的状态回调函数
            dry_run: True 则跳过 Slurm 渲染和视频拼接, 只生成代码
            interactive: True 则在 LLM 重试耗尽后询问用户 (skip/retry)
        """
        self._callback = callback
        ctx = PipelineContext(user_prompt=user_prompt, dry_run=dry_run, interactive=interactive)
        self._ctx = ctx
        state = State.INIT

        while state != State.DONE and state != State.ERROR:
            try:
                match state:
                    case State.INIT:
                        state = self._handle_init(ctx)
                    case State.PLANNING:
                        state = self._handle_planning(ctx)
                    case State.DETAILING:
                        state = self._handle_detailing(ctx)
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
                console.print(f"[bold red][Orchestrator][/] 错误: {e}", markup=False)
                # 清理已提交的作业
                self.cancel_all()
                state = State.ERROR
                raise

        if state == State.ERROR:
            raise RuntimeError(
                "流水线未能完成: 所有场景均失败或无法提交.\n"
                "请检查 Slurm 是否可用 (sbatch 命令), 或使用 --dry-run 先验证代码."
            )

        if ctx.dry_run:
            total = len(ctx.scene_states)
            ok = sum(1 for ss in ctx.scene_states.values() if not ss.failed)
            console.print(
                f"\n[bold green]Dry-run 完成![/] 共 {total} 个场景, {ok} 个代码已生成.\n"
                f"代码文件在 [bold]{settings.SCENES_DIR}[/] 目录下.\n"
                f"用 [bold]manim -qh <文件名> <类名>[/] 手动渲染."
            )
            return None

        if ctx.final_video is None:
            raise RuntimeError("流水线未能生成最终视频.")

        return ctx.final_video

    def _handle_init(self, ctx: PipelineContext) -> State:
        """初始化: 创建必要的目录, 检查 Slurm 可用性"""
        import shutil

        settings.SCENES_DIR.mkdir(parents=True, exist_ok=True)
        settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        settings.VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
        settings.OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

        # --- 预检: sbatch / ffmpeg 是否可用 ---
        missing = []
        for cmd, label in [("sbatch", "Slurm"), ("ffmpeg", "FFmpeg")]:
            if shutil.which(cmd) is None:
                missing.append(f"  - {cmd} ({label})")

        if missing:
            self._emit("preflight_warn", missing="\n".join(missing))
            console.print(
                f"[yellow][Orchestrator] 警告:[/] 以下命令未在 PATH 中找到:\n"
                f"{chr(10).join(missing)}\n"
                f"渲染和拼接阶段将失败. 如果只是在测试, 请使用 [bold]--dry-run[/] 模式."
            )

        return State.PLANNING

    def _handle_planning(self, ctx: PipelineContext) -> State:
        """阶段 1: 拆解场景概要 (轻量, 不涉及每场景细节)"""
        self._emit("stage_start", stage="planning")

        while True:
            try:
                ctx.outlines = self.planner.plan_outline(ctx.user_prompt)
                break
            except Exception as e:
                console.print(
                    f"[bold red][Orchestrator][/] 场景概要规划失败: {e}", markup=False
                )
                if not ctx.interactive:
                    return State.ERROR
                try:
                    answer = Confirm.ask(
                        "[bold]再重试规划?[/] (y = 重试, n = 放弃退出)",
                        default=True,
                        console=console,
                    )
                except (EOFError, KeyboardInterrupt):
                    answer = False
                if not answer:
                    return State.ERROR

        # 为每个概要预先创建 SceneState (plan 暂为空, detailing 阶段填充)
        for o in ctx.outlines:
            ctx.scene_states[o.scene_id] = SceneState(
                plan=ScenePlan(
                    scene_id=o.scene_id,
                    title=o.title,
                    duration_seconds=o.duration_seconds,
                    purpose=o.purpose,
                    math_concept=o.math_concept,
                    visual_design="",
                    camera_movement="",
                    visual_flow=[],
                    key_moments=[],
                    computation="",
                )
            )

        return State.DETAILING

    def _handle_detailing(self, ctx: PipelineContext) -> State:
        """阶段 2: 逐场景生成导演分镜 (单场景失败不影响别的)"""
        self._emit("stage_start", stage="detailing")

        total = len(ctx.outlines)
        for outline in ctx.outlines:
            ss = ctx.scene_states.get(outline.scene_id)
            if ss is None or ss.give_up or ss.failed:
                continue

            self._emit("scene_detailing", scene_id=outline.scene_id, title=outline.title)

            while True:
                try:
                    ss.plan = self.planner.plan_detail(outline, total, ctx.user_prompt)
                    break
                except Exception as e:
                    console.print(
                        f"[bold red][Orchestrator][/] Scene {outline.scene_id} "
                        f"导演分镜失败: {e}", markup=False
                    )
                    if self._ask_retry_or_skip(outline.scene_id, str(e)):
                        continue
                    ss.failed = True
                    break

        # 更新 ctx.scenes 供后续阶段使用
        ctx.scenes = [
            ss.plan for ss in ctx.scene_states.values()
            if not ss.failed and not ss.give_up
        ]

        if not ctx.scenes:
            return State.ERROR

        self._emit("plan_complete", scenes=ctx.scenes)
        return State.CODING

    def _handle_coding(self, ctx: PipelineContext) -> State:
        """阶段 2: 代码生成"""
        self._emit("stage_start", stage="coding")

        for scene_id, scene_state in ctx.scene_states.items():
            # 已有代码 (从 fixing 回来) 或已放弃的场景, 跳过
            if scene_state.code or scene_state.give_up or scene_state.failed:
                continue

            self._emit("scene_coding", scene_id=scene_id, title=scene_state.plan.title)

            while True:
                try:
                    scene_state.code = self.coder.generate_code(scene_state.plan)
                    code_file = settings.SCENES_DIR / f"scene_{scene_id}.py"
                    code_file.write_text(scene_state.code, encoding="utf-8")
                    self._emit("scene_coded", scene_id=scene_id, file_path=str(code_file))
                    break  # 成功, 退出 while
                except Exception as e:
                    console.print(
                        f"[bold red][Orchestrator][/] Scene {scene_id} 代码生成失败: {e}",
                        markup=False,
                    )
                    if self._ask_retry_or_skip(scene_id, str(e)):
                        continue  # 用户选择重试, while 循环继续
                    scene_state.failed = True
                    break
                self._emit("scene_failed", scene_id=scene_id)

        return State.REVIEWING

    def _handle_reviewing(self, ctx: PipelineContext) -> State:
        """阶段 3: 代码审查 (minor → 查找替换, major → Coder 重写)"""
        self._emit("stage_start", stage="reviewing")

        all_valid = True

        for scene_id, scene_state in ctx.scene_states.items():
            if scene_state.rendered or scene_state.give_up or scene_state.failed:
                continue

            # 单场景 review + fix 循环
            while True:
                # --- 1. 调用 Reviewer ---
                review_ok = False
                while True:
                    try:
                        result = self.reviewer.review(scene_state.code, scene_state.plan.title)
                        review_ok = True
                        break
                    except Exception as e:
                        console.print(
                            f"[bold red][Orchestrator][/] Scene {scene_id} 审查失败: {e}",
                            markup=False,
                        )
                        if self._ask_retry_or_skip(scene_id, str(e)):
                            continue
                        scene_state.failed = True
                        break
                if not review_ok:
                    break  # failed → next scene

                # --- 2. 处理结果 ---
                if result.is_valid:
                    self._emit("scene_review_pass", scene_id=scene_id)
                    scene_state.minor_fix_round = 0
                    break  # 本场景完成

                all_valid = False

                # --- 3a. Minor: 查找替换 ---
                if result.severity == "minor" and result.fixes:
                    scene_state.minor_fix_round += 1
                    if scene_state.minor_fix_round > 3:
                        console.print(
                            f"[yellow][Orchestrator][/] Scene {scene_id} "
                            f"小修 {scene_state.minor_fix_round} 轮未过, 升级为 major[/]"
                        )
                        scene_state.minor_fix_round = 0
                        # 强制走 major 分支
                        result.severity = "major"
                    else:
                        applied = 0
                        code = scene_state.code
                        for fix in result.fixes:
                            if fix.find in code:
                                code = code.replace(fix.find, fix.replace, 1)
                                applied += 1
                            else:
                                console.print(
                                    f"[yellow][Orchestrator][/] Scene {scene_id}: "
                                    f"未找到 '{fix.find[:60]}...', 跳过[/]",
                                    markup=False,
                                )
                        if applied > 0:
                            scene_state.code = code
                            code_file = settings.SCENES_DIR / f"scene_{scene_id}.py"
                            code_file.write_text(scene_state.code, encoding="utf-8")
                            console.print(
                                f"[dim][Orchestrator][/] Scene {scene_id}: "
                                f"应用 {applied}/{len(result.fixes)} 条修复 → 重新审查[/]"
                            )
                            continue  # re-review the fixed code

                # --- 3b. Major (或 minor 升级): Coder 重写 ---
                scene_state.review_round += 1
                if scene_state.review_round >= settings.MAX_REVIEW_ROUNDS:
                    self._emit(
                        "scene_review_skip",
                        scene_id=scene_id,
                        reason=f"已达最大审查轮次 ({settings.MAX_REVIEW_ROUNDS})",
                    )
                    break

                self._emit("scene_review_fail", scene_id=scene_id)
                while True:
                    try:
                        scene_state.code = self.coder.generate_code(
                            scene_state.plan,
                            feedback=result.feedback,
                            previous_code=scene_state.code,
                        )
                        code_file = settings.SCENES_DIR / f"scene_{scene_id}.py"
                        code_file.write_text(scene_state.code, encoding="utf-8")
                        break
                    except Exception as e:
                        console.print(
                            f"[bold red][Orchestrator][/] Scene {scene_id} 重新生成失败: {e}",
                            markup=False,
                        )
                        if self._ask_retry_or_skip(scene_id, str(e)):
                            continue
                        scene_state.failed = True
                        break
                if scene_state.failed:
                    break
                # loop back to review the regenerated code

        if not all_valid:
            needs_rereview = any(
                not ss.rendered and not ss.failed and not ss.give_up
                and ss.review_round < settings.MAX_REVIEW_ROUNDS
                for ss in ctx.scene_states.values()
            )
            if needs_rereview:
                return State.REVIEWING

        if ctx.dry_run:
            self._emit("dry_run_complete")
            return State.DONE
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

            # 诊断: 打印实际路径
            console.print(
                f"[dim][Orchestrator][/] DEBUG: code_file={code_file} exists={code_file.exists()}, "
                f"SCENES_DIR={settings.SCENES_DIR.resolve()}, cwd={Path.cwd()}[/]",
                markup=False,
            )

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
                    f"[bold red][Orchestrator][/] Scene {scene_id} 提交失败: {e}\n"
                    f"  [dim]脚本: {code_file}[/]\n"
                    f"  [dim]类名: {class_name}[/]",
                    markup=False,
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

            while True:
                try:
                    scene_state.code = self.auto_fixer.fix(scene_state.code, error_log)
                    code_file = settings.SCENES_DIR / f"scene_{scene_id}.py"
                    code_file.write_text(scene_state.code, encoding="utf-8")
                    scene_state.review_round = 0
                    scene_state.slurm_job = None
                    fixed_any = True
                    break
                except Exception as e:
                    console.print(
                        f"[bold red][Orchestrator][/] Scene {scene_id} 修复失败: {e}",
                        markup=False,
                    )
                    if self._ask_retry_or_skip(scene_id, str(e)):
                        continue
                    scene_state.failed = True
                    scene_state.slurm_job = None
                    break

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
