"""有限状态机：串联规划、代码生成、审查、Slurm 渲染、修复与拼接。"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from uuid import uuid4

from rich.console import Console
from rich.prompt import Confirm

from kd1_anime.exceptions import (
    ConfigError,
    LLMError,
    LLMResponseError,
    PipelineAbortedError,
    PipelineExhaustedError,
    RenderError,
    RunError,
    RunIntegrityError,
    RunLockError,
    RunNotFoundError,
    SlurmError,
    SlurmSubmitError,
    SlurmTimeoutError,
    ValidationError,
    VideoNotFoundError,
)
from kd1_anime.logging import get_logger

logger = get_logger(__name__)

from kd1_anime.agents.auto_fixer import AutoFixerAgent
from kd1_anime.agents.coder import CoderAgent
from kd1_anime.agents.planner import PlannerAgent, SceneOutline, ScenePlan
from kd1_anime.agents.reviewer import ReviewerAgent, ReviewResult
from kd1_anime.agents.validator import CodeValidationResult, validate_manim_code
from kd1_anime.cluster.slurm import FAILURE_STATES, MONITOR_ABORT_STATES, SlurmDispatcher, SlurmJob
from kd1_anime.config import resolve_runtime_path, settings
from kd1_anime.media.merger import VideoMerger
from kd1_anime.run_store import (
    MANIFEST_NAME,
    RunManifest,
    RunRepository,
    StoredSceneState,
    lock_run,
    restore_run_path,
    restore_slurm_job,
    sha256_text,
    store_slurm_job,
    write_manifest,
)

console = Console()
Callback = Callable[[str, dict], None]
FIXABLE_RENDER_STATES = {"FAILED", "TIMEOUT", "OUT_OF_MEMORY", "RUN_TIMEOUT"}


class State(Enum):
    INIT = auto()
    PLANNING = auto()
    DETAILING = auto()
    CODING = auto()
    REVIEWING = auto()
    DISPATCHING = auto()
    MONITORING = auto()
    FIXING = auto()
    MERGING = auto()
    DONE = auto()
    ERROR = auto()


@dataclass(frozen=True, slots=True)
class RunPaths:
    run_id: str
    root: Path
    scenes: Path
    logs: Path
    videos: Path
    output: Path

    @classmethod
    def create(cls) -> RunPaths:
        stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
        run_id = f"{stamp}-{uuid4().hex[:8]}"
        root = resolve_runtime_path(settings.WORKSPACE_DIR) / "runs" / run_id
        configured_output = settings.OUTPUT_FILE.expanduser()
        if configured_output == Path("output_final.mp4"):
            output = root / "output_final.mp4"
        else:
            output = resolve_runtime_path(configured_output)
        return cls(
            run_id=run_id,
            root=root,
            scenes=root / "scenes",
            logs=root / "logs",
            videos=root / "videos",
            output=output,
        )


@dataclass
class SceneState:
    plan: ScenePlan
    code: str = ""
    class_name: str = ""
    review_round: int = 0
    fix_attempts: int = 0
    slurm_job: SlurmJob | None = None
    rendered: bool = False
    give_up: bool = False
    failed: bool = False
    failure_reason: str = ""


@dataclass
class PipelineContext:
    user_prompt: str
    paths: RunPaths = field(default_factory=RunPaths.create)
    dry_run: bool = False
    interactive: bool = False
    auto_fix: bool = True
    outlines: list[SceneOutline] = field(default_factory=list)
    scenes: list[ScenePlan] = field(default_factory=list)
    scene_states: dict[int, SceneState] = field(default_factory=dict)
    final_video: Path | None = None
    
    # 增量渲染支持
    incremental: bool = False
    base_run_id: str | None = None
    base_manifest: RunManifest | None = None
    scenes_to_render: list[int] = field(default_factory=list)
    scenes_to_reuse: list[int] = field(default_factory=list)


class Orchestrator:
    def __init__(self) -> None:
        self.planner = PlannerAgent()
        self.auto_fixer = AutoFixerAgent()
        self.slurm = SlurmDispatcher()
        self.merger = VideoMerger()
        self._callback: Callback | None = None
        self._ctx: PipelineContext | None = None
        self._manifest: RunManifest | None = None

    def _emit(self, event: str, **data) -> None:
        if self._callback:
            self._callback(event, data)

    def cancel_all(self) -> None:
        if not self._ctx:
            return
        for state in self._ctx.scene_states.values():
            job = state.slurm_job
            if (
                job
                and not state.rendered
                and not job.cancelled
                and job.status not in {"COMPLETED", "CANCELLED", *FAILURE_STATES}
                and self.slurm.cancel_job(job.job_id)
            ):
                job.cancelled = True
                job.status = "CANCELLED"
                job.failure_reason = "本地流水线停止时已取消远端任务"

    def _ask_retry_or_skip(self, scene_id: int, error: str) -> bool:
        if not self._ctx or not self._ctx.interactive:
            return False
        console.print(
            f"Scene {scene_id} 操作失败：\n{error}",
            markup=False,
            style="yellow",
        )
        try:
            return Confirm.ask("再重试一次？", default=False, console=console)
        except (EOFError, KeyboardInterrupt):
            return False

    @staticmethod
    def _mark_failed(state: SceneState, reason: str) -> None:
        state.failed = True
        state.failure_reason = reason

    @staticmethod
    def _validate(code: str) -> CodeValidationResult:
        return validate_manim_code(code)

    @staticmethod
    def _write_private(path: Path, content: str) -> None:
        path.write_text(content, encoding="utf-8")
        path.chmod(0o600)

    def _checkpoint(
        self,
        ctx: PipelineContext,
        state: State,
        *,
        status: str = "running",
        error: str = "",
    ) -> None:
        scenes: dict[int, StoredSceneState] = {}
        for scene_id, scene in sorted(ctx.scene_states.items()):
            code_file = ""
            code_hash = ""
            if scene.code:
                code_path = ctx.paths.scenes / f"scene_{scene_id}.py"
                code_file = code_path.resolve().relative_to(ctx.paths.root.resolve()).as_posix()
                code_hash = sha256_text(scene.code)
            scenes[scene_id] = StoredSceneState(
                plan=scene.plan,
                code_file=code_file,
                code_sha256=code_hash,
                class_name=scene.class_name,
                review_round=scene.review_round,
                fix_attempts=scene.fix_attempts,
                slurm_job=(
                    store_slurm_job(scene.slurm_job, ctx.paths.root)
                    if scene.slurm_job
                    else None
                ),
                rendered=scene.rendered,
                give_up=scene.give_up,
                failed=scene.failed,
                failure_reason=scene.failure_reason,
            )
        manifest = RunManifest(
            run_id=ctx.paths.run_id,
            created_at=(self._manifest.created_at if self._manifest else datetime.now().astimezone()),
            status=status,
            state=state.name,
            user_prompt=ctx.user_prompt,
            dry_run=ctx.dry_run,
            interactive=ctx.interactive,
            auto_fix=ctx.auto_fix,
            output_path=str(ctx.paths.output),
            outlines=ctx.outlines,
            scenes=scenes,
            final_video=str(ctx.final_video) if ctx.final_video else None,
            error=error[-50_000:],
            incremental=ctx.incremental,
            base_run_id=ctx.base_run_id,
        )
        write_manifest(ctx.paths.root / MANIFEST_NAME, manifest)
        self._manifest = manifest

    @staticmethod
    def _context_from_manifest(manifest: RunManifest, root: Path) -> PipelineContext:
        root = root.resolve()
        output = Path(manifest.output_path).expanduser()
        if not output.is_absolute():
            raise ValueError("manifest.output_path 必须是绝对路径")
        paths = RunPaths(
            run_id=manifest.run_id,
            root=root,
            scenes=root / "scenes",
            logs=root / "logs",
            videos=root / "videos",
            output=output,
        )
        scene_states: dict[int, SceneState] = {}
        for scene_id, stored in manifest.scenes.items():
            code = ""
            if stored.code_file:
                code_path = restore_run_path(root, stored.code_file)
                expected_code_path = (root / "scenes" / f"scene_{scene_id}.py").resolve()
                if code_path != expected_code_path:
                    raise ValueError(f"Scene {scene_id} 的代码路径不是规范场景路径")
                if not code_path.is_file() or code_path.is_symlink():
                    raise ValueError(f"Scene {scene_id} 的代码文件不存在或不安全")
                code = code_path.read_text(encoding="utf-8")
                if sha256_text(code) != stored.code_sha256:
                    raise ValueError(f"Scene {scene_id} 的代码哈希与运行清单不一致")
            job = restore_slurm_job(stored.slurm_job, root) if stored.slurm_job else None
            scene_states[scene_id] = SceneState(
                plan=stored.plan,
                code=code,
                class_name=stored.class_name,
                review_round=stored.review_round,
                fix_attempts=stored.fix_attempts,
                slurm_job=job,
                rendered=stored.rendered,
                give_up=stored.give_up,
                failed=stored.failed,
                failure_reason=stored.failure_reason,
            )
        return PipelineContext(
            user_prompt=manifest.user_prompt,
            paths=paths,
            dry_run=manifest.dry_run,
            interactive=manifest.interactive,
            auto_fix=manifest.auto_fix,
            outlines=manifest.outlines,
            scenes=[scene_states[key].plan for key in sorted(scene_states)],
            scene_states=scene_states,
            final_video=Path(manifest.final_video) if manifest.final_video else None,
            incremental=manifest.incremental,
            base_run_id=manifest.base_run_id,
        )

    def _generate_validated_code(
        self,
        plan: ScenePlan,
        *,
        feedback: str = "",
        previous_code: str = "",
        stream: bool = False,
    ) -> tuple[str, str]:
        agent = CoderAgent()
        current_feedback = feedback
        current_previous = previous_code
        last_validation: CodeValidationResult | None = None
        for _ in range(settings.CODE_VALIDATION_ATTEMPTS):
            code = agent.generate_code(
                plan,
                feedback=current_feedback,
                previous_code=current_previous,
                stream=stream,
            )
            validation = self._validate(code)
            if validation.is_valid:
                return code, validation.scene_classes[0]
            last_validation = validation
            current_feedback = f"确定性校验未通过，必须修复以下问题：\n{validation.feedback}"
            current_previous = code
        raise ValidationError(
            "生成代码未通过确定性校验：\n"
            + (last_validation.feedback if last_validation else "未知错误"),
            hint="尝试简化场景或调整 prompt"
        )

    def run_incremental(
        self,
        user_prompt: str,
        base_run_id: str,
        callback: Callback | None = None,
        dry_run: bool = False,
        interactive: bool = False,
    ) -> Path | None:
        """增量渲染：只重新渲染受 prompt 变化影响的场景。"""
        if len(user_prompt) > settings.MAX_PROMPT_CHARS:
            raise ValueError(
                f"用户需求过长：{len(user_prompt)} 字符，最大允许 {settings.MAX_PROMPT_CHARS} 字符\n"
                f"提示：可以将需求拆分为多个较短的动画，或使用更简洁的描述"
            )
        
        # 加载基础运行的 manifest
        repository = RunRepository(settings.WORKSPACE_DIR)
        try:
            base_manifest = repository.load(base_run_id)
        except (OSError, ValueError) as exc:
            raise RunNotFoundError(f"无法加载基础运行 {base_run_id}: {exc}") from exc
        
        if base_manifest.status not in ("completed", "dry_run_complete"):
            raise RunError(f"基础运行 {base_run_id} 未完成（状态：{base_manifest.status}）")
        
        self._callback = callback
        ctx = PipelineContext(
            user_prompt=user_prompt,
            dry_run=dry_run,
            interactive=interactive,
            incremental=True,
            base_run_id=base_run_id,
            base_manifest=base_manifest,
        )
        self._ctx = ctx
        ctx.paths.root.mkdir(parents=True, exist_ok=True)
        ctx.paths.root.chmod(0o700)
        
        with lock_run(ctx.paths.root):
            return self._execute(ctx, State.INIT)

        def run(
        self,
        user_prompt: str,
        callback: Callback | None = None,
        dry_run: bool = False,
        interactive: bool = False,
    ) -> Path | None:
        if len(user_prompt) > settings.MAX_PROMPT_CHARS:
            raise ValueError(
                f"用户需求过长：{len(user_prompt)} 字符，最大允许 {settings.MAX_PROMPT_CHARS} 字符\n提示：可以将需求拆分为多个较短的动画，或使用更简洁的描述"
            )
        self._callback = callback
        ctx = PipelineContext(
            user_prompt=user_prompt,
            dry_run=dry_run,
            interactive=interactive,
        )
        self._ctx = ctx
        ctx.paths.root.mkdir(parents=True, exist_ok=True)
        ctx.paths.root.chmod(0o700)
        with lock_run(ctx.paths.root):
            return self._execute(ctx, State.INIT)

    def submit_existing_scene(
        self,
        source_code: str,
        class_name: str,
        *,
        scene_id: int = 1,
        wait: bool = False,
    ) -> tuple[SlurmJob, Path | None, str]:
        """提交用户已有的单 Scene 文件，并让它拥有完整的运行清单。"""

        validation = self._validate(source_code)
        if not validation.is_valid or class_name not in validation.scene_classes:
            raise ValueError("直接渲染代码未通过确定性校验")
        paths = RunPaths.create()
        for directory in (
            paths.root,
            paths.scenes,
            paths.logs,
            paths.videos,
            paths.output.parent,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        paths.root.chmod(0o700)
        for directory in (paths.scenes, paths.logs, paths.videos):
            directory.chmod(0o700)
        code_path = paths.scenes / f"scene_{scene_id}.py"
        self._write_private(code_path, source_code)
        prompt = f"Direct render of Scene class {class_name}"
        self._write_private(paths.root / "prompt.txt", prompt)
        plan = ScenePlan(
            scene_id=scene_id,
            title=f"Direct render: {class_name}",
            duration_seconds=1,
            purpose="渲染用户提供的 Manim Scene",
            math_concept="由用户提供的代码定义",
            visual_design="由用户提供的代码定义",
            camera_movement="由用户提供的代码定义",
            visual_flow=["执行用户提供的 Scene"],
            key_moments=["由用户提供的代码定义"],
            computation="由用户提供的代码定义",
        )
        scene_state = SceneState(plan=plan, code=source_code, class_name=class_name)
        ctx = PipelineContext(
            user_prompt=prompt,
            paths=paths,
            auto_fix=False,
            scenes=[plan],
            scene_states={scene_id: scene_state},
        )
        self._ctx = ctx
        self._manifest = None

        with lock_run(paths.root):
            self._emit("run_started", run_id=paths.run_id, run_dir=str(paths.root))
            if wait:
                try:
                    final_video = self._execute(ctx, State.DISPATCHING)
                except Exception as exc:
                    job = scene_state.slurm_job
                    suffix = f"\n错误日志: {job.log_err}" if job else ""
                    raise RuntimeError(f"{exc}{suffix}") from exc
            else:
                try:
                    self._checkpoint(ctx, State.DISPATCHING)
                    next_state = self._handle_dispatching(ctx)
                    self._checkpoint(ctx, next_state)
                except KeyboardInterrupt:
                    self.cancel_all()
                    self._checkpoint(
                        ctx,
                        State.DISPATCHING,
                        status="interrupted",
                        error="用户中断",
                    )
                    raise
                except Exception as exc:
                    self.cancel_all()
                    self._checkpoint(ctx, State.DISPATCHING, status="failed", error=str(exc))
                    raise
                if scene_state.slurm_job is None:
                    message = scene_state.failure_reason or "Slurm 提交失败"
                    self._checkpoint(ctx, State.ERROR, status="failed", error=message)
                    raise RuntimeError(message)
                final_video = None
        if scene_state.slurm_job is None:
            raise RuntimeError("直接渲染结束但没有 Slurm Job")
        return scene_state.slurm_job, final_video, paths.run_id

    def resume(
        self,
        run_id: str,
        callback: Callback | None = None,
        interactive: bool = False,
    ) -> Path | None:
        """从原子清单恢复，不重新提交仍有 Job ID 的场景。"""

        repository = RunRepository(settings.WORKSPACE_DIR)
        root = repository.run_root(run_id)
        # 先确认清单存在，避免拼错 run-id 时创建空目录；真正读取放在锁内，
        # 防止原进程在 load 与 flock 之间推进状态而导致使用过期 Job ID。
        if not repository.manifest_path(run_id).is_file():
            raise FileNotFoundError(f"找不到运行清单: {run_id}")
        with lock_run(root):
            manifest = repository.load(run_id)
            self._callback = callback
            self._manifest = manifest
            ctx = self._context_from_manifest(manifest, root)
            ctx.interactive = interactive
            self._ctx = ctx

            if manifest.status == "completed":
                if ctx.final_video and ctx.final_video.is_file():
                    return ctx.final_video
                raise RuntimeError("运行标记为完成，但最终视频不存在")
            if manifest.status == "dry_run_complete":
                return None
            try:
                state = State[manifest.state]
            except KeyError as exc:
                raise ValueError(f"运行清单包含未知 FSM 状态: {manifest.state}") from exc
            if state is State.ERROR:
                raise RuntimeError("该运行已进入不可恢复的 ERROR 状态，请创建新运行")

            cancelled_jobs = False
            for scene in ctx.scene_states.values():
                if scene.slurm_job and (
                    scene.slurm_job.cancelled or scene.slurm_job.status == "CANCELLED"
                ):
                    scene.slurm_job = None
                    cancelled_jobs = True
            if cancelled_jobs and state in {
                State.DISPATCHING,
                State.MONITORING,
                State.FIXING,
                State.MERGING,
                State.DONE,
            }:
                state = State.DISPATCHING

            self._emit("run_resumed", run_id=run_id, state=state.name)
            return self._execute(ctx, state)

    def _execute(self, ctx: PipelineContext, state: State) -> Path | None:
        try:
            while state not in {State.DONE, State.ERROR}:
                self._checkpoint(ctx, state)
                state = {
                    State.INIT: self._handle_init,
                    State.PLANNING: self._handle_planning,
                    State.DETAILING: self._handle_detailing,
                    State.CODING: self._handle_coding,
                    State.REVIEWING: self._handle_reviewing,
                    State.DISPATCHING: self._handle_dispatching,
                    State.MONITORING: self._handle_monitoring,
                    State.FIXING: self._handle_fixing,
                    State.MERGING: self._handle_merging,
                }[state](ctx)
                self._checkpoint(ctx, state)
        except KeyboardInterrupt:
            self.cancel_all()
            try:
                self._checkpoint(ctx, state, status="interrupted", error="用户中断")
            except Exception as checkpoint_error:
                console.print(f"[yellow]写入中断清单失败: {checkpoint_error}[/]", markup=False)
            raise
        except Exception as exc:
            self.cancel_all()
            try:
                self._checkpoint(ctx, state, status="failed", error=str(exc))
            except Exception as checkpoint_error:
                console.print(f"[yellow]写入失败清单失败: {checkpoint_error}[/]", markup=False)
            raise

        if state == State.ERROR:
            reasons = [
                f"Scene {sid}: {ss.failure_reason or '未完成'}"
                for sid, ss in ctx.scene_states.items()
                if not ss.rendered
            ]
            message = "流水线未能完成。\n" + "\n".join(reasons)
            self._checkpoint(ctx, state, status="failed", error=message)
            raise RuntimeError(message)
        if ctx.dry_run:
            self._checkpoint(ctx, State.DONE, status="dry_run_complete")
            self._emit("dry_run_complete", run_dir=str(ctx.paths.root))
            return None
        if ctx.final_video is None:
            message = "流水线结束但没有最终视频"
            self._checkpoint(ctx, State.ERROR, status="failed", error=message)
            raise RuntimeError(message)
        self._checkpoint(ctx, State.DONE, status="completed")
        return ctx.final_video

    def _handle_init(self, ctx: PipelineContext) -> State:
        for directory in (
            ctx.paths.root,
            ctx.paths.scenes,
            ctx.paths.logs,
            ctx.paths.videos,
            ctx.paths.output.parent,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        ctx.paths.root.chmod(0o700)
        for directory in (ctx.paths.scenes, ctx.paths.logs, ctx.paths.videos):
            directory.chmod(0o700)
        self._write_private(ctx.paths.root / "prompt.txt", ctx.user_prompt)
        self._emit("run_started", run_id=ctx.paths.run_id, run_dir=str(ctx.paths.root))

        if not ctx.dry_run:
            if not settings.SLURM_CONTAINER_IMAGE:
                warning = (
                    "未配置 SLURM_CONTAINER_IMAGE：LLM 生成代码将直接以当前用户身份运行。"
                    "共享集群建议启用 Apptainer，并设置 SLURM_REQUIRE_CONTAINER=true。"
                )
                if self._callback:
                    self._emit("security_warning", message=warning)
                else:
                    console.print(f"[bold yellow]安全警告:[/] {warning}")
            missing = [
                name for name in ("sbatch", "ffmpeg") if not __import__("shutil").which(name)
            ]
            if settings.SLURM_CONTAINER_IMAGE and not __import__("shutil").which("apptainer"):
                missing.append("apptainer")
            if missing:
                raise RuntimeError("运行环境缺少命令: " + ", ".join(missing))
        
        # 增量渲染分析
        if ctx.incremental:
            logger.info("增量渲染模式：分析场景变化...")
            self._emit("incremental_start", base_run_id=ctx.base_run_id)
        
        return State.PLANNING

    def _handle_planning(self, ctx: PipelineContext) -> State:
        self._emit("stage_start", stage="planning")
        while True:
            try:
                ctx.outlines = self.planner.plan_outline(ctx.user_prompt)
                break
            except LLMError as exc:
                logger.error(f"LLM 调用失败: {exc}")
                if not self._ask_retry_or_skip(0, str(exc)):
                    raise LLMResponseError(
                        f"场景概要规划失败: {exc}",
                        hint="检查 LLM API 配置和网络连接"
                    ) from exc
            except Exception as exc:
                logger.error(f"场景概要规划时发生未知错误: {exc}")
                if not self._ask_retry_or_skip(0, str(exc)):
                    raise PipelineError(f"场景概要规划失败: {exc}") from exc
        return State.DETAILING

    def _handle_detailing(self, ctx: PipelineContext) -> State:
        self._emit("stage_start", stage="detailing")
        if len(ctx.outlines) > settings.MAX_SCENES:
            raise RuntimeError(
                f"Planner 生成了 {len(ctx.outlines)} 个场景，超过 MAX_SCENES={settings.MAX_SCENES}\n提示：可以在需求中明确指定场景数量，或增加 MAX_SCENES 配置"
            )
        if not ctx.outlines:
            raise RuntimeError("Planner 没有生成任何场景概要")
        pending_outlines = [
            outline for outline in ctx.outlines if outline.scene_id not in ctx.scene_states
        ]
        errors: dict[int, str] = {}

        def detail_one(outline: SceneOutline) -> tuple[int, ScenePlan | None, str | None]:
            try:
                plan = PlannerAgent().plan_detail(
                    outline,
                    ctx.outlines,
                    ctx.user_prompt,
                    stream=False,
                )
                return outline.scene_id, plan, None
            except LLMError as exc:
                logger.error(f"Scene {outline.scene_id} 详细规划 LLM 调用失败: {exc}")
                return outline.scene_id, None, str(exc)
            except Exception as exc:
                logger.error(f"Scene {outline.scene_id} 详细规划时发生未知错误: {exc}")
                return outline.scene_id, None, str(exc)

        for outline in pending_outlines:
            self._emit("scene_detailing", scene_id=outline.scene_id, title=outline.title)
        if pending_outlines:
            workers = min(settings.LLM_PARALLEL_WORKERS, len(pending_outlines))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(detail_one, outline) for outline in pending_outlines]
                for future in as_completed(futures):
                    scene_id, plan, error = future.result()
                    if plan:
                        ctx.scene_states[scene_id] = SceneState(plan=plan)
                        self._checkpoint(ctx, State.DETAILING)
                        self._emit("scene_detailed", scene_id=scene_id, title=plan.title)
                    else:
                        errors[scene_id] = error or "未知错误"

        # 并发阶段结束后才在主线程询问，避免 worker 争用 stdin。
        outlines_by_id = {outline.scene_id: outline for outline in ctx.outlines}
        for scene_id, initial_error in errors.items():
            outline = outlines_by_id[scene_id]
            final_error = initial_error
            if self._ask_retry_or_skip(scene_id, initial_error):
                try:
                    plan = self.planner.plan_detail(
                        outline, ctx.outlines, ctx.user_prompt, stream=True
                    )
                    ctx.scene_states[scene_id] = SceneState(plan=plan)
                    self._checkpoint(ctx, State.DETAILING)
                    self._emit("scene_detailed", scene_id=scene_id, title=plan.title)
                    continue
                except Exception as exc:
                    final_error = str(exc)
            placeholder = ScenePlan(
                **outline.model_dump(),
                visual_design="失败",
                camera_movement="失败",
                visual_flow=["失败"],
                key_moments=["失败"],
                computation="失败",
            )
            ctx.scene_states[scene_id] = SceneState(
                plan=placeholder,
                failed=True,
                failure_reason=f"导演分镜失败: {final_error}",
            )
            self._checkpoint(ctx, State.DETAILING)
            self._emit("scene_failed", scene_id=scene_id, reason=final_error)

        ctx.scenes = [
            ctx.scene_states[scene_id].plan
            for scene_id in sorted(ctx.scene_states)
            if not ctx.scene_states[scene_id].failed
        ]
        if not ctx.scenes:
            return State.ERROR
        self._emit("plan_complete", scenes=ctx.scenes)
        return State.CODING

    def _handle_coding(self, ctx: PipelineContext) -> State:
        self._emit("stage_start", stage="coding")
        pending = [
            (sid, state)
            for sid, state in ctx.scene_states.items()
            if not state.code and not state.failed and not state.give_up
        ]
        if not pending:
            return State.REVIEWING
        workers = min(settings.LLM_PARALLEL_WORKERS, len(pending))
        errors: dict[int, str] = {}

        def code_one(scene_id: int, state: SceneState):
            try:
                code, class_name = self._generate_validated_code(state.plan, stream=False)
                return scene_id, code, class_name, None
            except Exception as exc:
                return scene_id, None, None, str(exc)

        for scene_id, state in pending:
            self._emit("scene_coding", scene_id=scene_id, title=state.plan.title)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(code_one, sid, state) for sid, state in pending]
            for future in as_completed(futures):
                scene_id, code, class_name, error = future.result()
                state = ctx.scene_states[scene_id]
                if code and class_name:
                    state.code = code
                    state.class_name = class_name
                    path = ctx.paths.scenes / f"scene_{scene_id}.py"
                    self._write_private(path, code)
                    self._checkpoint(ctx, State.CODING)
                    self._emit("scene_coded", scene_id=scene_id, file_path=str(path))
                else:
                    errors[scene_id] = error or "未知错误"

        for scene_id, initial_error in errors.items():
            state = ctx.scene_states[scene_id]
            final_error = initial_error
            if self._ask_retry_or_skip(scene_id, initial_error):
                try:
                    code, class_name = self._generate_validated_code(state.plan, stream=True)
                    state.code = code
                    state.class_name = class_name
                    path = ctx.paths.scenes / f"scene_{scene_id}.py"
                    self._write_private(path, code)
                    self._checkpoint(ctx, State.CODING)
                    self._emit("scene_coded", scene_id=scene_id, file_path=str(path))
                    continue
                except Exception as exc:
                    final_error = str(exc)
            self._mark_failed(state, f"代码生成失败: {final_error}")
            self._checkpoint(ctx, State.CODING)
            self._emit("scene_failed", scene_id=scene_id, reason=final_error)
        return State.REVIEWING

    def _handle_reviewing(self, ctx: PipelineContext) -> State:
        self._emit("stage_start", stage="reviewing")
        pending = [
            (sid, state)
            for sid, state in ctx.scene_states.items()
            if state.code and not state.rendered and not state.failed and not state.give_up
        ]
        if not pending:
            return State.DONE if ctx.dry_run else State.DISPATCHING
        workers = min(settings.LLM_PARALLEL_WORKERS, len(pending))
        results: dict[int, ReviewResult] = {}
        errors: dict[int, str] = {}

        def review_one(scene_id: int, state: SceneState):
            try:
                return scene_id, ReviewerAgent().review(state.code, state.plan), None
            except Exception as exc:
                return scene_id, None, str(exc)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(review_one, sid, state) for sid, state in pending]
            for future in as_completed(futures):
                scene_id, result, error = future.result()
                if result:
                    results[scene_id] = result
                else:
                    errors[scene_id] = error or "未知错误"

        for scene_id, initial_error in errors.items():
            state = ctx.scene_states[scene_id]
            final_error = initial_error
            if self._ask_retry_or_skip(scene_id, initial_error):
                try:
                    results[scene_id] = ReviewerAgent().review(state.code, state.plan)
                    continue
                except Exception as exc:
                    final_error = str(exc)
            self._mark_failed(state, f"代码审查失败: {final_error}")
            self._checkpoint(ctx, State.REVIEWING)
            self._emit("scene_failed", scene_id=scene_id, reason=final_error)

        code_changed = False
        for scene_id, review_result in results.items():
            state = ctx.scene_states[scene_id]
            effective_result = review_result
            if state.failed:
                continue
            if effective_result.is_valid:
                state.review_round = 0
                self._checkpoint(ctx, State.REVIEWING)
                self._emit("scene_review_pass", scene_id=scene_id)
                continue

            state.review_round += 1
            if state.review_round >= settings.MAX_REVIEW_ROUNDS:
                state.give_up = True
                state.failure_reason = "达到最大审查轮次，代码仍未通过"
                self._checkpoint(ctx, State.REVIEWING)
                self._emit("scene_give_up", scene_id=scene_id, reason=state.failure_reason)
                continue

            if effective_result.severity == "minor":
                candidate = state.code
                for fix in effective_result.fixes:
                    if candidate.count(fix.find) != 1:
                        candidate = ""
                        break
                    candidate = candidate.replace(fix.find, fix.replace, 1)
                validation = (
                    self._validate(candidate)
                    if candidate
                    else CodeValidationResult(False, ["fix.find 不是唯一匹配"])
                )
                if validation.is_valid:
                    state.code = candidate
                    state.class_name = validation.scene_classes[0]
                    self._write_private(ctx.paths.scenes / f"scene_{scene_id}.py", candidate)
                    code_changed = True
                    self._checkpoint(ctx, State.REVIEWING)
                    self._emit("scene_review_fail", scene_id=scene_id, severity="minor")
                    continue
                effective_result = ReviewResult(
                    is_valid=False,
                    severity="major",
                    feedback=(
                        "Reviewer 的局部修复无法安全应用，请重写。\n"
                        f"确定性校验：\n{validation.feedback}"
                    ),
                )

            try:
                code, class_name = self._generate_validated_code(
                    state.plan,
                    feedback=effective_result.feedback,
                    previous_code=state.code,
                    stream=False,
                )
                state.code = code
                state.class_name = class_name
                self._write_private(ctx.paths.scenes / f"scene_{scene_id}.py", code)
                code_changed = True
                self._checkpoint(ctx, State.REVIEWING)
                self._emit("scene_review_fail", scene_id=scene_id, severity="major")
            except Exception as exc:
                self._mark_failed(state, f"Reviewer 后重写失败: {exc}")
                self._checkpoint(ctx, State.REVIEWING)
                self._emit("scene_failed", scene_id=scene_id, reason=str(exc))

        if code_changed:
            return State.REVIEWING
        if ctx.dry_run:
            return State.DONE
        
        # 增量渲染分析：在代码生成完成后比较变化
        if ctx.incremental and ctx.base_manifest:
            self._compute_incremental_changes(ctx)
        
        return State.DISPATCHING

    def _compute_incremental_changes(self, ctx: PipelineContext) -> None:
        """计算增量渲染的变化，确定哪些场景需要重新渲染。"""
        if not ctx.incremental or not ctx.base_manifest:
            return
        
        base_root = RunRepository(settings.WORKSPACE_DIR).run_root(ctx.base_run_id)
        scenes_to_render = []
        scenes_to_reuse = []
        
        for scene_id, state in ctx.scene_states.items():
            base_scene = ctx.base_manifest.scenes.get(scene_id)
            if base_scene is None:
                # 新场景，需要渲染
                scenes_to_render.append(scene_id)
                logger.info(f"Scene {scene_id}: 新场景，需要渲染")
                continue
            
            # 比较代码 hash
            current_hash = sha256_text(state.code) if state.code else ""
            base_hash = base_scene.code_sha256
            
            if current_hash == base_hash and base_scene.rendered:
                # 代码未变化且已渲染，可以复用
                scenes_to_reuse.append(scene_id)
                logger.info(f"Scene {scene_id}: 代码未变化，复用旧视频")
                
                # 尝试复用旧视频
                old_video = get_reusable_video_path(
                    ctx.base_manifest,
                    scene_id,
                    base_root,
                )
                if old_video:
                    state.rendered = True
                    state.slurm_job = SlurmJob(
                        job_id=f"reused-{scene_id}",
                        scene_id=scene_id,
                        script_path=ctx.paths.scenes / f"scene_{scene_id}.py",
                        log_out=ctx.paths.logs / f"scene_{scene_id}_reused.out",
                        log_err=ctx.paths.logs / f"scene_{scene_id}_reused.err",
                        media_dir=old_video.parent,
                        scene_class_name=base_scene.class_name,
                        submitted_at=0,
                        status="REUSED",
                    )
                else:
                    # 无法复用，需要重新渲染
                    scenes_to_render.append(scene_id)
                    scenes_to_reuse.remove(scene_id)
                    logger.warning(f"Scene {scene_id}: 无法复用旧视频，需要重新渲染")
            else:
                # 代码变化，需要渲染
                scenes_to_render.append(scene_id)
                logger.info(f"Scene {scene_id}: 代码变化，需要重新渲染")
        
        ctx.scenes_to_render = scenes_to_render
        ctx.scenes_to_reuse = scenes_to_reuse
        
        self._emit(
            "incremental_analysis",
            total=len(ctx.scene_states),
            to_render=len(scenes_to_render),
            to_reuse=len(scenes_to_reuse),
        )
        
        logger.info(
            f"增量渲染分析完成: {len(scenes_to_render)} 个场景需要渲染, "
            f"{len(scenes_to_reuse)} 个场景可复用"
        )

        def _handle_dispatching(self, ctx: PipelineContext) -> State:
        self._emit("stage_start", stage="dispatching")
        active = [
            state
            for state in ctx.scene_states.values()
            if state.slurm_job
            and not state.rendered
            and not state.failed
            and not state.give_up
        ]
        limit = settings.SLURM_MAX_IN_FLIGHT
        available = max(0, limit - len(active)) if limit else len(ctx.scene_states)
        submitted = 0
        for scene_id, state in sorted(ctx.scene_states.items()):
            if submitted >= available:
                break
            if state.rendered or state.failed or state.give_up or state.slurm_job:
                continue
            source_path = ctx.paths.scenes / f"scene_{scene_id}.py"
            try:
                on_disk_code = source_path.read_text(encoding="utf-8")
            except OSError as exc:
                self._mark_failed(state, f"提交前无法读取场景代码: {exc}")
                self._checkpoint(ctx, State.DISPATCHING)
                self._emit("scene_failed", scene_id=scene_id, reason=state.failure_reason)
                continue
            if on_disk_code != state.code:
                self._mark_failed(state, "提交前代码一致性校验失败：磁盘文件已在流水线外被修改")
                self._checkpoint(ctx, State.DISPATCHING)
                self._emit("scene_failed", scene_id=scene_id, reason=state.failure_reason)
                continue
            validation = self._validate(on_disk_code)
            if not validation.is_valid:
                self._mark_failed(state, "提交前校验失败:\n" + validation.feedback)
                self._checkpoint(ctx, State.DISPATCHING)
                self._emit("scene_failed", scene_id=scene_id, reason=state.failure_reason)
                continue
            state.class_name = validation.scene_classes[0]
            try:
                state.slurm_job = self.slurm.submit_scene(
                    scene_id,
                    source_path,
                    state.class_name,
                    scenes_dir=ctx.paths.scenes,
                    logs_dir=ctx.paths.logs,
                    videos_dir=ctx.paths.videos,
                )
                submitted += 1
                # Job ID 返回后立刻持久化，避免进程在批次末尾崩溃导致重复提交。
                self._checkpoint(ctx, State.DISPATCHING)
                self._emit(
                    "scene_submitted",
                    scene_id=scene_id,
                    job_id=state.slurm_job.job_id,
                )
            except Exception as exc:
                self._mark_failed(state, f"Slurm 提交失败: {exc}")
                self._checkpoint(ctx, State.DISPATCHING)
                self._emit("scene_failed", scene_id=scene_id, reason=str(exc))
        has_active_jobs = any(
            state.slurm_job
            and not state.rendered
            and not state.failed
            and not state.give_up
            for state in ctx.scene_states.values()
        )
        if has_active_jobs:
            return State.MONITORING
        has_unsubmitted = any(
            state.code
            and not state.slurm_job
            and not state.rendered
            and not state.failed
            and not state.give_up
            for state in ctx.scene_states.values()
        )
        return State.DISPATCHING if has_unsubmitted else State.MERGING

    def _handle_monitoring(self, ctx: PipelineContext) -> State:
        self._emit("stage_start", stage="monitoring")
        jobs = {
            state.slurm_job.job_id: state.slurm_job
            for state in ctx.scene_states.values()
            if state.slurm_job and not state.rendered
        }
        if not jobs:
            return State.MERGING
        results = self.slurm.wait_for_all_jobs(jobs)
        failed = False
        for job_id, success in results.items():
            job = jobs[job_id]
            state = ctx.scene_states[job.scene_id]
            if success:
                state.rendered = True
                self._emit("scene_rendered", scene_id=job.scene_id)
            else:
                failed = True
                if not ctx.auto_fix:
                    self._mark_failed(
                        state, job.failure_reason or f"Slurm 状态: {job.status}"
                    )
                self._emit(
                    "scene_failed",
                    scene_id=job.scene_id,
                    reason=job.failure_reason or f"Slurm 状态: {job.status}",
                )
        self._checkpoint(ctx, State.MONITORING)
        if failed:
            return State.FIXING if ctx.auto_fix else State.MERGING
        has_unsubmitted = any(
            state.code
            and not state.slurm_job
            and not state.rendered
            and not state.failed
            and not state.give_up
            for state in ctx.scene_states.values()
        )
        return State.DISPATCHING if has_unsubmitted else State.MERGING

    def _handle_fixing(self, ctx: PipelineContext) -> State:
        self._emit("stage_start", stage="fixing")
        fixed = False
        for scene_id, state in ctx.scene_states.items():
            job = state.slurm_job
            if state.rendered or state.failed or state.give_up or not job:
                continue
            if job.status in MONITOR_ABORT_STATES - {"RUN_TIMEOUT"} or job.status not in (
                FIXABLE_RENDER_STATES | FAILURE_STATES
            ):
                state.give_up = True
                state.failure_reason = (
                    job.failure_reason or f"不可自动修复的 Slurm 状态: {job.status}"
                )
                self._checkpoint(ctx, State.FIXING)
                self._emit("scene_give_up", scene_id=scene_id, reason=state.failure_reason)
                continue
            if job.status not in FIXABLE_RENDER_STATES:
                state.give_up = True
                state.failure_reason = f"基础设施失败，不修改代码: {job.status}"
                self._checkpoint(ctx, State.FIXING)
                self._emit("scene_give_up", scene_id=scene_id, reason=state.failure_reason)
                continue
            state.fix_attempts += 1
            if state.fix_attempts > settings.MAX_FIX_ATTEMPTS:
                state.give_up = True
                state.failure_reason = "达到最大渲染修复次数"
                self._checkpoint(ctx, State.FIXING)
                self._emit("scene_give_up", scene_id=scene_id, reason=state.failure_reason)
                continue
            error_log = self.slurm.get_error_log(job=job)
            if not error_log:
                state.give_up = True
                state.failure_reason = "渲染失败且没有错误日志"
                self._checkpoint(ctx, State.FIXING)
                self._emit("scene_give_up", scene_id=scene_id, reason=state.failure_reason)
                continue
            if self.auto_fixer.is_infrastructure_error(error_log):
                state.give_up = True
                state.failure_reason = "检测到环境或 Slurm 配置错误，未让 LLM 重写业务代码"
                self._checkpoint(ctx, State.FIXING)
                self._emit("scene_give_up", scene_id=scene_id, reason=state.failure_reason)
                continue
            self._emit(
                "scene_fixing",
                scene_id=scene_id,
                attempt=state.fix_attempts,
                max_attempts=settings.MAX_FIX_ATTEMPTS,
            )
            try:
                candidate = self.auto_fixer.fix(state.code, error_log)
                validation = self._validate(candidate)
                if not validation.is_valid:
                    candidate, class_name = self._generate_validated_code(
                        state.plan,
                        feedback=(
                            "AutoFix 结果未通过确定性校验：\n"
                            f"{validation.feedback}\n\n原始渲染错误：\n{error_log}"
                        ),
                        previous_code=candidate,
                        stream=False,
                    )
                else:
                    class_name = validation.scene_classes[0]
                state.code = candidate
                state.class_name = class_name
                state.review_round = 0
                state.slurm_job = None
                self._write_private(ctx.paths.scenes / f"scene_{scene_id}.py", candidate)
                fixed = True
                self._checkpoint(ctx, State.FIXING)
            except Exception as exc:
                self._mark_failed(state, f"自动修复失败: {exc}")
                state.slurm_job = None
                self._checkpoint(ctx, State.FIXING)
                self._emit("scene_failed", scene_id=scene_id, reason=str(exc))
        if fixed:
            return State.REVIEWING
        has_unsubmitted = any(
            state.code
            and not state.slurm_job
            and not state.rendered
            and not state.failed
            and not state.give_up
            for state in ctx.scene_states.values()
        )
        return State.DISPATCHING if has_unsubmitted else State.MERGING

    def _handle_merging(self, ctx: PipelineContext) -> State:
        self._emit("stage_start", stage="merging")
        rendered_jobs = [
            state.slurm_job
            for state in ctx.scene_states.values()
            if state.rendered and state.slurm_job
        ]
        incomplete = [sid for sid, state in ctx.scene_states.items() if not state.rendered]
        if not rendered_jobs:
            return State.ERROR
        if incomplete and not settings.ALLOW_PARTIAL_OUTPUT:
            for sid in incomplete:
                state = ctx.scene_states[sid]
                if not state.failure_reason:
                    state.failure_reason = "场景未成功渲染"
            self._emit("partial_output_blocked", incomplete=incomplete)
            return State.ERROR
        
        resolved_output = ctx.paths.output.expanduser().resolve()
        output_is_run_local = resolved_output == ctx.paths.root.resolve() or (
            ctx.paths.root.resolve() in resolved_output.parents
        )
        if (
            output_is_run_local
            and resolved_output.is_file()
            and resolved_output.stat().st_size > 0
            and not settings.OVERWRITE_OUTPUT
        ):
            # FFmpeg 使用原子 replace；若进程恰在 replace 后、清单写入前退出，
            # 当前 run 内已存在的非空目标就是完成产物，可直接补写 DONE 检查点。
            ctx.final_video = resolved_output
        else:
            # 增量渲染模式：使用支持复用旧视频的方法
            if ctx.incremental and ctx.base_manifest and ctx.base_run_id:
                from kd1_anime.run_store import RunRepository
                base_root = RunRepository(settings.WORKSPACE_DIR).run_root(ctx.base_run_id)
                video_paths = self.merger.collect_incremental_videos(
                    ctx.scene_states,
                    ctx.paths.root,
                    ctx.base_manifest,
                    base_root,
                )
                ctx.final_video = self.merger.merge(video_paths, ctx.paths.output)
            else:
                ctx.final_video = self.merger.merge_jobs(
                    rendered_jobs,
                    output_path=ctx.paths.output,
                )
        self._checkpoint(ctx, State.MERGING)
        self._emit(
            "merge_complete",
            path=str(ctx.final_video),
            size_mb=ctx.final_video.stat().st_size / (1024 * 1024),
            partial=bool(incomplete),
            incomplete=incomplete,
        )
        return State.DONE
