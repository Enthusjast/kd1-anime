"""有限状态机：串联规划、代码生成、审查、Slurm 渲染、修复与拼接。"""

from __future__ import annotations

import re
import shutil
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from uuid import uuid4

from rich.console import Console
from rich.prompt import Confirm

from kd1_anime.agents.auto_fixer import AutoFixerAgent
from kd1_anime.agents.coder import CoderAgent
from kd1_anime.agents.planner import PlannerAgent, SceneOutline, ScenePlan
from kd1_anime.agents.reviewer import ReviewerAgent, ReviewResult
from kd1_anime.agents.validator import CodeValidationResult, validate_manim_code
from kd1_anime.cluster.slurm import (
    FAILURE_STATES,
    JobMonitor,
    SlurmDispatcher,
    SlurmJob,
)
from kd1_anime.config import resolve_runtime_path, settings
from kd1_anime.exceptions import (
    LLMError,
    LLMResponseError,
    PipelineError,
    RunError,
    RunNotFoundError,
    ValidationError,
)
from kd1_anime.logging import get_logger
from kd1_anime.media.merger import VideoMerger
from kd1_anime.rendering import RenderProfile, SceneArtifact, sha256_file
from kd1_anime.resources import ResourceCoordinator
from kd1_anime.run_store import (
    MANIFEST_NAME,
    RunManifest,
    RunRepository,
    StoredSceneState,
    get_reusable_video_path,
    lock_run,
    restore_run_path,
    restore_slurm_job,
    sha256_text,
    store_slurm_job,
    write_manifest,
)

logger = get_logger(__name__)
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
    EVALUATING = auto()  # 新增：评估状态
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
    def create(cls, output_path: Path | None = None) -> RunPaths:
        stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
        run_id = f"{stamp}-{uuid4().hex[:8]}"
        root = resolve_runtime_path(settings.WORKSPACE_DIR) / "runs" / run_id
        configured_output = (output_path or settings.OUTPUT_FILE).expanduser()
        if output_path is None and configured_output == Path("output_final.mp4"):
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
    artifact: SceneArtifact | None = None
    rendered: bool = False
    give_up: bool = False
    failed: bool = False
    failure_reason: str = ""
    # 导演分镜是否已完成(区别于占位 plan)。场景独立推进时据此判断下一步。
    plan_ready: bool = False
    # Reviewer major 反馈待重写时暂存; 调度器据此把场景重新排入编码阶段。
    rewrite_feedback: str = ""
    # 渲染错误指纹: 连续相同错误 → 环境问题, 提前放弃
    last_error_fp: str = ""
    identical_error_count: int = 0

    reviewed: bool = False


@dataclass
class PipelineContext:
    user_prompt: str
    original_prompt: str | None = None  # 原始用户提示，用于评估
    paths: RunPaths = field(default_factory=RunPaths.create)
    dry_run: bool = False
    interactive: bool = False
    auto_fix: bool = True
    outlines: list[SceneOutline] = field(default_factory=list)
    scenes: list[ScenePlan] = field(default_factory=list)
    scene_states: dict[int, SceneState] = field(default_factory=dict)
    final_video: Path | None = None
    final_video_sha256: str = ""
    render_profile: RenderProfile = field(default_factory=RenderProfile.current)
    manifest_revision: int = 0

    # 增量渲染支持
    incremental: bool = False
    base_run_id: str | None = None
    base_manifest: RunManifest | None = None
    scenes_to_render: list[int] = field(default_factory=list)
    scenes_to_reuse: list[int] = field(default_factory=list)

    # 评估-改进循环支持
    eval_round: int = 0
    scenes_to_improve: list[int] = field(default_factory=list)


class Orchestrator:
    def __init__(self, resource_coordinator: ResourceCoordinator | None = None) -> None:
        self.planner = PlannerAgent()
        self.slurm = SlurmDispatcher()
        self.merger = VideoMerger()
        self._callback: Callback | None = None
        self._ctx: PipelineContext | None = None
        self._manifest: RunManifest | None = None
        # 场景级并行调度时多个工作线程会并发写 manifest, 需要串行化
        self._state_lock = threading.RLock()
        self._stop_event = threading.Event()
        self._phase_lock = threading.Lock()
        self._emitted_phases: set[str] = set()
        self._resource_coordinator = resource_coordinator

    def _emit(self, event: str, **data) -> None:
        if self._callback:
            self._callback(event, data)

    def cancel_all(self) -> None:
        self._stop_event.set()
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
        # 交互式询问前暂停 Live 仪表盘，避免输入冲突
        from kd1_anime.dashboard import suspend_all

        with suspend_all():
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
    def _validate(code: str, *, renderer: str | None = None) -> CodeValidationResult:
        return validate_manim_code(code, renderer=renderer)

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
        with self._state_lock:
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
                    artifact=scene.artifact,
                    phase=self._scene_phase(scene),
                    reviewed=scene.reviewed,
                    rendered=scene.rendered,
                    give_up=scene.give_up,
                    failed=scene.failed,
                    failure_reason=scene.failure_reason,
                    plan_ready=scene.plan_ready,
                    rewrite_feedback=scene.rewrite_feedback,
                    last_error_fp=scene.last_error_fp,
                    identical_error_count=scene.identical_error_count,
                )
            ctx.manifest_revision = (
                max(
                    ctx.manifest_revision,
                    self._manifest.revision if self._manifest else 0,
                )
                + 1
            )
            manifest = RunManifest(
                revision=ctx.manifest_revision,
                run_id=ctx.paths.run_id,
                created_at=(
                    self._manifest.created_at if self._manifest else datetime.now().astimezone()
                ),
                status=status,
                state=state.name,
                user_prompt=ctx.user_prompt,
                dry_run=ctx.dry_run,
                interactive=ctx.interactive,
                auto_fix=ctx.auto_fix,
                output_path=str(ctx.paths.output),
                render_profile=ctx.render_profile,
                outlines=ctx.outlines,
                scenes=scenes,
                final_video=str(ctx.final_video) if ctx.final_video else None,
                final_video_sha256=ctx.final_video_sha256,
                error=error[-50_000:],
                incremental=ctx.incremental,
                base_run_id=ctx.base_run_id,
                eval_round=ctx.eval_round,
            )
            write_manifest(ctx.paths.root / MANIFEST_NAME, manifest)
            self._manifest = manifest

    @staticmethod
    def _scene_phase(scene: SceneState) -> str:
        if scene.rendered and scene.artifact:
            return "rendered"
        if scene.failed or scene.give_up:
            return "failed"
        if scene.slurm_job:
            return "monitoring"
        if scene.reviewed:
            return "reviewed"
        if scene.code:
            return "coded"
        if scene.plan_ready:
            return "detailed"
        return "pending"

    @staticmethod
    def _context_from_manifest(manifest: RunManifest, root: Path) -> PipelineContext:
        root = root.resolve()
        output = Path(manifest.output_path).expanduser()
        if not output.is_absolute():
            raise ValueError("manifest.output_path 必须是绝对路径")
        final_video = Path(manifest.final_video).expanduser() if manifest.final_video else None
        if final_video is not None and not final_video.is_absolute():
            raise ValueError("manifest.final_video 必须是绝对路径")
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
            if job is not None:
                if not code or not stored.code_sha256:
                    raise ValueError(f"Scene {scene_id} 存在 Slurm Job 但缺少代码身份")
                if job.scene_id != scene_id or job.scene_class_name != stored.class_name:
                    raise ValueError(f"Scene {scene_id} 的 Slurm Job 场景身份不一致")
                if job.code_sha256 != stored.code_sha256:
                    raise ValueError(f"Scene {scene_id} 的 Slurm Job 代码哈希不一致")
                if job.render_profile.digest() != manifest.render_profile.digest():
                    raise ValueError(f"Scene {scene_id} 的 Slurm Job 渲染配置不一致")
            artifact = stored.artifact
            if artifact is not None:
                if (
                    artifact.scene_id != scene_id
                    or artifact.scene_class_name != stored.class_name
                    or artifact.code_sha256 != stored.code_sha256
                ):
                    raise ValueError(f"Scene {scene_id} 的渲染产物身份不一致")
                if artifact.render_profile_sha256 != manifest.render_profile.digest():
                    raise ValueError(f"Scene {scene_id} 的渲染产物配置不一致")
            if stored.rendered and artifact is None:
                raise ValueError(f"Scene {scene_id} 标记为已渲染但缺少产物凭据")
            scene_states[scene_id] = SceneState(
                plan=stored.plan,
                code=code,
                class_name=stored.class_name,
                review_round=stored.review_round,
                fix_attempts=stored.fix_attempts,
                slurm_job=job,
                artifact=artifact,
                reviewed=stored.reviewed,
                rendered=stored.rendered,
                give_up=stored.give_up,
                failed=stored.failed,
                failure_reason=stored.failure_reason,
                plan_ready=getattr(stored, "plan_ready", False),
                rewrite_feedback=getattr(stored, "rewrite_feedback", ""),
                last_error_fp=getattr(stored, "last_error_fp", ""),
                identical_error_count=getattr(stored, "identical_error_count", 0),
            )
        context = PipelineContext(
            user_prompt=manifest.user_prompt,
            original_prompt=manifest.user_prompt,
            paths=paths,
            dry_run=manifest.dry_run,
            interactive=manifest.interactive,
            auto_fix=manifest.auto_fix,
            outlines=manifest.outlines,
            scenes=[scene_states[key].plan for key in sorted(scene_states)],
            scene_states=scene_states,
            final_video=final_video,
            final_video_sha256=manifest.final_video_sha256,
            incremental=manifest.incremental,
            base_run_id=manifest.base_run_id,
            render_profile=manifest.render_profile,
            manifest_revision=manifest.revision,
            eval_round=manifest.eval_round,
        )
        if manifest.incremental and manifest.base_run_id:
            context.base_manifest = RunRepository(settings.WORKSPACE_DIR).load(manifest.base_run_id)
        return context

    def _generate_validated_code(
        self,
        plan: ScenePlan,
        *,
        feedback: str = "",
        previous_code: str = "",
        stream: bool = False,
        renderer: str | None = None,
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
                renderer=renderer,
            )

            validation = self._validate(code, renderer=renderer)
            if validation.is_valid:
                return code, validation.scene_classes[0]
            last_validation = validation
            # 提供详细的修复指导
            feedback_parts = [f"确定性校验未通过，必须修复以下问题：\n{validation.feedback}"]

            # 如果是 TexTemplate 相关错误，提供正确示例
            if any("TexTemplate" in err or "tex_template" in err for err in validation.errors):
                feedback_parts.append("""
\n=== 正确的 TexTemplate 配置示例 ===
你必须在 construct() 方法开头添加以下代码：

    tex_template = TexTemplate(tex_compiler="xelatex", output_format=".xdv")
    tex_template.add_to_preamble(r"\\usepackage{ctex}")
    config.tex_template = tex_template

然后每个 Tex/MathTex 调用都必须传入 tex_template=tex_template：

    eq = MathTex(r"\\frac{a}{b}", tex_template=tex_template)
    text = Tex(r"中文文本", tex_template=tex_template)
""")

            current_feedback = "".join(feedback_parts)
            current_previous = code
        raise ValidationError(
            "生成代码未通过确定性校验：\n"
            + (last_validation.feedback if last_validation else "未知错误"),
            hint="尝试简化场景或调整 prompt",
        )

    def run_incremental(
        self,
        user_prompt: str,
        base_run_id: str,
        callback: Callback | None = None,
        dry_run: bool = False,
        interactive: bool = False,
        output_path: Path | None = None,
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
        self._manifest = None
        ctx = PipelineContext(
            user_prompt=user_prompt,
            original_prompt=user_prompt,
            dry_run=dry_run,
            interactive=interactive,
            incremental=True,
            base_run_id=base_run_id,
            base_manifest=base_manifest,
            paths=RunPaths.create(output_path),
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
        output_path: Path | None = None,
    ) -> Path | None:
        if len(user_prompt) > settings.MAX_PROMPT_CHARS:
            raise ValueError(
                f"用户需求过长：{len(user_prompt)} 字符，最大允许 {settings.MAX_PROMPT_CHARS} 字符\n提示：可以将需求拆分为多个较短的动画，或使用更简洁的描述"
            )
        self._callback = callback
        self._manifest = None
        ctx = PipelineContext(
            user_prompt=user_prompt,
            original_prompt=user_prompt,
            dry_run=dry_run,
            interactive=interactive,
            paths=RunPaths.create(output_path),
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

        validation = self._validate(source_code, renderer=RenderProfile.current().renderer)
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
        self._write_private(paths.root / "prompt.md", prompt)
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
        scene_state = SceneState(
            plan=plan,
            code=source_code,
            class_name=class_name,
            plan_ready=True,
            reviewed=True,
        )
        ctx = PipelineContext(
            user_prompt=prompt,
            original_prompt=prompt,
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
                    if (
                        ctx.final_video_sha256
                        and sha256_file(ctx.final_video) != ctx.final_video_sha256
                    ):
                        raise RuntimeError("运行标记为完成，但最终视频哈希与清单不一致")
                    return ctx.final_video
                raise RuntimeError("运行标记为完成，但最终视频不存在")
            if manifest.status == "dry_run_complete":
                return None
            try:
                state = State[manifest.state]
            except KeyError as exc:
                raise ValueError(f"运行清单包含未知 FSM 状态: {manifest.state}") from exc

            # resume 明确表示用户愿意再次尝试；先重置放弃标记，再处理 ERROR。
            # 旧逻辑先判断 ERROR，导致“所有场景都已放弃”时无法进入这里的
            # 重试分支，仪表盘提示可恢复但实际直接报“无可用场景”。
            reset_give_up = False
            for scene in ctx.scene_states.values():
                if scene.give_up:
                    scene.give_up = False
                    scene.review_round = 0
                    scene.fix_attempts = 0
                    scene.failure_reason = ""
                    reset_give_up = True

            if state is State.ERROR:
                # 允许从 ERROR 状态恢复：无场景时重跑概要规划；已有场景
                # 则让场景级 worker 根据 plan_ready/code/reviewed 自己选择
                # 分镜、编码或审查，不因某个失败快照而永久卡死。
                if not ctx.scene_states:
                    state = State.PLANNING
                else:
                    for scene in ctx.scene_states.values():
                        if scene.failed:
                            scene.failed = False
                            scene.failure_reason = ""
                    state = State.CODING
                self._emit("run_resuming_from_error", run_id=run_id, state=state.name)

            if reset_give_up:
                # resume 时仪表盘可能已激活, 避免直接打印破坏 Live 渲染
                with suppress(Exception):
                    from kd1_anime.dashboard import quiet

                    if not quiet():
                        console.print("[yellow]发现已放弃的场景，将重置并重试[/]")
                if state not in {State.CODING, State.REVIEWING, State.DISPATCHING}:
                    # 回到审查阶段重新评估已生成的代码
                    state = State.REVIEWING

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

            # 核对恢复的 Slurm 作业: 上次会话提交的作业可能早已结束/已不存在,
            # 直接监控会得到连续 UNKNOWN → CANCEL_FAILED → 场景永久判死。
            self._reconcile_restored_jobs(ctx)

            self._emit("run_resumed", run_id=run_id, state=state.name)
            return self._execute(ctx, state)

    def _execute(self, ctx: PipelineContext, state: State) -> Path | None:
        try:
            # ---- 准备: 目录 + 全局概要 (仅全新运行; resume 已从清单加载) ----
            if not ctx.scene_states:
                self._handle_init(ctx)
                self._checkpoint(ctx, State.PLANNING)
                self._plan_outline(ctx)
                ctx.scene_states = {
                    outline.scene_id: SceneState(plan=self._placeholder_plan(outline))
                    for outline in ctx.outlines
                }
                self._emit("plan_complete", scenes=ctx.scenes)
            elif ctx.scenes:
                # resume 等已有 scene_states 的入口: 补发 plan_complete 供 TUI 表格展示
                self._emit(
                    "plan_complete",
                    scenes=[
                        state.plan
                        for state in sorted(
                            ctx.scene_states.values(), key=lambda s: s.plan.scene_id
                        )
                    ],
                )

            # resume: 把已有场景的当前进度以事件补发给 TUI/仪表盘, 否则调度器
            # 会跳过 rendered/failed 场景 (不发任何事件), 仪表盘会误显示为"未开始"。
            self._emit_scene_snapshot(ctx)

            # ---- 场景级并行调度主循环 ----
            improve = True
            while improve:
                self._run_scheduler(ctx)
                if ctx.dry_run:
                    break
                self._merge(ctx)
                improve = self._eval(ctx)

            # ---- 收尾 ----
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
        except KeyboardInterrupt:
            self.cancel_all()
            try:
                self._checkpoint(
                    ctx,
                    self._latest_checkpoint_state(state),
                    status="interrupted",
                    error="用户中断",
                )
            except Exception as checkpoint_error:
                console.print(f"[yellow]写入中断清单失败: {checkpoint_error}[/]", markup=False)
            raise
        except Exception as exc:
            self.cancel_all()
            try:
                self._checkpoint(
                    ctx,
                    self._latest_checkpoint_state(state),
                    status="failed",
                    error=str(exc),
                )
            except Exception as checkpoint_error:
                console.print(f"[yellow]写入失败清单失败: {checkpoint_error}[/]", markup=False)
            raise

    def _latest_checkpoint_state(self, fallback: State) -> State:
        """异常收尾时保留最后已持久化阶段，不用 _execute 入参覆盖。"""

        if self._manifest is None:
            return fallback
        try:
            return State[self._manifest.state]
        except KeyError:
            return fallback

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
        self._write_private(ctx.paths.root / "prompt.md", ctx.user_prompt)
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
            missing = [name for name in ("sbatch", "ffmpeg", "ffprobe") if not shutil.which(name)]
            if settings.SLURM_REQUIRE_CONTAINER and not shutil.which("apptainer"):
                missing.append("apptainer")
            if missing:
                raise RuntimeError("运行环境缺少命令: " + ", ".join(missing))

        # 增量渲染分析
        if ctx.incremental:
            logger.info("增量渲染模式：分析场景变化...")
            self._emit("incremental_start", base_run_id=ctx.base_run_id)

        return State.PLANNING

    def _handle_dispatching(self, ctx: PipelineContext) -> State:
        """仅供 ``render --no-wait`` 使用；普通流水线走逐 Scene 调度器。"""

        self._emit("stage_start", stage="dispatching")
        active = [
            state
            for state in ctx.scene_states.values()
            if state.slurm_job and not state.rendered and not state.failed and not state.give_up
        ]
        limit = settings.SLURM_MAX_IN_FLIGHT
        available = max(0, limit - len(active)) if limit else len(ctx.scene_states)
        submitted = 0
        for scene_id, state in sorted(ctx.scene_states.items()):
            if submitted >= available:
                break
            if state.rendered or state.failed or state.give_up or state.slurm_job:
                continue
            self._scene_submit(ctx, scene_id, state)
            if state.slurm_job is not None:
                submitted += 1
        has_active_jobs = any(
            state.slurm_job and not state.rendered and not state.failed and not state.give_up
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

    # ------------------------------------------------------------------
    # 场景级并行调度：每个 Scene 独立推进并共享受限资源。
    # ------------------------------------------------------------------

    @staticmethod
    def _placeholder_plan(outline: SceneOutline) -> ScenePlan:
        """由概要构造占位 ScenePlan (detail 完成前使用)。"""
        return ScenePlan(
            scene_id=outline.scene_id,
            title=outline.title,
            duration_seconds=outline.duration_seconds,
            purpose=outline.purpose,
            math_concept=outline.math_concept,
            visual_design="…",
            camera_movement="…",
            visual_flow=["…"],
            key_moments=["…"],
            computation="…",
        )

    def _plan_outline(self, ctx: PipelineContext) -> None:
        """全局场景概要 (一次); 带交互重试。"""
        while True:
            try:
                outlines = self.planner.plan_outline(ctx.user_prompt)
                break
            except LLMError as exc:
                logger.error(f"LLM 调用失败: {exc}")
                if not self._ask_retry_or_skip(0, str(exc)):
                    raise LLMResponseError(
                        f"场景概要规划失败: {exc}",
                        hint="检查 LLM API 配置和网络连接",
                    ) from exc
            except Exception as exc:
                logger.error(f"场景概要规划时发生未知错误: {exc}")
                if not self._ask_retry_or_skip(0, str(exc)):
                    raise PipelineError(f"场景概要规划失败: {exc}") from exc
        ctx.outlines = outlines
        ctx.scenes = [self._placeholder_plan(o) for o in outlines]
        if len(outlines) > settings.MAX_SCENES:
            raise RuntimeError(
                f"Planner 生成了 {len(outlines)} 个场景，超过 MAX_SCENES={settings.MAX_SCENES}"
            )

    # ------------------------------------------------------------------
    # 场景级并行调度 (per-scene pipeline)
    # 每个 Scene 一个工作线程, 独立推进 分镜→编码→审查→提交→渲染→修复。
    # LLM 并发受 LLM_PARALLEL_WORKERS 信号量限制; 提交受 SLURM_MAX_IN_FLIGHT 名额限制。
    # ------------------------------------------------------------------

    def _reconcile_restored_jobs(self, ctx: PipelineContext) -> None:
        """resume 时核对清单里恢复的 Slurm 作业, 避免监控失效作业导致场景判死。

        中断后再次 resume, 上次会话提交的作业可能早已结束或已从集群消失;
        若直接监控会得到连续 UNKNOWN → UNKNOWN_TIMEOUT → scancel 失败 →
        CANCEL_FAILED ("禁止自动重提"), 场景被永久判死。这里在调度前核对:
        - COMPLETED 且视频存在 → 直接标记渲染完成 (复用上次结果)
        - COMPLETED 但视频缺失 → 清空, 重跑
        - GONE (squeue 可查但无记录且 sacct 无账务记录) → 作业已消失, 清空后重新提交
        - UNKNOWN (集群查询失败/作业不可见) → 保留 Job ID；未确认终态或成功取消前
          绝不自动重提，避免临时 Slurm 故障制造重复作业
        - FAILED/CANCELLED 等终态 → 保留, 交给监控触发自动修复
        - RUNNING/PENDING → 保留, 继续监控
        """
        for scene_id, state in sorted(ctx.scene_states.items()):
            job = state.slurm_job
            if job is None or state.rendered or state.failed or state.give_up:
                continue
            try:
                status = self.slurm.poll_all_statuses([job.job_id]).get(job.job_id, "UNKNOWN")
            except Exception:
                continue  # 集群查询异常 → 保守保留, 交给监控处理
            if status == "COMPLETED":
                if self.slurm.validate_completed_job(job):
                    state.artifact = self._artifact_from_job(ctx, state, job)
                    state.rendered = True
                    self._emit("scene_rendered", scene_id=scene_id)
                else:
                    job.status = "FAILED"
            elif status == "GONE":
                outcome = self.slurm._classify_gone(job)
                if outcome == "COMPLETED":
                    state.artifact = self._artifact_from_job(ctx, state, job)
                    state.rendered = True
                    self._emit("scene_rendered", scene_id=scene_id)
                elif outcome is None:
                    # 已确认不在调度器且没有任何运行痕迹，可安全重新提交。
                    state.slurm_job = None
                else:
                    job.status = "FAILED"
            elif status == "UNKNOWN":
                job.status = "UNKNOWN"

    def _emit_scene_snapshot(self, ctx: PipelineContext) -> None:
        """resume 后把已有场景的当前进度以事件形式补发给 TUI/仪表盘。

        调度器只对未完成场景发事件, 已 rendered / failed / give_up 的场景会被
        跳过, 导致仪表盘把它们显示为"未开始"。这里按清单记录的当前状态补发
        事件, 让仪表盘恢复后立即反映真实进度。
        """
        for state in sorted(ctx.scene_states.values(), key=lambda s: s.plan.scene_id):
            scene_id = state.plan.scene_id
            # 前置阶段按实际状态补发, 让已渲染场景也显示完整流水线 (分镜✓编码✓审查✓渲染✓)
            if state.plan_ready:
                self._emit("scene_detailed", scene_id=scene_id, title=state.plan.title)
            if state.code:
                self._emit("scene_coded", scene_id=scene_id)
            if state.reviewed:
                self._emit("scene_review_pass", scene_id=scene_id)
            # 终态事件
            if state.rendered:
                self._emit("scene_rendered", scene_id=scene_id)
            elif state.failed:
                self._emit("scene_failed", scene_id=scene_id, reason=state.failure_reason or "")
            elif state.give_up:
                self._emit("scene_give_up", scene_id=scene_id, reason=state.failure_reason or "")
            elif state.slurm_job is not None:
                self._emit("scene_submitted", scene_id=scene_id, job_id=state.slurm_job.job_id)

    def _run_scheduler(self, ctx: PipelineContext) -> None:
        """启动每个场景的独立流水线线程, 全部结束后返回。"""
        import threading

        self._llm_sem = (
            self._resource_coordinator.llm
            if self._resource_coordinator
            else threading.Semaphore(max(1, settings.LLM_PARALLEL_WORKERS))
        )
        self._slot_lock = threading.Lock()
        self._in_flight = 0
        self._reserved_existing_scenes: set[int] = set()
        self._stop_event.clear()
        with self._phase_lock:
            self._emitted_phases.clear()

        threads: list[threading.Thread] = []
        for scene_id, state in sorted(ctx.scene_states.items()):
            if state.rendered or state.failed or state.give_up:
                continue
            if state.slurm_job is not None:
                self._reserve_existing_slot(scene_id)
            thread = threading.Thread(
                target=self._scene_worker,
                args=(ctx, scene_id, state),
                name=f"scene-{scene_id}",
                daemon=True,
            )
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()

    def _scene_worker(self, ctx: PipelineContext, scene_id: int, state: SceneState) -> None:
        """单个 Scene 的完整流水线: 分镜→编码→审查→提交→渲染→(修复→重新提交)。"""
        acquired = scene_id in self._reserved_existing_scenes
        try:
            if self._stop_event.is_set():
                return
            # 1) 分镜
            if not state.plan_ready:
                self._phase_emit("detailing")
                self._scene_detail(ctx, scene_id, state)
                if state.failed or state.give_up:
                    return
            # 2) 编码 + 审查循环 (直到审查通过)
            while not state.reviewed:
                if self._stop_event.is_set() or state.failed or state.give_up:
                    return
                if not state.code or state.rewrite_feedback:
                    self._phase_emit("coding")
                    self._scene_code(ctx, scene_id, state)
                    if state.failed or state.give_up:
                        return
                self._phase_emit("reviewing")
                self._scene_review(ctx, scene_id, state)
            # 3) dry-run: 不提交渲染
            if ctx.dry_run:
                return
            # 4) 提交 + 渲染循环 (直到渲染成功/失败/放弃)
            while not state.rendered:
                if self._stop_event.is_set() or state.failed or state.give_up:
                    return
                if state.slurm_job is None:
                    if not acquired:
                        if not self._try_acquire_slot():
                            time.sleep(settings.MONITOR_POLL_INTERVAL)
                            continue
                        acquired = True
                    self._scene_submit(ctx, scene_id, state)
                    if state.slurm_job is None:
                        # 提交失败 → 释放名额
                        self._release_slot()
                        acquired = False
                        return
                ok = self._scene_wait_render(ctx, state)
                if ok:
                    return
                if state.failed or state.give_up:
                    return
                # 渲染失败 → 修复后重新提交 (名额保留)
                self._scene_fix(ctx, scene_id, state)
                # AutoFix 改变了代码，必须再次经过 Reviewer；major 反馈仍按
                # 正常编码循环重写，绝不能把未经复审的修复代码直接提交。
                while not state.reviewed:
                    if self._stop_event.is_set() or state.failed or state.give_up:
                        return
                    if state.rewrite_feedback:
                        self._phase_emit("coding")
                        self._scene_code(ctx, scene_id, state)
                    self._phase_emit("reviewing")
                    self._scene_review(ctx, scene_id, state)
        except Exception as exc:
            with self._state_lock:
                self._mark_failed(state, f"Scene {scene_id} 流水线异常: {exc}")
                self._checkpoint(ctx, State.MONITORING)
            self._emit("scene_failed", scene_id=scene_id, reason=str(exc))
        finally:
            # 无论成功/失败/异常, 都要释放 in-flight 名额, 避免其他场景死等
            if acquired:
                self._release_slot()
            if not self._stop_event.is_set():
                # 检查点失败不应让工作线程崩溃 (可能导致 join 永久等待)
                with suppress(Exception):
                    self._checkpoint(ctx, State.MONITORING)

    def _try_acquire_slot(self) -> bool:
        if self._resource_coordinator:
            return self._resource_coordinator.try_acquire_slurm()
        limit = settings.SLURM_MAX_IN_FLIGHT
        with self._slot_lock:
            if limit and self._in_flight >= limit:
                return False
            self._in_flight += 1
            return True

    def _reserve_existing_slot(self, scene_id: int) -> None:
        """让 resume 时已存在的远程作业先占用名额，防止继续超量提交。"""

        if self._resource_coordinator:
            self._resource_coordinator.register_existing_slurm()
        else:
            with self._slot_lock:
                self._in_flight += 1
        self._reserved_existing_scenes.add(scene_id)

    def _release_slot(self) -> None:
        if self._resource_coordinator:
            self._resource_coordinator.release_slurm()
            return
        with self._slot_lock:
            self._in_flight = max(0, self._in_flight - 1)

    def _phase_emit(self, stage: str) -> None:
        """每个阶段只发一次 stage_start (线程安全), 供 TUI/仪表盘显示阶段标题。"""
        with self._phase_lock:
            if stage in self._emitted_phases:
                return
            self._emitted_phases.add(stage)
        self._emit("stage_start", stage=stage)

    def _scene_detail(self, ctx: PipelineContext, scene_id: int, state: SceneState) -> None:
        self._emit(
            "scene_detailing",
            scene_id=scene_id,
            title=state.plan.title,
        )
        with self._llm_sem:
            outline = next(o for o in ctx.outlines if o.scene_id == scene_id)
            plan = PlannerAgent().plan_detail(
                outline,
                ctx.outlines,
                ctx.user_prompt,
                stream=False,
                renderer=ctx.render_profile.renderer,
            )
        with self._state_lock:
            state.plan = plan
            state.plan_ready = True
            self._checkpoint(ctx, State.DETAILING)
        self._emit("scene_detailed", scene_id=scene_id, title=plan.title)

    def _scene_code(self, ctx: PipelineContext, scene_id: int, state: SceneState) -> None:
        rewriting = bool(state.rewrite_feedback)
        self._emit(
            "scene_rewriting" if rewriting else "scene_coding",
            scene_id=scene_id,
            title=state.plan.title,
            reason=state.rewrite_feedback if rewriting else "",
        )
        with self._llm_sem:
            code, class_name = self._generate_validated_code(
                state.plan,
                feedback=state.rewrite_feedback or "",
                previous_code=state.code if state.rewrite_feedback else "",
                stream=False,
                renderer=ctx.render_profile.renderer,
            )
        path = ctx.paths.scenes / f"scene_{scene_id}.py"
        self._write_private(path, code)
        with self._state_lock:
            state.code = code
            state.class_name = class_name
            state.rewrite_feedback = ""
            state.reviewed = False
            state.artifact = None
            state.rendered = False
            self._checkpoint(ctx, State.CODING)
        self._emit("scene_coded", scene_id=scene_id, file_path=str(path))

    def _scene_review(self, ctx: PipelineContext, scene_id: int, state: SceneState) -> None:
        if settings.SKIP_REVIEW:
            with self._state_lock:
                state.reviewed = True
                self._apply_incremental_for_scene(ctx, scene_id, state)
                self._checkpoint(ctx, State.REVIEWING)
            self._emit("scene_review_skipped", scene_id=scene_id)
            if state.rendered and state.artifact and state.artifact.origin == "reused":
                self._emit("scene_reused", scene_id=scene_id)
            return
        self._emit("scene_reviewing", scene_id=scene_id)
        with self._llm_sem:
            result = ReviewerAgent().review(
                state.code,
                state.plan,
                renderer=ctx.render_profile.renderer,
            )
        self._apply_review_result(ctx, scene_id, state, result)

    def _scene_submit(self, ctx: PipelineContext, scene_id: int, state: SceneState) -> None:
        self._phase_emit("dispatching")
        source_path = ctx.paths.scenes / f"scene_{scene_id}.py"
        try:
            on_disk_code = source_path.read_text(encoding="utf-8")
        except OSError as exc:
            with self._state_lock:
                self._mark_failed(state, f"提交前无法读取场景代码: {exc}")
                self._checkpoint(ctx, State.DISPATCHING)
            self._emit("scene_failed", scene_id=scene_id, reason=state.failure_reason)
            return
        if on_disk_code != state.code:
            with self._state_lock:
                self._mark_failed(state, "提交前代码一致性校验失败：磁盘文件已在流水线外被修改")
                self._checkpoint(ctx, State.DISPATCHING)
            self._emit("scene_failed", scene_id=scene_id, reason=state.failure_reason)
            return
        validation = self._validate(on_disk_code, renderer=ctx.render_profile.renderer)
        if not validation.is_valid:
            with self._state_lock:
                self._mark_failed(state, "提交前校验失败:\n" + validation.feedback)
                self._checkpoint(ctx, State.DISPATCHING)
            self._emit("scene_failed", scene_id=scene_id, reason=state.failure_reason)
            return
        state.class_name = validation.scene_classes[0]
        try:
            job = self.slurm.submit_scene(
                scene_id,
                source_path,
                state.class_name,
                scenes_dir=ctx.paths.scenes,
                logs_dir=ctx.paths.logs,
                videos_dir=ctx.paths.videos,
                code_sha256=sha256_text(state.code),
                render_profile=ctx.render_profile,
            )
            with self._state_lock:
                state.slurm_job = job
                state.artifact = None
                state.rendered = False
                self._checkpoint(ctx, State.DISPATCHING)
            self._emit(
                "scene_submitted",
                scene_id=scene_id,
                job_id=job.job_id,
            )
        except Exception as exc:
            # 一旦拿到 Job ID 就绝不能把持久化异常伪装成提交失败并自动重提。
            if "job" in locals():
                with self._state_lock:
                    state.slurm_job = job
                    state.give_up = True
                    state.failure_reason = (
                        f"Slurm Job {job.job_id} 已提交，但本地检查点持久化失败: {exc}；"
                        "保留 Job ID 并禁止自动重提"
                    )
                self._emit("scene_failed", scene_id=scene_id, reason=state.failure_reason)
                return
            with self._state_lock:
                self._mark_failed(state, f"Slurm 提交失败: {exc}")
                self._checkpoint(ctx, State.DISPATCHING)
            self._emit("scene_failed", scene_id=scene_id, reason=str(exc))

    def _scene_wait_render(self, ctx: PipelineContext, state: SceneState) -> bool:
        """阻塞轮询当前作业直到结束; 返回是否渲染成功。"""
        job = state.slurm_job
        if job is None:
            return False
        self._phase_emit("monitoring")
        monitor = JobMonitor(self.slurm)
        monitor.add_job(job)
        while monitor.pending:
            if monitor.poll_once():
                break
            time.sleep(settings.MONITOR_POLL_INTERVAL)
        ok = monitor.results.get(job.job_id)
        if ok is None:
            with self._state_lock:
                state.give_up = True
                state.failure_reason = "渲染作业状态未知，已放弃"
                self._checkpoint(ctx, State.MONITORING)
            return False
        if ok:
            with self._state_lock:
                state.artifact = self._artifact_from_job(ctx, state, job)
                state.rendered = True
                self._checkpoint(ctx, State.MONITORING)
            self._emit("scene_rendered", scene_id=job.scene_id)
            return True
        if not ctx.auto_fix or job.status not in FIXABLE_RENDER_STATES:
            with self._state_lock:
                if not ctx.auto_fix:
                    self._mark_failed(state, job.failure_reason or f"Slurm 状态: {job.status}")
                else:
                    state.give_up = True
                    state.failure_reason = (
                        job.failure_reason or f"基础设施失败，不修改代码: {job.status}"
                    )
                self._checkpoint(ctx, State.MONITORING)
        self._emit(
            "scene_failed",
            scene_id=job.scene_id,
            reason=job.failure_reason or f"Slurm 状态: {job.status}",
        )
        return False

    @staticmethod
    def _error_fingerprint(error_log: str) -> str:
        """渲染错误日志的稳定指纹: 数字统一归一化、跳过进度行。

        同一环境/代码错误即使时间戳、帧号、进度百分比不同, 指纹也应相同,
        用于判断"修复后错误是否仍然一样"。
        """
        import hashlib

        normalized: list[str] = []
        for line in error_log.splitlines():
            low = line.lower()
            # 跳过 Manim 进度条行 (含 % 和 |) 与纯时间行
            if "%" in low and "|" in low:
                continue
            low = re.sub(r"\d+", "#", low).strip()
            if low:
                normalized.append(low)
        return hashlib.sha256("\n".join(normalized).encode("utf-8", errors="replace")).hexdigest()[
            :16
        ]

    def _give_up_reason(self, base: str, error_log: str) -> str:
        """放弃原因附带最近错误日志尾部, 方便用户直接定位根因。"""
        if not error_log:
            return base
        excerpt = "\n".join(error_log.splitlines()[-8:]).strip()
        if len(excerpt) > 600:
            excerpt = excerpt[-600:]
        return f"{base}\n最近错误日志(尾部):\n{excerpt}"

    def _scene_fix(self, ctx: PipelineContext, scene_id: int, state: SceneState) -> None:
        self._phase_emit("fixing")
        job = state.slurm_job
        if job is None:
            return
        # 与 Coder/Reviewer 一样每个 worker 独立构造 Agent，避免并发修复
        # 共享 OpenAI client/流式状态。
        fixer = AutoFixerAgent()
        error_log = self.slurm.get_error_log(job=job)
        if not error_log:
            with self._state_lock:
                state.give_up = True
                state.failure_reason = job.failure_reason or "渲染失败且没有错误日志"
                self._checkpoint(ctx, State.FIXING)
            self._emit("scene_give_up", scene_id=scene_id, reason=state.failure_reason)
            return
        if fixer.is_infrastructure_error(error_log):
            with self._state_lock:
                state.give_up = True
                state.failure_reason = self._give_up_reason(
                    "检测到环境或 Slurm 配置错误，未让 LLM 重写业务代码", error_log
                )
                self._checkpoint(ctx, State.FIXING)
            self._emit("scene_give_up", scene_id=scene_id, reason=state.failure_reason)
            return
        # 连续相同错误 → 判定为环境/配置问题, 提前放弃, 不再浪费修复次数
        fp = self._error_fingerprint(error_log)
        with self._state_lock:
            if fp and fp == state.last_error_fp:
                state.identical_error_count += 1
            else:
                state.identical_error_count = 1
                state.last_error_fp = fp
            # 连续相同错误 → 提前放弃, 避免修复器在同一个环境错误上空转。
            # 但必须叠加 fix_attempts>=2 门槛: 修复器至少要修过 2 次才允许据此放弃。
            if (
                state.identical_error_count >= settings.MAX_FIX_IDENTICAL_ERRORS
                and state.fix_attempts >= 2
            ):
                state.give_up = True
                state.failure_reason = self._give_up_reason(
                    f"连续 {state.identical_error_count} 次渲染错误完全相同且修复未能消除，"
                    "疑似环境/配置问题，已放弃",
                    error_log,
                )
                self._checkpoint(ctx, State.FIXING)
                terminal = True
            elif state.fix_attempts >= settings.MAX_FIX_ATTEMPTS:
                state.give_up = True
                state.failure_reason = self._give_up_reason("达到最大渲染修复次数", error_log)
                self._checkpoint(ctx, State.FIXING)
                terminal = True
            else:
                state.fix_attempts += 1
                # 在 LLM 调用前持久化次数，避免进程中断后无限重试同一轮。
                self._checkpoint(ctx, State.FIXING)
                terminal = False
            attempt = state.fix_attempts
        if terminal:
            self._emit("scene_give_up", scene_id=scene_id, reason=state.failure_reason)
            return
        self._emit(
            "scene_fixing",
            scene_id=scene_id,
            attempt=attempt,
            max_attempts=settings.MAX_FIX_ATTEMPTS,
        )
        with self._llm_sem:
            candidate = fixer.fix(
                state.code,
                error_log,
                renderer=ctx.render_profile.renderer,
            )
            validation = self._validate(candidate, renderer=ctx.render_profile.renderer)
            if not validation.is_valid:
                candidate, class_name = self._generate_validated_code(
                    state.plan,
                    feedback=(
                        "AutoFix 结果未通过确定性校验：\n"
                        f"{validation.feedback}\n\n原始渲染错误：\n{error_log}"
                    ),
                    previous_code=candidate,
                    stream=False,
                    renderer=ctx.render_profile.renderer,
                )
            else:
                class_name = validation.scene_classes[0]
        self._write_private(ctx.paths.scenes / f"scene_{scene_id}.py", candidate)
        with self._state_lock:
            state.code = candidate
            state.class_name = class_name
            state.review_round = 0
            state.reviewed = False
            state.slurm_job = None
            state.artifact = None
            state.rendered = False
            self._checkpoint(ctx, State.FIXING)
        self._emit(
            "scene_coded",
            scene_id=scene_id,
            file_path=str(ctx.paths.scenes / f"scene_{scene_id}.py"),
        )
        # 注意: identical_error_count 不在这里重置 —— 只有当"错误指纹变化"时才重置
        # (见上面的 else 分支), 从而让"修复后错误完全相同"能在第 2 次相同错误时提前放弃。

    def _apply_review_result(
        self, ctx: PipelineContext, scene_id: int, state: SceneState, result: ReviewResult
    ) -> bool:
        """应用单场景审查结果。"""
        if result.is_valid:
            with self._state_lock:
                state.review_round = 0
                state.reviewed = True
                self._apply_incremental_for_scene(ctx, scene_id, state)
                self._checkpoint(ctx, State.REVIEWING)
            self._emit("scene_review_pass", scene_id=scene_id)
            if state.rendered and state.artifact and state.artifact.origin == "reused":
                self._emit("scene_reused", scene_id=scene_id)
            return True

        with self._state_lock:
            state.review_round += 1
            review_round = state.review_round
            # Reviewer 已消耗一轮，在后续改写前先持久化计数。
            self._checkpoint(ctx, State.REVIEWING)
        original_feedback = result.feedback or ""
        fix_details = "\n".join(
            f"- [{fix.reason}] {fix.find!r} → {fix.replace!r}" for fix in result.fixes
        )

        if review_round >= settings.MAX_REVIEW_ROUNDS:
            with self._state_lock:
                state.give_up = True
                state.failure_reason = "达到最大审查轮次，代码仍未通过"
                self._checkpoint(ctx, State.REVIEWING)
            self._emit("scene_give_up", scene_id=scene_id, reason=state.failure_reason)
            return True

        if result.severity == "minor":
            candidate = state.code
            applied_count = 0
            for fix in result.fixes:
                if candidate.count(fix.find) == 1:
                    candidate = candidate.replace(fix.find, fix.replace, 1)
                    applied_count += 1
            validation = (
                self._validate(candidate, renderer=ctx.render_profile.renderer)
                if applied_count > 0
                else None
            )
            if validation and validation.is_valid:
                self._write_private(ctx.paths.scenes / f"scene_{scene_id}.py", candidate)
                with self._state_lock:
                    state.code = candidate
                    state.class_name = validation.scene_classes[0]
                    state.artifact = None
                    state.rendered = False
                    self._checkpoint(ctx, State.REVIEWING)
                self._emit(
                    "scene_coded",
                    scene_id=scene_id,
                    file_path=str(ctx.paths.scenes / f"scene_{scene_id}.py"),
                )
                self._emit("scene_review_fail", scene_id=scene_id, severity="minor")
                return True
            # minor 修复失败 → 升级为 major, 保留原始反馈
            result = ReviewResult(
                is_valid=False,
                severity="major",
                feedback=(
                    f"## Reviewer 审查意见（minor 修复未能全部应用）\n{original_feedback}\n\n"
                    f"## 修复建议详情\n{fix_details}\n\n"
                    f"## 确定性校验\n{validation.feedback if validation else '未生成有效代码'}"
                ),
            )

        # major → 排队重写 (下一轮调度进入编码阶段)
        with self._state_lock:
            state.rewrite_feedback = (
                f"## Reviewer 审查意见\n{original_feedback}\n\n"
                f"## 需修复的问题\n{fix_details}\n\n"
                f"请根据以上反馈逐项修正代码，保留正确部分，只修复指出的问题。"
            )
            state.artifact = None
            state.rendered = False
            self._checkpoint(ctx, State.REVIEWING)
        self._emit("scene_review_fail", scene_id=scene_id, severity="major")
        return True

    def _apply_incremental_for_scene(
        self, ctx: PipelineContext, scene_id: int, state: SceneState
    ) -> None:
        """Review 通过后，按代码与渲染配置身份安全复用旧产物。"""
        if not ctx.incremental or not ctx.base_manifest or not ctx.base_run_id:
            return
        base_scene = ctx.base_manifest.scenes.get(scene_id)
        if base_scene is None:
            ctx.scenes_to_render.append(scene_id)
            return
        artifact = base_scene.artifact
        if (
            base_scene.rendered
            and artifact is not None
            and artifact.verified
            and sha256_text(state.code) == base_scene.code_sha256
            and artifact.code_sha256 == base_scene.code_sha256
            and artifact.render_profile_sha256 == ctx.render_profile.digest()
        ):
            base_root = RunRepository(settings.WORKSPACE_DIR).run_root(ctx.base_run_id)
            old_video = get_reusable_video_path(ctx.base_manifest, scene_id, base_root)
            if old_video and sha256_file(old_video) == artifact.video_sha256:
                state.rendered = True
                state.slurm_job = None
                state.artifact = SceneArtifact(
                    origin="reused",
                    source_run_id=artifact.source_run_id,
                    job_id=artifact.job_id,
                    scene_id=scene_id,
                    scene_class_name=artifact.scene_class_name,
                    code_sha256=artifact.code_sha256,
                    render_profile_sha256=artifact.render_profile_sha256,
                    video_path=artifact.video_path,
                    video_sha256=artifact.video_sha256,
                    metadata=artifact.metadata,
                    verified=True,
                )
                if scene_id not in ctx.scenes_to_reuse:
                    ctx.scenes_to_reuse.append(scene_id)
                return
        if scene_id not in ctx.scenes_to_render:
            ctx.scenes_to_render.append(scene_id)

    def _artifact_from_job(
        self,
        ctx: PipelineContext,
        state: SceneState,
        job: SlurmJob,
    ) -> SceneArtifact:
        if (
            job.output_path is None or job.output_metadata is None
        ) and not self.slurm.validate_completed_job(job):
            raise RuntimeError(job.failure_reason or "渲染产物验证失败")
        if job.output_path is None or job.output_metadata is None:
            raise RuntimeError("渲染产物缺少路径或媒体元数据")
        try:
            relative_video = (
                job.output_path.resolve().relative_to(ctx.paths.root.resolve()).as_posix()
            )
        except ValueError as exc:
            raise RuntimeError("渲染产物不在当前 run 目录内") from exc
        code_hash = sha256_text(state.code)
        if job.code_sha256 != code_hash:
            raise RuntimeError("渲染作业记录的代码哈希与当前场景不一致")
        if job.render_profile.digest() != ctx.render_profile.digest():
            raise RuntimeError("渲染作业记录的配置与当前运行不一致")
        return SceneArtifact(
            origin="rendered",
            source_run_id=ctx.paths.run_id,
            job_id=job.job_id,
            scene_id=job.scene_id,
            scene_class_name=job.scene_class_name,
            code_sha256=code_hash,
            render_profile_sha256=ctx.render_profile.digest(),
            video_path=relative_video,
            video_sha256=sha256_file(job.output_path),
            metadata=job.output_metadata,
            verified=True,
        )

    @staticmethod
    def _artifact_video_path(ctx: PipelineContext, artifact: SceneArtifact) -> Path:
        if not artifact.verified:
            raise RuntimeError(f"Scene {artifact.scene_id} 的渲染产物未经验证")
        source_root = (
            ctx.paths.root
            if artifact.source_run_id == ctx.paths.run_id
            else RunRepository(settings.WORKSPACE_DIR).run_root(artifact.source_run_id)
        )
        video = restore_run_path(source_root, artifact.video_path)
        if not video.is_file() or video.stat().st_size <= 0:
            raise RuntimeError(f"Scene {artifact.scene_id} 的渲染产物不存在: {video}")
        if sha256_file(video) != artifact.video_sha256:
            raise RuntimeError(f"Scene {artifact.scene_id} 的渲染产物哈希不一致")
        return video

    def _merge(self, ctx: PipelineContext) -> None:
        """按代码、配置和视频哈希验证每个产物后再合并。"""
        self._emit("stage_start", stage="merging")
        rendered_artifacts: list[SceneArtifact] = []
        for scene_id, state in sorted(ctx.scene_states.items()):
            if not state.rendered:
                continue
            artifact = state.artifact
            if artifact is None:
                raise RuntimeError(f"Scene {scene_id} 标记为已渲染，但缺少产物凭据")
            if artifact.scene_id != scene_id:
                raise RuntimeError(f"Scene {scene_id} 的产物场景 ID 不一致")
            if artifact.scene_class_name != state.class_name:
                raise RuntimeError(f"Scene {scene_id} 的产物类名与当前代码不一致")
            if artifact.code_sha256 != sha256_text(state.code):
                raise RuntimeError(f"Scene {scene_id} 的产物代码哈希与当前代码不一致")
            if artifact.render_profile_sha256 != ctx.render_profile.digest():
                raise RuntimeError(f"Scene {scene_id} 的产物渲染配置与当前运行不一致")
            rendered_artifacts.append(artifact)
        incomplete = [sid for sid, state in ctx.scene_states.items() if not state.rendered]
        if not rendered_artifacts:
            raise RuntimeError("流水线未能完成。\n没有场景成功渲染")
        if incomplete and not settings.ALLOW_PARTIAL_OUTPUT:
            for sid in incomplete:
                state = ctx.scene_states[sid]
                if not state.failure_reason:
                    state.failure_reason = "场景未成功渲染"
            reasons = "\n".join(
                f"Scene {sid}: {ctx.scene_states[sid].failure_reason or '未完成'}"
                for sid in incomplete
            )
            self._emit("partial_output_blocked", incomplete=incomplete)
            raise RuntimeError("流水线未能完成。\n" + reasons)

        self._checkpoint(ctx, State.MERGING)
        resolved_output = ctx.paths.output.expanduser().resolve()
        output_is_run_local = resolved_output == ctx.paths.root.resolve() or (
            ctx.paths.root.resolve() in resolved_output.parents
        )
        # 评估-改进循环的第二次合并必须强制重新拼接, 不能复用上一轮的旧视频
        force_remerge = getattr(ctx, "eval_round", 0) > 0
        can_reuse_checkpointed_output = (
            ctx.final_video is not None
            and ctx.final_video.expanduser().resolve() == resolved_output
        )
        checkpointed_output_matches = (
            can_reuse_checkpointed_output
            and bool(ctx.final_video_sha256)
            and resolved_output.is_file()
            and sha256_file(resolved_output) == ctx.final_video_sha256
        )
        if (
            output_is_run_local
            and resolved_output.is_file()
            and resolved_output.stat().st_size > 0
            and not settings.OVERWRITE_OUTPUT
            and not force_remerge
            and checkpointed_output_matches
        ):
            ctx.final_video = resolved_output
        else:
            # run 内输出属于当前私有目录，即使旧检查点没有输入收据，也可以让
            # VideoMerger 写临时文件并在验证成功后原子替换；不能提前删除旧文件，
            # 否则新的 FFmpeg 失败会丢失上一次仍可用的结果。
            video_paths = [self._artifact_video_path(ctx, item) for item in rendered_artifacts]
            ctx.final_video = self.merger.merge(
                video_paths,
                ctx.paths.output,
                replace_existing=(
                    output_is_run_local
                    or (force_remerge and not output_is_run_local and checkpointed_output_matches)
                ),
                render_profile=ctx.render_profile,
            )
        ctx.final_video_sha256 = sha256_file(ctx.final_video)
        self._checkpoint(ctx, State.MERGING)
        self._emit(
            "merge_complete",
            path=str(ctx.final_video),
            size_mb=ctx.final_video.stat().st_size / (1024 * 1024),
            partial=bool(incomplete),
            incomplete=incomplete,
        )

    def _eval(self, ctx: PipelineContext) -> bool:
        """评估最终视频; 返回 True 表示触发改进 (需要重新调度低分场景)。"""
        if not settings.ENABLE_AUTO_EVAL:
            return False
        self._emit("stage_start", stage="evaluating")
        self._checkpoint(ctx, State.EVALUATING)
        eval_round = getattr(ctx, "eval_round", 0)
        if eval_round >= settings.MAX_EVAL_ROUNDS:
            self._emit(
                "eval_max_rounds_reached",
                rounds=eval_round,
                max_rounds=settings.MAX_EVAL_ROUNDS,
            )
            return False
        from kd1_anime.eval import Evaluator

        if not (ctx.final_video and ctx.final_video.exists()):
            self._emit("eval_skipped", reason="no_final_video")
            return False
        try:
            evaluator = Evaluator(
                enable_visual_eval=settings.ENABLE_VISUAL_EVAL,
                visual_eval_model=settings.EVAL_VISUAL_MODEL,
                output_dir=ctx.paths.root / "eval_reports",
            )
            scene_eval_results = {
                scene_id: evaluator.evaluate_code(state.code)
                for scene_id, state in ctx.scene_states.items()
                if state.code
            }
            eval_result = evaluator.evaluate_run(
                ctx.paths.run_id,
                ctx.paths.root,
                description=ctx.original_prompt or ctx.user_prompt,
                enable_visual=settings.ENABLE_VISUAL_EVAL,
            )
            eval_result.save(ctx.paths.root / "eval_result.json")
        except Exception as exc:
            # 可选评估在修改任何流水线状态前失败时，只记录并跳过。
            self._emit("eval_error", error=str(exc))
            return False

        overall_score = eval_result.overall_score
        code_values = [
            score.score for score in eval_result.scores if score.metric.value.startswith("code_")
        ]
        visual_values = [
            score.score
            for score in eval_result.scores
            if score.metric.value.startswith("visual_") or score.metric.value == "element_layout"
        ]
        self._emit(
            "eval_complete",
            overall_score=overall_score,
            code_score=(sum(code_values) / len(code_values) if code_values else None),
            visual_score=(sum(visual_values) / len(visual_values) if visual_values else None),
            errors=eval_result.errors,
            threshold=settings.EVAL_THRESHOLD,
        )
        if settings.ENABLE_VISUAL_EVAL and not visual_values:
            self._emit(
                "eval_skipped",
                reason="visual_metrics_unavailable",
                errors=eval_result.errors,
            )
            return False
        if overall_score is None:
            self._emit("eval_skipped", reason="no_valid_metrics", errors=eval_result.errors)
            return False
        if overall_score >= settings.EVAL_THRESHOLD:
            self._emit(
                "eval_passed",
                score=overall_score,
                threshold=settings.EVAL_THRESHOLD,
            )
            return False

        low_score_scenes = [
            scene_id
            for scene_id, scene_eval in scene_eval_results.items()
            if scene_eval.overall_score is not None
            and scene_eval.overall_score < settings.EVAL_THRESHOLD
        ]
        if not low_score_scenes:
            self._emit(
                "eval_improvement_skipped",
                reason="low_score_not_attributable_to_scene_code",
            )
            return False

        self._emit(
            "eval_below_threshold",
            score=overall_score,
            threshold=settings.EVAL_THRESHOLD,
            action="triggering_improvement",
        )
        with self._state_lock:
            ctx.eval_round = eval_round + 1
            ctx.scenes_to_improve = low_score_scenes
            for scene_id in low_score_scenes:
                state = ctx.scene_states[scene_id]
                state.rendered = False
                state.reviewed = False
                state.code = ""
                state.class_name = ""
                state.slurm_job = None
                state.artifact = None
                state.fix_attempts = 0
                state.rewrite_feedback = ""
            # 轮次和场景重置必须先持久化；检查点失败应终止流水线，不能被
            # 当作可忽略的视觉评估错误，否则 resume 会复用上一轮旧视频。
            self._checkpoint(ctx, State.EVALUATING)
        return True
