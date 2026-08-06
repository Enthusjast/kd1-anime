"""有限状态机：串联规划、代码生成、审查、Slurm 渲染、修复与拼接。"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
import re
import threading
import time
from uuid import uuid4

from rich.console import Console
from rich.prompt import Confirm

from kd1_anime.exceptions import (
    ConfigError,
    LLMError,
    LLMResponseError,
    PipelineAbortedError,
    PipelineExhaustedError,
    PipelineError,
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
from kd1_anime.cluster.slurm import (
    FAILURE_STATES,
    MONITOR_ABORT_STATES,
    JobMonitor,
    SlurmDispatcher,
    SlurmJob,
)
from kd1_anime.config import resolve_runtime_path, settings
from kd1_anime.media.merger import VideoMerger
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

console = Console()
Callback = Callable[[str, dict], None]
FIXABLE_RENDER_STATES = {"FAILED", "TIMEOUT", "OUT_OF_MEMORY", "RUN_TIMEOUT"}




def fix_tex_template(code: str) -> str:
    """自动修复 Manim 代码中缺失的 TexTemplate 配置。

    策略:
    1. 用 AST 精确定位 Tex/MathTex 调用（避免字符串/注释误匹配）
    2. 用 AST 查找是否已有 XeLaTeX 模板配置
    3. 仅在真正缺失时添加配置，避免重复添加
    """
    import ast as _ast

    # 如果代码不包含任何 Tex/MathTex，直接返回
    if not re.search(r"\b(?:Tex|MathTex)\b", code):
        return code

    try:
        tree = _ast.parse(code)
    except SyntaxError:
        # 语法错误的代码无法可靠修复，原样返回让校验器报错
        return code

    # --- 检查是否已有完整配置 ---
    has_xelatex_template = False
    has_ctex = False
    has_config_assignment = False
    template_var_name: str | None = None

    for node in _ast.walk(tree):
        if isinstance(node, _ast.Call):
            func_name = ""
            if isinstance(node.func, _ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, _ast.Attribute):
                func_name = node.func.attr
            if func_name == "TexTemplate":
                for kw in node.keywords:
                    if kw.arg == "tex_compiler" and isinstance(kw.value, _ast.Constant):
                        if kw.value.value == "xelatex":
                            has_xelatex_template = True
        if isinstance(node, _ast.Call) and isinstance(node.func, _ast.Attribute):
            if node.func.attr == "add_to_preamble":
                for arg in node.args:
                    if isinstance(arg, _ast.Constant) and "ctex" in str(arg.value):
                        has_ctex = True
        if isinstance(node, _ast.Assign):
            for target in node.targets:
                if isinstance(target, _ast.Attribute) and target.attr == "tex_template":
                    if isinstance(target.value, _ast.Name) and target.value.id == "config":
                        has_config_assignment = True

    # 找到模板变量名（第一个 tex_template = TexTemplate(...) 的赋值）
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, _ast.Name) and isinstance(node.value, _ast.Call):
                func_name = ""
                if isinstance(node.value.func, _ast.Name):
                    func_name = node.value.func.id
                elif isinstance(node.value.func, _ast.Attribute):
                    func_name = node.value.func.attr
                if func_name == "TexTemplate":
                    template_var_name = target.id
                    break

    # 如果配置完整，只需检查 tex_template 参数
    if has_xelatex_template and has_ctex and has_config_assignment and template_var_name:
        return _ensure_tex_template_param(code, template_var_name)

    # --- 配置不完整，需要添加 ---
    lines = code.split("\n")
    new_lines: list[str] = []
    in_construct = False
    added = False

    for line in lines:
        if re.match(r"\s*def construct\(self\)", line):
            in_construct = True
            new_lines.append(line)
            continue

        if in_construct and not added:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                indent = line[: len(line) - len(line.lstrip())]
                new_lines.append(f"{indent}# TexTemplate 配置（自动添加）")
                new_lines.append(
                    f'{indent}tex_template = TexTemplate(tex_compiler="xelatex", output_format=".xdv")'
                )
                new_lines.append(f'{indent}tex_template.add_to_preamble(r"\\usepackage{{ctex}}")')
                new_lines.append(f"{indent}config.tex_template = tex_template")
                added = True
                in_construct = False

        new_lines.append(line)

    result = "\n".join(new_lines)
    return _ensure_tex_template_param(result, "tex_template")


def _ensure_tex_template_param(code: str, template_var: str) -> str:
    """为缺少 tex_template= 参数的 Tex/MathTex 调用添加该参数。

    使用 AST 定位需要修复的调用行号，然后做精准的行级替换。
    """
    import ast as _ast

    try:
        tree = _ast.parse(code)
    except SyntaxError:
        return code

    lines = code.split("\n")
    lines_to_fix: set[int] = set()
    _lines_list: list[int] = []  # 0-indexed

    for node in _ast.walk(tree):
        if not isinstance(node, _ast.Call):
            continue
        func_name = ""
        if isinstance(node.func, _ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, _ast.Attribute):
            func_name = node.func.attr
        if func_name not in ("Tex", "MathTex"):
            continue
        # 检查是否已有 tex_template 参数
        has_param = any(kw.arg == "tex_template" for kw in node.keywords if kw.arg)
        if not has_param:
            lines_to_fix.add(node.lineno - 1)  # 转为 0-indexed

    if not lines_to_fix:
        return code

    fix_list = sorted(lines_to_fix)  # 转为有序列表，便于处理插入后的偏移
    for idx in fix_list:
        line = lines[idx]
        stripped = line.rstrip()
        # 单行调用: Tex(...) 或 MathTex(...) 在同一行，以 ) 结尾
        if stripped.endswith(")"):
            before = stripped[:-1].rstrip()
            if before.endswith("("):
                lines[idx] = stripped[:-1] + f"{template_var}={template_var})"
            else:
                lines[idx] = stripped[:-1] + f", {template_var}={template_var})"
        # 多行调用: 找到对应的闭合括号行，在最后一个参数后添加
        else:
            # 从当前行向下扫描，找到深度匹配的 ) 所在行
            depth = 0
            target_line_idx = idx
            for j in range(idx, len(lines)):
                target_line = lines[j]
                depth += target_line.count("(") - target_line.count(")")
                if depth <= 0 and j > idx:
                    target_line_idx = j
                    break
            # 在闭合行前插入 tex_template 参数
            close_line = lines[target_line_idx]
            if close_line.strip() == ")":
                indent = close_line[: len(close_line) - len(close_line.lstrip())]
                lines.insert(target_line_idx, f"{indent}    {template_var}={template_var},")
                # 修正后续待修复行的行号偏移
                for k, orig_idx in enumerate(fix_list):
                    if orig_idx > target_line_idx:
                        fix_list[k] += 1
            else:
                # 闭合行与其他参数同行，直接插入
                before_close = close_line.rstrip()[:-1].rstrip()
                lines[target_line_idx] = (
                    before_close + f", {template_var}={template_var}" + close_line.rstrip()[-1:]
                )

    return "\n".join(lines)



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
    def __init__(self) -> None:
        self.planner = PlannerAgent()
        self.auto_fixer = AutoFixerAgent()
        self.slurm = SlurmDispatcher()
        self.merger = VideoMerger()
        self._callback: Callback | None = None
        self._ctx: PipelineContext | None = None
        self._manifest: RunManifest | None = None
        # 场景级并行调度时多个工作线程会并发写 manifest, 需要串行化
        self._manifest_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._phase_lock = threading.Lock()
        self._emitted_phases: set[str] = set()

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
        with self._manifest_lock:
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
        for attempt in range(settings.CODE_VALIDATION_ATTEMPTS):
            code = agent.generate_code(
                plan,
                feedback=current_feedback,
                previous_code=current_previous,
                stream=stream,
            )
            
            # 自动修复 TexTemplate 问题
            code = fix_tex_template(code)
            
            validation = self._validate(code)
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
                # 允许从 ERROR 状态恢复：重置失败场景，回到 CODING 阶段重试
                has_renderable = any(
                    scene.code and not scene.give_up
                    for scene in ctx.scene_states.values()
                )
                has_pending = any(
                    not scene.code and not scene.failed and not scene.give_up
                    for scene in ctx.scene_states.values()
                )
                if has_renderable or has_pending:
                    # 重置失败状态，允许重试
                    for scene in ctx.scene_states.values():
                        if scene.failed:
                            scene.failed = False
                            scene.failure_reason = ""
                    # 根据场景状态决定从哪个阶段恢复
                    if has_pending:
                        state = State.CODING
                    else:
                        state = State.REVIEWING
                    self._emit("run_resuming_from_error", run_id=run_id, state=state.name)
                else:
                    raise RuntimeError(
                        "该运行已进入 ERROR 状态且无可用场景，请创建新运行。"
                        f"\n失败原因: {manifest.error[:200]}"
                    )



            # 重置 give_up 场景，允许 resume 后重试审查/生成
            reset_give_up = False
            for scene in ctx.scene_states.values():
                if scene.give_up:
                    scene.give_up = False
                    scene.review_round = 0
                    scene.fix_attempts = 0
                    scene.failure_reason = ""
                    reset_give_up = True
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

            self._emit("run_resumed", run_id=run_id, state=state.name)
            return self._execute(ctx, state)

    def _execute(self, ctx: PipelineContext, state: State) -> Path | None:
        try:
            # ---- 准备: 目录 + 全局概要 (仅全新运行; resume 已从清单加载) ----
            if not ctx.scene_states:
                self._handle_init(ctx)
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
            missing = [
                name for name in ("sbatch", "ffmpeg") if not __import__("shutil").which(name)
            ]
            if settings.SLURM_REQUIRE_CONTAINER and not __import__("shutil").which("apptainer"):
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
                        outline, ctx.outlines, ctx.user_prompt, stream=False
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
        
        # 确定需要生成代码的场景
        # 如果是评估改进循环，只处理需要改进的场景
        if ctx.scenes_to_improve:
            # 评估改进模式：只处理低分场景
            pending = [
                (sid, ctx.scene_states[sid])
                for sid in ctx.scenes_to_improve
                if sid in ctx.scene_states and not ctx.scene_states[sid].failed
            ]
            self._emit("eval_improvement_mode", scenes=ctx.scenes_to_improve)
            # 清空改进列表，避免下次重复处理
            ctx.scenes_to_improve = []
        else:
            # 正常模式：处理所有未生成代码的场景
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
                    code, class_name = self._generate_validated_code(state.plan, stream=False)
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
        
        # 如果配置为跳过审查，直接进入下一阶段
        if settings.SKIP_REVIEW:
            self._emit("review_skipped", reason="SKIP_REVIEW enabled")
            for state in ctx.scene_states.values():
                if state.code and not state.failed:
                    state.reviewed = True
            return State.DONE if ctx.dry_run else State.DISPATCHING
        
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

            # 构建完整反馈：保留原始审查意见，用于后续传给 Coder
            original_feedback = effective_result.feedback or ""
            fix_details = "\n".join(
                f"- [{fix.reason}] {fix.find!r} → {fix.replace!r}"
                for fix in effective_result.fixes
            )

            if effective_result.severity == "minor":
                candidate = state.code
                applied_count = 0
                for fix in effective_result.fixes:
                    # 尝试精确匹配，失败则尝试首次出现
                    if candidate.count(fix.find) == 1:
                        candidate = candidate.replace(fix.find, fix.replace, 1)
                        applied_count += 1
                    elif fix.find in candidate:
                        # find 出现多次，只替换首次出现
                        candidate = candidate.replace(fix.find, fix.replace, 1)
                        applied_count += 1
                    # 如果 find 不存在则跳过该 fix
                validation = self._validate(candidate) if applied_count > 0 else None
                if validation and validation.is_valid:
                    state.code = candidate
                    state.class_name = validation.scene_classes[0]
                    self._write_private(ctx.paths.scenes / f"scene_{scene_id}.py", candidate)
                    code_changed = True
                    self._checkpoint(ctx, State.REVIEWING)
                    self._emit("scene_review_fail", scene_id=scene_id, severity="minor")
                    continue
                # minor 修复失败，升级为 major，但保留原始反馈
                effective_result = ReviewResult(
                    is_valid=False,
                    severity="major",
                    feedback=(
                        f"## Reviewer 审查意见（minor 修复未能全部应用）\n"
                        f"{original_feedback}\n\n"
                        f"## 修复建议详情\n{fix_details}\n\n"
                        f"## 确定性校验\n{validation.feedback if validation else '未生成有效代码'}"
                    ),
                )

            # severity=major：将完整审查反馈传给 Coder 重写
            rewrite_feedback = (
                f"## Reviewer 审查意见\n{original_feedback}\n\n"
                f"## 需修复的问题\n{fix_details}\n\n"
                f"请根据以上反馈逐项修正代码，保留正确部分，只修复指出的问题。"
            )
            try:
                code, class_name = self._generate_validated_code(
                    state.plan,
                    feedback=rewrite_feedback,
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
        # 如果启用了评估，进入评估状态；否则直接完成
        if settings.ENABLE_AUTO_EVAL:
            return State.EVALUATING
        return State.DONE


    def _handle_evaluating(self, ctx: PipelineContext) -> State:
        """评估渲染结果，如果质量不达标则触发改进"""
        self._emit("stage_start", stage="evaluating")
        
        # 检查评估轮数限制
        eval_round = getattr(ctx, 'eval_round', 0)
        if eval_round >= settings.MAX_EVAL_ROUNDS:
            self._emit(
                "eval_max_rounds_reached",
                rounds=eval_round,
                max_rounds=settings.MAX_EVAL_ROUNDS,
            )
            return State.DONE
        
        try:
            from kd1_anime.eval import Evaluator
            
            evaluator = Evaluator(
                enable_visual_eval=settings.ENABLE_VISUAL_EVAL,
            )
            
            # 评估最终视频
            if ctx.final_video and ctx.final_video.exists():
                # 评估代码质量（存储每个场景的评估结果以便后续复用）
                code_scores = []
                scene_eval_results: dict[int, EvalResult] = {}
                for scene_id, state in ctx.scene_states.items():
                    if state.code:
                        code_result = evaluator.evaluate_code(state.code)
                        code_scores.append(code_result)
                        scene_eval_results[scene_id] = code_result
                
                # 计算平均代码分数
                avg_code_score = sum(r.overall_score for r in code_scores) / len(code_scores) if code_scores else 0
                
                # 尝试评估视觉效果（如果有截图）
                visual_score = 5.0  # 默认满分
                if settings.ENABLE_VISUAL_EVAL:
                    screenshot_paths = list(ctx.paths.root.glob("**/*.png"))
                    if screenshot_paths:
                        try:
                            visual_result = evaluator.evaluate_visual(
                                screenshot_paths[0],
                                ctx.original_prompt or "Mathematical animation",
                            )
                            visual_score = visual_result.overall_score
                        except Exception as e:
                            self._emit("visual_eval_failed", error=str(e))
                
                # 计算综合分数
                overall_score = (avg_code_score + visual_score) / 2
                
                self._emit(
                    "eval_complete",
                    overall_score=overall_score,
                    code_score=avg_code_score,
                    visual_score=visual_score,
                    threshold=settings.EVAL_THRESHOLD,
                )
                
                # 保存评估结果
                from kd1_anime.eval.metrics import EvalResult
                eval_result = EvalResult(
                    run_id=ctx.paths.run_id,
                    summary=f"Auto-evaluation: {overall_score:.2f}/5.00",
                )
                eval_result.save(ctx.paths.root / "eval_result.json")
                
                # 检查是否需要改进
                if overall_score < settings.EVAL_THRESHOLD:
                    self._emit(
                        "eval_below_threshold",
                        score=overall_score,
                        threshold=settings.EVAL_THRESHOLD,
                        action="triggering_improvement",
                    )
                    
                    # 增加评估轮数
                    ctx.eval_round = eval_round + 1
                    
                    # 找出低分的场景需要改进（复用已有评估结果）
                    low_score_scenes = [
                        scene_id
                        for scene_id, scene_eval in scene_eval_results.items()
                        if scene_eval.overall_score < settings.EVAL_THRESHOLD
                    ]
                    
                    # 标记需要改进的场景
                    ctx.scenes_to_improve = low_score_scenes
                    
                    # 重置场景状态以便重新生成
                    for scene_id in low_score_scenes:
                        state = ctx.scene_states[scene_id]
                        state.rendered = False
                        state.reviewed = False
                    
                    return State.CODING  # 返回到 CODING 阶段重新生成
                else:
                    self._emit(
                        "eval_passed",
                        score=overall_score,
                        threshold=settings.EVAL_THRESHOLD,
                    )
            else:
                self._emit("eval_skipped", reason="no_final_video")
            
        except Exception as e:
            self._emit("eval_error", error=str(e))
            # 评估失败不阻塞流程
        
        return State.DONE

    # ------------------------------------------------------------------
    # 场景级并行调度 (per-scene pipeline)
    # 每个 Scene 独立推进 分镜→编码→审查→提交→渲染→修复, 互不等待。
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

    def _run_scheduler(self, ctx: PipelineContext) -> None:
        """启动每个场景的独立流水线线程, 全部结束后返回。"""
        import threading

        self._llm_sem = threading.Semaphore(max(1, settings.LLM_PARALLEL_WORKERS))
        self._slot_lock = threading.Lock()
        self._in_flight = 0
        self._manifest_lock = threading.Lock()
        self._stop_event.clear()
        with self._phase_lock:
            self._emitted_phases.clear()

        threads: list[threading.Thread] = []
        for scene_id, state in sorted(ctx.scene_states.items()):
            if state.rendered or state.failed or state.give_up:
                continue
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
        acquired = False
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
        except Exception as exc:
            self._mark_failed(state, f"Scene {scene_id} 流水线异常: {exc}")
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
        limit = settings.SLURM_MAX_IN_FLIGHT
        with self._slot_lock:
            if limit and self._in_flight >= limit:
                return False
            self._in_flight += 1
            return True

    def _release_slot(self) -> None:
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
        with self._llm_sem:
            outline = next(o for o in ctx.outlines if o.scene_id == scene_id)
            plan = PlannerAgent().plan_detail(
                outline, ctx.outlines, ctx.user_prompt, stream=False
            )
        state.plan = plan
        state.plan_ready = True
        self._checkpoint(ctx, State.DETAILING)
        self._emit("scene_detailed", scene_id=scene_id, title=plan.title)

    def _scene_code(self, ctx: PipelineContext, scene_id: int, state: SceneState) -> None:
        with self._llm_sem:
            code, class_name = self._generate_validated_code(
                state.plan,
                feedback=state.rewrite_feedback or "",
                previous_code=state.code if state.rewrite_feedback else "",
                stream=False,
            )
        state.code = code
        state.class_name = class_name
        state.rewrite_feedback = ""
        path = ctx.paths.scenes / f"scene_{scene_id}.py"
        self._write_private(path, code)
        self._checkpoint(ctx, State.CODING)
        self._emit("scene_coded", scene_id=scene_id, file_path=str(path))
        self._apply_incremental_for_scene(ctx, scene_id, state)

    def _scene_review(self, ctx: PipelineContext, scene_id: int, state: SceneState) -> None:
        if settings.SKIP_REVIEW:
            state.reviewed = True
            return
        with self._llm_sem:
            result = ReviewerAgent().review(state.code, state.plan)
        self._apply_review_result(ctx, scene_id, state, result)

    def _scene_submit(self, ctx: PipelineContext, scene_id: int, state: SceneState) -> None:
        self._phase_emit("dispatching")
        source_path = ctx.paths.scenes / f"scene_{scene_id}.py"
        try:
            on_disk_code = source_path.read_text(encoding="utf-8")
        except OSError as exc:
            self._mark_failed(state, f"提交前无法读取场景代码: {exc}")
            self._emit("scene_failed", scene_id=scene_id, reason=state.failure_reason)
            return
        if on_disk_code != state.code:
            self._mark_failed(state, "提交前代码一致性校验失败：磁盘文件已在流水线外被修改")
            self._emit("scene_failed", scene_id=scene_id, reason=state.failure_reason)
            return
        validation = self._validate(on_disk_code)
        if not validation.is_valid:
            self._mark_failed(state, "提交前校验失败:\n" + validation.feedback)
            self._emit("scene_failed", scene_id=scene_id, reason=state.failure_reason)
            return
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
            state.give_up = True
            state.failure_reason = "渲染作业状态未知，已放弃"
            return False
        if ok:
            state.rendered = True
            self._emit("scene_rendered", scene_id=job.scene_id)
            return True
        if not ctx.auto_fix:
            self._mark_failed(state, job.failure_reason or f"Slurm 状态: {job.status}")
        elif job.status not in FIXABLE_RENDER_STATES:
            state.give_up = True
            state.failure_reason = (
                job.failure_reason or f"基础设施失败，不修改代码: {job.status}"
            )
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
        return hashlib.sha256("\n".join(normalized).encode("utf-8", errors="replace")).hexdigest()[:16]

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
        error_log = self.slurm.get_error_log(job=job)
        if not error_log:
            state.give_up = True
            state.failure_reason = "渲染失败且没有错误日志"
            self._emit("scene_give_up", scene_id=scene_id, reason=state.failure_reason)
            return
        if self.auto_fixer.is_infrastructure_error(error_log):
            state.give_up = True
            state.failure_reason = self._give_up_reason(
                "检测到环境或 Slurm 配置错误，未让 LLM 重写业务代码", error_log
            )
            self._emit("scene_give_up", scene_id=scene_id, reason=state.failure_reason)
            return
        # 连续相同错误 → 判定为环境/配置问题, 提前放弃, 不再浪费修复次数
        fp = self._error_fingerprint(error_log)
        if fp and fp == state.last_error_fp:
            state.identical_error_count += 1
        else:
            state.identical_error_count = 1
            state.last_error_fp = fp
        # 连续相同错误 → 提前放弃, 避免修复器在同一个环境错误上空转。
        # 但必须叠加 fix_attempts>=2 门槛: 修复器至少要修过 2 次才允许据此放弃,
        # 否则像 camera.frame / LaTeX 这类"可修代码错误"在第一次修复失败后
        # 就会被误判为环境问题而直接放弃 (此前只给 1 次机会)。
        if (
            state.identical_error_count >= settings.MAX_FIX_IDENTICAL_ERRORS
            and state.fix_attempts >= 2
        ):
            state.give_up = True
            state.failure_reason = self._give_up_reason(
                f"连续 {state.identical_error_count} 次渲染错误完全相同且修复未能消除，"
                "疑似环境/配置问题，已放弃", error_log
            )
            self._emit("scene_give_up", scene_id=scene_id, reason=state.failure_reason)
            return
        if state.fix_attempts >= settings.MAX_FIX_ATTEMPTS:
            state.give_up = True
            state.failure_reason = self._give_up_reason("达到最大渲染修复次数", error_log)
            self._emit("scene_give_up", scene_id=scene_id, reason=state.failure_reason)
            return
        state.fix_attempts += 1
        self._emit(
            "scene_fixing",
            scene_id=scene_id,
            attempt=state.fix_attempts,
            max_attempts=settings.MAX_FIX_ATTEMPTS,
        )
        with self._llm_sem:
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
        # 注意: identical_error_count 不在这里重置 —— 只有当"错误指纹变化"时才重置
        # (见上面的 else 分支), 从而让"修复后错误完全相同"能在第 2 次相同错误时提前放弃。
        self._write_private(ctx.paths.scenes / f"scene_{scene_id}.py", candidate)
        self._checkpoint(ctx, State.FIXING)

    def _apply_review_result(self, ctx: PipelineContext, scene_id: int, state: SceneState, result: ReviewResult) -> bool:
        """应用单场景审查结果 (mirror _handle_reviewing 的单场景逻辑)。"""
        if result.is_valid:
            state.review_round = 0
            state.reviewed = True
            self._checkpoint(ctx, State.REVIEWING)
            self._emit("scene_review_pass", scene_id=scene_id)
            return True

        state.review_round += 1
        original_feedback = result.feedback or ""
        fix_details = "\n".join(
            f"- [{fix.reason}] {fix.find!r} → {fix.replace!r}" for fix in result.fixes
        )

        if state.review_round >= settings.MAX_REVIEW_ROUNDS:
            state.give_up = True
            state.failure_reason = "达到最大审查轮次，代码仍未通过"
            self._checkpoint(ctx, State.REVIEWING)
            self._emit("scene_give_up", scene_id=scene_id, reason=state.failure_reason)
            return True

        if result.severity == "minor":
            candidate = state.code
            applied_count = 0
            for fix in result.fixes:
                if candidate.count(fix.find) == 1 or fix.find in candidate:
                    candidate = candidate.replace(fix.find, fix.replace, 1)
                    applied_count += 1
            validation = self._validate(candidate) if applied_count > 0 else None
            if validation and validation.is_valid:
                state.code = candidate
                state.class_name = validation.scene_classes[0]
                self._write_private(ctx.paths.scenes / f"scene_{scene_id}.py", candidate)
                self._checkpoint(ctx, State.REVIEWING)
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
        state.rewrite_feedback = (
            f"## Reviewer 审查意见\n{original_feedback}\n\n"
            f"## 需修复的问题\n{fix_details}\n\n"
            f"请根据以上反馈逐项修正代码，保留正确部分，只修复指出的问题。"
        )
        self._checkpoint(ctx, State.REVIEWING)
        self._emit("scene_review_fail", scene_id=scene_id, severity="major")
        return True

    def _apply_incremental_for_scene(self, ctx: PipelineContext, scene_id: int, state: SceneState) -> None:
        """增量渲染: 单场景代码就绪后判断是否可复用旧视频。"""
        if not ctx.incremental or not ctx.base_manifest or not ctx.base_run_id:
            return
        base_scene = ctx.base_manifest.scenes.get(scene_id)
        if base_scene is None:
            ctx.scenes_to_render.append(scene_id)
            return
        if base_scene.rendered and sha256_text(state.code) == base_scene.code_sha256:
            base_root = RunRepository(settings.WORKSPACE_DIR).run_root(ctx.base_run_id)
            old_video = get_reusable_video_path(ctx.base_manifest, scene_id, base_root)
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
                ctx.scenes_to_reuse.append(scene_id)
                self._emit("scene_reused", scene_id=scene_id)
                return
        ctx.scenes_to_render.append(scene_id)

    def _merge(self, ctx: PipelineContext) -> None:
        """合并所有已渲染场景; 失败条件直接抛错 (mirror _handle_merging)。"""
        self._emit("stage_start", stage="merging")
        rendered_jobs = [
            state.slurm_job
            for state in ctx.scene_states.values()
            if state.rendered and state.slurm_job
        ]
        incomplete = [sid for sid, state in ctx.scene_states.items() if not state.rendered]
        if not rendered_jobs:
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

        resolved_output = ctx.paths.output.expanduser().resolve()
        output_is_run_local = resolved_output == ctx.paths.root.resolve() or (
            ctx.paths.root.resolve() in resolved_output.parents
        )
        # 评估-改进循环的第二次合并必须强制重新拼接, 不能复用上一轮的旧视频
        force_remerge = getattr(ctx, "eval_round", 0) > 0
        if (
            output_is_run_local
            and resolved_output.is_file()
            and resolved_output.stat().st_size > 0
            and not settings.OVERWRITE_OUTPUT
            and not force_remerge
        ):
            ctx.final_video = resolved_output
        else:
            if ctx.incremental and ctx.base_manifest and ctx.base_run_id:
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

    def _eval(self, ctx: PipelineContext) -> bool:
        """评估最终视频; 返回 True 表示触发改进 (需要重新调度低分场景)。"""
        if not settings.ENABLE_AUTO_EVAL:
            return False
        self._emit("stage_start", stage="evaluating")
        eval_round = getattr(ctx, "eval_round", 0)
        if eval_round >= settings.MAX_EVAL_ROUNDS:
            self._emit(
                "eval_max_rounds_reached",
                rounds=eval_round,
                max_rounds=settings.MAX_EVAL_ROUNDS,
            )
            return False
        try:
            from kd1_anime.eval import Evaluator

            evaluator = Evaluator(enable_visual_eval=settings.ENABLE_VISUAL_EVAL)
            if not (ctx.final_video and ctx.final_video.exists()):
                self._emit("eval_skipped", reason="no_final_video")
                return False

            code_scores = []
            scene_eval_results: dict[int, object] = {}
            for scene_id, state in ctx.scene_states.items():
                if state.code:
                    code_result = evaluator.evaluate_code(state.code)
                    code_scores.append(code_result)
                    scene_eval_results[scene_id] = code_result

            avg_code_score = (
                sum(r.overall_score for r in code_scores) / len(code_scores)
                if code_scores
                else 0
            )
            visual_score = 5.0
            if settings.ENABLE_VISUAL_EVAL:
                screenshot_paths = list(ctx.paths.root.glob("**/*.png"))
                if screenshot_paths:
                    try:
                        visual_result = evaluator.evaluate_visual(
                            screenshot_paths[0],
                            ctx.original_prompt or "Mathematical animation",
                        )
                        visual_score = visual_result.overall_score
                    except Exception as e:
                        self._emit("visual_eval_failed", error=str(e))

            overall_score = (avg_code_score + visual_score) / 2
            self._emit(
                "eval_complete",
                overall_score=overall_score,
                code_score=avg_code_score,
                visual_score=visual_score,
                threshold=settings.EVAL_THRESHOLD,
            )

            from kd1_anime.eval.metrics import EvalResult

            eval_result = EvalResult(
                run_id=ctx.paths.run_id,
                summary=f"Auto-evaluation: {overall_score:.2f}/5.00",
            )
            eval_result.save(ctx.paths.root / "eval_result.json")

            if overall_score >= settings.EVAL_THRESHOLD:
                self._emit(
                    "eval_passed",
                    score=overall_score,
                    threshold=settings.EVAL_THRESHOLD,
                )
                return False

            self._emit(
                "eval_below_threshold",
                score=overall_score,
                threshold=settings.EVAL_THRESHOLD,
                action="triggering_improvement",
            )
            ctx.eval_round = eval_round + 1
            low_score_scenes = [
                scene_id
                for scene_id, scene_eval in scene_eval_results.items()
                if scene_eval.overall_score < settings.EVAL_THRESHOLD  # type: ignore[attr-defined]
            ]
            ctx.scenes_to_improve = low_score_scenes
            for scene_id in low_score_scenes:
                state = ctx.scene_states[scene_id]
                state.rendered = False
                state.reviewed = False
                state.code = ""
                state.class_name = ""
                state.slurm_job = None
                state.fix_attempts = 0
                state.rewrite_feedback = ""
            return True
        except Exception as e:
            self._emit("eval_error", error=str(e))
            return False
