"""
kd1-anime CLI 入口
使用 Typer 构建命令行界面

交互模式 (默认):
  kd1-anime                   → 启动 TUI 交互会话
  kd1-anime chat              → 同上

直接模式:
  kd1-anime generate "..."    → 直接生成 (无澄清)
  kd1-anime plan "..."        → 只生成场景规划
"""

import re
import shutil
from typing import Optional
from datetime import datetime, timedelta, timezone
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from kd1_anime.config import settings
from kd1_anime.run_store import RunManifest, RunRepository, lock_run

app = typer.Typer(
    name="kd1-anime",
    help="AI Agent 驱动的 Manim 数学动画自动渲染流水线",
    add_completion=False,
    invoke_without_command=True,
)
console = Console()


@app.callback()
def main_callback(
    ctx: typer.Context,
    api_key: str = typer.Option(
        None, "--api-key", "-k", help="LLM API Key (也可通过 .env 文件设置)"
    ),
    model: str = typer.Option(None, "--model", "-m", help="LLM 模型名称"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="只生成场景代码, 不提交 Slurm 渲染 (所有命令通用)"
    ),
):
    """kd1-anime — AI 驱动的数学动画生成器"""
    if api_key:
        settings.LLM_API_KEY = api_key
    if model:
        settings.LLM_MODEL = model

    # 没有子命令时默认启动 chat
    if ctx.invoked_subcommand is None:
        _start_chat(dry_run=dry_run)


@app.command()
def chat(
    dry_run: bool = typer.Option(False, "--dry-run", help="只生成代码不提交 Slurm"),
):
    """启动交互式会话 (默认命令)"""
    _start_chat(dry_run=dry_run)


@app.command()
def generate(
    prompt: str = typer.Argument(None, help="动画需求的自然语言描述"),
    output: Path = typer.Option(
        None, "--output", "-o", help="输出视频文件路径 (默认: output_final.mp4)"
    ),
    force: bool = typer.Option(False, "--force", help="允许覆盖已存在的输出文件"),
    partition: str = typer.Option(None, "--partition", "-p", help="Slurm 分区"),
    max_fix: int = typer.Option(None, "--max-fix", min=0, help="最大自动修复尝试次数 (默认: 3)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="只生成场景规划和代码,不提交 Slurm 任务"),
    incremental: str = typer.Option(
        None,
        "--incremental",
        "-i",
        help="增量渲染模式：基于指定的 run-id 只渲染变化的场景",
    ),
    file: Path = typer.Option(
        None,
        "--file",
        "-f",
        help="从文件读取 prompt (用于中文等易出问题的输入)",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
):
    """直接生成模式 (无需求澄清)"""
    if file:
        prompt = file.read_text(encoding="utf-8").strip()
    if not prompt:
        console.print("[bold red]错误:[/] 请提供 prompt 或通过 --file 指定文件\n使用 kd1-anime plan --help 查看帮助")
        raise typer.Exit(1)
    try:
        if partition:
            settings.SLURM_PARTITION = partition
        if max_fix is not None:
            settings.MAX_FIX_ATTEMPTS = max_fix
        if output:
            settings.OUTPUT_FILE = output
        settings.OVERWRITE_OUTPUT = force
    except ValueError as e:
        console.print(f"[bold red]配置错误:[/] {e}", markup=False)
        raise typer.Exit(1) from e

    try:
        settings.require_llm_key()
    except ValueError as e:
        console.print(f"[bold red]错误:[/] {e}", markup=False)
        raise typer.Exit(1) from e

    from kd1_anime.orchestrator import Orchestrator

    orchestrator = Orchestrator()
    try:
        if incremental:
            console.print(f"[cyan]增量渲染模式[/] 基于运行: {incremental}")
            final_video = orchestrator.run_incremental(
                prompt, incremental, dry_run=dry_run
            )
        else:
            final_video = orchestrator.run(prompt, dry_run=dry_run)
        
        if dry_run:
            console.print("\n[bold green]Dry-run 完成[/]")
        else:
            console.print(f"\n[bold green]成功![/] 输出文件: {final_video}")
    except KeyboardInterrupt as e:
        console.print("\n[yellow]用户中断 (已清理 Slurm 任务)[/]")
        raise typer.Exit(130) from e
    except Exception as e:
        console.print(f"\n[bold red]失败:[/] {e}", markup=False)
        raise typer.Exit(1) from e


@app.command()
def plan(
    prompt: str = typer.Argument(None, help="动画需求的自然语言描述"),
    file: Path = typer.Option(
        None,
        "--file",
        "-f",
        help="从文件读取 prompt",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
):
    """只生成场景规划,不执行渲染"""
    if file:
        prompt = file.read_text(encoding="utf-8").strip()
    if not prompt:
        console.print("[bold red]错误:[/] 请提供 prompt 或通过 --file 指定文件\n使用 kd1-anime plan --help 查看帮助")
        raise typer.Exit(1)
    try:
        settings.require_llm_key()
    except ValueError as e:
        console.print(f"[bold red]错误:[/] {e}", markup=False)
        raise typer.Exit(1) from e

    from kd1_anime.agents.planner import PlannerAgent

    try:
        planner = PlannerAgent()
        outlines = planner.plan_outline(prompt)
        scenes = [planner.plan_detail(o, outlines, prompt, stream=False) for o in outlines]
    except KeyboardInterrupt as e:
        console.print("\n[yellow]用户中断[/]")
        raise typer.Exit(130) from e
    except Exception as e:
        console.print(f"[bold red]规划失败:[/] {e}", markup=False)
        raise typer.Exit(1) from e

    console.print("\n[bold]场景规划结果:[/]")
    for scene in scenes:
        console.print(f"\n[cyan]Scene {scene.scene_id}:[/] {scene.title}")
        console.print(f"  数学概念: {scene.math_concept}")
        console.print(f"  时长: {scene.duration_seconds}s")
        console.print(f"  目的: {scene.purpose}")


@app.command()
def render(
    file: Path = typer.Argument(
        ...,
        help="Manim Python 文件路径",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    class_name: str = typer.Option(None, "--class", "-c", help="Manim Scene 类名 (默认自动识别)"),
    scene_id: int = typer.Option(1, "--scene-id", "-s", min=1, help="场景 ID, 用于命名"),
    wait: bool = typer.Option(False, "--wait", "-w", help="等待任务完成并显示进度"),
):
    """直接提交单个 .py 文件到 Slurm 渲染 (跳过 pipeline)"""
    from kd1_anime.agents.validator import validate_manim_code
    from kd1_anime.orchestrator import Orchestrator

    sid = scene_id
    source_code = file.read_text(encoding="utf-8")
    validation = validate_manim_code(source_code)
    if not validation.is_valid:
        console.print(
            "[bold red]代码校验失败:[/]\n" + validation.feedback,
            markup=False,
        )
        raise typer.Exit(1)
    if class_name and class_name not in validation.scene_classes:
        console.print(
            f"[bold red]错误:[/] 类 {class_name!r} 不在可渲染类列表 {validation.scene_classes}",
            markup=False,
        )
        raise typer.Exit(1)
    selected_class = class_name or validation.scene_classes[0]
    try:
        job, final_video, run_id = Orchestrator().submit_existing_scene(
            source_code,
            selected_class,
            scene_id=sid,
            wait=wait,
        )
    except Exception as e:
        console.print(f"[bold red]渲染任务失败:[/] {e}", markup=False)
        raise typer.Exit(1) from e

    console.print(f"[bold green]已提交[/] Job {job.job_id}")
    console.print(f"  Run ID: {run_id}")
    console.print(f"  脚本: {job.script_path}")
    console.print(f"  输出: {job.log_out}")
    console.print(f"  错误: {job.log_err}")
    if wait:
        console.print(f"[bold green]渲染完成![/] 输出文件: {final_video}")
    else:
        console.print(f"  后续运行: kd1-anime resume {run_id}")


def _scene_status(scene) -> str:
    if scene.rendered:
        return "rendered"
    if scene.failed:
        return "failed"
    if scene.give_up:
        return "give_up"
    if scene.slurm_job:
        return scene.slurm_job.status.lower()
    if scene.code_file:
        return "code_ready"
    return "planned"


def _print_run_details(manifest: RunManifest) -> None:
    console.print(f"Run ID:       {manifest.run_id}", markup=False)
    console.print(f"Status:       {manifest.status}", markup=False)
    console.print(f"FSM state:    {manifest.state}", markup=False)
    console.print(f"Created:      {manifest.created_at.astimezone().isoformat(timespec='seconds')}")
    console.print(f"Updated:      {manifest.updated_at.astimezone().isoformat(timespec='seconds')}")
    console.print(f"Output:       {manifest.final_video or manifest.output_path}", markup=False)

    table = Table(title="Scenes")
    table.add_column("ID", justify="right")
    table.add_column("Title")
    table.add_column("Status")
    table.add_column("Job ID")
    table.add_column("Review/Fix", justify="right")
    for scene_id, scene in sorted(manifest.scenes.items()):
        table.add_row(
            str(scene_id),
            Text(scene.plan.title),
            _scene_status(scene),
            scene.slurm_job.job_id if scene.slurm_job else "-",
            f"{scene.review_round}/{scene.fix_attempts}",
        )
    console.print(table)
    if manifest.error:
        console.print("Last error:", style="bold red")
        console.print(manifest.error, markup=False)


@app.command()
def status(
    run_id: str = typer.Argument(None, help="运行 ID；省略时列出最近运行"),
    limit: int = typer.Option(20, "--limit", min=1, max=200, help="列表最多显示条数"),
):
    """查看持久化运行状态，不调用 LLM 或 Slurm。"""

    repository = RunRepository(settings.WORKSPACE_DIR)
    if run_id:
        try:
            _print_run_details(repository.load(run_id))
        except (OSError, ValueError) as exc:
            console.print(f"[bold red]读取失败:[/] {exc}", markup=False)
            raise typer.Exit(1) from exc
        return

    manifests = repository.list()[:limit]
    if not manifests:
        console.print("没有可用的运行记录")
        return
    table = Table(title="Recent runs")
    table.add_column("Run ID")
    table.add_column("Updated")
    table.add_column("Status")
    table.add_column("State")
    table.add_column("Scenes", justify="right")
    for manifest in manifests:
        rendered = sum(scene.rendered for scene in manifest.scenes.values())
        table.add_row(
            manifest.run_id,
            manifest.updated_at.astimezone().strftime("%Y-%m-%d %H:%M"),
            manifest.status,
            manifest.state,
            f"{rendered}/{len(manifest.scenes)}",
        )
    console.print(table)


@app.command()
def resume(
    run_id: str = typer.Argument(..., help="要恢复的运行 ID"),
    force: bool = typer.Option(False, "--force", help="合并阶段允许覆盖原输出"),
    interactive: bool = typer.Option(False, "--interactive", help="失败时允许终端询问重试"),
):
    """恢复中断运行；复用已有代码和 Slurm Job ID。"""

    repository = RunRepository(settings.WORKSPACE_DIR)
    try:
        manifest = repository.load(run_id)
        if manifest.state in {"PLANNING", "DETAILING", "CODING", "REVIEWING", "FIXING"}:
            settings.require_llm_key()
        settings.OVERWRITE_OUTPUT = force
        from kd1_anime.orchestrator import Orchestrator

        final_video = Orchestrator().resume(run_id, interactive=interactive)
    except KeyboardInterrupt as exc:
        console.print("\n[yellow]用户中断 (已记录恢复点并清理 Slurm 任务)[/]")
        raise typer.Exit(130) from exc
    except (OSError, ValueError, RuntimeError) as exc:
        console.print(f"[bold red]恢复失败:[/] {exc}\n使用 kd1-anime status 查看可用运行", markup=False)
        raise typer.Exit(1) from exc
    if manifest.dry_run:
        console.print(f"[bold green]Dry-run 已完成[/] 运行目录: {repository.run_root(run_id)}")
    else:
        console.print(f"[bold green]运行已完成[/] 输出文件: {final_video}")


def _parse_retention(value: str) -> timedelta:
    match = re.fullmatch(r"(\d+)([dhm])", value.strip().lower())
    if not match:
        raise ValueError("保留期格式应为整数加 d/h/m，例如 30d、12h 或 90m")
    amount = int(match.group(1))
    unit = match.group(2)
    return {
        "d": timedelta(days=amount),
        "h": timedelta(hours=amount),
        "m": timedelta(minutes=amount),
    }[unit]


@app.command()
def clean(
    older_than: str = typer.Option("30d", "--older-than", help="清理多久以前的运行，如 30d"),
    yes: bool = typer.Option(False, "--yes", "-y", help="不再询问确认"),
    include_running: bool = typer.Option(
        False,
        "--include-running",
        help="也尝试清理状态为 running 的陈旧记录；仍会跳过持锁运行",
    ),
):
    """清理过期的私有运行目录，不删除目录外的自定义输出。"""

    try:
        retention = _parse_retention(older_than)
    except ValueError as exc:
        console.print(f"[bold red]参数错误:[/] {exc}\n示例: kd1-anime clean --older-than 30d", markup=False)
        raise typer.Exit(2) from exc
    repository = RunRepository(settings.WORKSPACE_DIR)
    cutoff = datetime.now(timezone.utc) - retention
    candidates = [
        manifest
        for manifest in repository.list()
        if manifest.updated_at <= cutoff and (include_running or manifest.status != "running")
    ]
    if not candidates:
        console.print("没有符合条件的运行目录")
        return
    console.print(f"将清理 {len(candidates)} 个运行目录（不会删除目录外的输出文件）")
    if not yes and not typer.confirm("继续？", default=False):
        raise typer.Abort()

    removed = 0
    skipped = 0
    for manifest in candidates:
        root = repository.run_root(manifest.run_id)
        try:
            with lock_run(root):
                shutil.rmtree(root)
            removed += 1
        except (OSError, RuntimeError) as exc:
            skipped += 1
            console.print(f"跳过 {manifest.run_id}: {exc}", markup=False, style="yellow")
    console.print(f"清理完成: 删除 {removed}, 跳过 {skipped}")


@app.command()
def batch(
    prompts_file: Path = typer.Argument(
        ...,
        help="包含 prompts 的文件路径（每行一个 prompt 或 JSON 格式）",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    max_parallel: int = typer.Option(
        3,
        "--max-parallel",
        "-j",
        help="最大并行任务数",
        min=1,
        max=10,
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="只生成场景代码，不提交 Slurm 渲染",
    ),
    output_dir: Path = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="输出目录",
    ),
):
    """批量并行处理多个动画项目。"""
    try:
        settings.require_llm_key()
    except ValueError as e:
        console.print(f"[bold red]错误:[/] {e}", markup=False)
        raise typer.Exit(1) from e
    
    from kd1_anime.batch import BatchProcessor, BatchConfig
    
    # 加载 prompts
    try:
        config = BatchConfig(
            max_parallel=max_parallel,
            dry_run=dry_run,
            output_dir=output_dir,
        )
        processor = BatchProcessor(config)
        processor.load_tasks_from_file(prompts_file)
        
        console.print(f"[cyan]加载了 {len(processor.tasks)} 个任务[/]")
        
        # 执行批量处理
        tasks = processor.execute_all()
        
        # 输出摘要
        summary = processor.generate_summary(tasks)
        console.print(summary)
        
        # 检查是否有失败的任务
        failed_count = sum(1 for t in tasks if t.status == "failed")
        if failed_count > 0:
            console.print(f"[bold red]{failed_count} 个任务失败[/]")
            raise typer.Exit(1)
    
    except Exception as e:
        console.print(f"[bold red]批量处理失败:[/] {e}", markup=False)
        raise typer.Exit(1) from e


@app.command(name="version")
def version_cmd():
    """显示版本信息"""
    try:
        from importlib.metadata import version

        current_version = version("kd1-anime")
    except Exception:
        current_version = "0.3.0-dev"
    console.print(f"kd1-anime v{current_version}")
    console.print("AI Agent 驱动的 Manim 数学动画自动渲染流水线")



@app.command()
def doctor():
    """检查环境依赖和配置是否完整。"""
    import shutil
    import subprocess
    
    console.print("[bold]kd1-anime 环境检查[/]\n")
    
    checks = []
    
    # 检查 Python 版本
    import sys
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    py_ok = sys.version_info >= (3, 10)
    checks.append(("Python >= 3.10", py_ok, py_version))
    
    # 检查 conda
    conda_path = shutil.which("conda")
    conda_ok = conda_path is not None
    checks.append(("conda", conda_ok, conda_path or "未找到"))
    
    # 检查 manim
    manim_ok = False
    manim_version = "未安装"
    try:
        result = subprocess.run(
            ["python3", "-c", "import manim; print(manim.__version__)"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            manim_ok = True
            manim_version = result.stdout.strip()
    except Exception:
        pass
    checks.append(("manim", manim_ok, manim_version))
    
    # 检查 ffmpeg
    ffmpeg_path = shutil.which("ffmpeg")
    ffmpeg_ok = ffmpeg_path is not None
    ffmpeg_version = "未找到"
    if ffmpeg_ok:
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                ffmpeg_version = result.stdout.split("\n")[0][:50]
        except Exception:
            ffmpeg_version = "已找到"
    checks.append(("ffmpeg", ffmpeg_ok, ffmpeg_version))
    
    # 检查 sbatch (Slurm)
    sbatch_path = shutil.which("sbatch")
    sbatch_ok = sbatch_path is not None
    checks.append(("sbatch (Slurm)", sbatch_ok, sbatch_path or "未找到"))
    
    # 检查 xelatex
    xelatex_path = shutil.which("xelatex")
    xelatex_ok = xelatex_path is not None
    checks.append(("xelatex", xelatex_ok, xelatex_path or "未找到"))
    
    # 检查 apptainer (可选)
    apptainer_path = shutil.which("apptainer")
    apptainer_ok = apptainer_path is not None
    checks.append(("apptainer (可选)", apptainer_ok, apptainer_path or "未找到"))
    
    # 检查 LLM 配置
    llm_ok = bool(settings.LLM_API_KEY and settings.LLM_MODEL)
    llm_info = "已配置" if llm_ok else "未配置 (需要 LLM_API_KEY 和 LLM_MODEL)"
    checks.append(("LLM 配置", llm_ok, llm_info))
    
    # 显示结果
    table = Table(title="环境检查结果")
    table.add_column("检查项", style="cyan")
    table.add_column("状态", justify="center")
    table.add_column("详情", style="dim")
    
    all_ok = True
    for name, ok, detail in checks:
        status = "[green]✓[/]" if ok else "[red]✗[/]"
        if not ok and name not in ["apptainer (可选)"]:
            all_ok = False
        table.add_row(name, status, detail)
    
    console.print(table)
    console.print()
    
    if all_ok:
        console.print("[bold green]所有必要依赖已就绪！[/]")
    else:
        console.print("[bold yellow]部分依赖缺失，请参考文档安装：[/]")
        console.print("  https://github.com/Enthusjast/kd1-anime#readme")


def _start_chat(dry_run: bool = False) -> None:
    """启动 TUI 交互会话"""
    from kd1_anime.tui import ChatSession

    session = ChatSession(dry_run=dry_run)
    session.run()


if __name__ == "__main__":
    app()


@app.command()
def evaluate(
    run_id: Optional[str] = typer.Argument(None, help="运行 ID (留空则评估最近的运行)"),
    code: Optional[str] = typer.Option(None, "--code", "-c", help="直接评估代码字符串"),
    code_file: Optional[Path] = typer.Option(None, "--code-file", "-f", help="评估代码文件"),
    image: Optional[Path] = typer.Option(None, "--image", "-i", help="评估渲染截图"),
    description: str = typer.Option("", "--desc", "-d", help="动画描述"),
    visual: bool = typer.Option(True, "--visual/--no-visual", help="是否进行视觉评估"),
    visual_model: Optional[str] = typer.Option(None, "--visual-model", help="视觉评估模型"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="输出报告路径"),
    compare: Optional[str] = typer.Option(None, "--compare", help="对比的基准运行 ID"),
    json_output: bool = typer.Option(False, "--json", help="JSON 格式输出"),
):
    """评估动画生成质量
    
    支持多种评估模式：
    - 评估完整运行: kd1-anime evaluate <run-id>
    - 评估代码: kd1-anime evaluate --code "..." 或 --code-file scene.py
    - 评估截图: kd1-anime evaluate --image screenshot.png
    - 对比运行: kd1-anime evaluate <run-id> --compare <baseline-id>
    """
    from kd1_anime.eval import Evaluator
    
    evaluator = Evaluator(
        enable_visual_eval=visual and image is not None,
        visual_eval_model=visual_model,
    )
    
    try:
        # 对比模式
        if compare and run_id:
            console.print(f"[bold]对比运行 {compare} 和 {run_id}[/]")
            comparison = evaluator.compare_runs(compare, run_id)
            
            if json_output:
                console.print_json(json.dumps(comparison.to_dict(), indent=2))
            else:
                _print_comparison(comparison)
            return
        
        # 评估代码字符串
        if code:
            console.print("[bold]评估代码质量[/]")
            result = evaluator.evaluate_code(code)
        
        # 评估代码文件
        elif code_file:
            if not code_file.exists():
                console.print(f"[red]文件不存在: {code_file}[/]")
                raise typer.Exit(1)
            
            console.print(f"[bold]评估代码文件: {code_file}[/]")
            code_content = code_file.read_text(encoding='utf-8')
            result = evaluator.evaluate_code(code_content)
        
        # 评估截图
        elif image:
            if not image.exists():
                console.print(f"[red]图片不存在: {image}[/]")
                raise typer.Exit(1)
            
            console.print(f"[bold]评估视觉效果: {image}[/]")
            result = evaluator.evaluate_visual(image, description)
        
        # 评估运行
        elif run_id:
            console.print(f"[bold]评估运行: {run_id}[/]")
            result = evaluator.evaluate_run(run_id, description=description, enable_visual=visual)
        
        # 评估最近的运行
        else:
            # 查找最近的运行
            runs_dir = Path("workspace/runs")
            if not runs_dir.exists():
                console.print("[red]没有找到运行记录[/]")
                raise typer.Exit(1)
            
            run_dirs = sorted(runs_dir.iterdir(), key=lambda d: d.stat().st_mtime, reverse=True)
            if not run_dirs:
                console.print("[red]没有找到运行记录[/]")
                raise typer.Exit(1)
            
            recent_run = run_dirs[0].name
            console.print(f"[bold]评估最近的运行: {recent_run}[/]")
            result = evaluator.evaluate_run(recent_run, description=description, enable_visual=visual)
        
        # 输出结果
        if json_output:
            console.print_json(json.dumps(result.to_dict(), indent=2))
        else:
            _print_eval_result(result)
        
        # 保存报告
        if output:
            result.save(output)
            console.print(f"\n[dim]报告已保存到: {output}[/]")
        
    except Exception as e:
        console.print(f"[red]评估失败: {e}[/]")
        raise typer.Exit(1)


def _print_eval_result(result: 'EvalResult'):
    """打印评估结果"""
    from rich.table import Table
    from rich.panel import Panel
    
    # 总分面板
    score_color = "green" if result.overall_score >= 4 else "yellow" if result.overall_score >= 3 else "red"
    panel = Panel(
        f"[bold {score_color}]{result.overall_score:.2f}[/] / 5.00",
        title="总分",
        border_style=score_color,
    )
    console.print(panel)
    
    # 详细分数表格
    table = Table(title="详细评分")
    table.add_column("指标", style="cyan")
    table.add_column("分数", justify="center")
    table.add_column("等级", justify="center")
    table.add_column("说明")
    
    for score in result.scores:
        score_str = f"{score.score}/5"
        if score.score >= 4:
            score_str = f"[green]{score_str}[/]"
        elif score.score >= 3:
            score_str = f"[yellow]{score_str}[/]"
        else:
            score_str = f"[red]{score_str}[/]"
        
        table.add_row(
            score.metric.value,
            score_str,
            score.level.value,
            score.justification[:50] + "..." if len(score.justification) > 50 else score.justification,
        )
    
    console.print(table)
    
    # 摘要
    if result.summary:
        console.print(f"\n[dim]{result.summary}[/]")


def _print_comparison(comparison: 'ComparisonResult'):
    """打印对比结果"""
    from rich.table import Table
    
    diff = comparison.score_diff
    diff_color = "green" if diff > 0 else "red" if diff < 0 else "white"
    diff_str = f"+{diff:.2f}" if diff > 0 else f"{diff:.2f}"
    
    console.print(f"\n[bold]基准:[/] {comparison.baseline_run_id} ({comparison.baseline_result.overall_score:.2f})")
    console.print(f"[bold]当前:[/] {comparison.current_run_id} ({comparison.current_result.overall_score:.2f})")
    console.print(f"[bold {diff_color}]差异:[/] {diff_str}")
    
    if comparison.improvements:
        console.print("\n[green]改进:[/]")
        for item in comparison.improvements:
            console.print(f"  ✓ {item}")
    
    if comparison.regressions:
        console.print("\n[red]退化:[/]")
        for item in comparison.regressions:
            console.print(f"  ✗ {item}")


@app.command()
def test_llm(
    json_mode: bool = typer.Option(True, "--json-mode/--no-json-mode", help="测试 JSON 模式"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="详细输出"),
):
    """测试 LLM 端点连接和功能
    
    检测当前配置的 LLM 端点是否正常工作，包括 JSON 模式支持。
    """
    from kd1_anime.agents.base import BaseAgent
    
    console.print(Rule("LLM 端点测试", style="bold blue"))
    console.print()
    
    # 显示配置
    console.print("[bold]当前配置:[/]")
    console.print(f"  模型: {settings.LLM_MODEL}")
    console.print(f"  Base URL: {settings.LLM_BASE_URL}")
    console.print(f"  JSON 模式: {'启用' if json_mode else '禁用'}")
    console.print()
    
    # 创建测试 Agent
    class TestAgent(BaseAgent):
        name = "TestAgent"
    
    agent = TestAgent()
    
    # 测试 1: 基本连接
    console.print("[bold]测试 1: 基本连接[/]")
    try:
        response = agent.call_llm(
            system_prompt="你是一个助手。",
            user_message="回复 'OK' 两个字母。",
            temperature=0.0,
            max_tokens=10,
        )
        if response.strip():
            console.print(f"  [green]✓ 连接成功[/] 响应: {response.strip()[:50]}")
        else:
            console.print("  [red]✗ 响应为空[/]")
            return
    except Exception as e:
        console.print(f"  [red]✗ 连接失败: {e}[/]")
        return
    
    console.print()
    
    # 测试 2: JSON 模式
    if json_mode:
        console.print("[bold]测试 2: JSON 模式[/]")
        try:
            # 使用 call_llm_json 测试
            from pydantic import BaseModel
            
            class TestResponse(BaseModel):
                status: str
                message: str
            
            result = agent.call_llm_json(
                system_prompt="返回一个 JSON 对象，包含 status 和 message 字段。",
                user_message="status 设为 'ok'，message 设为 '测试成功'。",
                response_model=TestResponse,
                temperature=0.0,
            )
            console.print(f"  [green]✓ JSON 模式正常[/]")
            console.print(f"    解析结果: status={result.status}, message={result.message}")
        except Exception as e:
            console.print(f"  [yellow]⚠ JSON 模式异常: {e}[/]")
            console.print("    建议: 在 .env 中设置 LLM_USE_JSON_MODE=false")
    
    console.print()
    console.print("[bold green]测试完成[/]")
