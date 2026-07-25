"""
kd1-anime CLI 入口
使用 Typer 构建命令行界面

交互模式 (默认):
  python main.py              → 启动 TUI 交互会话
  python main.py chat         → 同上

直接模式:
  python main.py generate "..." → 直接生成 (无澄清)
  python main.py plan "..."     → 只生成场景规划
"""

import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中 (解决间歇性 ModuleNotFoundError)
_proj = Path(__file__).resolve().parent
if str(_proj) not in sys.path:
    sys.path.insert(0, str(_proj))

import typer
from rich.console import Console

from config import settings

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
    model: str = typer.Option(
        None, "--model", "-m", help="LLM 模型名称"
    ),
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
    dry_run: bool = typer.Option(
        False, "--dry-run", help="只生成代码不提交 Slurm"
    ),
):
    """启动交互式会话 (默认命令)"""
    _start_chat(dry_run=dry_run)


@app.command()
def generate(
    prompt: str = typer.Argument(None, help="动画需求的自然语言描述"),
    output: Path = typer.Option(
        None, "--output", "-o", help="输出视频文件路径 (默认: output_final.mp4)"
    ),
    partition: str = typer.Option(
        None, "--partition", "-p", help="Slurm 分区"
    ),
    max_fix: int = typer.Option(
        None, "--max-fix", help="最大自动修复尝试次数 (默认: 3)"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="只生成场景规划和代码,不提交 Slurm 任务"
    ),
    file: Path = typer.Option(
        None, "--file", "-f", help="从文件读取 prompt (用于中文等易出问题的输入)", exists=True,
    ),
):
    """直接生成模式 (无需求澄清)"""
    if file:
        prompt = file.read_text(encoding="utf-8").strip()
    if not prompt:
        console.print("[bold red]错误:[/] 请提供 prompt 或通过 --file 指定文件")
        raise typer.Exit(1)
    if partition:
        settings.SLURM_PARTITION = partition
    if max_fix is not None:
        settings.MAX_FIX_ATTEMPTS = max_fix
    if output:
        settings.OUTPUT_FILE = output

    try:
        settings.require_llm_key()
    except ValueError as e:
        console.print(f"[bold red]错误:[/] {e}", markup=False)
        raise typer.Exit(1)

    if dry_run:
        _dry_run(prompt)
        return

    from orchestrator import Orchestrator

    orchestrator = Orchestrator()
    try:
        final_video = orchestrator.run(prompt)
        console.print(f"\n[bold green]成功![/] 输出文件: {final_video}")
    except KeyboardInterrupt:
        console.print("\n[yellow]用户中断 (已清理 Slurm 任务)[/]")
        raise typer.Exit(130)
    except Exception as e:
        console.print(f"\n[bold red]失败:[/] {e}", markup=False)
        raise typer.Exit(1)


@app.command()
def plan(
    prompt: str = typer.Argument(None, help="动画需求的自然语言描述"),
    file: Path = typer.Option(
        None, "--file", "-f", help="从文件读取 prompt", exists=True,
    ),
):
    """只生成场景规划,不执行渲染"""
    if file:
        prompt = file.read_text(encoding="utf-8").strip()
    if not prompt:
        console.print("[bold red]错误:[/] 请提供 prompt 或通过 --file 指定文件")
        raise typer.Exit(1)
    try:
        settings.require_llm_key()
    except ValueError as e:
        console.print(f"[bold red]错误:[/] {e}", markup=False)
        raise typer.Exit(1)

    from agents.planner import PlannerAgent

    planner = PlannerAgent()
    outlines = planner.plan_outline(prompt)
    scenes = [
        planner.plan_detail(o, len(outlines), prompt)
        for o in outlines
    ]

    console.print("\n[bold]场景规划结果:[/]")
    for scene in scenes:
        console.print(f"\n[cyan]Scene {scene.scene_id}:[/] {scene.title}")
        console.print(f"  数学概念: {scene.math_concept}")
        console.print(f"  时长: {scene.duration_seconds}s")
        console.print(f"  目的: {scene.purpose}")


@app.command()
def render(
    file: Path = typer.Argument(..., help="Manim Python 文件路径", exists=True),
    class_name: str = typer.Option(
        "Scene", "--class", "-c", help="Manim Scene 类名 (默认: Scene)"
    ),
    scene_id: int = typer.Option(
        None, "--scene-id", "-s", help="场景 ID, 用于命名 (默认: 0)"
    ),
    wait: bool = typer.Option(
        False, "--wait", "-w", help="等待任务完成并显示进度"
    ),
):
    """直接提交单个 .py 文件到 Slurm 渲染 (跳过 pipeline)"""
    from cluster.slurm import SlurmDispatcher

    sid = scene_id or 0
    settings.SCENES_DIR.mkdir(parents=True, exist_ok=True)
    settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    settings.VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

    dispatcher = SlurmDispatcher()

    try:
        job = dispatcher.submit_scene(
            scene_id=sid,
            python_file=file.resolve(),
            scene_class_name=class_name,
        )
    except Exception as e:
        console.print(f"[bold red]提交失败:[/] {e}", markup=False)
        raise typer.Exit(1)

    console.print(f"[bold green]已提交[/] Job {job.job_id}")
    console.print(f"  脚本: {job.script_path}")
    console.print(f"  输出: {job.log_out}")
    console.print(f"  错误: {job.log_err}")

    if wait:
        success = dispatcher.wait_for_job(job.job_id, sid)
        if success:
            console.print(f"[bold green]渲染完成![/] 视频在 workspace/videos/scene_{sid}/")
        else:
            console.print(f"[bold red]渲染失败[/], 查看日志: {job.log_err}")
            # 显示错误日志
            error_tail = dispatcher.get_error_log(sid, job.job_id)
            if error_tail:
                console.print(f"\n[dim]错误日志 (最后几行):[/]\n{error_tail}", markup=False)
            raise typer.Exit(1)


@app.command(name="version")
def version_cmd():
    """显示版本信息"""
    console.print("kd1-anime v0.2.0")
    console.print("AI Agent 驱动的 Manim 数学动画自动渲染流水线")


def _start_chat(dry_run: bool = False) -> None:
    """启动 TUI 交互会话"""
    from tui import ChatSession

    session = ChatSession(dry_run=dry_run)
    session.run()


def _dry_run(prompt: str) -> None:
    """Dry-run 模式: 只生成规划和代码,不提交渲染"""
    from agents.planner import PlannerAgent
    from agents.coder import CoderAgent
    from agents.reviewer import ReviewerAgent

    planner = PlannerAgent()
    coder = CoderAgent()
    reviewer = ReviewerAgent()

    console.print("[bold yellow]Dry-run 模式: 只生成规划和代码,不提交渲染[/]\n")

    outlines = planner.plan_outline(prompt)
    scenes = [
        planner.plan_detail(o, len(outlines), prompt)
        for o in outlines
    ]

    settings.SCENES_DIR.mkdir(parents=True, exist_ok=True)

    for scene in scenes:
        console.print(f"\n--- Scene {scene.scene_id}: {scene.title} ---")
        code = coder.generate_code(scene)

        result = reviewer.review(code, scene.title)
        if not result.is_valid:
            console.print(f"[yellow]审查反馈:[/] {result.feedback}", markup=False)
            code = coder.generate_code(scene, feedback=result.feedback, previous_code=code)

        code_file = settings.SCENES_DIR / f"scene_{scene.scene_id}.py"
        code_file.write_text(code, encoding="utf-8")
        console.print(f"[green]已保存:[/] {code_file}")

    console.print(f"\n[bold green]Dry-run 完成![/] 共生成 {len(scenes)} 个场景文件")


if __name__ == "__main__":
    app()
