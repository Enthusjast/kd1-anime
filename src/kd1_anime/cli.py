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

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import typer
from rich.console import Console
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from kd1_anime.config import settings
from kd1_anime.eval.metrics import ComparisonResult, EvalResult
from kd1_anime.run_store import (
    RESUME_LLM_STATES,
    RunManifest,
    RunRepository,
    lock_run,
    restore_run_path,
)

app = typer.Typer(
    name="kd1-anime",
    help="AI Agent 驱动的 Manim 数学动画自动渲染流水线",
    add_completion=False,
    invoke_without_command=True,
)
console = Console()
rag_app = typer.Typer(help="管理本地 RAG 知识库", add_completion=False)
app.add_typer(rag_app, name="rag")
cache_app = typer.Typer(help="管理本地 LLM 响应缓存", add_completion=False)
app.add_typer(cache_app, name="cache")


@cache_app.command("status")
def cache_status():
    """查看缓存路径、条目数和调用统计，不显示响应内容。"""

    from kd1_anime.llm_cache import LLMResponseCache

    data = LLMResponseCache().summary()
    console.print(f"路径: {data['path']}", markup=False)
    console.print(f"响应条目: {data['entries']}")
    console.print(f"调用事件: {data['events']}")
    console.print(f"本进程统计: {json.dumps(data['stats'], ensure_ascii=False)}")


@cache_app.command("clear")
def cache_clear(
    yes: bool = typer.Option(False, "--yes", "-y", help="不再询问确认"),
):
    """清除本地 LLM 响应缓存。"""

    from kd1_anime.llm_cache import LLMResponseCache

    cache = LLMResponseCache()
    summary = cache.summary()
    if (
        not yes
        and summary["entries"]
        and not typer.confirm(f"清除 {summary['entries']} 个缓存响应？", default=False)
    ):
        raise typer.Abort()
    console.print(f"已清除 {cache.clear()} 个缓存响应")


@rag_app.command("index")
def rag_index(
    docs_dir: Path | None = typer.Option(None, "--docs-dir", help="Manim 文档目录"),
    examples_dir: Path | None = typer.Option(None, "--examples-dir", help="Manim 示例目录"),
    recipes_dir: Path | None = typer.Option(None, "--recipes-dir", help="Manim Recipe 目录"),
    rebuild: bool = typer.Option(False, "--rebuild", help="忽略已有索引，强制重新计算 Embedding"),
):
    """索引 Manim 文档和示例。"""

    from kd1_anime.rag.service import RagService

    if docs_dir is not None and not docs_dir.is_dir():
        console.print(f"[red]文档目录不存在: {docs_dir}[/]", markup=False)
        raise typer.Exit(1)
    if examples_dir is not None and not examples_dir.is_dir():
        console.print(f"[red]示例目录不存在: {examples_dir}[/]", markup=False)
        raise typer.Exit(1)
    if recipes_dir is not None and not recipes_dir.is_dir():
        console.print(f"[red]Recipe 目录不存在: {recipes_dir}[/]", markup=False)
        raise typer.Exit(1)
    try:
        service = RagService()
        result = service.build_index(
            docs_dir=docs_dir,
            examples_dir=examples_dir,
            recipes_dir=recipes_dir,
            rebuild=rebuild,
        )
    except Exception as exc:
        console.print(f"[red]RAG 索引失败:[/] {exc}", markup=False)
        raise typer.Exit(1) from exc
    action = "重建" if rebuild else "生成"
    console.print(
        f"[green]RAG 索引{action}完成[/]：{result.chunk_count} 个分块，"
        f"{result.source_file_count} 个源文件\n索引: {result.info.index_path}\n"
        f"SHA-256: {result.info.index_sha256}"
    )
    if result.skipped_files:
        console.print(f"[yellow]跳过 {len(result.skipped_files)} 个文件[/]")


@rag_app.command("status")
def rag_status():
    """查看 RAG 配置、索引和服务状态（不发起网络请求）。"""

    from kd1_anime.rag.service import RagService

    data = RagService().runtime_status()
    console.print(f"状态: {data['status']}")
    console.print(f"启用: {'是' if data['enabled'] else '否'}")
    console.print(f"索引: {data['index_path']}")
    console.print(f"文档目录: {data['docs_dir'] or '未配置'}")
    console.print(f"示例目录: {data['examples_dir'] or '未配置'}")
    console.print(f"Recipe 目录: {data.get('recipes_dir') or '未配置'}")
    console.print(f"Embedding: {data['embedding_model'] or '未配置'}")
    console.print(f"Reranker: {data['reranker_model'] or '未配置'}")
    if data["index"]:
        index = data["index"]
        console.print(
            f"分块: {index['chunk_count']}，维度: {index['embedding_dimension']}，"
            f"索引 SHA-256: {index['index_sha256']}"
        )
    if data["index_error"]:
        console.print(f"[yellow]索引错误:[/] {data['index_error']}", markup=False)


@rag_app.command("search")
def rag_search(
    query: str = typer.Argument(..., help="检索问题"),
    top_k: int | None = typer.Option(None, "--top-k", min=1, max=100, help="覆盖默认 Top-K"),
):
    """检索本地知识库并显示参考片段。"""

    from kd1_anime.rag.service import RagService

    try:
        service = RagService()
        result = service.search(query, stage="cli-search", top_k=top_k)
    except Exception as exc:
        console.print(f"[red]RAG 检索失败:[/] {exc}", markup=False)
        raise typer.Exit(1) from exc
    console.print(f"状态: {result.receipt.status}")
    if result.receipt.warning:
        console.print(f"[yellow]提示:[/] {result.receipt.warning}", markup=False)
    if not result.context:
        console.print("没有检索到参考内容。")
        return
    console.print(result.context, markup=False)


def _ensure_llm_api_available() -> None:
    """在进入会话/流水线前验证 LLM 配置、网络和模型路由。"""

    from kd1_anime.agents.base import BaseAgent

    try:
        BaseAgent().check_api_available()
    except Exception as exc:
        console.print(f"LLM API 不可用: {exc}", style="bold red", markup=False)
        raise typer.Exit(1) from exc


def _ensure_visual_llm_api_available(
    *,
    model_override: str | None = None,
    endpoint_required: bool = False,
) -> bool:
    """验证独立视觉端点；流水线探测失败时降级为 unknown，显式评估则失败。"""

    from kd1_anime.eval.visual_eval import VisualEvaluator

    try:
        settings.visual_llm_profile(model_override=model_override).require()
    except Exception as exc:
        console.print(f"视觉 LLM 配置不可用: {exc}", style="bold red", markup=False)
        raise typer.Exit(1) from exc
    try:
        VisualEvaluator(model_override).check_api_available()
    except Exception as exc:
        if endpoint_required:
            console.print(f"视觉 LLM API 不可用: {exc}", style="bold red", markup=False)
            raise typer.Exit(1) from exc
        console.print(
            f"视觉 LLM API 当前不可用；主流水线将继续，视觉结果会标记为 unknown。 原因: {exc}",
            style="yellow",
            markup=False,
        )
        return False
    return True


def _ensure_rag_apis_available() -> None:
    """在启用 RAG 的生成入口前验证索引、Embedding 与 Reranker。"""

    if not settings.RAG_ENABLED:
        return
    from kd1_anime.rag.service import RagService

    try:
        service = RagService()
        require_index = getattr(service, "require_index", None)
        if callable(require_index):
            require_index()
        service.probe()
    except Exception as exc:
        console.print(
            f"RAG 不可用: {exc}",
            style="bold red",
            markup=False,
        )
        raise typer.Exit(1) from exc


def _manifest_requires_generation_apis(manifest: RunManifest) -> bool:
    """判断恢复是否还会进入需要主 LLM/RAG 的阶段。"""

    if manifest.state in RESUME_LLM_STATES:
        return True
    if manifest.status == "dry_run_complete" and any(
        not scene.reviewed or scene.failed or scene.give_up for scene in manifest.scenes.values()
    ):
        return True

    # MONITORING/DISPATCHING 本身通常只轮询 Slurm，但 AutoFix 会在作业
    # 失败后重新调用主模型。恢复时无法预知作业下一次的终态，因此只要
    # 仍有未完成场景且允许自动修复，就提前做 API 预检，避免运行数分钟
    # 后才因缺少凭据失败。
    if manifest.auto_fix and manifest.state in {"DISPATCHING", "MONITORING"}:
        return any(not scene.rendered and not scene.give_up for scene in manifest.scenes.values())
    return False


def _cancel_jobs_before_clean(manifest: RunManifest) -> None:
    """删除陈旧 run 前取消其已知远端 Job，避免留下孤儿任务。"""
    from kd1_anime.cluster.slurm import FAILURE_STATES, SlurmDispatcher

    terminal_statuses = {"COMPLETED", "CANCELLED", *FAILURE_STATES}
    jobs = [
        scene.slurm_job
        for scene in manifest.scenes.values()
        if scene.slurm_job
        and not scene.slurm_job.cancelled
        and scene.slurm_job.status not in terminal_statuses
    ]
    if not jobs:
        return
    if getattr(manifest, "backend", "slurm") == "local":
        # 本地进程句柄不持久化；不能在新进程中凭 Job ID/PID 猜测并删除其
        # 正在写入的 run 目录。正常完成/取消的本地 Job 不会进入 jobs。
        raise RuntimeError("运行包含未完成的本地渲染任务，无法安全确认并删除；请先恢复或停止它")
    dispatcher = SlurmDispatcher()
    try:
        statuses = dispatcher.poll_all_statuses([job.job_id for job in jobs])
    except Exception as exc:
        raise RuntimeError(f"无法确认运行中的 Slurm Job，拒绝删除运行目录: {exc}") from exc
    failed: list[str] = []
    uncertain: list[str] = []
    for job in jobs:
        status = statuses.get(job.job_id, "UNKNOWN")
        if status == "GONE" or status == "COMPLETED" or status == "CANCELLED":
            continue
        if status in FAILURE_STATES:
            continue
        if status == "UNKNOWN":
            uncertain.append(job.job_id)
            continue
        if not dispatcher.cancel_job(job.job_id):
            failed.append(job.job_id)
    if uncertain:
        raise RuntimeError(
            "无法确认以下 Slurm Job 是否仍在运行，拒绝删除运行目录: " + ", ".join(uncertain)
        )
    if failed:
        raise RuntimeError(
            "检测到仍在运行的 Slurm Job，且取消失败，拒绝删除运行目录: " + ", ".join(failed)
        )


def _ensure_generation_apis(*, dry_run: bool) -> None:
    """生成入口统一预检，避免不同 CLI 命令出现不一致行为。"""

    _ensure_llm_api_available()
    if settings.ENABLE_VISUAL_EVAL and not dry_run:
        _ensure_visual_llm_api_available(endpoint_required=False)
    _ensure_rag_apis_available()


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
    ctx.ensure_object(dict)
    ctx.obj["dry_run"] = dry_run
    if api_key:
        settings.LLM_API_KEY = api_key
    if model:
        settings.LLM_MODEL = model

    # 没有子命令时默认启动 chat
    if ctx.invoked_subcommand is None:
        _ensure_generation_apis(dry_run=dry_run)
        _start_chat(dry_run=dry_run)


@app.command()
def chat(
    ctx: typer.Context,
    dry_run: bool = typer.Option(False, "--dry-run", help="只生成代码不提交 Slurm"),
):
    """启动交互式会话 (默认命令)"""
    effective_dry_run = dry_run or bool((ctx.obj or {}).get("dry_run"))
    _ensure_generation_apis(dry_run=effective_dry_run)
    _start_chat(dry_run=effective_dry_run)


@app.command()
def generate(
    ctx: typer.Context,
    prompt: str = typer.Argument(None, help="动画需求的自然语言描述"),
    output: Path = typer.Option(
        None, "--output", "-o", help="输出视频文件路径 (默认: output_final.mp4)"
    ),
    force: bool = typer.Option(False, "--force", help="允许覆盖已存在的输出文件"),
    partition: str = typer.Option(None, "--partition", "-p", help="Slurm 分区"),
    max_fix: int = typer.Option(
        None, "--max-fix", min=0, help="最大自动修复尝试次数 (默认: 5, 上限 20)"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="只生成场景规划和代码,不提交 Slurm 任务"),
    smoke: bool = typer.Option(
        False,
        "--smoke",
        help="显式执行本地低质量 Smoke/Frame Canary（可与 --dry-run 配合）",
    ),
    approve_plan: bool = typer.Option(
        False,
        "--approve-plan",
        help="计划审查后暂停确认；非交互环境下视为显式批准",
    ),
    incremental: str = typer.Option(
        None,
        "--incremental",
        "-i",
        help="增量渲染模式：基于指定的 run-id 只渲染变化的场景",
    ),
    resume: str = typer.Option(
        None,
        "--resume",
        "-r",
        help="恢复中断的运行：指定 run-id 继续生成",
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
    plan_file: Path = typer.Option(
        None,
        "--plan",
        help="从结构化计划 JSON 继续生成（不能与 prompt、--file、--resume、--incremental 混用）",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    backend: str = typer.Option(
        None,
        "--backend",
        help="渲染后端：slurm（默认）或 local（本地前台渲染）",
    ),
):
    """直接生成模式 (无需求澄清)"""
    dry_run = dry_run or bool((ctx.obj or {}).get("dry_run"))
    if file:
        prompt = file.read_text(encoding="utf-8").strip()
    if plan_file and (prompt or file or resume or incremental):
        console.print(
            "[bold red]错误:[/] --plan 不能与 prompt、--file、--resume 或 --incremental 混用"
        )
        raise typer.Exit(1)
    if backend and backend not in {"slurm", "local"}:
        console.print("[bold red]错误:[/] --backend 只能是 slurm 或 local", markup=False)
        raise typer.Exit(1)
    if resume and backend:
        console.print(
            "[bold red]错误:[/] resume 必须使用运行清单中的渲染后端，不能覆盖 --backend",
            markup=False,
        )
        raise typer.Exit(1)
    if not prompt and not resume and not plan_file:
        console.print(
            "[bold red]错误:[/] 请提供 prompt 或通过 --file 指定文件\n使用 kd1-anime plan --help 查看帮助"
        )
        raise typer.Exit(1)
    try:
        if partition:
            settings.SLURM_PARTITION = partition
        if max_fix is not None:
            settings.MAX_FIX_ATTEMPTS = max_fix
        if output:
            settings.OUTPUT_FILE = output
        # 不要让 Typer 的默认 False 覆盖 .env 中显式配置的 true；只有用户
        # 明确传入 --force 时才开启覆盖。
        if force:
            settings.OVERWRITE_OUTPUT = True
        if backend:
            settings.RENDER_BACKEND = backend
    except ValueError as e:
        console.print(f"[bold red]配置错误:[/] {e}", markup=False)
        raise typer.Exit(1) from e

    try:
        # 纯渲染监控、合并或已完成运行不需要重新调用 LLM；恢复命令应能在
        # 用户暂时未配置 API Key 时继续处理已有代码和远端 Job。
        requires_llm = not resume
        if resume:
            manifest = RunRepository(settings.WORKSPACE_DIR).load(resume)
            # 恢复任务的 dry-run 属性以 manifest 为准；不能因为用户省略
            # --dry-run 就把 dry-run 运行误报为普通视频生成成功。
            dry_run = manifest.dry_run
            requires_llm = _manifest_requires_generation_apis(manifest)
        if requires_llm:
            _ensure_llm_api_available()
        if resume:
            if manifest.visual_eval_profile.enabled and manifest.status not in {
                "completed",
                "dry_run_complete",
            }:
                _ensure_visual_llm_api_available(
                    model_override=manifest.visual_eval_profile.model,
                    endpoint_required=False,
                )
        elif settings.ENABLE_VISUAL_EVAL and not dry_run:
            _ensure_visual_llm_api_available(endpoint_required=False)
        if not resume or requires_llm:
            _ensure_rag_apis_available()
    except (OSError, ValueError) as e:
        console.print(f"[bold red]错误:[/] {e}", markup=False)
        raise typer.Exit(1) from e

    from kd1_anime.orchestrator import Orchestrator

    orchestrator = Orchestrator()
    try:
        if resume:
            console.print(f"[cyan]恢复运行[/] {resume}")
            final_video = orchestrator.resume(resume, interactive=True)
        elif plan_file:
            console.print(f"[cyan]从计划文件继续[/] {plan_file}")
            final_video = orchestrator.run_from_plan(
                plan_file,
                dry_run=dry_run,
                output_path=output,
                approve_plan=approve_plan,
                smoke=smoke,
                backend=backend,
            )
        elif incremental:
            console.print(f"[cyan]增量渲染模式[/] 基于运行: {incremental}")
            final_video = orchestrator.run_incremental(
                prompt,
                incremental,
                dry_run=dry_run,
                smoke=smoke,
                backend=backend,
            )
        else:
            run_kwargs = {"dry_run": dry_run}
            if approve_plan:
                run_kwargs["approve_plan"] = True
            if smoke:
                run_kwargs["smoke"] = True
            if backend:
                run_kwargs["backend"] = backend
            final_video = orchestrator.run(prompt, **run_kwargs)

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
    review: bool = typer.Option(
        True,
        "--review/--no-review",
        help="生成计划后执行数学、可实现性和连续性合同审查（默认启用）",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="额外导出结构化计划 JSON（运行清单始终写入 ~/.kd1-anime/workspace）",
    ),
):
    """只生成场景规划，不执行渲染；默认同时审查计划。"""
    if file:
        prompt = file.read_text(encoding="utf-8").strip()
    if not prompt:
        console.print(
            "[bold red]错误:[/] 请提供 prompt 或通过 --file 指定文件\n使用 kd1-anime plan --help 查看帮助"
        )
        raise typer.Exit(1)
    _ensure_generation_apis(dry_run=True)

    try:
        from kd1_anime.orchestrator import Orchestrator

        orchestrator = Orchestrator()
        scenes = orchestrator.plan_only(
            prompt,
            interactive=False,
            review=review,
            preflight=False,
        )
        context = orchestrator._ctx
        if output is not None and context is not None:
            from kd1_anime.run_store import atomic_write_json

            payload = {
                "schema_version": 1,
                "run_id": context.paths.run_id,
                "user_prompt": context.user_prompt,
                "lesson_spec": context.lesson_spec.model_dump(mode="json"),
                "teaching_graph": context.teaching_graph.model_dump(mode="json"),
                "continuity_bible": (
                    context.continuity_bible.model_dump(mode="json")
                    if context.continuity_bible is not None
                    else None
                ),
                "items": [scene.model_dump(mode="json") for scene in scenes],
            }
            atomic_write_json(output.expanduser().resolve(), payload)
            console.print(f"结构化计划已导出: {output.expanduser().resolve()}", markup=False)
    except KeyboardInterrupt as e:
        console.print("\n[yellow]用户中断[/]")
        raise typer.Exit(130) from e
    except Exception as e:
        console.print(f"[bold red]规划失败:[/] {e}", markup=False)
        raise typer.Exit(1) from e

    console.print(
        "\n[bold]场景规划结果[/]"
        + ("（已完成计划审查）" if review else "（预览模式，未执行计划审查）")
        + ":"
    )
    for scene in scenes:
        console.print(f"\n[cyan]Scene {scene.scene_id}:[/] {scene.title}")
        console.print(f"  数学概念: {scene.math_concept}")
        console.print(f"  时长: {scene.duration_seconds}s")
        console.print(f"  目的: {scene.purpose}")


@app.command()
def render(
    ctx: typer.Context,
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
    backend: str = typer.Option(
        None,
        "--backend",
        help="渲染后端：slurm（默认）或 local（本地前台渲染）",
    ),
):
    """直接提交单个 .py 文件到 Slurm 渲染 (跳过 pipeline)"""
    from kd1_anime.agents.validator import validate_manim_code
    from kd1_anime.orchestrator import Orchestrator

    sid = scene_id
    if backend and backend not in {"slurm", "local"}:
        console.print("[bold red]错误:[/] --backend 只能是 slurm 或 local", markup=False)
        raise typer.Exit(1)
    if backend == "local" and not wait and not bool((ctx.obj or {}).get("dry_run")):
        console.print("[bold red]错误:[/] 本地渲染后端必须使用 --wait 前台运行", markup=False)
        raise typer.Exit(1)
    source_code = file.read_text(encoding="utf-8")
    validation = validate_manim_code(source_code, renderer=settings.MANIM_RENDERER)
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
    if bool((ctx.obj or {}).get("dry_run")):
        console.print(
            f"[bold green]Dry-run:[/] 代码校验通过，不提交 Scene {selected_class} 的 Slurm 任务"
        )
        return
    try:
        job, final_video, run_id = Orchestrator().submit_existing_scene(
            source_code,
            selected_class,
            scene_id=sid,
            wait=wait,
            backend=backend,
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
    if scene.failed:
        return "failed"
    if scene.give_up:
        return "give_up"
    if scene.rendered:
        visual_status = getattr(scene, "visual_status", "skipped")
        return "rendered" if visual_status == "skipped" else f"rendered/{visual_status}"
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
    console.print(f"Backend:      {getattr(manifest, 'backend', 'slurm')}", markup=False)
    integrity_errors = manifest.integrity_errors()
    if integrity_errors:
        console.print(
            "清单完整性警告: " + "; ".join(integrity_errors),
            style="yellow",
            markup=False,
        )

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
    verification = Table(title="Verification")
    verification.add_column("Scene")
    verification.add_column("Static")
    verification.add_column("Execution")
    verification.add_column("Visual")
    for scene_id, scene in sorted(manifest.scenes.items()):
        verification.add_row(
            str(scene_id),
            scene.static_verification.status,
            scene.execution_verification.status,
            scene.visual_verification.status,
        )
    console.print(verification)
    if manifest.error:
        console.print("Last error:", style="bold red")
        console.print(manifest.error, markup=False)


@app.command()
def status(
    run_id: str = typer.Argument(None, help="运行 ID；省略时列出最近运行"),
    limit: int = typer.Option(20, "--limit", min=1, max=200, help="列表最多显示条数"),
    json_output: bool = typer.Option(False, "--json", help="以机器可读 JSON 输出"),
):
    """查看持久化运行状态，不调用 LLM 或 Slurm。"""

    repository = RunRepository(settings.WORKSPACE_DIR)
    if run_id:
        try:
            manifest = repository.load(run_id)
            if json_output:
                console.print_json(manifest.model_dump_json())
            else:
                _print_run_details(manifest)
        except Exception as exc:
            console.print(f"[bold red]读取失败:[/] {exc}", markup=False)
            raise typer.Exit(1) from exc
        return

    manifests = repository.list()[:limit]
    if repository.list_errors:
        console.print(
            f"[yellow]警告: {len(repository.list_errors)} 个运行清单无法读取，"
            "可用 status <run-id> 查看具体错误。[/]"
        )
    if not manifests:
        console.print("没有可用的运行记录")
        return
    if json_output:
        console.print_json(
            json.dumps(
                [manifest.model_dump(mode="json") for manifest in manifests],
                ensure_ascii=False,
            )
        )
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
def stats(
    run_id: str = typer.Argument(None, help="运行 ID；省略时汇总最近运行"),
    limit: int = typer.Option(20, "--limit", min=1, max=200, help="最多汇总的运行数"),
    json_output: bool = typer.Option(False, "--json", help="以机器可读 JSON 输出"),
):
    """查看离线生成统计，不调用 LLM 或 Slurm。"""

    from kd1_anime.stats import collect_stats

    try:
        report = collect_stats(settings.WORKSPACE_DIR, run_id, limit=limit)
    except Exception as exc:
        console.print(f"[bold red]统计失败:[/] {exc}", markup=False)
        raise typer.Exit(1) from exc
    if json_output:
        console.print_json(json.dumps(report, ensure_ascii=False))
        return
    runs = report["runs"]
    if not runs:
        console.print("没有可用的运行记录")
        return
    table = Table(title="Pipeline statistics")
    for column in (
        "Run ID",
        "Status",
        "Scenes",
        "Plan reviews",
        "Code reviews",
        "Fixes",
        "Fallbacks",
    ):
        table.add_column(column)
    for item in runs:
        scenes = item["scenes"]
        table.add_row(
            item["run_id"],
            item["status"],
            f"{scenes['rendered']}/{item['scene_count']}",
            str(item["plan_review_attempts"]),
            str(item["review_attempts"]),
            str(item["fix_attempts"]),
            str(scenes["safe_fallback"]),
        )
    console.print(table)
    category_counts = Counter()
    for item in runs:
        category_counts.update(item["failure_categories"])
    if category_counts:
        console.print(
            "失败分类: "
            + ", ".join(f"{key}={value}" for key, value in sorted(category_counts.items()))
        )
    if report["read_errors"]:
        console.print(f"[yellow]有 {len(report['read_errors'])} 个运行清单读取失败[/]")


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
        if _manifest_requires_generation_apis(manifest):
            _ensure_llm_api_available()
            _ensure_rag_apis_available()
        if manifest.visual_eval_profile.enabled and manifest.status not in {
            "completed",
            "dry_run_complete",
        }:
            _ensure_visual_llm_api_available(
                model_override=manifest.visual_eval_profile.model,
                endpoint_required=False,
            )
        if force:
            settings.OVERWRITE_OUTPUT = True
        from kd1_anime.orchestrator import Orchestrator

        final_video = Orchestrator().resume(run_id, interactive=interactive)
    except typer.Exit:
        raise
    except KeyboardInterrupt as exc:
        console.print("\n[yellow]用户中断 (已记录恢复点并清理 Slurm 任务)[/]")
        raise typer.Exit(130) from exc
    except Exception as exc:
        console.print(
            f"[bold red]恢复失败:[/] {exc}\n使用 kd1-anime status 查看可用运行", markup=False
        )
        raise typer.Exit(1) from exc
    if manifest.dry_run:
        console.print(f"[bold green]Dry-run 已完成[/] 运行目录: {repository.run_root(run_id)}")
    else:
        console.print(f"[bold green]运行已完成[/] 输出文件: {final_video}")


@app.command()
def retry(
    run_id: str = typer.Argument(..., help="运行 ID"),
    scene_id: int = typer.Option(..., "--scene-id", "-s", min=1, help="只重试这个场景"),
    interactive: bool = typer.Option(False, "--interactive", help="失败时允许终端询问重试"),
):
    """只重试一个场景，保留同一运行中其它已完成场景。"""

    repository = RunRepository(settings.WORKSPACE_DIR)
    try:
        manifest = repository.load(run_id)
        if manifest.dry_run:
            raise ValueError("dry-run 运行请使用 resume，而不是 retry")
        if _manifest_requires_generation_apis(manifest):
            _ensure_llm_api_available()
            _ensure_rag_apis_available()
        if manifest.visual_eval_profile.enabled:
            _ensure_visual_llm_api_available(
                model_override=manifest.visual_eval_profile.model,
                endpoint_required=False,
            )
        from kd1_anime.orchestrator import Orchestrator

        final_video = Orchestrator().resume(
            run_id,
            interactive=interactive,
            retry_scene_id=scene_id,
        )
    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[bold red]场景重试失败:[/] {exc}", markup=False)
        raise typer.Exit(1) from exc
    console.print(f"[bold green]场景重试完成[/] 输出文件: {final_video}")


@app.command()
def logs(
    run_id: str = typer.Argument(..., help="运行 ID"),
    scene_id: int | None = typer.Option(None, "--scene-id", "-s", min=1),
    lines: int = typer.Option(80, "--lines", "-n", min=1, max=5000),
    stderr: bool = typer.Option(False, "--stderr", help="只显示 stderr"),
):
    """查看某次运行的渲染日志尾部，不调用 LLM 或提交任务。"""

    repository = RunRepository(settings.WORKSPACE_DIR)
    try:
        manifest = repository.load(run_id)
        root = repository.run_root(run_id)
        if scene_id is not None:
            if scene_id not in manifest.scenes:
                raise ValueError(f"运行 {run_id} 不包含 Scene {scene_id}")
            selected = [(scene_id, manifest.scenes[scene_id])]
        else:
            selected = sorted(manifest.scenes.items())
        for current_id, scene in selected:
            paths: list[tuple[str, Path]] = []
            if scene.slurm_job is not None:
                paths.extend(
                    [
                        ("stderr", restore_run_path(root, scene.slurm_job.log_err)),
                        ("stdout", restore_run_path(root, scene.slurm_job.log_out)),
                    ]
                )
            else:
                paths.extend(
                    ("stdout", path)
                    for path in sorted((root / "logs").glob(f"scene_{current_id}_*.out"))
                )
                paths.extend(
                    ("stderr", path)
                    for path in sorted((root / "logs").glob(f"scene_{current_id}_*.err"))
                )
            if stderr:
                paths = [item for item in paths if item[0] == "stderr"]
            if not paths:
                console.print(f"Scene {current_id}: 没有日志", markup=False)
                continue
            for label, path in paths:
                if not path.is_file():
                    continue
                content = path.read_text(encoding="utf-8", errors="replace").splitlines()
                console.print(f"\n--- Scene {current_id} {label}: {path} ---", markup=False)
                console.print("\n".join(content[-lines:]), markup=False)
    except (OSError, ValueError) as exc:
        console.print(f"[red]读取日志失败:[/] {exc}", markup=False)
        raise typer.Exit(1) from exc


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
        console.print(
            f"[bold red]参数错误:[/] {exc}\n示例: kd1-anime clean --older-than 30d", markup=False
        )
        raise typer.Exit(2) from exc
    repository = RunRepository(settings.WORKSPACE_DIR)
    cutoff = datetime.now(timezone.utc) - retention
    candidates = [
        manifest
        for manifest in repository.list()
        if manifest.updated_at <= cutoff and (include_running or manifest.status != "running")
    ]
    if repository.list_errors:
        console.print(
            f"[yellow]警告: {len(repository.list_errors)} 个运行清单损坏，已跳过清理。[/]"
        )
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
            if not root.is_dir() or root.is_symlink():
                skipped += 1
                continue
            with lock_run(root):
                # 候选列表生成与真正删除之间可能有很长的用户确认窗口；
                # 加锁后重新读取，避免误删刚恢复/刚更新的运行。
                current = repository.load(manifest.run_id)
                if current.updated_at > cutoff or (
                    not include_running and current.status == "running"
                ):
                    skipped += 1
                    continue
                # 清理的安全边界是“持锁 + 确认并取消远端作业”，不应因为
                # 某个旧版本的瞬时字段不完整而跳过取消，留下孤儿 Job。
                # 清单完整性问题仍展示给用户，但不会被当成删除许可；下面
                # 的作业状态核对失败时依然会保留整个 run 目录。
                integrity_errors = current.integrity_errors()
                if integrity_errors:
                    console.print(
                        f"{manifest.run_id} 清单存在 {len(integrity_errors)} 个一致性提示，"
                        "清理前仍会先核对并取消远端作业",
                        markup=False,
                        style="yellow",
                    )
                _cancel_jobs_before_clean(current)
                shutil.rmtree(root)
            removed += 1
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            skipped += 1
            console.print(f"跳过 {manifest.run_id}: {exc}", markup=False, style="yellow")
    console.print(f"清理完成: 删除 {removed}, 跳过 {skipped}")


@app.command()
def batch(
    ctx: typer.Context,
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
    backend: str = typer.Option(
        None,
        "--backend",
        help="渲染后端：slurm（默认）或 local（本地前台渲染）",
    ),
):
    """批量并行处理多个动画项目。"""
    dry_run = dry_run or bool((ctx.obj or {}).get("dry_run"))
    if backend and backend not in {"slurm", "local"}:
        console.print("[bold red]错误:[/] --backend 只能是 slurm 或 local", markup=False)
        raise typer.Exit(1)
    _ensure_generation_apis(dry_run=dry_run)

    from kd1_anime.batch import BatchConfig, BatchProcessor

    # 加载 prompts
    try:
        config = BatchConfig(
            max_parallel=max_parallel,
            dry_run=dry_run,
            output_dir=output_dir,
            backend=backend,
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
        interrupted_count = sum(1 for t in tasks if t.status == "interrupted")
        if failed_count > 0:
            console.print(f"[bold red]{failed_count} 个任务失败[/]")
            raise typer.Exit(1)
        if interrupted_count > 0:
            raise typer.Exit(130)

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[bold red]批量处理失败:[/] {e}", markup=False)
        raise typer.Exit(1) from e


@app.command(name="version")
def version_cmd():
    """显示版本信息"""
    # 使用源码包自己的版本常量，避免直接运行 ``python main.py`` 时被
    # 环境中遗留的旧 editable-install 元数据误导；发布 wheel 时该常量
    # 与 pyproject.toml 同步更新。
    from kd1_anime import __version__

    current_version = __version__
    console.print(f"kd1-anime v{current_version}")
    console.print("AI Agent 驱动的 Manim 数学动画自动渲染流水线")


@app.command()
def doctor(
    deep: bool = typer.Option(False, "--deep", help="额外执行 Slurm 客户端版本探测"),
    probe: bool = typer.Option(
        False,
        "--probe",
        help="运行一次最小 FFmpeg、XeLaTeX 和当前 Manim renderer 探测（不会提交 Slurm）",
    ),
    strict: bool = typer.Option(
        True,
        "--strict/--no-strict",
        help="依赖检查失败时返回非零退出码（默认启用）",
    ),
    probe_llm: bool = typer.Option(
        False,
        "--probe-llm",
        help="发送一次最小 LLM 请求验证网络、鉴权和模型路由",
    ),
    probe_visual_llm: bool = typer.Option(
        False,
        "--probe-visual-llm",
        help="发送图片消息验证独立视觉 LLM 的网络、鉴权、模型和多模态能力",
    ),
    probe_rag: bool = typer.Option(
        False,
        "--probe-rag",
        help="发送最小 Embedding 和 Reranker 请求验证 RAG 服务",
    ),
    security_strict: bool = typer.Option(
        False,
        "--security-strict",
        help="要求生成代码使用存在的 Apptainer 镜像并启用 fail-closed 策略",
    ),
):
    """检查环境依赖和配置是否完整。"""
    console.print("[bold]kd1-anime 环境检查[/]\n")

    checks = []

    # 检查 Python 版本
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
            [sys.executable, "-c", "import manim; print(manim.__version__)"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
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
                ["ffmpeg", "-version"], capture_output=True, text=True, timeout=10, check=False
            )
            if result.returncode == 0:
                ffmpeg_version = result.stdout.split("\n")[0][:50]
        except Exception:
            ffmpeg_version = "已找到"
    checks.append(("ffmpeg", ffmpeg_ok, ffmpeg_version))
    ffprobe_path = shutil.which("ffprobe")
    checks.append(("ffprobe", ffprobe_path is not None, ffprobe_path or "未找到"))

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

    if deep and sbatch_ok:
        try:
            result = subprocess.run(
                [sbatch_path, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            checks.append(
                (
                    "Slurm 客户端",
                    result.returncode == 0,
                    (result.stdout or result.stderr).strip(),
                )
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            checks.append(("Slurm 客户端", False, str(exc)))

    if probe:
        _run_doctor_probes(checks)

    # 检查 LLM 配置；默认只做本地配置检查，避免 doctor 在离线环境意外
    # 产生 API 请求。需要真实验证时显式使用 --probe-llm。
    try:
        settings.main_llm_profile().require()
    except ValueError as exc:
        llm_ok = False
        llm_info = str(exc)
    else:
        llm_ok = True
        llm_info = f"已配置模型 {settings.LLM_MODEL}"
    checks.append(("LLM 配置", llm_ok, llm_info))
    if probe_llm:
        if not llm_ok:
            checks.append(("LLM API 探测", False, "配置不完整，未发送请求"))
        else:
            from kd1_anime.agents.base import BaseAgent

            started = datetime.now().timestamp()
            try:
                BaseAgent().check_api_available()
            except Exception as exc:
                checks.append(("LLM API 探测", False, str(exc)))
            else:
                elapsed_ms = (datetime.now().timestamp() - started) * 1000
                checks.append(("LLM API 探测", True, f"可用，耗时 {elapsed_ms:.0f} ms"))

    visual_profile = settings.visual_llm_profile()
    try:
        visual_profile.require()
    except ValueError as exc:
        visual_config_ok = False
        visual_config_detail = str(exc)
    else:
        visual_config_ok = True
        visual_config_detail = f"已配置模型 {visual_profile.model}"
    visual_required = settings.ENABLE_VISUAL_EVAL or probe_visual_llm
    checks.append(
        (
            "视觉 LLM 配置",
            visual_config_ok if visual_required else True,
            visual_config_detail if visual_required else "未启用（独立配置可留空）",
        )
    )
    if probe_visual_llm:
        if not visual_config_ok:
            checks.append(("视觉 LLM API 探测", False, "配置不完整，未发送请求"))
        else:
            from kd1_anime.eval.visual_eval import VisualEvaluator

            started = datetime.now().timestamp()
            try:
                VisualEvaluator().check_api_available()
            except Exception as exc:
                checks.append(("视觉 LLM API 探测", False, str(exc)))
            else:
                elapsed_ms = (datetime.now().timestamp() - started) * 1000
                checks.append(("视觉 LLM API 探测", True, f"可用，耗时 {elapsed_ms:.0f} ms"))

    from kd1_anime.rag.service import RagService

    rag_service = RagService()
    rag_status = rag_service.runtime_status()
    rag_required = settings.RAG_ENABLED or probe_rag
    rag_config_ok = (
        rag_status["embedding_configured"]
        and rag_status["reranker_configured"]
        and (not settings.RAG_ENABLED or rag_status["index"] is not None)
        and not rag_status["index_error"]
    )
    rag_detail = (
        f"{rag_status['status']}，Embedding={rag_status['embedding_model'] or '未配置'}，"
        f"Reranker={rag_status['reranker_model'] or '未配置'}"
        if rag_required
        else "未启用（独立配置可留空）"
    )
    if rag_required and rag_status["index_error"]:
        rag_detail += f"；{rag_status['index_error']}"
    checks.append(
        (
            "RAG 配置",
            rag_config_ok if rag_required else True,
            rag_detail,
        )
    )
    if probe_rag:
        started = datetime.now().timestamp()
        try:
            rag_service.probe()
        except Exception as exc:
            checks.append(("RAG 服务探测", False, str(exc)))
        else:
            elapsed_ms = (datetime.now().timestamp() - started) * 1000
            checks.append(
                ("RAG 服务探测", True, f"Embedding/Reranker 可用，耗时 {elapsed_ms:.0f} ms")
            )
    container_configured = bool(settings.SLURM_CONTAINER_IMAGE)
    if not container_configured:
        isolation_ok = not settings.SLURM_REQUIRE_CONTAINER and not security_strict
        checks.append(
            (
                "生成代码隔离",
                isolation_ok,
                (
                    "SLURM_REQUIRE_CONTAINER=true 但未配置镜像"
                    if settings.SLURM_REQUIRE_CONTAINER
                    else "未配置容器；兼容模式可运行，但共享集群建议启用严格隔离"
                ),
            )
        )
    else:
        container_path = Path(settings.SLURM_CONTAINER_IMAGE).expanduser()
        image_ok = container_path.is_file()
        fail_closed_ok = not security_strict or settings.SLURM_REQUIRE_CONTAINER
        isolation_ok = image_ok and apptainer_ok and fail_closed_ok
        details: list[str] = []
        if not image_ok:
            details.append(f"镜像不存在: {container_path}")
        if not apptainer_ok:
            details.append("未找到 apptainer")
        if not fail_closed_ok:
            details.append("严格模式要求 SLURM_REQUIRE_CONTAINER=true")
        checks.append(
            (
                "生成代码隔离",
                isolation_ok,
                "; ".join(details) if details else "Apptainer 镜像和命令均可用",
            )
        )

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
        if strict:
            raise typer.Exit(1)


def _run_doctor_probes(checks: list[tuple[str, bool, str]]) -> None:
    """执行不提交 Slurm 的本地最小能力探测。"""

    def run_probe(name: str, command: list[str], *, env: dict[str, str] | None = None) -> None:
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=90,
                check=False,
                env=env,
            )
            detail = (result.stdout or result.stderr).strip().splitlines()
            checks.append((name, result.returncode == 0, detail[-1][:120] if detail else "完成"))
        except (OSError, subprocess.TimeoutExpired) as exc:
            checks.append((name, False, str(exc)))

    with tempfile.TemporaryDirectory(prefix="kd1-doctor-") as directory:
        root = Path(directory)

        ffmpeg = shutil.which("ffmpeg")
        ffprobe = shutil.which("ffprobe")
        if ffmpeg and ffprobe:
            video = root / "probe.mp4"
            run_probe(
                "FFmpeg 最小编码探测",
                [
                    ffmpeg,
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=c=black:s=16x16:d=0.2",
                    "-c:v",
                    "mpeg4",
                    "-pix_fmt",
                    "yuv420p",
                    str(video),
                ],
            )
            run_probe(
                "ffprobe 最小视频探测",
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=nw=1:nk=1",
                    str(video),
                ],
            )
        else:
            checks.append(("FFmpeg/ffprobe 最小探测", False, "缺少 ffmpeg 或 ffprobe"))

        xelatex = shutil.which("xelatex")
        if xelatex:
            tex = root / "probe.tex"
            tex.write_text(
                "\\documentclass{article}\n\\begin{document}\nprobe\\end{document}\n",
                encoding="utf-8",
            )
            run_probe(
                "XeLaTeX .xdv 探测",
                [
                    xelatex,
                    "-no-pdf",
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    "-output-directory",
                    str(root),
                    str(tex),
                ],
            )
            checks.append(
                ("XeLaTeX .xdv 产物", (root / "probe.xdv").is_file(), str(root / "probe.xdv"))
            )
        else:
            checks.append(("XeLaTeX .xdv 探测", False, "未找到 xelatex"))

        dvisvgm = shutil.which("dvisvgm")
        checks.append(("dvisvgm", dvisvgm is not None, dvisvgm or "未找到"))

        scene = root / "doctor_scene.py"
        scene.write_text(
            "from manim import *\n"
            'tex_template = TexTemplate(tex_compiler="xelatex", output_format=".xdv")\n'
            'tex_template.add_to_preamble(r"\\usepackage{ctex}")\n'
            "config.tex_template = tex_template\n"
            "class DoctorTexProbe(Scene):\n"
            "    def construct(self):\n"
            '        chinese = Tex("中文测试", tex_template=tex_template)\n'
            '        formula = MathTex(r"a^2+b^2=c^2", tex_template=tex_template)\n'
            "        formula.next_to(chinese, DOWN)\n"
            "        self.add(chinese, formula)\n"
            "        self.wait(0.1)\n",
            encoding="utf-8",
        )
        env = os.environ.copy()
        env["MANIM_RENDERER"] = settings.MANIM_RENDERER
        env["MANIM_OPENGL_PLATFORM"] = settings.MANIM_OPENGL_PLATFORM
        # PyOpenGL 实际读取的是 PYOPENGL_PLATFORM；只设置 Manim 自定义
        # 变量会让 doctor 在 GLX 环境中误报通过，而 Slurm 的 EGL 作业仍失败。
        if settings.MANIM_RENDERER == "opengl":
            env["PYOPENGL_PLATFORM"] = settings.MANIM_OPENGL_PLATFORM
        manim_command = [
            sys.executable,
            "-m",
            "manim",
            "render",
            "--renderer",
            settings.MANIM_RENDERER,
            f"-q{settings.MANIM_QUALITY}",
            "--resolution",
            f"{settings.MANIM_PIXEL_WIDTH},{settings.MANIM_PIXEL_HEIGHT}",
            "--fps",
            str(settings.MANIM_FRAME_RATE),
            "--disable_caching",
            "--media_dir",
            str(root / "manim_media"),
            str(scene),
            "DoctorTexProbe",
        ]
        if settings.MANIM_RENDERER == "opengl":
            # OpenGL 默认可能只播放而不写成品；探针必须覆盖真正的文件输出路径。
            manim_command.insert(-2, "--write_to_movie")
        run_probe(
            f"Manim {settings.MANIM_RENDERER} + XeLaTeX/CJK 最小渲染",
            manim_command,
            env=env,
        )
        try:
            rendered_videos = list((root / "manim_media").rglob("DoctorTexProbe.mp4"))
        except OSError:
            rendered_videos = []
        rendered_video = None
        for path in rendered_videos:
            try:
                if path.is_file() and path.stat().st_size > 0:
                    rendered_video = path
                    break
            except OSError:
                continue
        checks.append(
            (
                "Manim TeX/CJK 最小渲染产物",
                rendered_video is not None,
                str(rendered_video) if rendered_video else "未找到 DoctorTexProbe.mp4",
            )
        )


def _start_chat(dry_run: bool = False) -> None:
    """启动 TUI 交互会话"""
    from kd1_anime.tui import ChatSession

    session = ChatSession(dry_run=dry_run)
    exit_code = session.run()
    if exit_code:
        raise typer.Exit(exit_code)


@app.command()
def evaluate(
    run_id: str | None = typer.Argument(None, help="运行 ID (留空则评估最近的运行)"),
    scene_id: int | None = typer.Option(
        None,
        "--scene-id",
        "-s",
        min=1,
        help="只评估指定运行中某个 Scene 的精确渲染产物",
    ),
    code: str | None = typer.Option(None, "--code", "-c", help="直接评估代码字符串"),
    code_file: Path | None = typer.Option(None, "--code-file", "-f", help="评估代码文件"),
    image: Path | None = typer.Option(None, "--image", "-i", help="评估渲染截图"),
    description: str = typer.Option("", "--desc", "-d", help="动画描述"),
    visual: bool | None = typer.Option(
        None,
        "--visual/--no-visual",
        help="是否进行视觉评估（默认使用 ENABLE_VISUAL_EVAL 配置）",
    ),
    visual_model: str | None = typer.Option(None, "--visual-model", help="视觉评估模型"),
    output: Path | None = typer.Option(None, "--output", "-o", help="输出报告路径"),
    compare: str | None = typer.Option(None, "--compare", help="对比的基准运行 ID"),
    json_output: bool = typer.Option(False, "--json", help="JSON 格式输出"),
):
    """评估动画生成质量

    支持多种评估模式：
    - 评估完整运行: kd1-anime evaluate <run-id>
    - 评估单个场景: kd1-anime evaluate <run-id> --scene-id 2
    - 评估代码: kd1-anime evaluate --code "..." 或 --code-file scene.py
    - 评估截图: kd1-anime evaluate --image screenshot.png
    - 对比运行: kd1-anime evaluate <run-id> --compare <baseline-id>
    """
    from kd1_anime.eval import Evaluator

    # 一个命令只能有一个评估目标。此前多个参数同时传入时会按内部
    # if/elif 顺序静默忽略，容易让用户误评估了另一个对象。
    explicit_targets = [
        ("run-id", run_id is not None),
        ("--code", code is not None),
        ("--code-file", code_file is not None),
        ("--image", image is not None),
    ]
    selected_targets = [name for name, selected in explicit_targets if selected]
    if scene_id is not None and run_id is None:
        console.print("[red]--scene-id 必须与运行 ID 一起使用[/]")
        raise typer.Exit(2)
    if compare is not None:
        if run_id is None:
            console.print("[red]--compare 必须与运行 ID 一起使用[/]")
            raise typer.Exit(2)
        if len(selected_targets) != 1 or selected_targets != ["run-id"]:
            console.print("[red]--compare 不能与 --code、--code-file 或 --image 混用[/]")
            raise typer.Exit(2)
        if scene_id is not None:
            console.print("[red]--scene-id 暂不支持与 --compare 混用[/]")
            raise typer.Exit(2)
    elif len(selected_targets) > 1:
        console.print("[red]评估目标互斥：run-id、--code、--code-file、--image 只能选择一个[/]")
        raise typer.Exit(2)
    if code_file is not None and not code_file.is_file():
        console.print(f"[red]文件不存在: {code_file}[/]")
        raise typer.Exit(1)
    if image is not None and not image.is_file():
        console.print(f"[red]图片不存在: {image}[/]")
        raise typer.Exit(1)

    # 截图评估本身就是视觉评估。即使配置默认关闭视觉评估，也不能让
    # 文档中公开的 `evaluate --image` 进入一个必然抛出“视觉评估已禁用”的路径。
    if scene_id is not None:
        if visual is False:
            console.print("[red]--scene-id 视觉评估不能与 --no-visual 一起使用[/]")
            raise typer.Exit(2)
        visual_enabled = True
    elif image is not None:
        if visual is False:
            console.print("[red]--image 需要启用视觉评估，不能与 --no-visual 一起使用[/]")
            raise typer.Exit(2)
        visual_enabled = True
    elif code is not None or code_file is not None:
        # 纯代码目标没有可信视频/关键帧，不能因为全局默认开启视觉评估
        # 就做一次无关的多模态 API 请求。
        if visual is True or visual_model is not None:
            console.print("[red]--visual/--visual-model 只能用于运行或图片评估[/]")
            raise typer.Exit(2)
        visual_enabled = False
    else:
        visual_enabled = settings.ENABLE_VISUAL_EVAL if visual is None else visual
    if visual_enabled:
        _ensure_visual_llm_api_available(
            model_override=visual_model,
            endpoint_required=True,
        )
    evaluator = Evaluator(
        enable_visual_eval=visual_enabled,
        visual_eval_model=visual_model,
    )

    try:
        # 对比模式
        if compare is not None and run_id is not None:
            console.print(f"[bold]对比运行 {compare} 和 {run_id}[/]")
            comparison = evaluator.compare_runs(compare, run_id)

            if json_output:
                console.print_json(json.dumps(comparison.to_dict(), indent=2))
            else:
                _print_comparison(comparison)
            return

        # 评估代码字符串
        if code is not None:
            console.print("[bold]评估代码质量[/]")
            result = evaluator.evaluate_code(code)

        # 评估代码文件
        elif code_file is not None:
            console.print(f"[bold]评估代码文件: {code_file}[/]")
            code_content = code_file.read_text(encoding="utf-8")
            result = evaluator.evaluate_code(code_content)

        # 评估截图
        elif image is not None:
            console.print(f"[bold]评估视觉效果: {image}[/]")
            result = evaluator.evaluate_visual(image, description)

        # 评估运行
        elif run_id is not None:
            console.print(f"[bold]评估运行: {run_id}[/]")
            if scene_id is not None:
                result = evaluator.evaluate_run_scene(
                    run_id,
                    scene_id,
                    description=description,
                )
            else:
                result = evaluator.evaluate_run(
                    run_id, description=description, enable_visual=visual_enabled
                )

        # 评估最近的运行
        else:
            # 查找最近的运行
            manifests = RunRepository(settings.WORKSPACE_DIR).list()
            if not manifests:
                console.print("[red]没有找到运行记录[/]")
                raise typer.Exit(1)
            recent_run = manifests[0].run_id
            console.print(f"[bold]评估最近的运行: {recent_run}[/]")
            result = evaluator.evaluate_run(
                recent_run, description=description, enable_visual=visual_enabled
            )

        # 输出结果
        if json_output:
            console.print_json(json.dumps(result.to_dict(), indent=2))
        else:
            _print_eval_result(result)

        # 保存报告
        if output:
            result.save(output)
            console.print(f"\n[dim]报告已保存到: {output}[/]")

    except Exception as exc:
        console.print(f"[red]评估失败: {exc}[/]")
        raise typer.Exit(1) from exc


def _print_eval_result(result: EvalResult):
    """打印评估结果"""
    from rich.panel import Panel
    from rich.table import Table

    # 总分面板
    if result.overall_score is None:
        console.print(Panel("[yellow]未知[/]", title="总分", border_style="yellow"))
        if result.errors:
            for category, message in result.errors.items():
                console.print(f"[yellow]{category}:[/] {message}", markup=False)
        return
    score_color = (
        "green" if result.overall_score >= 4 else "yellow" if result.overall_score >= 3 else "red"
    )
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
            score.justification[:50] + "..."
            if len(score.justification) > 50
            else score.justification,
        )

    console.print(table)

    # 摘要
    if result.summary:
        console.print(f"\n[dim]{result.summary}[/]")


def _print_comparison(comparison: ComparisonResult):
    """打印对比结果"""
    diff = comparison.score_diff
    if diff is None:
        diff_color = "yellow"
        diff_str = "未知"
    else:
        diff_color = "green" if diff > 0 else "red" if diff < 0 else "white"
        diff_str = f"+{diff:.2f}" if diff > 0 else f"{diff:.2f}"

    baseline = comparison.baseline_result.overall_score
    current = comparison.current_result.overall_score
    console.print(
        f"\n[bold]基准:[/] {comparison.baseline_run_id} ({baseline:.2f})"
        if baseline is not None
        else f"\n[bold]基准:[/] {comparison.baseline_run_id} (未知)"
    )
    console.print(
        f"[bold]当前:[/] {comparison.current_run_id} ({current:.2f})"
        if current is not None
        else f"[bold]当前:[/] {comparison.current_run_id} (未知)"
    )
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
    failed = False

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
            raise typer.Exit(1)
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"  [red]✗ 连接失败: {e}[/]")
        raise typer.Exit(1) from e

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
            console.print("  [green]✓ JSON 模式正常[/]")
            console.print(f"    解析结果: status={result.status}, message={result.message}")
        except Exception as e:
            console.print(f"  [yellow]⚠ JSON 模式异常: {e}[/]")
            console.print("    建议: 在 .env 中设置 LLM_USE_JSON_MODE=false")
            failed = True

    console.print()
    if failed:
        console.print("[bold yellow]测试完成，但有检查失败[/]")
        raise typer.Exit(1)
    console.print("[bold green]测试完成[/]")


if __name__ == "__main__":
    app()
