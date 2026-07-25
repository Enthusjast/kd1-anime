"""
TUI 交互模块
提供类似 Claude Code 的终端交互体验:
  用户输入 → 需求澄清 → 确认 → 流式生成 (带状态指示)

输入方式:
  使用 prompt_toolkit (multiline=True):
  - Enter 换行, Esc+Enter 或 Alt+Enter 提交
  - 粘贴多行文本不会被截断
"""

import sys

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.prompt import Confirm
from rich.status import Status
from rich.table import Table
from rich.markdown import Markdown
from rich.text import Text

from config import settings

console = Console()

# ---------------------------------------------------------------------------
# 多行输入 — prompt_toolkit (multiline=True)
# Enter 换行, Esc+Enter 或 Alt+Enter 提交, 粘贴多行不截断
# ---------------------------------------------------------------------------

from prompt_toolkit import PromptSession

_prompt_session = PromptSession(multiline=True)


def _read_multiline(prompt_text: str = "") -> str:
    try:
        return _prompt_session.prompt(prompt_text).strip()
    except EOFError:
        return ""
    except KeyboardInterrupt:
        raise

CLARIFIER_SYSTEM_PROMPT = """你是一个专业的数学动画需求分析师.用户想用 Manim 制作数学动画,你的任务是通过对话明确他们的需求.

## 工作流程 — 每轮对话遵循以下步骤

### 第一步: 评估信息缺口
阅读用户的初始描述和所有历史对话,列出:
- 已明确的信息有哪些
- 还缺什么关键信息来生成高质量动画规划
- 不要问用户已经明确说过的事情

### 第二步: 提一个问题
从信息缺口中选择当前最关键的一个提问.
问题必须具体、可回答,不要像问卷调查一样笼统.

### 第三步 (收到回答后): 更新评估
将新信息加入"已明确"列表,重新执行第一步.
如果新回答引出了需要进一步明确的细节,将其加入缺口列表.

## 你需要收集的信息维度 (按需,不必全问)
- 动画目标: 严格跟随文稿 vs 仅表现核心概念 vs 混合 (有画面描述处还原,无描述处自行策划)
- 核心数学/物理概念及其深度
- 目标受众和前置知识水平
- 视频总时长预期
- 对特定动画风格的偏好
- 是否需要特定的颜色方案或视觉主题

## 动画风格参考 (仅在用户询问或需要二选一时使用,不要主动描述风格细节)
- 3Blue1Brown 风: 深色纯色背景, 清晰几何图形, 数学变换动画, 颜色意义编码
- 现代教学风: 简洁浅色背景, 清晰图形, 适度文字
- 炫酷展示风: 强视觉冲击, 粒子光效, 动态镜头

## 结束条件
当你判断信息足够生成完整的动画规划时,输出:
{"READY": true, "prompt": "整合后的完整需求描述,包含所有已确认的细节"}

还需要信息时直接提问,不要输出 JSON.
"""


class Clarifier:
    """需求澄清对话管理"""

    def __init__(self):
        from agents.base import BaseAgent

        class _Agent(BaseAgent):
            name = "Clarifier"

        self.agent = _Agent()
        self.history: list[dict] = [
            {"role": "system", "content": CLARIFIER_SYSTEM_PROMPT},
        ]

    def ask(self, user_input: str) -> str:
        """发送用户输入,获取 LLM 回复 (流式输出, 带重试询问)"""
        self.history.append({"role": "user", "content": user_input})

        while True:
            try:
                console.print("[dim cyan]AI:[/] ", end="")
                response = self.agent.call_llm(messages=self.history, stream=True)
                console.print()
                self.history.append({"role": "assistant", "content": response})
                return response
            except Exception as e:
                console.print(f"\n[yellow]Clarifier LLM 错误: {e}[/]", markup=False)
                try:
                    answer = Confirm.ask(
                        "[bold]再试一次?[/] (y = 重试, n = 退出)",
                        default=True,
                        console=console,
                    )
                except (EOFError, KeyboardInterrupt):
                    answer = False
                if not answer:
                    # 移除刚才加入但未得到回复的 user message
                    self.history.pop()
                    raise RuntimeError("Clarifier 失败, 用户选择退出") from e

    def extract_ready(self, response: str) -> str | None:
        """
        如果 LLM 表示需求已明确, 返回精炼后的 prompt; 否则返回 None.

        健壮解析: 先去 markdown fence, 再用括号配平提取 JSON, 容忍
        散文包裹 / 换行格式化 / 嵌套花括号.
        """
        import json

        text = response.strip()
        # 快速短路: 完全没有大括号, 不可能是 READY 信号
        if "{" not in text:
            return None

        # 复用 BaseAgent 的健壮 JSON 提取 (去 fence + 括号配平)
        json_str = self.agent._extract_json(text)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return None

        if isinstance(data, dict) and data.get("READY"):
            return data.get("prompt", "") or None

        return None


def _setup_terminal() -> None:
    """配置终端以正确处理多字节字符输入."""
    import locale

    # 1. 确保 locale 正确 (Python I/O 据此选择编码)
    try:
        locale.setlocale(locale.LC_ALL, "")
    except locale.Error:
        pass

    # 2. 确保 stdin 使用 UTF-8 (修复损坏的 surrogate 字节)
    if hasattr(sys.stdin, "buffer"):
        try:
            sys.stdin.reconfigure(encoding="utf-8", errors="surrogateescape")
        except (AttributeError, OSError):
            pass


def _clean_input(text: str) -> str:
    """清理终端输入中的损坏字符.

    浏览器终端 backspace 中文时可能产生三种垃圾:
    1. 被插入字面的控制字符 (\\x08 BS, \\x7f DEL, \\x00 NUL)
    2. 无法解码的孤立字节 → Python surrogateescape 保留为 \\udcXX
    3. 残缺字节与相邻字符重组为错误但合法的 Unicode
    """
    import re

    # 删除 ANSI 转义序列
    text = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", text)

    # 删除低值控制字符 (BS, DEL, NUL 等, 保留 \\t \\n)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # surrogateescape 产生的孤立 surrogate → 删除
    text = re.sub(r"[\udc80-\udcff]", "", text)

    # 最后一道: encode→decode 清理残留的无效序列
    text = text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
    text = re.sub(r"�+", "", text)

    return text


class ChatSession:
    """交互式会话管理"""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.clarifier: Clarifier | None = None

    def run(self) -> None:
        """启动完整的交互会话"""
        _setup_terminal()
        try:
            self._show_banner()

            # 检查 API Key (先于任何 Agent 构造)
            if not settings.LLM_API_KEY:
                console.print(
                    Panel(
                        "[bold red]未设置 LLM_API_KEY[/]\n\n"
                        "请通过以下方式之一设置:\n"
                        "  1. .env 文件: LLM_API_KEY=your_key\n"
                        "  2. 环境变量: export LLM_API_KEY=your_key\n"
                        "  3. 命令行参数: --api-key your_key",
                        title="配置错误",
                        border_style="red",
                    )
                )
                return

            # 获取初始需求
            user_prompt = self._get_initial_prompt()
            if not user_prompt:
                return

            # 此刻才构造 Clarifier (Key 已确认存在)
            self.clarifier = Clarifier()

            # 需求澄清 + 确认 (可循环: 不满意就继续讨论)
            while True:
                refined_prompt = self._run_clarification(user_prompt)
                if not refined_prompt:
                    return

                if self._confirm_prompt(refined_prompt):
                    break

                # 用户选 n → 让用户说出哪里不满意, 再喂给 Clarifier
                console.print("[dim]哪里需要调整? (输入后回车)[/]")
                feedback = _read_multiline(">>> ")
                if not feedback or feedback.lower() in ("quit", "exit", "q"):
                    console.print("[dim]已退出[/]")
                    return
                user_prompt = (
                    f"用户对当前需求描述不满意: {feedback}\n\n"
                    f"当前的需求描述是: {refined_prompt}\n\n"
                    f"请根据用户的反馈进一步澄清和完善."
                )

            # 执行生成流水线
            self._run_pipeline(refined_prompt)

        except KeyboardInterrupt:
            console.print("\n[dim]已取消[/]")

    def _show_banner(self) -> None:
        """显示欢迎横幅"""
        banner = Text()
        banner.append("  ╔═══════════════════════════════════════╗\n", style="bold cyan")
        banner.append("  ║       ", style="bold cyan")
        banner.append("kd1-anime", style="bold white")
        banner.append(" · ", style="dim cyan")
        banner.append("Manim 动画生成器", style="bold cyan")
        banner.append("       ║\n", style="bold cyan")
        banner.append("  ╚═══════════════════════════════════════╝", style="bold cyan")
        console.print(banner)
        console.print(
            "  输入你想制作的数学动画描述,我会帮你生成.\n"
            "  输入 [bold cyan]quit[/] 退出.\n",
        )

    def _get_initial_prompt(self) -> str | None:
        """获取用户的初始需求描述

        使用 prompt_toolkit (若可用), 支持多行粘贴和 Alt+Enter 换行.
        回退到原生 input().
        """
        console.print(Rule("描述你的需求", style="dim"))
        console.print()
        console.print("[dim]提示: 可直接粘贴多行文本; Enter 换行, Esc+Enter 提交[/]")
        console.print()
        while True:
            try:
                raw = _read_multiline(">>> ")
            except EOFError:
                console.print("\n[dim]已退出[/]")
                return None
            except KeyboardInterrupt:
                console.print("\n[dim]已取消[/]")
                return None
            user_input = _clean_input(raw).strip()

            if user_input.lower() in ("quit", "exit", "q"):
                console.print("[dim]已退出[/]")
                return None

            if not user_input:
                # 空输入不退出, 重新提示
                continue

            return user_input

    def _run_clarification(self, initial_prompt: str) -> str | None:
        """运行需求澄清对话"""
        max_rounds = getattr(settings, "MAX_CLARIFY_ROUNDS", 6)

        console.print(Rule("需求澄清", style="dim"))
        console.print()

        try:
            # 第一轮: LLM 根据初始描述提问
            response = self.clarifier.ask(initial_prompt)
        except RuntimeError:
            # Clarifier 失败, 用户选择退出 → 直接用初始 prompt 走 pipeline
            console.print(
                "[yellow]跳过需求澄清, 使用原始描述继续.[/]"
            )
            self._show_refined(initial_prompt)
            return initial_prompt

        # 检查是否已经足够明确
        refined = self.clarifier.extract_ready(response)
        if refined:
            self._show_refined(refined)
            return refined

        # 开始多轮对话
        console.print()

        for round_num in range(1, max_rounds + 1):
            try:
                raw = _read_multiline("? ")
            except EOFError:
                console.print("\n[dim]已退出[/]")
                return None
            except KeyboardInterrupt:
                console.print("\n[dim]已取消[/]")
                return None
            user_input = _clean_input(raw).strip()

            if user_input.lower() in ("quit", "exit", "q"):
                console.print("[dim]已退出[/]")
                return None

            if not user_input:
                continue

            try:
                response = self.clarifier.ask(user_input)
            except RuntimeError:
                fallback = (
                    f"{initial_prompt}\n\n澄清过程因 LLM 错误中断,"
                    f"已收集到 {round_num} 轮对话信息."
                )
                console.print(
                    f"\n[yellow]Clarifier 失败, 使用已收集到的信息继续.[/]",
                    markup=False,
                )
                self._show_refined(fallback)
                return fallback

            refined = self.clarifier.extract_ready(response)
            if refined:
                self._show_refined(refined)
                return refined

            console.print()

        # 达到最大轮次仍未 READY
        console.print(
            f"[yellow]已达到最大澄清轮次 ({max_rounds}), 使用当前收集到的信息继续.[/]"
        )
        fallback = f"{initial_prompt}\n\n澄清过程中补充的细节:\n{response}"
        self._show_refined(fallback)
        return fallback

    @staticmethod
    def _show_refined(refined: str) -> None:
        """统一展示精炼后的需求"""
        console.print()
        console.print(
            Panel(
                Markdown(refined),
                title="[bold green]已明确的需求[/]",
                border_style="green",
            )
        )

    def _confirm_prompt(self, refined_prompt: str) -> bool:
        """让用户确认需求"""
        console.print()
        answer = Confirm.ask(
            "[bold]开始生成?[/]",
            default=True,
            console=console,
        )
        if not answer:
            console.print("[dim]已取消. 重新描述需求或输入 quit 退出.[/]")
        return answer

    def _run_pipeline(self, prompt: str) -> None:
        """执行生成流水线,带进度指示"""
        console.print()
        console.print(Rule("开始生成", style="bold magenta"))
        console.print()

        from orchestrator import Orchestrator

        orchestrator = Orchestrator()

        try:
            final_video = orchestrator.run(
                prompt, callback=self._pipeline_callback, dry_run=self.dry_run,
                interactive=True,
            )
            self._show_completion(final_video)
        except KeyboardInterrupt:
            console.print("\n[bold yellow]用户中断,正在清理...[/]")
            raise
        except Exception as e:
            # 错误消息可能包含 [/] 等 Rich markup 标记, 必须关闭 markup
            console.print(f"\n[bold red]生成失败:[/] {e}", markup=False)
            raise

    @staticmethod
    def _escape_markup(text: str) -> str:
        """转义 Rich markup 标记, 防止 LLM/系统输出中的 [] 被误解析"""
        return str(text).replace("[", "\\[")

    @staticmethod
    def _pipeline_callback(event: str, data: dict) -> None:
        """流水线状态回调 — 简洁输出当前步骤"""
        esc = ChatSession._escape_markup
        match event:
            case "stage_start":
                stage = data.get("stage", "")
                match stage:
                    case "planning":
                        console.print(Rule("[bold magenta]场景概要[/]", style="magenta"))
                    case "detailing":
                        console.print(Rule("[bold magenta]导演分镜[/]", style="magenta"))
                    case "coding":
                        console.print(Rule("[bold magenta]代码生成[/]", style="magenta"))
                    case "reviewing":
                        console.print(Rule("[bold magenta]代码审查[/]", style="magenta"))
                    case "dispatching":
                        console.print(Rule("[bold magenta]提交渲染[/]", style="magenta"))
                    case "monitoring":
                        console.print(Rule("[bold magenta]监控渲染[/]", style="magenta"))
                    case "fixing":
                        console.print(Rule("[bold magenta]自动修复[/]", style="magenta"))
                    case "merging":
                        console.print(Rule("[bold magenta]视频拼接[/]", style="magenta"))

            case "plan_complete":
                scenes = data.get("scenes", [])
                table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
                table.add_column("#", style="cyan", justify="center")
                table.add_column("场景", style="green")
                table.add_column("时长", style="magenta", justify="center")
                table.add_column("概念", style="yellow")
                table.add_column("叙事", style="dim")
                for scene in scenes:
                    table.add_row(
                        str(scene.scene_id),
                        scene.title,
                        f"{scene.duration_seconds}s",
                        scene.math_concept,
                        scene.purpose,
                    )
                console.print(table)
                console.print()

            case "scene_detailing":
                scene_id = data.get("scene_id", "?")
                title = data.get("title", "")
                console.print(f"  [dim]▸[/] Scene {scene_id}: [cyan]{esc(title)}[/] 导演分镜...", end="")

            case "scene_coding":
                scene_id = data.get("scene_id", "?")
                title = data.get("title", "")
                console.print(f"  [dim]▸[/] Scene {scene_id}: [green]{esc(title)}[/] 代码生成...", end="")

            case "scene_coded":
                file_path = data.get("file_path", "")
                console.print(f" [bold green]✓[/] [dim]{file_path}[/]")

            case "scene_review_pass":
                scene_id = data.get("scene_id", "?")
                console.print(f"  [dim]▸[/] Scene {scene_id}: [bold green]审查通过 ✓[/]")

            case "scene_review_fail":
                scene_id = data.get("scene_id", "?")
                console.print(f"  [dim]▸[/] Scene {scene_id}: [yellow]需修正[/]")

            case "scene_submitted":
                scene_id = data.get("scene_id", "?")
                job_id = data.get("job_id", "?")
                console.print(f"  [dim]▸[/] Scene {scene_id} → Job {job_id}")

            case "scene_rendered":
                scene_id = data.get("scene_id", "?")
                console.print(f"  [dim]▸[/] Scene {scene_id}: [bold green]渲染成功 ✓[/]")

            case "scene_failed":
                scene_id = data.get("scene_id", "?")
                console.print(f"  [dim]▸[/] Scene {scene_id}: [bold red]渲染失败 ✗[/]")

            case "scene_fixing":
                scene_id = data.get("scene_id", "?")
                attempt = data.get("attempt", 0)
                console.print(f"  [dim]▸[/] Scene {scene_id}: [yellow]修复尝试 {attempt}[/]")

            case "merge_complete":
                path = data.get("path", "")
                size_mb = data.get("size_mb", 0)
                console.print(f"\n  [bold]输出:[/] {path} [dim]({size_mb:.1f} MB)[/]")

    @staticmethod
    def _show_completion(output_path) -> None:
        """显示完成信息"""
        if output_path is None:
            return  # 流水线失败, orchestrator 已输出错误
        console.print()
        console.print(Rule("[bold green]完成[/]", style="green"))
        size_mb = output_path.stat().st_size / (1024 * 1024)
        console.print(
            Panel(
                f"[bold]{output_path}[/]\n"
                f"[dim]{size_mb:.1f} MB[/]",
                title="[bold green]✓ 最终视频[/]",
                border_style="green",
            )
        )
