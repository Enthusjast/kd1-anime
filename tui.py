"""
TUI 交互模块
提供类似 Claude Code 的终端交互体验:
  用户输入 → 需求澄清 → 确认 → 流式生成 (带状态指示)
"""

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.prompt import Prompt, Confirm
from rich.status import Status
from rich.table import Table
from rich.markdown import Markdown
from rich.text import Text

from config import settings

console = Console()

CLARIFIER_SYSTEM_PROMPT = """你是一个专业的数学动画需求分析师.用户想用 Manim 制作数学动画,你的任务是通过对话明确他们的需求.

## 你的工作方式

1. 仔细理解用户的初步需求
2. 每次只问一个最关键的问题来澄清需求
3. 问完问题后等待用户回答,不要自己假设答案
4. 常见需要澄清的方面:
   - 数学概念的深度和范围
   - 动画的风格 (3Blue1Brown 风格? 简洁教学? 炫酷展示?)
   - 目标受众 (高中生? 大学生? 研究人员?)
   - 视频时长预期
   - 是否需要特定的颜色方案
   - 是否有特别想要的视觉效果

## 何时结束澄清

当你收集到足够信息时,输出以下 JSON (不要包裹在代码块中):
{"READY": true, "prompt": "整合后的完整需求描述,包含所有已确认的细节"}

如果还需要更多信息,直接输出你的问题 (不要输出 JSON).
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
        """发送用户输入,获取 LLM 回复 (流式输出, 避免长调用界面冻结)"""
        self.history.append({"role": "user", "content": user_input})

        console.print("[dim cyan]AI:[/] ", end="")
        response = self.agent.call_llm(messages=self.history, stream=True)
        console.print()

        self.history.append({"role": "assistant", "content": response})
        return response

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


class ChatSession:
    """交互式会话管理"""

    def __init__(self):
        # 延迟构造 Clarifier: 它会实例化 BaseAgent (含 OpenAI client),
        # 在 API Key 缺失时不应在构造阶段崩溃, 而是先给友好提示.
        self.clarifier: Clarifier | None = None

    def run(self) -> None:
        """启动完整的交互会话"""
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

            # 需求澄清
            refined_prompt = self._run_clarification(user_prompt)
            if not refined_prompt:
                return

            # 确认需求
            if not self._confirm_prompt(refined_prompt):
                return

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
        """获取用户的初始需求描述"""
        console.print(Rule("描述你的需求", style="dim"))
        user_input = Prompt.ask(
            "\n[bold cyan]>>>[/]",
            console=console,
        ).strip()

        if user_input.lower() in ("quit", "exit", "q"):
            console.print("[dim]已退出[/]")
            return None

        if not user_input:
            console.print("[red]输入不能为空[/]")
            return None

        return user_input

    def _run_clarification(self, initial_prompt: str) -> str | None:
        """运行需求澄清对话"""
        max_rounds = getattr(settings, "MAX_CLARIFY_ROUNDS", 6)

        console.print(Rule("需求澄清", style="dim"))
        console.print()

        # 第一轮: LLM 根据初始描述提问 (流式输出, 自带进度反馈)
        response = self.clarifier.ask(initial_prompt)

        # 检查是否已经足够明确
        refined = self.clarifier.extract_ready(response)
        if refined:
            self._show_refined(refined)
            return refined

        # 开始多轮对话
        console.print()

        for round_num in range(1, max_rounds + 1):
            user_input = Prompt.ask(
                "[bold cyan]?[/]",
                console=console,
            ).strip()

            if user_input.lower() in ("quit", "exit", "q"):
                console.print("[dim]已退出[/]")
                return None

            if not user_input:
                continue

            response = self.clarifier.ask(user_input)

            refined = self.clarifier.extract_ready(response)
            if refined:
                self._show_refined(refined)
                return refined

            console.print()

        # 达到最大轮次仍未 READY: 用最后一轮回复作为需求 (兜底, 避免无限循环)
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
            final_video = orchestrator.run(prompt, callback=self._pipeline_callback)
            self._show_completion(final_video)
        except KeyboardInterrupt:
            console.print("\n[bold yellow]用户中断,正在清理...[/]")
            raise
        except Exception as e:
            console.print(f"\n[bold red]生成失败:[/] {e}")
            raise

    @staticmethod
    def _pipeline_callback(event: str, data: dict) -> None:
        """流水线状态回调 — 简洁输出当前步骤"""
        match event:
            case "stage_start":
                stage = data.get("stage", "")
                match stage:
                    case "planning":
                        console.print(Rule("[bold magenta]场景规划[/]", style="magenta"))
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
                table.add_column("数学概念", style="yellow")
                for scene in scenes:
                    table.add_row(
                        str(scene.scene_id),
                        scene.title,
                        f"{scene.duration_seconds}s",
                        scene.math_concept,
                    )
                console.print(table)
                console.print()

            case "scene_coding":
                scene_id = data.get("scene_id", "?")
                title = data.get("title", "")
                console.print(f"  [dim]▸[/] Scene {scene_id}: [green]{title}[/] ...", end="")

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
